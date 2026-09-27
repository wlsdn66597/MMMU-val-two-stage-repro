#!/usr/bin/env python3
"""Auditable output policies, including the complete two-stage MMMU baseline."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import eval_mmmu as val
import eval_mmmu_pro as pro
from mc_parser import PARSER_POLICY, parse_pro
from vendor.mmmu_eval_utils import parse_open_response, eval_open, eval_multi_choice

MODES = ("free", "constrained", "two-stage")
REASON_INSTRUCTION = (
    "Work out the problem briefly, using the supplied images. Identify the relevant visual "
    "evidence and perform only the necessary calculations. Avoid repeating alternatives. "
    "This is a working draft; a separate step will select the final option."
)
SELECT_INSTRUCTION = (
    "Now select the single best option for the ORIGINAL question using the original images "
    "and the working draft above. The draft may be incomplete or incorrect; do not blindly "
    "copy it. Output only one valid option letter."
)
OPEN_REASON_INSTRUCTION = REASON_INSTRUCTION.replace("select the final option", "produce the final answer")
OPEN_SELECT_INSTRUCTION = (
    "Now answer the ORIGINAL question using the original images and the working draft above. "
    "The draft may be incomplete or incorrect; do not blindly copy it. "
    "Output only the concise final answer, including units if needed. Do not include an explanation."
)


def score_answer(gold, parsed, question_type):
    if parsed is None:
        return False
    return bool(eval_open(gold, parsed) if question_type == "open" else eval_multi_choice(gold, parsed))


def paired_input_fingerprint(details):
    """Fields recorded by both legacy MMMU-Pro runs and new policy runs."""
    core = {key: details[key] for key in
            ("messages_without_image_bytes", "images", "option_count") if key in details}
    if "messages_without_image_bytes" not in core or "images" not in core:
        raise ValueError("Input audit is missing messages or images")
    return val.canonical_sha256(core)


def select_indices(ids, types, count, seed, include_open=False):
    """Outcome-independent, stable sample; preserve the legacy MC-only default."""
    allowed = {"multiple-choice", "open"} if include_open else {"multiple-choice"}
    indices = [i for i, kind in enumerate(types) if kind in allowed]
    indices.sort(key=lambda i: hashlib.sha256(f"{seed}:{ids[i]}".encode()).hexdigest())
    return indices[:count] if count else indices


def stage_messages(messages, suffix, instruction=REASON_INSTRUCTION):
    result = deepcopy(messages)
    last = result[-1]["content"][-1]
    if last.get("type") != "text" or not last["text"].endswith(suffix):
        raise ValueError("Cannot locate the direct instruction; refusing to alter input silently")
    last["text"] = last["text"][:-len(suffix)] + instruction
    return result


def generate_policy(llm, messages, choices, mode, suffix, args, params_class, structured_class,
                    question_type="multiple-choice"):
    """Never pass labels to generation. Keep both stages, including truncated drafts."""
    letters = list(choices)
    is_open = question_type == "open"
    if question_type not in ("multiple-choice", "open"):
        raise ValueError(f"Unknown question type: {question_type}")
    if not is_open and (not 2 <= len(letters) <= 26 or letters != [chr(65+i) for i in range(len(letters))]):
        raise ValueError("Expected contiguous A-Z option letters")
    stages = []

    def call(conversation, budget, constrained, name):
        kwargs = dict(val.RECIPE, n=1, max_tokens=budget)
        if constrained:
            kwargs["structured_outputs"] = structured_class(choice=letters)
        params = params_class(**kwargs)
        start = time.perf_counter()
        outputs = llm.chat(conversation, sampling_params=params, use_tqdm=False)
        seconds = time.perf_counter() - start
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError("Expected one request and one completion")
        output = outputs[0]
        generated = output.outputs[0]
        if (output.prompt_token_ids is not None and hasattr(args, "max_model_len")
                and len(output.prompt_token_ids) + budget > args.max_model_len):
            raise RuntimeError("Requested output budget does not fit the context; refusing silent budget reduction")
        stages.append({"stage": name, "raw_response": generated.text,
                       "requested_max_tokens": budget,
                       "output_tokens": len(generated.token_ids),
                       "input_tokens": len(output.prompt_token_ids) if output.prompt_token_ids is not None else None,
                       "finish_reason": generated.finish_reason, "stop_reason": generated.stop_reason,
                       "seconds": seconds, "sampling_params": str(params),
                       "constrained_choices": letters if constrained else None})
        return generated

    if mode == "two-stage":
        draft_messages = stage_messages(messages, suffix, OPEN_REASON_INSTRUCTION if is_open else REASON_INSTRUCTION)
        draft = call(draft_messages, args.reasoning_tokens, False, "reasoning")
        # Retain the original images. There is no second copy of the image payload.
        final_messages = draft_messages + [
            {"role": "assistant", "content": draft.text},
            {"role": "user", "content": OPEN_SELECT_INSTRUCTION if is_open else SELECT_INSTRUCTION}]
        final = call(final_messages, getattr(args, "open_final_tokens", 128) if is_open else args.final_tokens,
                     not is_open, "final")
    else:
        final = call(messages, args.max_tokens if mode == "free" else
                     (getattr(args, "open_final_tokens", 128) if is_open else args.final_tokens),
                     mode == "constrained" and not is_open, "final")
    if is_open:
        # Only the final response is scored, never the working draft. No guessed recovery.
        parsed = (sorted(parse_open_response(final.text), key=lambda value: (type(value).__name__, str(value)))
                  if final.text.strip() and final.finish_reason != "length" else None)
        parsed = parsed or None
        info = {"mode": "official_open" if parsed is not None else
                ("truncated_unparsed" if final.finish_reason == "length" else "empty_unparsed"),
                "candidates": parsed or []}
    elif mode != "free":
        # Do not let the permissive MC parser hide a broken constraint backend.
        parsed = final.text.strip()
        if parsed not in choices:
            raise RuntimeError(f"Structured output contract violated: {final.text!r}")
        info = {"mode": "exact_letter", "candidates": [parsed]}
    else:
        parsed, info = parse_pro(final.text, choices, final.finish_reason)
    return {"raw_response": final.text, "parsed_answer": parsed, "parsing": info,
            "finish_reason": final.finish_reason, "stop_reason": final.stop_reason,
            "output_tokens": sum(stage["output_tokens"] for stage in stages),
            "input_tokens": sum(stage["input_tokens"] for stage in stages)
                if all(stage["input_tokens"] is not None for stage in stages) else None,
            "final_output_tokens": stages[-1]["output_tokens"],
            "batch_seconds": sum(stage["seconds"] for stage in stages),
            "reasoning_length_limited": any(stage["stage"] == "reasoning" and
                                            stage["finish_reason"] == "length" for stage in stages),
            "any_stage_length_limited": any(stage["finish_reason"] == "length" for stage in stages),
            "stages": stages}


def load_selection(args):
    from datasets import load_dataset
    selection, fingerprints, seen = [], {}, set()
    data_root = getattr(args, "data_root", None) or ("MMMU/MMMU" if args.benchmark == "mmmu-val" else pro.DATASET)
    revision = val.DATA_REV if args.benchmark == "mmmu-val" else pro.DATA_REV
    kwargs = {"revision": revision} if not Path(data_root).expanduser().is_dir() else {}
    if getattr(args, "require_full", False):
        official = "MMMU/MMMU" if args.benchmark == "mmmu-val" else pro.DATASET
        if (kwargs and data_root != official) or (not kwargs and Path(data_root).expanduser().name != revision):
            raise ValueError(f"Full baseline data must be {official} or its pinned {revision} snapshot directory")
    data_root = str(Path(data_root).expanduser()) if not kwargs else data_root
    if args.benchmark == "mmmu-val":
        for subject in val.SUBJECTS:
            dataset = load_dataset(data_root, subject, split="validation", **kwargs)
            ids = list(dataset["id"])
            if len(ids) != 30 or len(set(ids)) != 30 or seen.intersection(ids):
                raise ValueError(f"Invalid MMMU subject: {subject}")
            seen.update(ids)
            fingerprints[subject] = dataset._fingerprint
            if set(dataset["question_type"]) - {"multiple-choice", "open"}:
                raise ValueError(f"Unknown question type in {subject}")
            indices = select_indices(ids, dataset["question_type"], args.per_subject, args.selection_seed,
                                     getattr(args, "include_open", False))
            selection.extend((dataset, i, subject) for i in indices)
    else:
        dataset = load_dataset(data_root, pro.SETTINGS[args.setting], split="test", **kwargs)
        ids = list(dataset["id"])
        if len(ids) != 1730 or len(set(ids)) != 1730:
            raise ValueError("Expected 1730 unique MMMU-Pro IDs")
        seen.update(ids)
        fingerprints[args.setting] = dataset._fingerprint
        subjects = list(dataset["subject"])
        for subject in sorted(set(subjects)):
            types = ["multiple-choice" if s == subject else "excluded" for s in subjects]
            indices = select_indices(ids, types, args.per_subject, args.selection_seed)
            selection.extend((dataset, i, subject) for i in indices)
    if not selection:
        raise ValueError("Empty selection")
    return selection, fingerprints, len(seen)


def configuration(args):
    config = {"benchmark": "MMMU validation MC" if args.benchmark == "mmmu-val" else "MMMU-Pro test",
            "pipeline_version": "output-policy-v1", "parser_policy": PARSER_POLICY,
            "dataset_revision": val.DATA_REV if args.benchmark == "mmmu-val" else pro.DATA_REV,
            "setting": args.setting, "mode": args.mode, "sampling_recipe": val.RECIPE,
            "max_tokens": args.max_tokens if args.mode == "free" else args.final_tokens,
            "reasoning_tokens": args.reasoning_tokens if args.mode == "two-stage" else 0,
            "reasoning_instruction": REASON_INSTRUCTION if args.mode == "two-stage" else None,
            "select_instruction": SELECT_INSTRUCTION if args.mode == "two-stage" else None,
            "prompt": val.MC_TEMPLATE if args.benchmark == "mmmu-val" else pro.prompt_for(args.setting),
            "constraint": "per-row choice letters" if args.mode != "free" else None,
            "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
            "max_model_len": args.max_model_len, "batch_size": 1,
            "dtype": "bfloat16", "gpu_memory_utilization": args.gpu_memory_utilization,
            "selection_seed": args.selection_seed, "per_subject": args.per_subject,
            "model_path": args.model_path, "model_revision": args.model_revision}
    if getattr(args, "include_open", False):
        config.update(pipeline_version="output-policy-v2", include_open=True,
                      benchmark="MMMU validation" if args.benchmark == "mmmu-val" else "MMMU-Pro test",
                      open_final_tokens=args.open_final_tokens, open_prompt=val.OPEN_TEMPLATE,
                      open_reasoning_instruction=OPEN_REASON_INSTRUCTION,
                      open_select_instruction=OPEN_SELECT_INSTRUCTION,
                      open_parser_revision=val.PARSER_REV, open_truncation_policy="unparsed-and-incorrect",
                      image_layout_version="numbered-images-prefix-v1", quantization=None,
                      require_full=getattr(args, "require_full", False))
        # Checkpoint identity is in manifest.arguments; evaluation must stay fixed after training.
        config.pop("model_path")
        config.pop("model_revision")
    return config


def summarize(rows):
    n = len(rows)
    return {"n": n, "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / n,
            "unparsed": sum(r["parsed_answer"] is None for r in rows),
            "length_limited": sum(r["finish_reason"] == "length" for r in rows),
            "reasoning_length_limited": sum(r["reasoning_length_limited"] for r in rows),
            "any_stage_length_limited": sum(r["any_stage_length_limited"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "inference_seconds": sum(r["batch_seconds"] for r in rows),
            "calls": sum(len(r["stages"]) for r in rows)}


def run(args, outdir, manifest):
    selection, fingerprints, source_n = load_selection(args)
    selected_ids = [ds[i]["id"] for ds, i, _ in selection]
    from baseline_contract import check_coverage
    coverage = [{"id": ds[i]["id"], "subject": subject,
                 "question_type": ds[i]["question_type"] if args.benchmark == "mmmu-val" else "multiple-choice"}
                for ds, i, subject in selection]
    if getattr(args, "require_full", False):
        check_coverage(coverage, args.benchmark)
    manifest["selected_question_types"] = dict(Counter(row["question_type"] for row in coverage))
    config = configuration(args)
    config["selected_ids_sha256"] = val.canonical_sha256(selected_ids)
    manifest.update(dataset_fingerprints=fingerprints, source_n=source_n, selected_n=len(selection),
                    evaluation_signature={"config": config, "sha256": val.canonical_sha256(config)})
    checked = None
    if getattr(args, "checked_inputs", None):
        from baseline_contract import read_jsonl
        checkdir = Path(args.checked_inputs)
        checkmanifest = json.loads((checkdir / "manifest.json").read_text(encoding="utf-8"))
        checked = read_jsonl(checkdir / "inputs.jsonl")
        if (checkmanifest.get("status") != "inputs_checked_no_inference"
                or checkmanifest.get("evaluation_signature") != manifest["evaluation_signature"]
                or checkmanifest.get("dataset_fingerprints") != fingerprints
                or [r["id"] for r in checked] != selected_ids):
            raise ValueError("Preflight does not match this run's exact selection/configuration/data")
    val.write_json(outdir / "selected_ids.json", selected_ids)
    val.write_json(outdir / "manifest.json", manifest)
    print(f"[data] {len(selection)} questions; {manifest['selected_question_types']}; "
          f"{args.benchmark}/{args.setting}; {args.mode}", flush=True)
    llm = None
    model_seconds = 0
    if not args.check_only:
        from huggingface_hub import snapshot_download
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        model_path = (str(Path(args.model_path).expanduser().resolve()) if Path(args.model_path).expanduser().is_dir()
                      else snapshot_download(args.model_path, revision=args.model_revision))
        model_config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
        if model_config.get("quantization_config"):
            raise ValueError("This comparison requires an unquantized checkpoint")
        if getattr(args, "require_full", False) and model_config.get("model_type") != "qwen3_vl":
            raise ValueError("The full baseline requires a Qwen3-VL checkpoint")
        start = time.perf_counter()
        llm = LLM(model=model_path, tokenizer=model_path, dtype="bfloat16", seed=val.RECIPE["seed"],
                  max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                  limit_mm_per_prompt={"image": 7}, max_num_seqs=1,
                  mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
                  generation_config="vllm")
        model_seconds = time.perf_counter() - start
        manifest["resolved_model_path"] = model_path
        manifest["model_config"] = model_config
        (outdir / "chat_template.txt").write_text(llm.get_tokenizer().get_chat_template(), encoding="utf-8")
    rows, image_counts = [], Counter()
    with (outdir / "inputs.jsonl").open("w", encoding="utf-8") as audit, \
         (outdir / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
        for ds, index, subject in selection:
            ex = ds[index]
            question_type = ex["question_type"] if args.benchmark == "mmmu-val" else "multiple-choice"
            is_open = question_type == "open"
            if args.benchmark == "mmmu-val":
                messages, choices, details = val.build_message(ex, "direct")
                suffix = (val.OPEN_TEMPLATE if is_open else val.MC_TEMPLATE).split("\n\n")[-1]
            else:
                messages, choices, details = pro.build_message(ex, args.setting)
                suffix = pro.prompt_for(args.setting)
            if not is_open and ex["answer"] not in choices:
                raise ValueError(f"Gold answer not in option set: {ex['id']}")
            if not is_open and not 2 <= len(choices) <= 26:
                raise ValueError(f"Unsupported option count: {ex['id']}: {len(choices)}")
            if is_open and not ex["answer"]:
                raise ValueError(f"Missing open answer: {ex['id']}")
            image_counts[len(details["images"])] += 1
            # Common input hash excludes labels and output policy; validates paired inputs.
            input_hash = paired_input_fingerprint(details)
            if checked is not None and checked[len(rows)]["input_sha256"] != input_hash:
                raise ValueError(f"Preflight input changed: {ex['id']}")
            entry = {"id": ex["id"], "subject": subject, "question_type": question_type, "choices": choices,
                     **details, "input_sha256": input_hash}
            if args.mode == "two-stage":
                entry["reasoning_messages_without_image_bytes"] = stage_messages(
                    details["messages_without_image_bytes"], suffix,
                    OPEN_REASON_INSTRUCTION if is_open else REASON_INSTRUCTION)
                entry["final_user_instruction"] = OPEN_SELECT_INSTRUCTION if is_open else SELECT_INSTRUCTION
            audit.write(json.dumps(entry, ensure_ascii=False) + "\n")
            audit.flush()
            if args.check_only:
                continue
            generated = generate_policy(llm, messages, choices, args.mode, suffix, args,
                                        SamplingParams, StructuredOutputsParams, question_type)
            row = {"id": ex["id"], "subject": subject, "question_type": question_type,
                   "answer": ex["answer"], "choices": choices, "option_count": len(choices),
                   "mode": args.mode, "setting": args.setting, "input_sha256": input_hash,
                   "batch_id": ex["id"], "batch_size": 1, **generated}
            row["correct"] = score_answer(row["answer"], row["parsed_answer"], question_type)
            predictions.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            predictions.flush()
            rows.append(row)
            print(f"[progress] {len(rows)}/{len(selection)} calls={len(row['stages'])} "
                  f"tokens={row['output_tokens']} seconds={row['batch_seconds']:.1f}", flush=True)
    manifest["image_count_distribution"] = dict(image_counts)
    if args.check_only:
        manifest["status"] = "inputs_checked_no_inference"
        return None
    if len(rows) != len(selection) or len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError("Incomplete or duplicate predictions")
    manifest["status"] = "complete"
    from baseline_contract import aggregate_rows
    return {**summarize(rows), **aggregate_rows(rows), "model_load_seconds": model_seconds, "mode": args.mode,
            "complete_900": args.benchmark == "mmmu-val" and len(rows) == 900,
            "image_count_distribution": dict(image_counts)}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--benchmark", choices=("mmmu-val", "mmmu-pro"), default="mmmu-val")
    parser.add_argument("--setting", choices=("standard", *pro.SETTINGS), default="standard")
    parser.add_argument("--per-subject", type=int, default=4, help="Stable sample per subject; 0 = all eligible")
    parser.add_argument("--include-open", action="store_true")
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--evaluation-profile", type=Path)
    parser.add_argument("--data-root", help="HF dataset ID or local dataset snapshot directory")
    parser.add_argument("--checked-inputs", type=Path, help="Preflight output directory to verify before inference")
    parser.add_argument("--selection-seed", type=int, default=3407)
    parser.add_argument("--max-tokens", type=int, default=8192, help="Free generation ceiling")
    parser.add_argument("--reasoning-tokens", type=int, default=1024)
    parser.add_argument("--final-tokens", type=int, default=16)
    parser.add_argument("--open-final-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--min-pixels", type=int, default=1003520)
    parser.add_argument("--max-pixels", type=int, default=4014080)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.85)
    parser.add_argument("--monitor-gpu", default="0")
    parser.add_argument("--model-path", default=val.MODEL)
    parser.add_argument("--model-revision", default=val.MODEL_REV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    argv = sys.argv[1:] if argv is None else argv
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.evaluation_profile:
        from baseline_contract import load_profile
        profile = load_profile(preliminary.evaluation_profile)
        parser.set_defaults(**profile["locked_arguments"])
    args = parser.parse_args(argv)
    if args.check_only and args.checked_inputs:
        parser.error("--checked-inputs applies to inference, not --check-only")
    if args.evaluation_profile:
        from baseline_contract import validate_profile
        validate_profile(profile, args)
    if args.require_full and (args.per_subject != 0 or not args.include_open):
        parser.error("Full baseline requires --per-subject 0 --include-open")
    if (args.benchmark == "mmmu-val") != (args.setting == "standard"):
        parser.error("mmmu-val uses --setting standard; mmmu-pro requires standard-4, standard-10 or vision")
    if args.per_subject < 0 or not 0 < args.min_pixels <= args.max_pixels or not 0 < args.gpu_memory_utilization < 1:
        parser.error("Invalid sample size, pixels or GPU utilization")
    if min(args.max_tokens, args.reasoning_tokens, args.final_tokens, args.open_final_tokens) < 1:
        parser.error("All output budgets must be positive")
    if max(args.max_tokens, args.reasoning_tokens + max(args.final_tokens, args.open_final_tokens)) >= args.max_model_len:
        parser.error("Context must exceed output budgets; leave room for images, question and stage-2 instruction")
    return args


def main():
    args = arguments()
    outdir = args.output_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    packages = {}
    for name in ("torch", "vllm", "transformers", "datasets", "Pillow", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": vars(args), "packages": packages, "python": sys.version,
                "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("eval_output_policy.py", "eval_mmmu.py", "eval_mmmu_pro.py", "mc_parser.py",
                                             "baseline_contract.py", "vendor/mmmu_eval_utils.py")}}
    if args.evaluation_profile:
        from baseline_contract import load_profile
        profile = load_profile(args.evaluation_profile)
        val.write_json(outdir / "evaluation_profile.json", profile)
        manifest["evaluation_profile"] = {"name": profile["profile_name"], "sha256": val.canonical_sha256(profile)}
    (outdir / "requirements.freeze.txt").write_text(val.command_output([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")
    (outdir / "environment.txt").write_text(val.command_output(["nvidia-smi"]), encoding="utf-8")
    val.write_json(outdir / "manifest.json", manifest)
    monitor = val.GpuMonitor(args.monitor_gpu)
    monitor.thread.start()
    try:
        summary = run(args, outdir, manifest)
        if summary is not None:
            summary.update(gpu=monitor.stop(), total_seconds=time.perf_counter()-start)
            val.write_json(outdir / "summary.json", summary)
            if args.evaluation_profile:
                from baseline_contract import write_report
                write_report(outdir, manifest, summary)
            print(f"[done] accuracy={summary['accuracy']:.4f} unparsed={summary['unparsed']}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(gpu=monitor.stop(), total_seconds=time.perf_counter()-start)
        val.write_json(outdir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
