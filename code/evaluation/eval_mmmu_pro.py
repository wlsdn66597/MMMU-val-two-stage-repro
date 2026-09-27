#!/usr/bin/env python3
"""Evaluate one pinned MMMU-Pro test setting with vLLM chat."""
import argparse
import ast
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import sys
import time
import traceback
from mc_parser import PARSER_POLICY, parse_pro

from eval_mmmu import (
    DOMAIN_CAT2SUB_CAT, GpuMonitor, MODEL, MODEL_REV, RECIPE, RECIPE_URL,
    canonical_sha256, command_output, image_as_rgb, write_json,
)


DATASET = "MMMU/MMMU_Pro"
DATA_REV = "563f3e84bb3b90893083a1f039cfa13077f2302b"
PIPELINE_VERSION = "mmmu-pro-v1"
IMAGE_LAYOUT_VERSION = "numbered-images-prefix-v1"
EXPECTED_ROWS = 1730
SETTINGS = {
    "standard-4": "standard (4 options)",
    "standard-10": "standard (10 options)",
    "vision": "vision",
}
STANDARD_DIRECT_PROMPT = "Answer with the option letter from the given choices directly."
VISION_DIRECT_PROMPT = (
    "Answer with the option letter from the given choices directly. "
    "The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of options."
)
PROMPT_SOURCE = "https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml"


def parse_options(value):
    options = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(options, (list, tuple)):
        raise ValueError("options must be a list or serialized list")
    return [str(option) for option in options]


def prompt_for(setting):
    return VISION_DIRECT_PROMPT if setting == "vision" else STANDARD_DIRECT_PROMPT


def pro_parse(raw, choices):
    return parse_pro(raw, choices)


def build_message(ex, setting):
    options = parse_options(ex["options"])
    # Setting names describe benchmark construction, not a row-level invariant.
    # The pinned data includes 5-option standard-4 and up to 12-option
    # standard-10/vision rows. Only the A-Z single-letter representation is assumed.
    if not 2 <= len(options) <= 26:
        raise ValueError(f"{ex['id']}: unsupported option count {len(options)}")
    choices = {chr(65 + index): option for index, option in enumerate(options)}
    if setting == "vision":
        images = [(1, ex.get("image"))]
        text = VISION_DIRECT_PROMPT
    else:
        images = [(index, ex.get(f"image_{index}")) for index in range(1, 8)
                  if ex.get(f"image_{index}") is not None]
        available = {index for index, _ in images}
        reference_text = ex["question"] + "\n" + "\n".join(options)
        references = {int(value) for value in re.findall(r"<image\s+(\d+)>", reference_text, re.I)}
        if not references.issubset(available):
            raise ValueError(f"Missing image reference: {ex['id']}: {references - available}")

        def relabel(value):
            return re.sub(r"<image\s+(\d+)>", r"[Image \1]", value, flags=re.I)

        listed = "\n".join(f"({letter}) {relabel(option)}" for letter, option in choices.items())
        text = f"Question: {relabel(ex['question'])}\n\nChoices:\n{listed}\n\n{STANDARD_DIRECT_PROMPT}"
    if not images or any(image is None for _, image in images):
        raise ValueError(f"No image: {ex['id']}")
    content, audit_content, metadata = [], [], []
    for number, image in images:
        label = {"type": "text", "text": f"Image {number}:"}
        buffer = io.BytesIO()
        image_as_rgb(image).save(buffer, format="PNG")
        png = buffer.getvalue()
        content.extend([label, {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}}])
        audit_content.extend([label, {"type": "image_url", "image_url": {
            "url": f"<image_{number}: PNG omitted>"}}])
        metadata.append({"number": number, "width": image.width, "height": image.height,
                         "png_sha256": hashlib.sha256(png).hexdigest()})
    content.append({"type": "text", "text": text})
    audit_content.append({"type": "text", "text": text})
    return ([{"role": "user", "content": content}], choices, {
        "messages_without_image_bytes": [{"role": "user", "content": audit_content}],
        "images": metadata,
        "option_count": len(options),
        "choices": choices,
    })


def load_profile(path):
    path = Path(path).expanduser().resolve()
    profile = json.loads(path.read_text(encoding="utf-8"))
    required = {"profile_name", "pipeline_version", "dataset_revision", "parser_policy",
                "image_layout_version", "sampling_recipe", "locked_arguments"}
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"Evaluation profile is missing keys: {missing}")
    return profile, path, canonical_sha256(profile)


def validate_profile(profile, args):
    invariants = {
        "pipeline_version": PIPELINE_VERSION,
        "dataset_revision": DATA_REV,
        "parser_policy": PARSER_POLICY,
        "image_layout_version": IMAGE_LAYOUT_VERSION,
        "sampling_recipe": RECIPE,
    }
    mismatches = []
    for key, actual in invariants.items():
        if profile.get(key) != actual:
            mismatches.append(f"profile.{key}: expected {actual!r}, got {profile.get(key)!r}")
    for key, expected in profile["locked_arguments"].items():
        actual = getattr(args, key, None)
        if actual != expected:
            mismatches.append(f"--{key.replace('_', '-')}: expected {expected!r}, got {actual!r}")
    if mismatches:
        raise ValueError("Evaluation profile mismatch:\n  " + "\n  ".join(mismatches))


def evaluation_signature(args, profile_hash=None):
    config = {
        "benchmark": "MMMU-Pro test",
        "setting": args.setting,
        "dataset_revision": DATA_REV,
        "pipeline_version": PIPELINE_VERSION,
        "parser_policy": PARSER_POLICY,
        "image_layout_version": IMAGE_LAYOUT_VERSION,
        "prompt": prompt_for(args.setting),
        "sampling_recipe": RECIPE,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": "bfloat16",
        "quantization": None,
        "max_images_per_prompt": 7,
        "profile_sha256": profile_hash,
    }
    return {"sha256": canonical_sha256(config), "config": config}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=tuple(SETTINGS), required=True)
    parser.add_argument("--model-path", default=MODEL)
    parser.add_argument("--model-revision", default=MODEL_REV)
    parser.add_argument("--data-root", default=DATASET,
                        help="MMMU/MMMU_Pro or the exact pinned HF snapshot directory")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = all 1,730; positive = smoke subset")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=9048)
    parser.add_argument("--min-pixels", type=int, default=1003520)
    parser.add_argument("--max-pixels", type=int, default=4014080)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.85)
    parser.add_argument("--monitor-gpu", default="0")
    parser.add_argument("--evaluation-profile", default=None)
    args = parser.parse_args()
    if not 0 <= args.limit <= EXPECTED_ROWS:
        parser.error(f"limit must be between 0 and {EXPECTED_ROWS}")
    if args.batch_size < 1 or not 0 < args.min_pixels <= args.max_pixels:
        parser.error("invalid batch size or image pixel limits")
    if not 0 < args.max_tokens < args.max_model_len:
        parser.error("max-tokens must be positive and smaller than max-model-len")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be in (0,1)")
    profile_info = None
    if args.evaluation_profile:
        try:
            profile_info = load_profile(args.evaluation_profile)
            validate_profile(profile_info[0], args)
            args.evaluation_profile = str(profile_info[1])
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(str(exc))
    return args, profile_info


def aggregate(rows, seconds_by_subject):
    by_subject = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)
    subjects = []
    for subject in sorted(by_subject):
        selected = by_subject[subject]
        correct = sum(row["correct"] for row in selected)
        subjects.append({"subject": subject, "n": len(selected), "correct": correct,
                         "accuracy": correct / len(selected), "seconds": seconds_by_subject.get(subject, 0.0)})
    subject_map = {row["subject"]: row for row in subjects}
    domains = []
    for domain, names in DOMAIN_CAT2SUB_CAT.items():
        selected = [subject_map[name] for name in names if name in subject_map]
        if not selected:
            continue
        n = sum(row["n"] for row in selected)
        correct = sum(row["correct"] for row in selected)
        domains.append({"domain": domain, "n": n, "correct": correct,
                        "accuracy": correct / n, "seconds": sum(row["seconds"] for row in selected)})
    n = len(rows)
    correct = sum(row["correct"] for row in rows)
    return subjects, domains, correct, correct / n


def run(args, outdir, manifest):
    from datasets import load_dataset

    data_local = Path(args.data_root).expanduser().is_dir()
    data_root = str(Path(args.data_root).expanduser().resolve()) if data_local else args.data_root
    if data_local and Path(data_root).name != DATA_REV:
        raise ValueError("Local data must be the exact pinned MMMU-Pro snapshot directory")
    if not data_local and args.data_root != DATASET:
        raise ValueError(f"Dataset must be {DATASET}")
    kwargs = {"split": "test", "cache_dir": args.cache_dir}
    if not data_local:
        kwargs["revision"] = DATA_REV
    load_start = time.perf_counter()
    dataset = load_dataset(data_root, SETTINGS[args.setting], **kwargs)
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {len(dataset)}")
    ids = list(dataset["id"])
    if len(set(ids)) != EXPECTED_ROWS:
        raise ValueError("MMMU-Pro IDs are not unique")
    manifest["dataset_fingerprint"] = dataset._fingerprint
    manifest["dataset_load_seconds"] = time.perf_counter() - load_start
    selected = dataset.select(range(args.limit)) if args.limit else dataset
    write_json(outdir / "manifest.json", manifest)
    print(f"[data] Verified {EXPECTED_ROWS} unique IDs. Selected {len(selected)} ({args.setting}).", flush=True)
    if args.check_only:
        with (outdir / "inputs.jsonl").open("w", encoding="utf-8") as audit:
            for ex in selected:
                _, _, details = build_message(ex, args.setting)
                audit.write(json.dumps({"id": ex["id"], "subject": ex["subject"], **details},
                                       ensure_ascii=False) + "\n")
        manifest["status"] = "inputs_checked_no_inference"
        write_json(outdir / "manifest.json", manifest)
        return None

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    model_local = Path(args.model_path).expanduser().is_dir()
    model_path = (str(Path(args.model_path).expanduser().resolve()) if model_local else
                  snapshot_download(args.model_path, revision=args.model_revision))
    manifest["resolved_model_path"] = model_path
    manifest["assignment_model"] = (
        (args.model_path == MODEL and args.model_revision == MODEL_REV)
        or (model_local and Path(model_path).name == MODEL_REV)
    )
    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    if config.get("quantization_config"):
        raise ValueError("Quantized checkpoints are not supported")
    model_start = time.perf_counter()
    llm = LLM(model=model_path, tokenizer=model_path, dtype="bfloat16", seed=RECIPE["seed"],
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              limit_mm_per_prompt={"image": 7}, max_num_seqs=args.batch_size,
              mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
              generation_config="vllm")
    model_seconds = time.perf_counter() - model_start
    tokenizer = llm.get_tokenizer()
    (outdir / "chat_template.txt").write_text(tokenizer.get_chat_template(), encoding="utf-8")
    params = SamplingParams(**RECIPE, max_tokens=args.max_tokens, n=1)
    (outdir / "sampling_params.txt").write_text(str(params), encoding="utf-8")
    rows, seconds_by_subject = [], defaultdict(float)
    with (outdir / "predictions.jsonl").open("w", encoding="utf-8") as predictions, \
         (outdir / "inputs.jsonl").open("w", encoding="utf-8") as audit:
        for offset in range(0, len(selected), args.batch_size):
            examples = [selected[index] for index in range(offset, min(offset + args.batch_size, len(selected)))]
            prepared = [build_message(ex, args.setting) for ex in examples]
            messages = [item[0] for item in prepared]
            for ex, (_, _, details) in zip(examples, prepared):
                audit.write(json.dumps({"id": ex["id"], "subject": ex["subject"], **details},
                                       ensure_ascii=False) + "\n")
            audit.flush()
            start = time.perf_counter()
            outputs = llm.chat(messages, sampling_params=params, use_tqdm=False)
            batch_seconds = time.perf_counter() - start
            if len(outputs) != len(examples):
                raise RuntimeError("vLLM output count mismatch")
            per_subject = batch_seconds / len(examples)
            for ex, (_, choices, _), output in zip(examples, prepared, outputs):
                generated = output.outputs[0]
                parsed, info = parse_pro(generated.text, choices, generated.finish_reason)
                correct = parsed is not None and parsed == ex["answer"]
                row = {"id": ex["id"], "subject": ex["subject"],
                       "question_type": "multiple-choice", "answer": ex["answer"],
                       "option_count": len(choices),
                       "choices": choices,
                       "raw_response": generated.text, "parsed_answer": parsed,
                       "correct": bool(correct), "parsing": info,
                       "output_tokens": len(generated.token_ids),
                       "input_tokens": len(output.prompt_token_ids) if output.prompt_token_ids is not None else None,
                       "finish_reason": generated.finish_reason, "stop_reason": generated.stop_reason,
                       "batch_id": f"{args.setting}:{offset}", "batch_size": len(examples),
                       "batch_seconds": batch_seconds, "setting": args.setting}
                predictions.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                rows.append(row)
                seconds_by_subject[ex["subject"]] += per_subject
            predictions.flush()
            if len(rows) % 100 == 0 or len(rows) == len(selected):
                print(f"[progress] {len(rows)}/{len(selected)}", flush=True)
    if len(rows) != len(selected) or len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Incomplete or duplicate predictions")
    subjects, domains, correct, accuracy = aggregate(rows, seconds_by_subject)
    return {"setting": args.setting, "n": len(rows), "correct": correct, "accuracy": accuracy,
            "complete_1730": len(rows) == EXPECTED_ROWS, "subjects": subjects, "domains": domains,
            "option_count_distribution": dict(sorted(Counter(row["option_count"] for row in rows).items())),
            "macro_subject_accuracy": sum(row["accuracy"] for row in subjects) / len(subjects),
            "unparsed": sum("unparsed" in row["parsing"]["mode"] for row in rows),
            "ambiguous_mc": sum(len(set(row["parsing"]["candidates"])) > 1 for row in rows),
            "length_limited": sum(row["finish_reason"] == "length" for row in rows),
            "model_load_seconds": model_seconds}


def make_report(outdir, manifest, summary):
    args = manifest["arguments"]
    lines = [f"# MMMU-Pro {args['setting']} Baseline", "",
             "> Test-only benchmark. Preserve this pre-intervention result and do not tune on item-level errors.", "",
             "## Configuration", "", "| Item | Value |", "|---|---|",
             f"| Dataset revision | `{DATA_REV}` |", f"| Model | `{args['model_path']}` |",
             f"| Prompt source | {PROMPT_SOURCE} |", f"| Prompt | {prompt_for(args['setting'])} |",
             f"| Sampling | `{RECIPE}` |", f"| min/max pixels | {args['min_pixels']} / {args['max_pixels']} |",
             f"| max tokens / context | {args['max_tokens']} / {args['max_model_len']} |",
             f"| Batch | {args['batch_size']} |", "", "## Result", "",
             "| Setting | N | Correct | Accuracy |", "|---|---:|---:|---:|",
             f"| {args['setting']} | {summary['n']} | {summary['correct']} | {100*summary['accuracy']:.2f}% |",
             "", "Official MMMU-Pro aggregation is instance-level (micro) accuracy.",
             f"Observed option counts: `{summary['option_count_distribution']}`. Setting names are nominal; scoring uses each row's actual options.", "",
             "## Domains", "", "| Domain | N | Correct | Accuracy |", "|---|---:|---:|---:|"]
    lines += [f"| {row['domain']} | {row['n']} | {row['correct']} | {100*row['accuracy']:.2f}% |"
              for row in summary["domains"]]
    lines += ["", "## Subjects", "", "| Subject | N | Correct | Accuracy |", "|---|---:|---:|---:|"]
    lines += [f"| {row['subject']} | {row['n']} | {row['correct']} | {100*row['accuracy']:.2f}% |"
              for row in summary["subjects"]]
    lines += ["", "## Diagnostics", "",
              f"- Unparsed: {summary['unparsed']}/{summary['n']}",
              f"- Ambiguous: {summary['ambiguous_mc']}/{summary['n']}",
              f"- Length limited: {summary['length_limited']}/{summary['n']}"]
    (outdir / "report_draft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args, profile_info = arguments()
    start = time.perf_counter()
    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    packages = {}
    for name in ("torch", "transformers", "vllm", "datasets", "huggingface-hub", "Pillow", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    profile, profile_path, profile_hash = profile_info if profile_info else (None, None, None)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "arguments": vars(args), "packages": packages, "python": sys.version,
                "platform": platform.platform(),
                "command": shlex.join(["python", "-u", "eval_mmmu_pro.py"] + sys.argv[1:]),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "mc_parser_sha256": hashlib.sha256(Path(__file__).with_name('mc_parser.py').read_bytes()).hexdigest(),
                "data_revision": DATA_REV, "recipe": RECIPE, "recipe_source": RECIPE_URL,
                "prompt": prompt_for(args.setting), "prompt_source": PROMPT_SOURCE,
                "evaluation_profile": ({"name": profile["profile_name"], "path": str(profile_path),
                                        "sha256": profile_hash} if profile else None),
                "evaluation_signature": evaluation_signature(args, profile_hash),
                "environment": {key: os.environ.get(key) for key in (
                    "CUDA_VISIBLE_DEVICES", "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_WORKER_MULTIPROC_METHOD")}}
    write_json(outdir / "manifest.json", manifest)
    if profile:
        write_json(outdir / "evaluation_profile.json", profile)
    (outdir / "requirements.freeze.txt").write_text(
        command_output([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")
    (outdir / "environment.txt").write_text(command_output(["nvidia-smi"]), encoding="utf-8")
    monitor = GpuMonitor(args.monitor_gpu)
    monitor.thread.start()
    try:
        summary = run(args, outdir, manifest)
        gpu = monitor.stop()
        total = time.perf_counter() - start
        if summary is not None:
            summary.update({"gpu": gpu, "total_seconds": total})
            write_json(outdir / "summary.json", summary)
            make_report(outdir, manifest, summary)
            manifest["status"] = "complete" if summary["complete_1730"] else "development_subset_complete"
            print(f"[done] {summary['n']} questions, accuracy {100*summary['accuracy']:.2f}%", flush=True)
        manifest.update({"gpu": gpu, "total_seconds": total})
        write_json(outdir / "manifest.json", manifest)
        print(f"[output] {outdir}", flush=True)
    except BaseException as exc:
        manifest.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
                         "gpu": monitor.stop(), "total_seconds": time.perf_counter() - start})
        write_json(outdir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
