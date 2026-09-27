"""Coverage, fixed-profile and artifact checks for the complete two-stage baseline."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import eval_mmmu as val
import eval_mmmu_pro as pro

PROFILE_PATH = Path(__file__).parent / "configs" / "two_stage4096_v1.json"
RUNS = (("mmmu_val", "mmmu-val", "standard"),
        ("standard-4", "mmmu-pro", "standard-4"),
        ("standard-10", "mmmu-pro", "standard-10"),
        ("vision", "mmmu-pro", "vision"))


def load_profile(path=PROFILE_PATH):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if (profile.get("profile_name") != "two_stage4096_v1"
            or profile.get("sampling_recipe") != val.RECIPE
            or profile.get("mmmu_revision") != val.DATA_REV
            or profile.get("mmmu_pro_revision") != pro.DATA_REV
            or profile.get("open_parser_revision") != val.PARSER_REV):
        raise ValueError("Unknown or incompatible two-stage profile")
    return profile


def validate_profile(profile, args):
    errors = [f"{key}: expected {value!r}, got {getattr(args, key, None)!r}"
              for key, value in profile["locked_arguments"].items()
              if getattr(args, key, None) != value]
    if errors:
        raise ValueError("Fixed evaluation profile mismatch: " + "; ".join(errors))


def check_coverage(rows, benchmark):
    expected = 900 if benchmark == "mmmu-val" else 1730
    ids = [r["id"] for r in rows]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"Expected {expected} unique questions, got {len(ids)} rows / {len(set(ids))} IDs")
    kinds = Counter(r["question_type"] for r in rows)
    if benchmark == "mmmu-val":
        if Counter(r["subject"] for r in rows) != Counter({s: 30 for s in val.SUBJECTS}):
            raise ValueError("MMMU must contain exactly 30 rows in each of the 30 subjects")
        if kinds != Counter({"multiple-choice": 847, "open": 53}):
            raise ValueError(f"Expected 847 MC and 53 open questions: {dict(kinds)}")
    elif kinds != Counter({"multiple-choice": 1730}):
        raise ValueError("MMMU-Pro must contain 1730 MC questions")


def aggregate_rows(rows):
    def group_stats(group):
        n = len(group)
        correct = sum(bool(r["correct"]) for r in group)
        return {"n": n, "correct": correct, "accuracy": correct / n if n else None}
    subjects = [{"subject": subject, **group_stats([r for r in rows if r["subject"] == subject])}
                for subject in sorted({r["subject"] for r in rows})]
    types = {kind: group_stats([r for r in rows if r["question_type"] == kind])
             for kind in sorted({r["question_type"] for r in rows})}
    domains = [{"domain": domain, **group_stats([r for r in rows if r["subject"] in names])}
               for domain, names in val.DOMAIN_CAT2SUB_CAT.items()]
    return {"subjects": subjects, "domains": domains, "question_types": types,
            "micro_accuracy": group_stats(rows)["accuracy"],
            "macro_accuracy": sum(s["accuracy"] for s in subjects)/len(subjects)}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_run(folder, benchmark, setting, require_base=False):
    # Validate actual prediction rows, not just a summary's self-reported n.
    from eval_output_policy import configuration, score_answer, summarize, paired_input_fingerprint
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(**manifest["arguments"])
    profile = load_profile()
    validate_profile(profile, args)
    if manifest.get("status") != "complete" or (args.benchmark, args.setting) != (benchmark, setting):
        raise ValueError(f"Incomplete run or wrong setting: {folder}")
    if manifest.get("evaluation_profile", {}).get("sha256") != val.canonical_sha256(profile):
        raise ValueError("Run does not match the frozen profile")
    if json.loads((folder / "evaluation_profile.json").read_text(encoding="utf-8")) != profile:
        raise ValueError("Stored profile differs from the frozen profile")
    if manifest.get("model_config", {}).get("quantization_config"):
        raise ValueError("Quantized checkpoint is not allowed")
    if require_base and (args.model_path != val.MODEL or args.model_revision != val.MODEL_REV):
        raise ValueError("Assignment baseline requires the pinned official base checkpoint")
    rows = read_jsonl(folder / "predictions.jsonl")
    inputs = read_jsonl(folder / "inputs.jsonl")
    check_coverage(rows, benchmark)
    check_coverage(inputs, benchmark)
    ids = [r["id"] for r in rows]
    if ids != json.loads((folder / "selected_ids.json").read_text(encoding="utf-8")) or ids != [r["id"] for r in inputs]:
        raise ValueError("Selected, audited and predicted IDs differ")
    if manifest.get("source_n") != len(rows) or manifest.get("selected_n") != len(rows):
        raise ValueError("Source/selected counts do not match the full prediction set")
    config = configuration(args)
    config["selected_ids_sha256"] = val.canonical_sha256(ids)
    if manifest.get("evaluation_signature") != {"config": config, "sha256": val.canonical_sha256(config)}:
        raise ValueError("Evaluation signature mismatch")
    for row, entry in zip(rows, inputs):
        if (row["subject"], row["question_type"]) != (entry["subject"], entry["question_type"]):
            raise ValueError(f"Metadata mismatch: {row['id']}")
        if row["input_sha256"] != entry["input_sha256"] or entry["input_sha256"] != paired_input_fingerprint(entry):
            raise ValueError(f"Input fingerprint mismatch: {row['id']}")
        if row["correct"] != score_answer(row["answer"], row["parsed_answer"], row["question_type"]):
            raise ValueError(f"Scoring mismatch: {row['id']}")
        if [stage["stage"] for stage in row["stages"]] != ["reasoning", "final"]:
            raise ValueError(f"Missing generation stage: {row['id']}")
        final = row["stages"][-1]
        if (row["raw_response"] != final["raw_response"] or row["finish_reason"] != final["finish_reason"]
                or row["output_tokens"] != sum(stage["output_tokens"] for stage in row["stages"])):
            raise ValueError(f"Final-stage or token accounting mismatch: {row['id']}")
    for key, value in {**summarize(rows), **aggregate_rows(rows)}.items():
        if summary.get(key) != value:
            raise ValueError(f"Summary does not match predictions: {key}")
    if benchmark == "mmmu-val" and not summary.get("complete_900"):
        raise ValueError("Missing complete_900 flag")
    return manifest, summary, rows


def write_report(outdir, manifest, summary):
    from eval_output_policy import (REASON_INSTRUCTION, SELECT_INSTRUCTION,
                                    OPEN_REASON_INSTRUCTION, OPEN_SELECT_INSTRUCTION)
    args = manifest["arguments"]
    config = manifest["evaluation_signature"]["config"]
    is_val = args["benchmark"] == "mmmu-val"
    lines = ["# MMMU validation baseline" if is_val else f"# MMMU-Pro {args['setting']} baseline", "",
             "Protocol: `two_stage4096_v1`. Separate 4096-token working draft and final answer.", "",
             f"Model: `{args['model_path']}`; revision: `{args['model_revision']}`. BF16; no quantization.",
             f"Data source: `{args.get('data_root') or ('MMMU/MMMU' if is_val else pro.DATASET)}`.",
             f"Dataset revision (Hub source): `{config['dataset_revision']}`. Local sources require provenance verification.",
             f"N={summary['n']}; correct={summary['correct']}; micro accuracy={summary['micro_accuracy']:.4%}; "
             f"subject macro accuracy={summary['macro_accuracy']:.4%}.",
             f"Unparsed={summary['unparsed']}; final length-limited={summary['length_limited']}; "
             f"draft length-limited={summary['reasoning_length_limited']}.",
             "Unparsed answers remain incorrect in the full denominator. Zero final truncation does not mean the draft finished.",
             f"Inference={summary['inference_seconds']/60:.2f} min; end-to-end={summary['total_seconds']/60:.2f} min; "
             f"model load={summary['model_load_seconds']:.2f} s; calls={summary['calls']}.",
             f"GPU sampled device peak: `{json.dumps(summary['gpu'], ensure_ascii=False)}`.", "",
             "## Question types", "", "| Type | N | Correct | Accuracy |", "|---|---:|---:|---:|"]
    for kind, stat in summary["question_types"].items():
        lines.append(f"| {kind} | {stat['n']} | {stat['correct']} | {stat['accuracy']:.2%} |")
    for title, key, label in (("Subjects", "subjects", "subject"), ("Domains", "domains", "domain")):
        lines += ["", f"## {title}", "", "| Group | N | Correct | Accuracy |", "|---|---:|---:|---:|"]
        for stat in summary[key]:
            if stat["n"]:
                lines.append(f"| {stat[label]} | {stat['n']} | {stat['correct']} | {stat['accuracy']:.2%} |")
    lines += ["", "## Prompts and generation", "",
              "Full actual per-item messages and image hashes are in `inputs.jsonl`; both completions and sampling parameters are in `predictions.jsonl`.",
              "Base prompt (its final direct-answer instruction is replaced for the working draft):", "```text",
              config["prompt"], "```", "Open base prompt:", "```text", val.OPEN_TEMPLATE, "```"]
    for name, text in (("MC draft", REASON_INSTRUCTION), ("MC final", SELECT_INSTRUCTION),
                       ("Open draft", OPEN_REASON_INSTRUCTION), ("Open final", OPEN_SELECT_INSTRUCTION)):
        lines += [name + ":", "```text", text, "```"]
    lines += ["", "```json", json.dumps(config, ensure_ascii=False, indent=2), "```", "",
              "Sampling follows [Qwen evaluation reproduction](https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction). "
              "MC final selection additionally restricts the distribution to the item's actual option letters (2–26). "
              "Open final answers are unconstrained short text (128 tokens), scored using the vendored official MMMU open parser/evaluator. "
              "Empty or length-truncated open final answers are incorrect; drafts are never scored as final answers.",
              "4096 is a bounded draft budget selected for this protocol, not a course-mandated value. "
              "MC final budget is 16, context is 16384 including input and both stages. Image range is 1280×28×28 to 5120×28×28. "
              "Batch size is one. All settings stay fixed before/after training; this method comparison is not equal-compute to prior free generation.",
              "", "## Reproduction and environment", "", "```json",
              json.dumps({"arguments": args, "python": manifest["python"], "packages": manifest["packages"],
                          "code_sha256": manifest["code_sha256"], "signature": manifest["evaluation_signature"]["sha256"]},
                         ensure_ascii=False, indent=2, default=str), "```",
              "See environment.txt, requirements.freeze.txt, chat_template.txt and evaluation_profile.json.", "",
              "Prior MMMU-Pro test experiments informed this protocol; disclose them as exploratory test ablations. "
              "This is not an untouched held-out test. Keep subsequent training/tuning on separate development data."]
    if is_val:
        gap = 100 * summary["macro_accuracy"] - 67.4
        lines += ["", "## Course reference and gap analysis (≤1000 characters)", "",
                  f"Course reference: 67.4%; observed difference: {gap:+.2f} percentage points. "
                  "The two-stage protocol and MC constrained selection differ from the reference; this score alone cannot identify a cause.",
                  "TODO: Add an evidence-based ≤1000-character gap analysis after reviewing this run, plus team/student details. "
                  "Do not claim the reference is reproduced or infer a causal OCR deficit from aggregate setting gaps."]
    (Path(outdir) / "report_draft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
