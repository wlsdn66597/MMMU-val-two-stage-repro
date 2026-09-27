#!/usr/bin/env python3
"""MMMU validation evaluation using vLLM chat; no training or environment changes."""
import argparse
import ast
import base64
from collections import Counter
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
import subprocess
import sys
import threading
import time
import traceback

from PIL import Image
from mc_parser import PARSER_POLICY as MC_PARSER_POLICY, parse_mc

# Must be set before importing vLLM; matches the working server environment.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATA_REV = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
PARSER_REV = "aa9b70da92c2825b3d544d1a11b36856bd92f6c3"
EVALUATION_PIPELINE_VERSION = "mmmu-val-v1"
IMAGE_LAYOUT_VERSION = "numbered-images-prefix-v1"
SUBJECTS = """Accounting Agriculture Architecture_and_Engineering Art Art_Theory
Basic_Medical_Science Biology Chemistry Clinical_Medicine Computer_Science
Design Diagnostics_and_Laboratory_Medicine Economics Electronics Energy_and_Power
Finance Geography History Literature Manage Marketing Materials Math
Mechanical_Engineering Music Pharmacy Physics Psychology Public_Health Sociology""".split()
DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science", "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health",
    ],
    "Humanities and Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech and Engineering": [
        "Agriculture", "Architecture_and_Engineering", "Computer_Science", "Electronics",
        "Energy_and_Power", "Materials", "Mechanical_Engineering",
    ],
}
RECIPE = dict(temperature=0.7, top_p=0.8, top_k=20,
              repetition_penalty=1.0, presence_penalty=1.5, seed=3407)
RECIPE_URL = "https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction"
MC_TEMPLATE = (
    "Question: {question}\n\nChoices:\n{choices}\n\n"
    "Answer with the single best option letter only. Do not include an explanation or any other text."
)
OPEN_TEMPLATE = (
    "Question: {question}\n\n"
    "Answer concisely with only the final answer. Do not include an explanation."
)
COT_MC_TEMPLATE = (
    "Question: {question}\n\nChoices:\n{choices}\n\n"
    "Solve the problem step by step. End with `Final answer: (X)`, replacing X with "
    "the single best option letter."
)
COT_OPEN_TEMPLATE = (
    "Question: {question}\n\n"
    "Solve the problem step by step. End with `Final answer: <answer>`."
)
BRIEF_COT_MC_TEMPLATE = (
    "Question: {question}\n\nChoices:\n{choices}\n\n"
    "Reason briefly in at most three short steps. Then write exactly one final line in the form "
    "`Final answer: (X)`, replacing X with the single best option letter. Stop immediately after that line."
)
BRIEF_COT_OPEN_TEMPLATE = (
    "Question: {question}\n\n"
    "Reason briefly in at most three short steps. Then write exactly one final line in the form "
    "`Final answer: <answer>`. Stop immediately after that line."
)


def prompt_templates(style):
    if style == "direct":
        return MC_TEMPLATE, OPEN_TEMPLATE
    if style == "cot":
        return COT_MC_TEMPLATE, COT_OPEN_TEMPLATE
    if style == "cot-brief":
        return BRIEF_COT_MC_TEMPLATE, BRIEF_COT_OPEN_TEMPLATE
    raise ValueError(f"Unknown prompt style: {style}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def canonical_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_evaluation_profile(path):
    path = Path(path).expanduser().resolve()
    profile = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "profile_name", "pipeline_version", "dataset_revision", "parser_revision",
        "image_layout_version", "sampling_recipe", "locked_arguments",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"Evaluation profile is missing keys: {missing}")
    return profile, path, canonical_sha256(profile)


def validate_evaluation_profile(profile, args):
    """Reject a run when a locked evaluation setting differs from the named profile."""
    invariants = {
        "mc_parser_policy": MC_PARSER_POLICY,
        "pipeline_version": EVALUATION_PIPELINE_VERSION,
        "dataset_revision": DATA_REV,
        "parser_revision": PARSER_REV,
        "image_layout_version": IMAGE_LAYOUT_VERSION,
        "sampling_recipe": RECIPE,
    }
    mismatches = []
    for key, actual in invariants.items():
        if profile.get(key) != actual:
            mismatches.append(f"profile.{key}: expected current code {actual!r}, got {profile.get(key)!r}")
    for key, expected in profile["locked_arguments"].items():
        actual = getattr(args, key, None)
        if actual != expected:
            mismatches.append(f"--{key.replace('_', '-')}: expected {expected!r}, got {actual!r}")
    if mismatches:
        raise ValueError("Evaluation profile mismatch:\n  " + "\n  ".join(mismatches))


def evaluation_signature(args, profile_sha256=None):
    """Settings that must match for a controlled model/checkpoint comparison."""
    mc_prompt, open_prompt = prompt_templates(args.prompt_style)
    config = {
        "mc_parser_policy": MC_PARSER_POLICY,
        "pipeline_version": EVALUATION_PIPELINE_VERSION,
        "dataset_revision": DATA_REV,
        "parser_revision": PARSER_REV,
        "image_layout_version": IMAGE_LAYOUT_VERSION,
        "sampling_recipe": RECIPE,
        "prompt_style": args.prompt_style,
        "mc_prompt": mc_prompt,
        "open_prompt": open_prompt,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": "bfloat16",
        "quantization": None,
        "max_images_per_prompt": 7,
        "profile_sha256": profile_sha256,
    }
    return {"sha256": canonical_sha256(config), "config": config}


def image_as_rgb(img):
    """Preserve palette/alpha images by compositing transparency onto white."""
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if not has_alpha:
        return img.convert("RGB")
    rgba = img.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")


def mc_parse(raw, choices):
    return parse_mc(raw, choices, allow_answer_marker=True)


def build_message(ex, prompt_style="direct"):
    """Numbered images first, followed by the question; labels preserve references."""
    images = [(i, ex.get(f"image_{i}")) for i in range(1, 8)
              if ex.get(f"image_{i}") is not None]
    if not images:
        raise ValueError(f"No images: {ex['id']}")
    question_type = ex["question_type"]
    if question_type not in ("multiple-choice", "open"):
        raise ValueError(f"Unknown question type: {question_type}")
    options = ex.get("options", [])
    if isinstance(options, str):
        options = ast.literal_eval(options) if options.strip() else []
    choices = {chr(65 + i): str(value) for i, value in enumerate(options)}
    if question_type == "multiple-choice" and len(choices) < 2:
        raise ValueError(f"Invalid options: {ex['id']}")
    available = {i for i, _ in images}
    reference_text = ex["question"] + "\n" + "\n".join(choices.values())
    refs = {int(i) for i in re.findall(r"<image\s+(\d+)>", reference_text, re.I)}
    if not refs.issubset(available):
        raise ValueError(f"Missing image reference: {ex['id']}: {refs - available}")
    def relabel(text):
        return re.sub(r"<image\s+(\d+)>", r"[Image \1]", text, flags=re.I)
    mc_template, open_template = prompt_templates(prompt_style)
    template = mc_template if question_type == "multiple-choice" else open_template
    text = template.format(question=relabel(ex["question"]), choices="\n".join(
        f"({key}) {relabel(value)}" for key, value in choices.items()))
    content, metadata, audit_content = [], [], []
    for i, img in images:
        label = {"type": "text", "text": f"Image {i}:"}
        buffer = io.BytesIO()
        image_as_rgb(img).save(buffer, format="PNG")
        png = buffer.getvalue()
        uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        content.extend([label, {"type": "image_url", "image_url": {"url": uri}}])
        metadata.append(dict(number=i, width=img.width, height=img.height,
                             png_sha256=hashlib.sha256(png).hexdigest()))
        audit_content.extend([label, {"type": "image_url", "image_url": {"url": f"<image_{i}: PNG omitted>"}}])
    content.append({"type": "text", "text": text})
    audit_content.append({"type": "text", "text": text})
    return ([{"role": "user", "content": content}], choices,
            {"messages_without_image_bytes": [{"role": "user", "content": audit_content}],
             "images": metadata})


class GpuMonitor:
    """Sample whole-device used VRAM, including vLLM workers and other processes."""
    def __init__(self, gpu):
        self.gpu = gpu
        self.peak_mib = None
        self.samples = 0
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output([
                    "nvidia-smi", "-i", self.gpu, "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"], text=True, timeout=5)
                value = float(output.strip())
                self.peak_mib = max(self.peak_mib or 0, value)
                self.samples += 1
            except Exception as exc:
                self.error = str(exc)
            self.stop_event.wait(1.0)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        return {"peak_device_used_mib_sampled": self.peak_mib, "samples": self.samples,
                "interval_seconds": 1.0, "physical_gpu": self.gpu, "error": self.error,
                "scope": "whole device including other processes; sampled peak, not exact allocation peak"}


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception as exc:
        return f"Unavailable: {exc}"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "--model_path", default=MODEL,
                        help="HF repo ID or local HF snapshot/checkpoint directory")
    parser.add_argument("--model-revision", default=MODEL_REV,
                        help="Set explicitly for a different remote fine-tuned checkpoint")
    parser.add_argument("--data-root", "--data_root", default="MMMU/MMMU",
                        help="MMMU/MMMU or local snapshot directory at the required revision")
    parser.add_argument("--cache-dir", default=None, help="Optional datasets cache, not the snapshot path")
    parser.add_argument("--output-dir", required=True, help="Must not exist; avoids mixing runs")
    parser.add_argument("--limit-per-subject", type=int, default=0,
                        help="0 = full 900; positive = development subset only")
    parser.add_argument("--check-only", action="store_true", help="Check selected inputs without loading model")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-style", choices=("direct", "cot", "cot-brief"), default="direct",
                        help="direct answer-only prompt or step-by-step prompt with an explicit final answer")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=589824)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--monitor-gpu", default="0", help="Physical index/UUID as used by nvidia-smi")
    parser.add_argument("--evaluation-profile", default=None,
                        help="JSON profile whose locked evaluation settings must match this run")
    args = parser.parse_args()
    if not 0 <= args.limit_per_subject <= 30:
        parser.error("limit-per-subject must be between 0 and 30")
    if not 0 < args.max_tokens < args.max_model_len:
        parser.error("max-tokens must be positive and smaller than max-model-len")
    if args.batch_size < 1 or not 0 < args.min_pixels <= args.max_pixels:
        parser.error("invalid batch size or image pixel limits")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be in (0,1)")
    profile_info = None
    if args.evaluation_profile:
        try:
            profile_info = load_evaluation_profile(args.evaluation_profile)
            validate_evaluation_profile(profile_info[0], args)
            args.evaluation_profile = str(profile_info[1])
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(str(exc))
    return args, profile_info


def make_report(outdir, manifest, summary):
    args = manifest["arguments"]
    rows = summary["subjects"]
    domains = summary["domains"]
    question_types = summary["question_types"]
    peak = summary["gpu"]["peak_device_used_mib_sampled"]
    peak_text = f"{peak / 1024:.3f} GiB (1 s sampled whole-device used VRAM)" if peak is not None else "NOT MEASURED"
    complete = summary["complete_900"]
    heading = "MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct"
    lines = [f"# {heading}", "", "> Draft: fill team details, rationale and evidence-based gap analysis."]
    if not complete:
        lines += ["> DEVELOPMENT SUBSET — not a valid 900-question submission."]
    if not manifest["assignment_model"]:
        lines += ["> CUSTOM CHECKPOINT — not the assignment's required base model."]
    prompt_style = args.get("prompt_style", "direct")
    mc_template, open_template = prompt_templates(prompt_style)
    lines += ["", "- Team: TODO", "- Members: TODO", f"- Date: {manifest['started_utc']}",
              "", "## 1. Environment / Reproducibility", "", "| Item | Value |", "|---|---|",
              f"| Model | {args['model_path']} |",
              f"| Resolved local checkpoint | {manifest['resolved_model_path']} |",
              f"| Model revision | {args['model_revision']} |",
              f"| Data revision | {DATA_REV} |", "| Dtype | bfloat16; no quantization |",
              f"| Backend | vLLM {manifest['packages'].get('vllm')} |",
              f"| GPU | See environment.txt; physical device {args['monitor_gpu']} |",
              f"| Peak VRAM | {peak_text} |",
              f"| Wall time | {summary['total_seconds']:.2f} seconds, including setup/data/model load |",
              f"| Model initialization | {summary['model_load_seconds']:.2f} seconds |",
              "| Dependencies | requirements.freeze.txt, environment.txt |",
              "", "Reproduce with a fresh output directory (all other settings unchanged):", "", "```bash",
              manifest["command"], "```", "",
              "Backend rationale: vLLM chat already passed the server smoke test; small batches limit memory use.",
              "", "## 2. Prompt", "", f"Style: `{prompt_style}`.",
              "Source: team-designed template (not claimed to be Qwen's benchmark prompt).",
              "Numbered images precede the text; `<image N>` references become `[Image N]`. No answer/explanation is included.",
              "The pinned model's chat template is saved as chat_template.txt; per-question inputs are in inputs.jsonl.",
              "", "Multiple choice:", "```text", mc_template, "```", "", "Open-ended:", "```text", open_template, "```",
              ("Reason: constrain the response to the directly scored answer and avoid unfinished repetitive reasoning."
               if prompt_style == "direct" else
               "Reason: test whether explicit reasoning improves accuracy while retaining a strict final-answer delimiter."),
              "", "## 3. Generation Settings", "", "### 3.1 Sampling recipe", "",
              "| Parameter | Value |", "|---|---|", "| do_sample | True (vLLM: temperature > 0) |"]
    lines += [f"| {key} | {value} |" for key, value in RECIPE.items()]
    lines += [f"\nSource: {RECIPE_URL}", "", "### 3.2 Generation budget / image resolution", "",
              "| Parameter | Value |", "|---|---|"]
    lines += [f"| {key} | {args[key]} |" for key in (
        "max_tokens", "max_model_len", "min_pixels", "max_pixels", "batch_size", "gpu_memory_utilization")]
    lines += ["", "Initial engineering limits, not claimed optimal. TODO: justify using measured VRAM, runtime and truncation results.",
              "", "## 4. Scoring / Parsing", "",
              f"MMMU source commit: {PARSER_REV}; see THIRD_PARTY.md.",
              "MC precedence: the last explicit `Final answer`, exact letter-only response, parenthesized letter,",
              "standalone letter, then option text (responses longer than five words). Multiple candidates use the last occurrence.",
              "A length-truncated MC response is accepted only when it contains an explicit final answer or is exactly one letter.",
              "All other unparsed responses are WRONG;",
              "the official parser's random-choice fallback is deliberately removed. Open-ended parsing/scoring uses the vendored official code.",
              "Empty responses are WRONG. No LLM judge. All attempted questions remain in the denominator.",
              "", "## 5. Results", "", "### 5.1 Question type", "",
              "| Type | Data Num | Correct | Acc (%) |", "|---|---:|---:|---:|"]
    lines += [f"| {r['question_type']} | {r['n']} | {r['correct']} | {100*r['accuracy']:.2f} |"
              for r in question_types]
    lines += ["", "### 5.2 Core discipline", "",
              "| Discipline | Data Num | Correct | Acc (%) | Time (s) |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['domain']} | {r['n']} | {r['correct']} | {100*r['accuracy']:.2f} | {r['seconds']:.2f} |"
              for r in domains]
    lines += ["", "### 5.3 Subject", "", "| No. | Subject | Data Num | Acc (%) | Time (s) |",
              "|---|---|---:|---:|---:|"]
    lines += [f"| {i} | {r['subject']} | {r['n']} | {100*r['accuracy']:.2f} | {r['seconds']:.2f} |"
              for i, r in enumerate(rows, 1)]
    lines += [f"| | **Overall (macro avg)** | **{summary['n']}** | **{100*summary['macro_accuracy']:.2f}** | |",
              "", "Overall = mean of 30 subject accuracies, computed before rounding.",
              "", "## 6. Official Comparison", ""]
    if complete:
        score = 100 * summary["macro_accuracy"]
        lines += ["| | Overall (%) |", "|---|---|", "| Official course reference | 67.4 |",
                  f"| Our result | {score:.2f} |", f"| Difference (percentage points) | {score-67.4:+.2f} |"]
    else:
        lines += ["Not compared to 67.4: this is a development subset."]
    lines += ["", "## 7. Gap Analysis", "", "TODO: no more than 1,000 characters; observations, counts, controlled comparison and limitations.",
              "", "## 8. Notes / Limitations", "",
              f"- Unparsed/empty responses: {summary['unparsed']}/{summary['n']}.",
              f"- Multiple MC candidates: {summary['ambiguous_mc']}/{summary['n']} (inspect raw outputs).",
              f"- Length-limit terminations: {summary['length_limited']}/{summary['n']} (not all imply a missing final answer).",
              "- VLLM_USE_FLASHINFER_SAMPLER=0: inherited server workaround for missing nvcc.",
              "- VRAM includes all processes on the monitored GPU and may miss peaks between samples.",
              "- Fixed seed is recorded, but changing hardware/backend/batching may change sampled outputs.",
              "- Per-subject times exclude shared model initialization; see summary.json."]
    (outdir / "report_draft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args, outdir, manifest):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from vendor.mmmu_eval_utils import parse_open_response, eval_open, eval_multi_choice

    data_local = Path(args.data_root).expanduser().is_dir()
    data_root = str(Path(args.data_root).expanduser().resolve()) if data_local else args.data_root
    if data_local and Path(data_root).name != DATA_REV:
        raise ValueError("For local data, provide the official HF snapshot directory named with the required data revision")
    if not data_local and args.data_root != "MMMU/MMMU":
        raise ValueError("Assignment dataset must be MMMU/MMMU")
    load_start = time.perf_counter()
    datasets, fingerprints, seen = {}, {}, set()
    for subject in SUBJECTS:
        kwargs = dict(split="validation", cache_dir=args.cache_dir)
        if not data_local:
            kwargs["revision"] = DATA_REV
        dataset = load_dataset(data_root, subject, **kwargs)
        if len(dataset) != 30:
            raise ValueError(f"{subject}: expected 30, got {len(dataset)}")
        ids = list(dataset["id"])
        if len(set(ids)) != 30 or seen.intersection(ids):
            raise ValueError(f"Duplicate IDs in {subject}")
        seen.update(ids)
        fingerprints[subject] = dataset._fingerprint
        datasets[subject] = dataset.select(range(args.limit_per_subject)) if args.limit_per_subject else dataset
    if len(seen) != 900:
        raise ValueError("Expected 900 unique validation IDs")
    manifest["dataset_fingerprints"] = fingerprints
    manifest["dataset_load_seconds"] = time.perf_counter() - load_start
    write_json(outdir / "manifest.json", manifest)
    print(f"[data] Verified 900 unique IDs across 30 subjects. Selected {sum(map(len, datasets.values()))}.", flush=True)
    if args.check_only:
        counts = Counter()
        with (outdir / "inputs.jsonl").open("w", encoding="utf-8") as audit:
            for subject, dataset in datasets.items():
                for ex in dataset:
                    _, _, details = build_message(ex, getattr(args, "prompt_style", "direct"))
                    counts[ex["question_type"]] += 1
                    audit.write(json.dumps(dict(id=ex["id"], subject=subject, **details), ensure_ascii=False) + "\n")
        manifest.update(status="inputs_checked_no_inference", selected_question_types=dict(counts))
        write_json(outdir / "manifest.json", manifest)
        print(f"[check-only] Inputs OK: {dict(counts)}. Model/GPU inference NOT tested.", flush=True)
        return None

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
        raise ValueError("Quantized checkpoints are not supported in this assignment runner")
    write_json(outdir / "manifest.json", manifest)
    from vllm import LLM, SamplingParams
    model_start = time.perf_counter()
    print("[model] Loading BF16 model; initial compilation may take several minutes.", flush=True)
    llm = LLM(
        model=model_path, tokenizer=model_path, dtype="bfloat16", seed=RECIPE["seed"],
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 7}, max_num_seqs=args.batch_size,
        mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        generation_config="vllm",
    )
    model_seconds = time.perf_counter() - model_start
    tokenizer = llm.get_tokenizer()
    (outdir / "chat_template.txt").write_text(tokenizer.get_chat_template(), encoding="utf-8")
    params = SamplingParams(**RECIPE, max_tokens=args.max_tokens, n=1)
    (outdir / "sampling_params.txt").write_text(str(params), encoding="utf-8")
    all_rows, subject_rows = [], []
    with (outdir / "predictions.jsonl").open("w", encoding="utf-8") as predictions, \
         (outdir / "inputs.jsonl").open("w", encoding="utf-8") as audit:
        for subject, dataset in datasets.items():
            subject_start = time.perf_counter()
            subject_records = []
            print(f"[subject] {subject}: {len(dataset)} questions", flush=True)
            for offset in range(0, len(dataset), args.batch_size):
                examples = [dataset[i] for i in range(offset, min(offset + args.batch_size, len(dataset)))]
                prepared = [build_message(ex, getattr(args, "prompt_style", "direct")) for ex in examples]
                messages = [p[0] for p in prepared]
                for ex, (_, _, details) in zip(examples, prepared):
                    audit.write(json.dumps(dict(id=ex["id"], subject=subject, **details), ensure_ascii=False) + "\n")
                audit.flush()
                start = time.perf_counter()
                outputs = llm.chat(messages, sampling_params=params, use_tqdm=False)
                batch_seconds = time.perf_counter() - start
                if len(outputs) != len(examples):
                    raise RuntimeError("vLLM output count mismatch")
                for ex, (_, choices, _), output in zip(examples, prepared, outputs):
                    generated = output.outputs[0]
                    raw = generated.text
                    if ex["question_type"] == "multiple-choice":
                        parsed, info = mc_parse(raw, choices)
                        if generated.finish_reason == "length" and info["mode"] not in (
                            "explicit_final", "exact_letter"
                        ):
                            info = dict(mode="truncated_unparsed", candidates=info["candidates"])
                            parsed = None
                        correct = parsed is not None and eval_multi_choice(ex["answer"], parsed)
                    else:
                        parsed = sorted(parse_open_response(raw), key=lambda x: (type(x).__name__, str(x))) if raw.strip() else []
                        info = dict(mode="official_open" if parsed else "unparsed", candidates=[])
                        correct = bool(parsed) and eval_open(ex["answer"], parsed)
                    row = dict(id=ex["id"], subject=subject, question_type=ex["question_type"],
                               answer=ex["answer"], raw_response=raw, parsed_answer=parsed,
                               correct=bool(correct), parsing=info,
                               output_tokens=len(generated.token_ids),
                               input_tokens=len(output.prompt_token_ids) if output.prompt_token_ids is not None else None,
                               finish_reason=generated.finish_reason, stop_reason=generated.stop_reason,
                               batch_id=f"{subject}:{offset}", batch_size=len(examples), batch_seconds=batch_seconds)
                    predictions.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    subject_records.append(row)
                    all_rows.append(row)
                predictions.flush()
                print(f"  {offset + len(examples)}/{len(dataset)}  batch={batch_seconds:.1f}s", flush=True)
            n = len(subject_records)
            correct_n = sum(row["correct"] for row in subject_records)
            subject_rows.append(dict(subject=subject, n=n, correct=correct_n, accuracy=correct_n/n,
                                     seconds=time.perf_counter()-subject_start))
            print(f"[result] {subject}: {correct_n}/{n} = {100*correct_n/n:.2f}%", flush=True)
            write_json(outdir / "progress.json", dict(subjects=subject_rows, n_completed=len(all_rows)))
    if len({row["id"] for row in all_rows}) != len(all_rows):
        raise RuntimeError("Duplicate output IDs")
    expected = 30 * (args.limit_per_subject or 30)
    if len(all_rows) != expected:
        raise RuntimeError(f"Expected {expected} outputs, got {len(all_rows)}")
    macro = sum(row["accuracy"] for row in subject_rows) / 30
    micro = sum(row["correct"] for row in all_rows) / len(all_rows)
    if abs(macro - micro) > 1e-12:
        raise RuntimeError("Macro/micro mismatch on balanced data")
    subject_by_name = {row["subject"]: row for row in subject_rows}
    domains = []
    for domain, domain_subjects in DOMAIN_CAT2SUB_CAT.items():
        selected = [subject_by_name[name] for name in domain_subjects]
        domain_n = sum(row["n"] for row in selected)
        domain_correct = sum(row["correct"] for row in selected)
        domains.append(dict(domain=domain, n=domain_n, correct=domain_correct,
                            accuracy=domain_correct / domain_n,
                            seconds=sum(row["seconds"] for row in selected)))
    question_types = []
    for question_type in ("multiple-choice", "open"):
        selected = [row for row in all_rows if row["question_type"] == question_type]
        correct_n = sum(row["correct"] for row in selected)
        question_types.append(dict(question_type=question_type, n=len(selected), correct=correct_n,
                                   accuracy=correct_n / len(selected) if selected else 0.0))
    return dict(n=len(all_rows), correct=sum(row["correct"] for row in all_rows),
                subjects=subject_rows, domains=domains,
                question_types=question_types, macro_accuracy=macro, micro_accuracy=micro,
                complete_900=len(all_rows) == 900,
                unparsed=sum("unparsed" in row["parsing"]["mode"] for row in all_rows),
                ambiguous_mc=sum(len(set(row["parsing"]["candidates"])) > 1 for row in all_rows),
                length_limited=sum(row["finish_reason"] == "length" for row in all_rows),
                model_load_seconds=model_seconds)


def main():
    args, profile_info = arguments()
    start = time.perf_counter()
    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    packages = {}
    for name in ("torch", "transformers", "vllm", "datasets", "huggingface-hub", "Pillow", "numpy", "qwen-vl-utils"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    mc_prompt, open_prompt = prompt_templates(args.prompt_style)
    profile, profile_path, profile_hash = profile_info if profile_info else (None, None, None)
    signature = evaluation_signature(args, profile_hash)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), status="running",
                    arguments=vars(args), packages=packages, python=sys.version, platform=platform.platform(),
                    command=shlex.join(["python", "-u", "eval_mmmu.py"] + sys.argv[1:]),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    mc_parser_sha256=hashlib.sha256(Path(__file__).with_name('mc_parser.py').read_bytes()).hexdigest(),
                    mc_parser_policy=MC_PARSER_POLICY,
                    recipe=RECIPE, recipe_source=RECIPE_URL, data_revision=DATA_REV,
                    parser_revision=PARSER_REV, mc_prompt=mc_prompt, open_prompt=open_prompt,
                    evaluation_pipeline_version=EVALUATION_PIPELINE_VERSION,
                    image_layout_version=IMAGE_LAYOUT_VERSION,
                    evaluation_profile=(dict(name=profile["profile_name"], path=str(profile_path),
                                             sha256=profile_hash) if profile else None),
                    evaluation_signature=signature,
                    environment={key: os.environ.get(key) for key in (
                        "CUDA_VISIBLE_DEVICES", "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_WORKER_MULTIPROC_METHOD")})
    write_json(outdir / "manifest.json", manifest)
    if profile:
        write_json(outdir / "evaluation_profile.json", profile)
    (outdir / "requirements.freeze.txt").write_text(command_output([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")
    (outdir / "environment.txt").write_text(command_output(["nvidia-smi"]), encoding="utf-8")
    monitor = GpuMonitor(args.monitor_gpu)
    monitor.thread.start()
    try:
        summary = run(args, outdir, manifest)
        gpu = monitor.stop()
        total_seconds = time.perf_counter() - start
        if summary is not None:
            summary.update(gpu=gpu, total_seconds=total_seconds)
            write_json(outdir / "summary.json", summary)
            make_report(outdir, manifest, summary)
            manifest["status"] = "complete" if summary["complete_900"] else "development_subset_complete"
            print(f"[done] {summary['n']} questions, macro accuracy {100*summary['macro_accuracy']:.2f}%", flush=True)
        manifest.update(total_seconds=total_seconds, gpu=gpu)
        write_json(outdir / "manifest.json", manifest)
        print(f"[output] {outdir}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), traceback=traceback.format_exc(),
                        gpu=monitor.stop(), total_seconds=time.perf_counter()-start)
        write_json(outdir / "manifest.json", manifest)
        print("[failed] Partial records retained. No complete score/report claimed. See manifest.json.", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
