#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/run_mmmu_val_baseline.sh --model-path CHECKPOINT_OR_HF_ID --data-root MMMU_SNAPSHOT_OR_HF_ID [--output-root NEW_DIRECTORY]" >&2
  exit 2
}

model_path=""
data_root=""
output_root="results/mmmu_val_two_stage4096"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model-path|--model_path)
      [ "$#" -ge 2 ] || usage
      model_path="$2"
      shift 2 ;;
    --data-root|--data_root)
      [ "$#" -ge 2 ] || usage
      data_root="$2"
      shift 2 ;;
    --output-root|--output_root)
      [ "$#" -ge 2 ] || usage
      output_root="$2"
      shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$model_path" ] && [ -n "$data_root" ] && [ -n "$output_root" ] || usage

# Resolve caller-supplied local directories before moving to the repository root.
# Hugging Face IDs remain unchanged and use the pinned revisions in the profile.
if [ -d "$model_path" ]; then
  model_path="$(cd "$model_path" && pwd -P)"
fi
if [ -d "$data_root" ]; then
  data_root="$(cd "$data_root" && pwd -P)"
fi

cd "$(dirname "$0")/.."
if [ -e "$output_root/check" ] || [ -e "$output_root/run" ]; then
  echo "Output already exists. Choose a new --output-root." >&2
  exit 2
fi
mkdir -p "$output_root"

common=(
  --benchmark mmmu-val --setting standard --mode two-stage
  --evaluation-profile code/evaluation/configs/two_stage4096_v1.json
  --model-path "$model_path"
  --model-revision ebb281ec70b05090aa6165b016eac8ec08e71b17
  --data-root "$data_root"
)

python -u code/evaluation/eval_output_policy.py "${common[@]}" \
  --check-only --output-dir "$output_root/check"

python -u code/evaluation/eval_output_policy.py "${common[@]}" \
  --checked-inputs "$output_root/check" --output-dir "$output_root/run"

python - "$output_root/run/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
if summary.get("n") != 900 or not summary.get("complete_900"):
    raise SystemExit("Incomplete MMMU validation run")
print(f"[done] 900 questions; accuracy={summary['accuracy']:.2%}; summary={sys.argv[1]}")
PY
