# Sources and parser changes

- Official assignment: https://gist.github.com/neur-lab/38deabdfcde9e6dbacf362ab8059eb41
- Submission template: https://gist.github.com/neur-lab/483852e1f9d8d52f54627e600677c2f9
- Qwen evaluation recipe: https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction
- Qwen model: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- vLLM chat API: https://docs.vllm.ai/en/latest/api/vllm/entrypoints/llm/
- MMMU parser source: https://github.com/MMMU-Benchmark/MMMU/blob/aa9b70da92c2825b3d544d1a11b36856bd92f6c3/mmmu/utils/eval_utils.py

`vendor/mmmu_eval_utils.py` is an unmodified copy from the commit above.
The upstream license is included as `vendor/MMMU_LICENSE`.

## MMMU-Pro

- Dataset: `MMMU/MMMU_Pro`, revision
  `563f3e84bb3b90893083a1f039cfa13077f2302b` (Apache-2.0).
- Dataset card: https://huggingface.co/datasets/MMMU/MMMU_Pro
- Official direct prompts: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml
- Official evaluation reference: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/evaluate.py

`eval_mmmu_pro.py` uses the official direct prompt text and instance-level accuracy. Its multiple-choice
parser is deterministic: it preserves the official answer-pattern precedence but treats an unparsed response
as wrong instead of applying the official random-choice fallback.

`mc_parse()` in `eval_mmmu.py` adapts the upstream multiple-choice parser:
explicit final-answer and exact-letter responses are checked first, followed by the
same bracket / standalone-letter / option-text matching order;
the last match is selected when several candidate options are found.
Differences: no-match returns `None` (scored wrong), not a random option;
empty option strings are not treated as matches; parse mode and candidate list are recorded.
Open-ended extraction, normalization and scoring use the unmodified upstream functions.
Empty model responses are always wrong. Predictions are never repaired using the gold answer.
Length-truncated MC responses without an explicit final answer are scored wrong instead of
using option letters mentioned in unfinished reasoning.
Open-ended candidate lists are sorted only for stable JSON output; scoring is unchanged.

This is a documented team pipeline, not a claim to reproduce every detail of Qwen's internal evaluation.
Image labels and question templates are newly designed; both are saved in the run artifacts.
