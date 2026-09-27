# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명·팀원**: 기입 필요
- **작성일**: 2026-09-27
- **채택 방식**: 학습 전 `two_stage4096_v1`; 이 문서는 MMMU validation만 보고함

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`; BF16, 비양자화 |
| 데이터 | `MMMU/MMMU` validation, revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` |
| 추론 백엔드 | vLLM 0.28.0의 로컬 `LLM.chat` 호출 (`generation_config="vllm"`) |
| 사용 GPU | NVIDIA GeForce RTX 4090 |
| 실측 peak VRAM | 22,057 MiB (약 21.54 GiB); 1초 간격으로 표본 추출한 GPU 전체 사용량의 최고치 |
| 추론 시간 | 약 136.5분 (900문항의 모델 호출 시간 합계) |
| 총 소요 시간 | 약 138.8분 (모델 로딩·데이터 처리 포함) |
| 의존성·환경 | [requirements.txt](../requirements.txt); Python 3.10.12; PyTorch 2.13.0+cu130, vLLM 0.28.0, Transformers 5.16.1, Datasets 5.0.1, NumPy 2.2.6, Pillow 12.3.0, huggingface-hub 1.29.0 |
| 실행 커맨드 | `bash scripts/run_mmmu_val_baseline.sh --model-path Qwen/Qwen3-VL-4B-Instruct --data-root MMMU/MMMU --output-root results/mmmu_val_two_stage4096` ([스크립트](../scripts/run_mmmu_val_baseline.sh)) |

peak VRAM은 모델 프로세스만의 정확한 최대 할당량이 아니라 다른 프로세스도 포함할 수 있는 GPU 전체 사용량의 표본 최고치다. 이 저장소의 `requirements.txt`에는 원 실행에서 확인한 직접 의존성 버전과 CUDA 13.0용 PyTorch wheel 출처를 기록했다. 실행 당시 전체 패키지 목록과 GPU·드라이버 기록은 원본 실행 폴더 `results/two_stage4096_v1/mmmu_val/`의 `requirements.freeze.txt`와 `environment.txt`에 저장되어 있으며 아직 이 저장소에 포함하지 않았다. 따라서 `requirements.txt`만으로 모든 간접 의존성까지 고정되지는 않는다.

원 실행은 실험 저장소의 `scripts/run_two_stage_baseline4096.sh`가 MMMU val과 MMMU-Pro 세 조건을 순차 수행했다. 이 저장소에는 원 실행의 코드 해시와 일치하는 평가 코드 및 고정 프로필을 포함했다. 저장소를 클론하고 [README의 설치 절차](../README.md)를 마친 뒤, 아래 명령은 **같은 고정 프로필로 MMMU val만** 검사하고 실행한다. `--model-path`와 `--data-root`에는 Hugging Face repo ID 또는 실행 컴퓨터의 실제 snapshot 경로를 전달한다. 원본 서버의 절대경로는 필요하지 않다.

```bash
bash scripts/run_mmmu_val_baseline.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --data-root MMMU/MMMU \
  --output-root results/mmmu_val_two_stage4096
```

## 2. 프롬프트

원본 이미지를 번호 순서로 먼저 전달한다. 투명한 이미지의 배경은 흰색으로 합성한 뒤 RGB PNG로 직렬화한다. 각 이미지는 한 문자열에 삽입되는 것이 아니라 `text: "Image {i}:"`와 `image_url: {image_i}`라는 별도 멀티모달 content 항목 두 개로 전달된다. 질문/선택지의 `<image {i}>`는 `[Image {i}]`로 바꾼다. 아래는 가변 부분을 `{}`로 표시한 **메시지 구성의 읽기 쉬운 표기**이며, 실제 이미지 데이터는 text가 아닌 `image_url` 항목이다. 정답 레이블은 입력하지 않는다.

객관식 **1단계 user 메시지**:

```text
Image 1: {image_1}
...
Image N: {image_N}
Question: {question}

Choices:
(A) {option_A}
(B) {option_B}
...

Work out the problem briefly, using the supplied images. Identify the relevant visual evidence and perform only the necessary calculations. Avoid repeating alternatives. This is a working draft; a separate step will select the final option.
```

객관식 **2단계**: 위 user 메시지(원본 이미지 포함), `assistant: {working_draft}`, 아래 user 메시지 순서.

```text
Now select the single best option for the ORIGINAL question using the original images and the working draft above. The draft may be incomplete or incorrect; do not blindly copy it. Output only one valid option letter.
```

주관식은 선택지 블록 없이 `Question: {question}`을 전달한다. 1단계는 아래 문구로 끝난다.

```text
Work out the problem briefly, using the supplied images. Identify the relevant visual evidence and perform only the necessary calculations. Avoid repeating alternatives. This is a working draft; a separate step will produce the final answer.
```

주관식 2단계에도 원본 이미지·질문과 `assistant: {working_draft}`를 전달한 뒤 다음 user 메시지를 붙인다.

```text
Now answer the ORIGINAL question using the original images and the working draft above. The draft may be incomplete or incorrect; do not blindly copy it. Output only the concise final answer, including units if needed. Do not include an explanation.
```

- **출처**: 실험 코드의 `eval_mmmu.py`, `eval_output_policy.py`에서 직접 설계.
- **선택 이유**: 시각적 근거·필요한 계산을 쓰는 과정과 최종 답의 형식을 분리한다. Qwen 공식 MMMU 프롬프트와 동일하지 않다.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---:|
| `do_sample` | 별도 인자로 전달하지 않음; vLLM `temperature=0.7`로 샘플링 |
| `temperature` | 0.7 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 3407 |

두 호출 모두 [Qwen 공개 평가 설정](https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction)의 temperature·top-p·top-k·penalty 값을 따른다. 시드 3407은 이 실험의 고정값이며, [공개 MMMU 스크립트](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)의 엔진 시드 42와 다르다. 객관식 2단계는 유효 선택지 문자만 허용하는 제한 디코딩을 추가하므로 공식 자유 생성과 출력 분포가 다르다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---:|
| 1단계 `max_new_tokens` | 4096 |
| 2단계 `max_new_tokens` — 객관식 / 주관식 | 16 / 128 |
| `max_model_len` | 16384 |
| `min_pixels` / `max_pixels` | 1,003,520 / 4,014,080 (= 1280×28² / 5120×28²) |
| 배치 / GPU memory fraction | 1문항씩 / 0.85 |

프로필의 `max_tokens=8192`는 free 모드용 필드이며, 이번 two-stage 초안 상한은 `reasoning_tokens=4096`이다. 초안이 잘려도 2단계를 실행한다. 이번 실행에서 초안 잘림은 120건, 최종 답 잘림은 0건이었다. 8192-token 초안은 같은 900문항에서 63.00%(+0.22%p)였으나 추론 시간이 약 223분으로 늘어 4096을 기준선으로 채택했다.

## 4. 채점(파싱) 방식

- **객관식**: 2단계에서 해당 문항의 유효 선택지 문자 하나를 제한 디코딩으로 생성한다. 출력 전체를 `strip()`한 값이 유효 선택지 문자와 정확히 같아야 하며, 아니면 실행을 오류로 중단한다. 유효한 문자는 정답 문자와 비교한다. 이 모드에서는 일반 `mc_parser.py`를 호출하지 않고 무작위 fallback도 없다.
- **주관식**: 2단계의 최종 답만 vendored MMMU `parse_open_response`/`eval_open`으로 채점한다. 초안은 채점하지 않는다.
- **실패 처리**: 주관식 최종 답이 비었거나 길이 제한으로 끝나면 미파싱·오답이다. 객관식 최종 답이 유효 문자 형식을 어기면 오답으로 보정하지 않고 실행을 중단한다. 완료된 평가에서는 전체 900문항을 분모에 유지했다. 외부 judge API는 사용하지 않았다.
- **실측**: 최종 답 미파싱 0건, 최종 답 잘림 0건. 이 수치는 초안 잘림 120건이나 오답이 없다는 뜻이 아니다.

## 5. 결과

MMMU validation의 **900개 고유 ID**(30과목×30문항; 객관식 847, 주관식 53)를 평가했다. 과목별 정답 수는 전달받은 900행 replay 예측 파일의 원래 `baseline_correct` 필드를 재집계했다. 합계 565개가 원 실행 요약과 일치한다.

| No. | Subject | Data Num | Correct | Acc |
|---:|---|---:|---:|---:|
| 1 | Accounting | 30 | 21 | 70.00% |
| 2 | Agriculture | 30 | 16 | 53.33% |
| 3 | Architecture_and_Engineering | 30 | 16 | 53.33% |
| 4 | Art | 30 | 19 | 63.33% |
| 5 | Art_Theory | 30 | 24 | 80.00% |
| 6 | Basic_Medical_Science | 30 | 22 | 73.33% |
| 7 | Biology | 30 | 11 | 36.67% |
| 8 | Chemistry | 30 | 14 | 46.67% |
| 9 | Clinical_Medicine | 30 | 21 | 70.00% |
| 10 | Computer_Science | 30 | 19 | 63.33% |
| 11 | Design | 30 | 22 | 73.33% |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 13 | 43.33% |
| 13 | Economics | 30 | 25 | 83.33% |
| 14 | Electronics | 30 | 15 | 50.00% |
| 15 | Energy_and_Power | 30 | 18 | 60.00% |
| 16 | Finance | 30 | 20 | 66.67% |
| 17 | Geography | 30 | 13 | 43.33% |
| 18 | History | 30 | 23 | 76.67% |
| 19 | Literature | 30 | 25 | 83.33% |
| 20 | Manage | 30 | 21 | 70.00% |
| 21 | Marketing | 30 | 23 | 76.67% |
| 22 | Materials | 30 | 18 | 60.00% |
| 23 | Math | 30 | 19 | 63.33% |
| 24 | Mechanical_Engineering | 30 | 13 | 43.33% |
| 25 | Music | 30 | 9 | 30.00% |
| 26 | Pharmacy | 30 | 21 | 70.00% |
| 27 | Physics | 30 | 18 | 60.00% |
| 28 | Psychology | 30 | 22 | 73.33% |
| 29 | Public_Health | 30 | 25 | 83.33% |
| 30 | Sociology | 30 | 19 | 63.33% |
| | **Overall (macro avg)** | **900** | **565** | **62.78%** |

`Overall = mean(30개 과목 accuracy)`. 모든 과목이 30문항이므로 micro accuracy `565/900`과 같다. 객관식은 546/847(64.46%), 주관식은 19/53(35.85%)이다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---:|
| [Qwen3-VL Technical Report, Table 4](https://arxiv.org/pdf/2511.21631)의 4B-Instruct | 67.40% |
| 우리 two-stage 기준선 | 62.78% |
| 차이 (우리 결과 − 공식) | −4.62%p |

이는 참고 수치 비교다. Qwen의 [공개 MMMU 실행 코드](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)는 기본적으로 `MMMU_DEV_VAL` 데이터와 별도 judge 기반 답 추출을 사용한다. 여기서는 validation 900문항만 로드하고 judge API 없이 채점한다. 이미지 표현·프롬프트·채점 정책과 두 번 호출·선택지 제한 디코딩도 다르므로 동일 조건의 재현 오차라고 해석하지 않는다.

## 7. 격차 분석

우리 결과는 공식 67.4%보다 4.62%p 낮다. 두 번 호출하는 프롬프트, 최종 선택지 제한 디코딩, 이미지 전처리·채점 방식이 공식 평가와 달라 차이를 한 원인에 귀속할 수 없다. 내부적으로는 120문항의 초안이 4096토큰에서 잘렸고 이 중 49문항(40.83%)을 맞았다. 초안이 끝난 780문항은 516문항(66.15%)을 맞았지만 난도 차이가 섞여 있어 잘림의 인과 효과를 뜻하지 않는다. 주관식은 19/53(35.85%)으로 낮다. 최종 답 미파싱은 0건이어서 이 실행의 격차를 파서 실패만으로 설명하기 어렵다.

## 8. 기타 특이사항 / 한계

- 이 수치는 **학습 전 모델을 고정된 two-stage 파이프라인으로 평가한 기준선**이다. 학습 후에도 같은 파이프라인을 적용해야 모델 변화의 효과를 비교할 수 있다.
- 4096→8192 paired 비교는 565→567 정답(+0.22%p, exact McNemar `p=0.8238`)이었다. 추가 추론 시간에 비해 개선 근거가 약해 4096을 선택했다.
- 이 보고서의 과목별 표와 환경 수치는 원본 실행의 `summary.json`, `manifest.json`, `predictions.jsonl`과 대조했다. `requirements.freeze.txt`와 `environment.txt`는 원본 서버에서 존재를 확인했으나 이 재현 저장소에는 포함하지 않았다.
- 과거 MMMU-Pro test 실험을 프로토콜 검토에 사용한 이력이 있다. MMMU-Pro를 untouched holdout으로 주장하거나 그 개별 오답에 맞춰 추가 조정하지 않는다.
