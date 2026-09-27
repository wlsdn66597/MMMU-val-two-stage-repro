# MMMU-val two-stage baseline reproduction

Qwen3-VL-4B-Instruct의 MMMU validation 900문항(객관식 847, 주관식 53)을 고정된 two-stage 설정으로 평가한다. 1차 초안은 최대 4096토큰, 2차 최종 답은 객관식 16토큰·주관식 128토큰이다. 모델 가중치와 데이터셋은 이 저장소에 포함하지 않는다.

## 환경 설치

원 실행 환경은 Linux, Python 3.10.12, NVIDIA RTX 4090, PyTorch 2.13.0+cu130, vLLM 0.28.0이다. CUDA 13.0을 지원하는 NVIDIA 드라이버와 약 22 GiB 이상의 사용 가능한 GPU 메모리가 필요하다. [requirements.txt](requirements.txt)는 원 실행에서 확인한 직접 의존성 버전과 CUDA 13.0용 PyTorch 공식 wheel 출처를 지정한다.

```bash
git clone https://github.com/wlsdn66597/MMMU-val-two-stage-repro.git
cd MMMU-val-two-stage-repro
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 실행

저장소 루트에서 실행한다. 모델은 Qwen3-VL-4B-Instruct의 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, 데이터는 MMMU validation의 revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`을 사용한다. 아래 빈 따옴표 안에 **실행할 컴퓨터의 실제 snapshot 경로**를 각각 입력한다.

```bash
MODEL_SNAPSHOT=""
MMMU_SNAPSHOT=""

bash scripts/run_mmmu_val_baseline.sh \
  --model-path "$MODEL_SNAPSHOT" \
  --data-root "$MMMU_SNAPSHOT" \
  --output-root results/mmmu_val_two_stage4096
```

상대경로를 전달하면 명령을 실행한 디렉터리를 기준으로 해석한다. 데이터 snapshot 디렉터리의 이름은 위 MMMU revision과 같아야 한다. 모델·데이터를 아직 내려받지 않았다면 두 경로 대신 Hugging Face ID `Qwen/Qwen3-VL-4B-Instruct`와 `MMMU/MMMU`를 각각 전달할 수도 있다. 이 경우 코드가 고정 revision을 내려받는다.

스크립트는 먼저 900문항 입력을 확인하고 `--output-root/check/`에 기록한 다음 추론 결과를 `--output-root/run/`에 저장한다. 완료 시 `run/summary.json`과 `run/predictions.jsonl`을 확인한다. 기존 실행 결과와 프롬프트·채점 방식은 [기준선 보고서](reports/mmmu_baseline.md)에 정리했다.

직접 의존성은 고정했지만 원 실행 서버의 전체 `pip freeze`는 포함하지 않았다. GPU·드라이버와 간접 의존성이 다르면 수치가 완전히 같다고 보장할 수 없다.
