# 9채널 입력 특징 추가 실험

기존 파일은 수정하지 않는다. 새 전용 실행기만 아래 두 입력을 활성화한다.
기존 `src` 소스와 기존 실험의 소스 해시는 그대로 유지된다.

| 설정 | 행렬 입력 | 목적 |
|---|---|---|
| 기존 실행기 | 로그 대각 1 + 중심을 포함한 행 정규화 9 = 10 | 기존 대조군 |
| `row9` | 로그 대각 1 + 주변의 `A_ij / d_i` 8 = 9 | 중심 채널 제거 효과 |
| `symmetric9` | 로그 대각 1 + 주변의 `A_ij / sqrt(d_i*d_j)` 8 = 9 | 중심 제거와 대칭 정규화 |

여기서 `d_i = max(A_ii, 1e-14)`이고 로그 채널은
`log(d_i / median(d) + 1e-12)`이다. 모든 실제 대각값이 양수인지 확인한다.
양의 대각만 검사하는 것은 SPD 인증이 아니다. 결합의 부호를 유지하며,
격자 밖이나 없는 연결은 0이다. 채널 순서는 로그 대각 다음에
`(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)`이다.
입력은 실제 각 level의 행렬에서 계산하며 Galerkin 행렬에 대한 미분도 유지한다.
두 정규화 모두 원래 선형계의 행렬 자체는 바꾸지 않는다.

기하 특징 3, support mask 16을 그대로 붙여 첫 층의 입력은 **28채널**이다.
5개 3×3 합성곱, 폭 8, ReLU, 16개 출력 슬롯, fixed bilinear support의
행당 최대 4개 연결, injection, 경계 행합, L1 제한을 유지한다.
mask 16→4 압축이나 후보 선택 변경은 이 실험에 포함하지 않는다.
원래 입력과 마찬가지로 8방향 밖의 coarse fill-in을 모두 표현하지는 않는다.
첫 합성곱 계산량 감소가 실제 수렴 시간 개선을 보장하지 않는다.

## 실행 조건과 비교 방법

각 모드에 smoke, 560개, 1,120개 설정을 추가했다.
파일명은 `configs/v6_7_bamg_energy_cnn_feature9_{row9|symmetric9}_{q1|1120_q1|smoke_q1}.json`이다.
기존 대응 설정과 다른 항목은 `models.weight.feature_mode` 하나다.
따라서 연구 설정은 λ=10·30·50 × scratch/warm × seed 3개로 **모드당 18개 학습**이다.
15 supervised epochs, 8 fine-tuning epochs, validation 280개, 기존 routing과
수치해석 비교군 및 기존 10채널 CNN 비교군을 유지한다.

별개의 출력 폴더를 만든 것만으로 동일한 TRAIN/validation이 보장되지는 않는다.
기존 생성기가 주변 실험을 중복 제외 목록에 넣기 때문이다. 특징의 효과를 비교하려면
아래처럼 **같은 기존 routing 실험을 `--development-study`로 지정**한다.
새 adapter는 특징 모드만 다른 것을 허용하고, 데이터 seed·학습 예산·나머지 모델
구조·validation·teacher·source 동일성 검사를 유지한다. 기존 기준 실험의 파일은 읽기만 한다.
기준 실험은 데이터 준비가 끝나 있어야 하며, 기존 규칙대로 final/OOD claim을 열기 전이어야 한다.
진행 중인 기준 실험을 참조해도 되지만 타이밍 측정은 다른 학습과 동시에 실행하지 않는 편이 좋다.

이미 final/OOD를 연 실험은 개발용 기준으로 재사용하지 않는다. 새로운 개발 cohort를
먼저 만들고, 그 cohort를 세 입력 방식이 공유해야 한다. 새 폴더만 쓴다고 기존에 본
holdout이 새로운 독립 검증 데이터로 바뀌는 것은 아니다.

## Lenovo / WSL 실행 예시

기존 학습 checkout을 건드리지 않으려면 새 worktree를 만든다.
아래 경로는 기존 routing 문서의 폴더 배치를 따른 예시이며 실제 폴더명이 다르면 바꾼다.

```bash
cd /mnt/c/Users/user/Desktop/adaptive-multigrid-bamg
git fetch origin residual-u-net-teacher
git worktree add --detach ../adaptive-multigrid-feature9 FETCH_HEAD
cd ../adaptive-multigrid-feature9
source ../adaptive-multigrid-bamg/.venv-bamg/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

첫 번째 실험:

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_feature9.py run \
  --config configs/v6_7_bamg_energy_cnn_feature9_row9_q1.json \
  --source-study ../adaptive-multigrid-bamg/artifacts/initial_weights/mac_bamg \
  --teacher-comparison ../adaptive-multigrid-bamg/artifacts/bamg_energy_teacher_lambda_sweep_q1 \
  --development-study ../adaptive-multigrid-cnn-routing/artifacts/bamg_energy_cnn_routing_560_q1 \
  --output artifacts/bamg_feature9_row9_560_q1
```

두 번째 실험은 위 명령의 config를
`configs/v6_7_bamg_energy_cnn_feature9_symmetric9_q1.json`, output을
`artifacts/bamg_feature9_symmetric9_560_q1`로 바꾼다. 나머지 경로는 동일하게 유지한다.
중단 후에는 **동일한 전체 명령에 `--resume`을 추가**한다.
기존 10채널 학습의 출력 폴더나 다른 특징 모드의 출력 폴더에 이어 쓰면 안 된다.
두 방식 모두 새 실행기 안에서 teacher label을 한 case씩 읽는 기존 메모리 절약 경로를 사용한다.

각각 개발 평가가 완료된 뒤, 해당 output을 사용한다:

```bash
python scripts/run_v6_7_bamg_energy_cnn_feature9.py report \
  --output artifacts/bamg_feature9_row9_560_q1
python scripts/run_v6_7_bamg_energy_cnn_feature9.py freeze \
  --output artifacts/bamg_feature9_row9_560_q1
python -u scripts/run_v6_7_bamg_energy_cnn_feature9.py evaluate-final \
  --output artifacts/bamg_feature9_row9_560_q1
python scripts/run_v6_7_bamg_energy_cnn_feature9.py report-final \
  --output artifacts/bamg_feature9_row9_560_q1
```

final 평가 중단 후에는 `evaluate-final`에 `--resume`을 추가한다.
`report-final --group ood_grid`로 개별 OOD 결과를 볼 수 있다.
모드 선택은 개발 validation으로 끝내고, final/OOD 결과로 다시 고르지 않는다.
1,120개 확장은 해당 `1120_q1.json`, 새 output, 같은 모드의 560개 개발 기준을 사용한다.

## Warm start와 재현성

기존 29입력 체크포인트에서 첫 합성곱의 입력 인덱스 1, 즉 정규화 중심 채널만
제외하고 나머지 텐서를 그대로 복사한다. 로그 대각·주변 결합·기하·mask의 순서를
보존한다. optimizer와 학습 진행도는 가져오지 않으며 모든 층을 다시 학습한다.
중심 채널과 zero padding의 영향이 사라지므로 **동일 함수로의 변환은 아니다**.
특히 symmetric9는 결합 입력의 의미도 달라진다. 이것은 새 실험의 초기화이고,
기존 10채널 실험의 resume가 아니다. scratch 결과를 함께 비교해야 한다.
`initialization.json`에 원본/대상 spec, 전달 방식과 `function_preserving: false`를 기록한다.

새 체크포인트의 `weight_spec.feature_mode`가 입력 방식을 식별한다.
전용 실행기는 run/resume/validation/freeze/final 모두 같은 특징을 사용한다.
`old_cnn` 비교군은 기존 10채널 특징과 기존 체크포인트 로더를 사용한다.
새 특징 코드·adapter·실행기·메모리 adapter의 해시를 새 study manifest에 추가하여
코드나 설정이 달라지면 resume/final 검증이 거부한다. 기존 실행기의 해시는 바꾸지 않는다.
보고서에 기록된 결과를 읽는 `report`는 재계산하지 않는다.

추가 API는 `scripts/bamg_feature9.py`의 `direct_features9`, `Feature9WeightCNN`이다.
전체 연구 실행은 전용 CLI를 사용한다. Python에서 직접 연결할 때는
`source_reader(), memory_mode(), feature9_mode()` context를 사용하며,
한 프로세스에서 여러 연구를 동시에 thread로 실행하지 않는다.
context가 끝나거나 예외가 발생하면 기존 함수 연결을 복구한다.

## 검증 명령

```bash
python -m pytest tests/test_bamg_feature9.py tests/test_bamg_feature9_study.py
python -m pytest
sha256sum -c THREE_PILLARS_SOURCE.sha256
sha256sum -c P_HEADROOM_SOURCE.sha256
sha256sum -c WORLD_MODEL_SOURCE.sha256
sha256sum -c HS_WORLD_SOURCE.sha256
```

테스트는 수식·부호·경계 padding·autograd·checkpoint·기존 경로 복원·cohort 공유와
두 모드의 실제 소규모 학습/재개/validation/freeze/final/OOD를 확인한다.
소규모 테스트의 성공은 560개/1,120개 연구 성능 개선의 증거가 아니다.
