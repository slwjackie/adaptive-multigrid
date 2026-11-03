# 고정 bilinear support의 BAMG teacher와 Weight CNN

`support_mode: "fixed_bilinear"`은 **teacher 라벨 생성, 지도학습, 잔차 미세조정,
배포의 모든 단계**에서 같은 기하학적 연결을 사용한다. 각 행의 연결 수는
위치에 따라 1/2/4개다. `k=4` top-k와는 다르며, 이 모드에서는 k가 연결을
선택하지 않는다. 경계 행합, identity injection, signed L1 cap은 유지된다.

Teacher는 기존 BAMG의 relaxed-vector LS와 bootstrap enrichment를 사용하되,
연결 선택 없이 가까운 연결의 값만 최적화한다. Weight CNN도 해당 mask 안에서
값을 직접 생성한다. Bilinear 값에 보정을 더하는 decoder로 바꾸지 않는다.
Classical line_alt, pre/post sweeps, full coarsening, 실제 Galerkin, coarse LU는 같다.

초기 Q1 A가 9-point이고 모든 level이 이 support를 사용하면 coarse A도 9-point
범위에 남는다. 이로써 먼 coarse 연결 때문에 batched line solve가 꺼지는
경로를 피한다. 수렴 실패 제거 또는 EM5보다 빠른 warm solve를 보장하는 것은 아니다.

## CNN 재학습과 기존 체크포인트

- Teacher 자체(`teacher` 명령, `bootstrap_C` 비교군)를 시험하는 데 CNN 학습은 필요 없다.
- 기존 자유-mask CNN은 새 mask에서 학습된 모델이 아니다. 이전 성능을 그대로
  인정하거나 이전 `benchmark.json`에 새 결과를 섞지 않는다.
- 새 모드에는 Mask CNN이 없다. Weight CNN 하나만 지도학습하고 미세조정한다.
- 호환되는 기존 BAMG Weight CNN은 `--init-weight`로 초기값을 재사용할 수 있다.
  이는 추가 학습의 시작점이며, 추가 학습이 필요 없다는 뜻은 아니다.
- `supervised_models.pt`, `deployment_uniform.pt`, `deployment_terminal.pt`,
  `pretrain.pt`, `finetune_*.pt`의 weight tensors를 지원한다. 모델 spec이 정확히
  일치해야 한다. 다른 구조의 과거 residual CNN은 자동 변환하지 않는다.
- optimizer, 이전 update 수, Mask CNN, 이전 mask는 가져오지 않는다.
  `weight_initialization.json`에 원본 SHA-256을 기록하고 원본은 수정하지 않는다.
- `--resume`은 **새 모드로 이미 시작한 동일 실험**의 중단 후 재개다.
  `--init-weight`와 동시에 사용할 수 없다.

기하학적 mask는 A나 기준 Weight CNN에 의존하지 않는다. 따라서 mask 생성에
별도 CNN 추론 또는 기준 Galerkin 계층을 만들지 않는다. 같은 격자의 mask와
후보 인덱스는 캐시를 공유하여 체크포인트 중복 저장도 줄인다. 실제 student
계층은 현재 Weight CNN 값으로 다시 구성한다.

## 실행

저장소 루트에서, 기존 Python 환경을 활성화한 상태로 실행한다. 원본 실험의
source manifest는 변경하지 말고 **반드시 새 출력 디렉터리**를 사용한다.
기존 결과를 재현해야 한다면 이전 commit의 별도 checkout을 보관한다.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

# Q1 n=15, 7 PDE family: 배선/학습/벤치마크 smoke 확인
python scripts/run_v6_7_bamg_study.py run \
  --config configs/v6_7_bamg_compact_smoke_q1.json \
  --output artifacts/bamg_compact_smoke_q1

# 새 Weight CNN으로 시작: 6→12 cycle + 중간 warm 체크포인트 선택
python scripts/run_v6_7_bamg_study.py run \
  --config configs/v6_7_bamg_compact_research_q1.json \
  --output artifacts/bamg_compact_warm612_q1

# 새 실험이 중단된 경우: init-weight 없이 재개
python scripts/run_v6_7_bamg_study.py run \
  --config configs/v6_7_bamg_compact_research_q1.json \
  --output artifacts/bamg_compact_warm612_q1 \
  --resume

```

이전 체크포인트의 hidden/depth/activation/geometry/mask_mode 등이 새 설정과
다르면 명시적으로 거절한다. 설정을 원본 spec에 맞춘 후 새 실험을 준비한다.
초기값 없이 학습하려면 `--init-weight`를 생략한다. 설정의 updates는 새로
수행할 학습량이며 warm start라고 자동으로 줄이지 않는다.
기존 가중치로 시작하려면 최초 `run`에
`--init-weight artifacts/bamg_research_q1/deployment_terminal.pt`를 추가한다.

## 잔차 미세조정과 중간 warm 모델 선택

본실험의 각 uniform/terminal 학습은 같은 지도학습 모델에서 독립적으로 시작한다.
각각 총 1,000 updates이며, 1–500회는 6 V-cycle, 501–1,000회는 12 V-cycle의
잔차 손실로 Weight CNN만 학습한다. 기존 mask, smoother, 수렴 tolerance는 유지한다.
Smoke는 배선 확인용 2 updates(6회, 12회)이며 성능 판단용 학습량이 아니다.

`warm_validation`은 학습 손실과 별개인 개발용 모델 선택이다.

- 본실험은 update 0(지도학습 초기값), 매 100 updates, 마지막 update에서 평가한다.
  `every: 200`도 설정 가능하다. 마지막 update가 주기의 배수가 아니어도 평가한다.
- 기존 validation에서 **PDE family × 격자 크기마다 1개**를 case ID 순서로 고정 선택한다.
  n=31/63 × 7 family = 14개이며 uniform/terminal 모두 같은 A와 RHS를 사용한다.
  TRAIN과 겹치는 case/operator를 거절한다. Final/OOD에는 접근하지 않는다.
- 매 평가마다 현재 P와 실제 Galerkin 행렬·분해를 새로 준비한다. Setup은 시간
  점수에서 제외한다. Untimed warmup 1 cycle 뒤, 같은 RHS를 매번 x=0에서 3회 푼다.
- 각 측정은 6/12 cycle에서 멈추지 않고
  `||r|| <= max(1e-12, 1e-8*||r0||)`까지 실행한다. `mg.max_cycles=150` 또는
  측정당 120초 제한에 걸리면 실패다. 시간 제한은 cycle 사이에서 확인하는 방식이며
  이미 실행 중인 factorization/cycle을 강제로 중단하는 기능은 아니다.
- **실패 측정 수가 적은 모델을 먼저 선택**하고, 같으면 case별 반복 중앙값의
  산술평균 warm 시간으로 비교한다. 실패 측정도 최소 120초의 페널티로 포함한다.
  동점이면 기존 모델을 유지한다. Fallback 성공을 raw 성공으로 계산하지 않는다.
- 모든 후보가 실패해도 가장 실패가 적은 후보는 저장하되 `eligible=false`로 표시한다.
  최종 전체 validation의 기존 실패율 자격 심사는 별도로 유지한다.
- 평가는 `eval()`과 `no_grad()`로 실행하며 학습 RNG 상태를 복원한다. Warm 시간은
  미분하지 않는다. 더 긴 rollout의 잔차로 역전파하고, 실제 시간으로 모델을 고른다.

| 파일 (uniform 예시; terminal도 동일) | 용도 |
| --- | --- |
| `finetune_uniform.pt` | **latest**: 마지막 저장 update의 모델·Adam·RNG·학습 이력. 재개는 항상 여기서 수행 |
| `finetune_uniform_best_warm.pt` | **best_warm**: 선택된 update의 가중치와 성능·평가 조건. Optimizer 없음 |
| `deployment_uniform.pt` | 학습 종료 후 **best_warm 가중치**를 내보낸 배포 모델. 최종 benchmark도 이 모델 사용 |
| `finetune_uniform_warm_validation.json` | 미세조정 종료 후 중간 평가 이력과 선택된 update를 확인하는 작은 보고서 |

Latest 안에도 최우수 가중치와 평가 이력을 보관한다. 두 파일 저장 사이 중단이
생겨도 latest에서 best 파일을 복원할 수 있다. 재개 후 이전 최우수 결과를 유지하며,
이미 끝난 평가를 반복하지 않는다. Latest 저장 주기(10 updates)와 별개로 평가가
끝난 update에도 latest를 저장한다. 가장 좋은 모델이 update 0일 수도 있다.
전체 잔차 궤적·해 벡터·계층 행렬을 중간 평가 기록에 저장하지 않아 큰 파일의
중복을 피한다. 실행 중 성능 요약은 `Warm validation update=... best_update=...`로 출력된다.

긴 rollout과 중간 평가 때문에 **오프라인 학습 시간은 늘어난다**. 최종 warm 풀이에
그 비용이 들어가지는 않는다. 14개 선택용 validation에서 빠르다는 것이 전체
validation이나 final/OOD의 우위를 보장하지 않는다. 최종 비교는 기존 전체 validation을
사용하고, 연구 결론은 별도로 보존된 final/OOD로 확인한다.

기존 실험은 source/configuration manifest가 달라지므로 그대로 `--resume`하지 않는다.
새 출력 폴더로 시작하거나 호환되는 가중치만 `--init-weight`로 이전한다.
기존 자유-mask 설정은 `warm_validation`을 생략하면 종전의 마지막 update 선택을 유지한다.

CNN 없이 compact teacher 라벨만 먼저 만들려면:

```bash
python scripts/run_v6_7_bamg_study.py prepare \
  --config configs/v6_7_bamg_compact_smoke_q1.json \
  --output artifacts/bamg_compact_teacher_probe
python scripts/run_v6_7_bamg_study.py teacher \
  --output artifacts/bamg_compact_teacher_probe
```

## 평가와 범위

기존 7개 비교군을 유지하되, 새 모드의 CNN 이름은
`bootstrap_weight_cnn_supervised`, `bootstrap_weight_cnn_finetuned_uniform`,
`bootstrap_weight_cnn_finetuned_terminal`이다. `ls_once_C`와 `bootstrap_C`도
새 모드에서는 **compact support teacher**를 의미하며 설정/manifest에 구분된다.
과거 자유-mask 결과와 이름만 보고 합치지 않는다.

모델 선택은 기존의 raw warm tolerance solve와 실패율 정책을 유지한다.
Fallback으로 회복한 사례를 raw 성공으로 바꾸지 않는다. 새 CNN은 기존
체크포인트를 초기값으로 사용해도 새 validation이 필요하다.

이번 변경은 고정 support 경로를 끝까지 일치시키는 비교 실험이다.
Teacher의 LS+ridge, bootstrap 후보의 짧은 잔차 기반 선택은 기존 구현을
유지한다. 별도로 제안된 LS+energy 공동 목적함수, EM5 기반 추가 테스트 벡터,
teacher 후보 자체의 full-tolerance 시간 선택을 구현했다고 해석하지 않는다.
따라서 연결 구조를 고정한 효과를 다른 목적함수 변경과 구분해서 평가할 수 있다.

## 검증

```bash
python -m pytest tests/test_bamg_compact.py tests/test_bamg_teacher.py \
  tests/test_bamg_transfer.py tests/test_bamg_solver.py \
  tests/test_bamg_training.py tests/test_bamg_study.py \
  tests/test_bamg_warm_validation.py -ra
```

새 검증에는 n=31/63 Q1 bootstrap의 support·9-point 구조·line batching,
모델 없는 geometry reference, Weight-only 학습과 exact resume,
옛 checkpoint의 weight-only 초기화, 6→12 cycle의 CLI 실행과 재개, 실패 우선 선택,
학습 RNG 독립성, 초기/중간/최종 평가, best/latest 분리와 배포 연결이 포함된다.
이 테스트는 연구 규모에서의 속도 향상 측정이 아니다.
