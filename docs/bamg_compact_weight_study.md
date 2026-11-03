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

# 기존 학습 Weight CNN을 초기값으로 쓰는 새 Q1 본실험
python scripts/run_v6_7_bamg_study.py run \
  --config configs/v6_7_bamg_compact_research_q1.json \
  --output artifacts/bamg_compact_research_q1 \
  --init-weight artifacts/bamg_research_q1/deployment_terminal.pt

# 새 실험이 중단된 경우: init-weight 없이 재개
python scripts/run_v6_7_bamg_study.py run \
  --config configs/v6_7_bamg_compact_research_q1.json \
  --output artifacts/bamg_compact_research_q1 \
  --resume
```

이전 체크포인트의 hidden/depth/activation/geometry/mask_mode 등이 새 설정과
다르면 명시적으로 거절한다. 설정을 원본 spec에 맞춘 후 새 실험을 준비한다.
초기값 없이 학습하려면 `--init-weight`를 생략한다. 설정의 updates는 새로
수행할 학습량이며 warm start라고 자동으로 줄이지 않는다.

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
  tests/test_bamg_training.py tests/test_bamg_study.py -ra
```

새 검증에는 n=31/63 Q1 bootstrap의 support·9-point 구조·line batching,
모델 없는 geometry reference, Weight-only 학습과 exact resume,
옛 checkpoint의 weight-only 초기화, 새 CLI 전체 실행과 재개가 포함된다.
이 테스트는 연구 규모에서의 속도 향상 측정이 아니다.
