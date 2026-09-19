# 기존 Strong Classical 기준 Neural MG 연구 실행 안내

이 경로는 기존 A-only deterministic selector를 **수정하지 않고** 그 선택 C*(A)에 맞춰
새 smoother와 sparse transfer를 학습·비교합니다. 구현 완료와 속도 우위는 별개입니다.
실제 작은 실험 결과와 미검증 범위는 [최신 설치 검증](docs/STRONG_AWARE_COMPLETION_KR.md)과
[첫 smoke 기록](docs/STRONG_AWARE_RESULTS_KR.md)에 있습니다.
기존 35-case audit, 이미 관찰한 14개 문제, 과거 certificate는 새 final 근거로 사용하지 않습니다.

## 실행 위치와 보존 범위

```bash
cd /Users/kimjiin/Desktop/adaptive_neural_multigrid_v6_7
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

기존 `.venv`와 이미 빌드된 native kernel을 사용합니다. escnn/e2cnn 등 새 의존성은
설치하지 않습니다. 기존 `strong.py`, selector rules, classical strategy bank,
stopping/safety, C++/OpenMP 코드는 유지합니다. 원래 모델·결과·certificate는 덮어쓰지 않습니다.
새 실행의 출력은 `artifacts/my_v6_7_strong_aware/<run_name>` 아래에만 생성합니다.

이번 전달물의 `smoke_01`은 별도 staging 디렉터리에서 실행한 실제 기록입니다.
복사된 JSON의 원래 경로·source hash를 바꿔 provenance를 꾸미지 않습니다.
이 기록을 설치 위치에서 `--resume`하지 말고 새 run 디렉터리를 사용하세요.
설치 위치에서 수행한 추가 검증은 별도 `installed_validation`/generation 기록으로 구분합니다.

## 먼저 작은 실행

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_v6_7_strong_aware.py \
  --config configs/v6_7_strong_aware_smoke.json \
  --output-dir artifacts/my_v6_7_strong_aware/my_smoke \
  --phase develop
```

Smoke는 새 train 7개(n=15), validation 14개(n=15/31), 모델당 optimizer update 7회입니다.
Learned prefix 2회 + classical tail 1회의 **full multilevel V-cycle**을 unroll합니다.
Cold 비교는 14개 operator에서 3회 반복합니다. Cold/warm/실제 multiple-RHS 비교는
명시된 validation 2개(n=15/31의 near-isotropic 사례)에서 RHS 1/4/16/64개를 각각 풉니다.
여러 RHS를 단일 solve 시간으로 외삽하지 않습니다. 이 작은 subset은 family 전체를 대표하지 않습니다.

원하는 단계까지만 실행할 수도 있습니다. 단계는 앞에서부터 순서대로 수행합니다.

```bash
.venv/bin/python scripts/run_v6_7_strong_aware.py \
  --config configs/v6_7_strong_aware_research.json \
  --output-dir artifacts/my_v6_7_strong_aware/research_01 --phase baseline
```

`data → baseline → direct → smoother → p → teacher → kd → joint → policy → evaluate → generation`
순서입니다. `baseline`은 기존 H_S/H_P/H_SP 재현이며, 그 뒤부터 새 가중치를 학습합니다.
`develop`은 이 development 단계 전체를 실행하지만 final/OOD는 열지 않습니다.

## 장시간 전체 연구 명령 — 이번 작업에서 자동 실행하지 않음

```bash
.venv/bin/python scripts/run_v6_7_strong_aware.py \
  --config configs/v6_7_strong_aware_research.json \
  --output-dir artifacts/my_v6_7_strong_aware/research_01 \
  --phase develop --resume
```

위 명령은 앞의 `baseline` 실행을 이어가는 예시입니다. 처음 시작한다면 `--resume`을 빼세요.
이미 완료한 update와 비교는 같은 protocol의 저장된 결과로 재사용합니다.
중단된 학습은 `resume.pt`의 optimizer/state/다음 sample 위치에서 이어집니다.
Source, config, rules, hardware, checkpoint digest가 달라지면 같은 run을 이어가지 않습니다.
**한 run에서는 같은 config 파일을 사용하고 `--phase`만 바꾸세요.** Preset 파일을 서로
바꿔 가며 resume하면 실험 조건 변경으로 거부됩니다.

| Config | 용도 / 기본 종료 단계 |
| --- | --- |
| `v6_7_strong_aware_direct.json` | C*(A)-aware direct 학생 / direct |
| `v6_7_strong_aware_smoother_ablation.json` | ordinary/D4/O2 / smoother |
| `v6_7_strong_aware_p_ablation.json` | CNN-P/GNN/edge-MLP / p |
| `v6_7_strong_aware_teacher.json` | offline teacher 비교 / teacher |
| `v6_7_strong_aware_distillation.json` | 조건부 KD / kd |
| `v6_7_strong_aware_research.json` | 전체 development 후 별도 final |
| `v6_7_strong_aware_a100.json` | CUDA FP32 generation 설정; A100 실측 결과 아님 |
| `v6_7_strong_aware_smoke.json` | 작은 경로 검증; final 금지 |

전체 설정은 train **280 unique operators**(7 families × 4 grids × 10), validation 56개입니다.
각 direct/teacher/joint 후보는 1,120 updates, prefix 4 + tail 2 V-cycles를 사용합니다.
KD가 허용되면 같은 크기의 direct 학생과 **같은 총 1,120 updates**를 사용하며 앞의 560회는
KD+task, 뒤의 560회는 task-only입니다. 총 예산에 KD와 fine-tuning을 숨겨 더하지 않습니다.
모델마다 같은 operator 순서와 C*(A)를 사용합니다. D4는 parameter 수가 같아도 계산량이 더 큽니다.

Smoke 시간으로 n=127/255 비용을 선형 예측할 수 없습니다. Full unroll의 sparse Galerkin
gradient와 큰 teacher, 56개 validation, 실제 64-RHS 반복 평가 때문에 전체 작업은 장시간
연구 작업입니다. `checkpoints/*/training.json`의 실측 시간과 단계별 `progress.json`을
확인하며 실행하세요. 최종 평가의 solve 수와 시간은 grid/fill/수렴에 크게 의존합니다.

## 바뀐 모델과 공정 비교 계약

| 역할 | 후보 | 실제 사용 head parameter 수 |
| --- | --- | ---: |
| S control | ordinary residual5 CNN, width 16 | 10,906 |
| S compact direct/KD control | compact3 CNN, width 16 | 6,266 |
| S equivariant | D4 CNN, width 16 | 10,906 |
| S steerable | O2 model, width 18 | 10,093 |
| P control | 기존 body를 유지한 CNN-P, width 16 | 11,008 |
| P student | 2-layer small GNN, width 16 | 4,961 |
| P student | sparse edge-MLP, width 16 | 1,409 |
| S teacher | deeper residual/dilated CNN, width 48 | 254,266 |
| P teacher | large GNN, width 64 | 120,065 |

S 후보는 동일한 열 개 operator feature, 9-point support, direction/gain 분리,
gain bound, learned level, replacement work, FP64 numerical kernel을 사용합니다.
과거 checkpoint는 원래 loader로 별도 재현합니다. 새 ordinary control은 기존 body에
공통 direction/bounded-gain 계약을 적용한 대조군이므로 과거 checkpoint와 구분합니다.

D4는 회전·반사 여덟 view의 Reynolds 평균이며 **encoder work가 8배**입니다.
O2는 scalar/order-1/order-2 field의 analytic steerable 연산을 사용합니다.
연속 표현·kernel의 equivariance와 사각 격자에서 임의각 회전 후의 전체 network
equivariance는 다릅니다. 후자는 interpolation/경계/9-point support로 오차가 발생합니다.
`validation_rotation_resampling.json`은 해당 오차 진단이며 PDE rotation-OOD 성능 검증이 아닙니다.
수학적 제약과 테스트 범위는 [architecture 문서](docs/STRONG_AWARE_ARCHITECTURE_KR.md)에 있습니다.

P는 sparse A connectivity로부터 classical candidate edge의 보정값을 생성하여 최종 P 하나에
합칩니다. R=Pᵀ, Ac=PᵀAP, coarse injection, 기존 row sum을 유지합니다.
새 CNN/GNN/edge/teacher 모두 기본 **row nnz≤8**, P/classical nnz ratio≤4,
Ac/classical ratio≤3, 전체 C_op≤3의 같은 cap을 사용합니다. Historical P는 과거 설정 그대로
재현하므로 공정한 새 P architecture 선택에는 `p_cnn_direct` 대조군을 사용합니다.
Cap을 넘는 learned hierarchy는 거부하고 Classical로 복구하며 그 시간도 총 시간에 포함합니다.
Tight cap 8이 기존 P의 support보다 작으면 zero-head여도 pruning이 생길 수 있습니다.
따라서 모든 설정에서 zero initialization이 classical P와 같다고 가정하지 않습니다.
`support: expanded`로 6×6 candidate support ablation을 새 config/run에서 수행할 수 있습니다.
Dense/global P는 생성하지 않습니다.

학습 loss는 full-V-cycle residual, 안정성, sparse support/fill proxy를 포함합니다.
KD는 정규화 direction + log-gain Huber 또는 동일 candidate-edge weight 차이를 추가합니다.
Training operator에서 측정한 고정 branch work ratio는 비용 기록/상수 proxy이며 clock gradient가 아닙니다.
P의 soft occupancy/fill proxy만 미분 가능하고, 실제 cap 및 최종 cost는 runtime에서 따로 검증합니다.
마지막 보완에서는 학습 중 제안된 P에도 실제 sparse nnz/row magnitude cap을 검사합니다.
제약을 넘으면 task V-cycle은 Classical fallback으로 계산하고, 별도의 미분 가능한
feasibility repair loss로 P를 수정합니다. 무효한 learned hierarchy의 contraction을 배포 가능한
학습 성능으로 기록하지 않습니다. 각 update의 task_branch와 proposal_feasibility가 남습니다.
기존 smoke_01은 이 보완 전 기록이며, 새 검증은 기존 checkpoint에서 이어서 별도 저장합니다.

## Teacher·KD·joint와 정책 채택

Teacher는 direct 학생보다 validation contraction이 기본 5% 이상 좋아야 KD 후보를 만듭니다.
비교 cohort의 성공 보존과 실제 neural 사용도 검사합니다. 개선이 없으면
`distillation_decisions.json`에 `created: false`를 남기고 KD checkpoint를 만들지 않습니다.
직접 학습 checkpoint를 복사해 distilled라고 표시하지 않습니다.
KD는 동일 크기 direct 학생보다 setup 포함 시간이 기본 3% 이상 개선되어야 채택합니다.
Teacher의 큰 추론 비용은 offline 결과로 별도 기록하며 student runtime에 섞지 않습니다.
Production에서는 teacher factory 표지를 검사해 실행을 금지하고, offline에서 만든 cached
teacher bank를 context 밖에서 재사용하는 것도 막습니다.

Joint는 H_S 성능 보존과 H_SP 개선을 모두 요구합니다. 실패하면 독립 S/P로 rollback합니다.
H_S/H_P/H_SP는 독립 branch로 남으며 H_SP를 강제하지 않습니다. Validation에서 가장 나은
후보를 골랐다는 사실만으로 production 채택을 선언하지 않습니다.

새 expert마다 training/validation에서 spatial detector와 temporal controller label을
재생성하고 refit합니다. 기존 tiny WHEN/WHERE 경로와 CLASSICAL LOCK은 유지합니다.
WHICH는 A-only rule/size bucket의 conservative cost utility를 사용하며 자료가 부족하면
Classical로 abstain합니다. Cold/cached 상태와 expected RHS 1/4/16/64를 구분합니다.
Label 생성의 reuse utility는 cold/warm 측정에 기반한 **offline 추정**이고, 최종 multi-RHS
표는 서로 다른 b를 실제로 푼 시간입니다. 두 값을 같은 측정으로 표시하지 않습니다.

## Cache와 장치 시간

한 model/config의 **실제 level A_l마다** heavy generator를 한 번 실행하고 FP64 bank에
cache합니다. Hierarchy 전체가 무조건 단 한 번의 forward라는 의미는 아닙니다.
같은 A의 여러 RHS에서 재사용하고, H_S→H_SP 전환 시 바뀌지 않은 root A의 smoother도
재사용합니다. P가 바뀌어 coarse A_l이 달라진 경우에만 새 smoother를 생성합니다.
Heavy NN은 매 cycle에 실행하지 않으며 tiny controller/detector만 online입니다.

Numerical solve는 계속 CPU FP64입니다. CPU/MPS/CUDA 비교는 FP32 operator generation,
FP64 변환·CPU 전송, sparse assembly, factorization을 실제 측정합니다.
Import/checkpoint loading은 별도이며 setup timing에는 넣지 않습니다.
Generation 실패는 `failed`와 이유/실제 경과시간을 남기고 빠른 성공으로 집계하지 않습니다.
MPS가 샌드박스 안에서 보이지 않으면 해당 프로세스의 `not_measured`로 기록합니다.
실제 Mac에서 수행한 CPU/MPS 검증은 별도 artifact에 남기며 CUDA/A100은 측정하지 않았습니다.

## Final/OOD를 여는 명령 — architecture를 고정한 뒤 한 번만

```bash
.venv/bin/python scripts/run_v6_7_strong_aware.py \
  --config configs/v6_7_strong_aware_research.json \
  --output-dir artifacts/my_v6_7_strong_aware/research_01 --phase freeze

.venv/bin/python scripts/run_v6_7_strong_aware.py \
  --config configs/v6_7_strong_aware_research.json \
  --output-dir artifacts/my_v6_7_strong_aware/research_01 --phase final
```

Freeze는 weights, architecture 후보, selector rules, policy, source, solver/hardware/protocol의
digest를 묶습니다. Final 명령은 별도 단일 claim/process lease를 얻어 final/OOD를 처음 생성합니다.
중단된 final만 같은 freeze와 `--resume`으로 이어갈 수 있고, 이미 끝난 final로 다시 튜닝할 수 없습니다.
Smoke에서 freeze/final을 시도하면 거부합니다.

Full plan은 final 56개와 grid n=255, intermediate rotation angle, anisotropy strength,
contrast, topology OOD를 포함합니다. `dataset_plan.json`은 specification이며 development 때
그 final operator를 assemble/solve하지 않습니다. 기존 provenance에서 발견한 normalized
operator digest와 train/validation/final 사이 중복을 검사하고 잘못된 provenance는 거부합니다.

Main final은 fixed/strong Classical, old fixed-baseline-aware H_S, 새 H_S/H_P/H_SP와 adaptive를
동일 A,b,x0,tolerance,budget에서 비교합니다. Cold/warm/multiple RHS를 구분합니다.
Architecture와 oracle reference는 사전 고정된 별도 표이며 이 결과로 다시 모델을 고르지 않습니다.
Ordinary/D4/O2 등의 frozen student는 OOD에서도 cold 비교하여 rotation generalization을 평가합니다.
Offline Best Classical은 후보를 전부 실행한 **사후 reference**이며 검색 비용을 기록합니다.
Deployable selector 또는 최종 adaptive speedup의 baseline을 oracle로 바꾸지 않습니다.

Certificate는 untouched final만 사용합니다. Strong C 성공 보존, 새 실패 0, 실제 neural 사용,
최소 20개/5반복, 기본 speedup≥1/(1−0.05)=1.05263, bootstrap CI 하한>1 및 정답 오차
보존을 요구합니다. 실패하면 새 certificate는 false이고 production model을 내보내지 않습니다.
성공한 경우만 `certified_policy`를 생성합니다. 기존 certificate를 수정하지 않습니다.

## 결과 파일

| 위치 | 내용 |
| --- | --- |
| `run_manifest.json`, `frozen_rules.json` | 코드/환경/config/selector 고정 근거 |
| `development_manifest.json`, `dataset_plan.json` | split·operator digest·C*(A), unopened final 계획 |
| `current_baseline_registry.json` | 원본 checkpoint 경로와 hash |
| `checkpoints/<model>/` | 독립 initial, resume, candidate, training 기록 |
| `teacher_decisions.json`, `distillation_decisions.json`, `joint_decision.json` | 채택/거부와 비교 cohort |
| `selected_students/`, `policy/deployable_policy/` | validation 후보와 refit policy; 성능 인증 아님 |
| `validation/<comparison>/` | raw repeat 기록, JSON/CSV, counters, 시간·오차·complexity |
| `tables/` | smoother/P/teacher/branch/cold-warm-multi 비교 |
| `generation/` | 실제 device/FP32 generation·FP64 cache 시간 |
| `research_freeze.json`, `final_evaluation/` | 명시적 full final을 실행할 때만 생성 |

Speedup은 `T_strong_C / T_method`이며 1보다 커야 빠릅니다.
실패한 solve를 speedup cohort에 넣지 않고 전체 success/failure 수를 함께 제시합니다.
CI는 operator 간 재표집이며 반복마다 변하는 시간 불확실성을 모두 설명하지는 않습니다.
Neural 사용 0인 adaptive 결과를 Neural 가속으로 해석하지 않습니다.

## 기존 smoke에서 이어서 검증하기

```bash
.venv/bin/python scripts/verify_v6_7_strong_aware.py \
  --run-dir artifacts/my_v6_7_strong_aware/smoke_01 \
  --output-dir artifacts/my_v6_7_strong_aware/continued_validation

.venv/bin/python scripts/benchmark_v6_7_strong_generation.py \
  --run-dir artifacts/my_v6_7_strong_aware/smoke_01 \
  --output-dir artifacts/my_v6_7_strong_aware/device_generation \
  --devices cpu mps cuda --repeats 5 --include-teachers
```

첫 명령은 기존 GNN checkpoint에서 repair 학습 2 update만 이어서 확인하고,
기존 selected student를 다시 학습하거나 바꾸지 않은 채 비교합니다.
둘째는 validation n=15 near-isotropic/n=31 rotated operator에서 각 실제 역할의 head만
측정합니다. Teacher는 명시적 offline scope입니다. 기존 기록을 덮어쓰지 않습니다.
