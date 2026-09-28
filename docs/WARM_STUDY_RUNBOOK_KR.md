# Warm-first Neural Multigrid 연구 실행 안내

## 범위와 기존 코드와의 차이

기준: main `ba22fb7d569767a635e79c93ce928bde872e4f56`의 three-pillars workflow.
새 실행 파일은 `scripts/run_v6_7_warm_study.py`이다. 기존 three-pillars CLI, H_P/H_SP,
classical selector와 bank는 유지한다. 새 CLI는 H_S expert 개발 후 policy를 보정하는
독립 연구 경로다. 이 변경은 성능 우월성 인증 또는 논문의 재현 완료 선언이 아니다.

| 항목 | 이전 | 새 경로 |
|---|---|---|
| 주측정 | warm RHS=1, setup 포함 multiple | 별도 RHS로 준비한 뒤 실제 다양한 RHS를 푸는 warm_multiple 추가 |
| H_S | A에서 단일 9-point stencil 생성 | 기존 H0 + 5-parameter H1 + 2/3-stage cached residual correction |
| 수치 stage | 한 correction | stage마다 갱신한 residual을 다시 보정 |
| 학습 | full-cycle loss | 선택적 one-sided no-harm, coarse-complement loss, multi-RHS augmentation |
| hierarchy 학습 | fine 중심 | explicit smoother/transfer levels, coarse-to-fine expert freezing |
| replacement | 1C -> 1H | 2C -> 1H, pre/post 교체를 독립 설정 |
| policy | exact-n table | 연속 크기 feature ridge cost regression + tune optimism margin |
| policy 데이터 | 단일 RHS cold/warm 기반 추정 | 별도 fit/tune/validation에서 실제 distinct RHS batch 측정 |
| 실험 순서 | expert/policy 연결 유연 | expert 선택/고정 이후 policy fitting, 이후 독립 validation과 freeze |
| 진단 | 단일 demo | 저장된 정확한 A/RHS로 16개 classical 후보 재측정 |

같은 A에 대해 C/H_S/Adaptive가 공유하는 classical parent C*(A), FP64 true residual,
tolerance, 전체 cycle budget, rollback/CLASSICAL_LOCK을 유지한다. 외부 Krylov를 추가하지 않았다.
Neural runtime 제어기 비용과 probe 비용은 측정에 포함한다.

## A. 설치와 측정 준비

기존 로컬 수정은 먼저 commit 또는 별도 보관한다. 아래는 저장소 루트에서 실행한다.

```bash
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python scripts/build_native_stencil.py --no-openmp
python -m pytest -ra
python -m pytest tests/test_warm_study.py -ra
python scripts/run_v6_7_warm_study.py --help
```

Native compiler가 없으면 build 명령을 생략하고 CSR로 실행할 수 있다. 실험 도중에는
backend/thread/library/source를 바꾸지 않는다. **이전 RUN은 새 소스로 resume하지 않는다.**
소스 해시가 다르면 의도적으로 거부한다. 예전 checkpoint/benchmark는 보존하고 새 RUN을 만든다.

## B. 처음부터 끝까지 smoke

```bash
RUN="artifacts/warm_study_smoke_$(date +%Y%m%d_%H%M%S)"
echo "$RUN"

python scripts/run_v6_7_warm_study.py calibrate \
  --config configs/v6_7_warm_study_smoke.json --run-dir "$RUN"

python scripts/run_v6_7_warm_study.py train --run-dir "$RUN" \
  --variants H0 H1 H2 H2_NH H2_L H3 H2_2C H2_PREPOST

python scripts/run_v6_7_warm_study.py benchmark --run-dir "$RUN" \
  --variants H0 H1 H2 H2_NH H2_L H3 H2_2C H2_PREPOST \
  --tag architecture --repeats 3 --warmups 1 \
  --rhs-counts 1 4 --regimes warm_multiple multiple

python scripts/run_v6_7_warm_study.py select --run-dir "$RUN" --tag architecture
python scripts/run_v6_7_warm_study.py policy-fit --run-dir "$RUN"

python scripts/run_v6_7_warm_study.py policy-validate --run-dir "$RUN" \
  --probes 0 1 2 --tag policy_validation --repeats 3 --warmups 1 \
  --rhs-counts 1 4 --regimes warm_multiple multiple
```

Smoke는 n=7,15, variant당 8 optimizer updates. policy-fit/tune/validation은 각각
4개의 새 operator다. 성능 평가에 충분한 표본/학습량이 아니다. Smoke RUN은 final 금지.
`select`는 측정된 warm evidence에서 new failure 없는 variant 중 기하평균 점수가 가장 큰 것을
선택한다. score>1을 요구하지 않는다. 즉 느린 variant가 선택되어도 그것은 코드 오류나
성능 인증이 아니다. 학술적 판단 후 직접 선택하려면 `--variant H2_NH`처럼 지정한다.
선택한 이후 같은 RUN에서 expert를 재학습하거나 architecture를 바꿀 수 없다.

## C. 본 연구: expert를 먼저 개발

```bash
RUN="artifacts/warm_study_research_$(date +%Y%m%d_%H%M%S)"
echo "$RUN"
python scripts/run_v6_7_warm_study.py calibrate \
  --config configs/v6_7_warm_study_research.json --run-dir "$RUN"

python scripts/run_v6_7_warm_study.py train --run-dir "$RUN" \
  --variants H0 H1 H2 H2_NH

python scripts/run_v6_7_warm_study.py benchmark --run-dir "$RUN" \
  --variants H0 H1 H2 H2_NH --tag architecture_initial \
  --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 \
  --regimes warm_multiple multiple
```

여기서 아직 `select`하지 않으면 추가 variant를 학습할 수 있다.

```bash
python scripts/run_v6_7_warm_study.py train --run-dir "$RUN" \
  --variants H2_L H3 H2_2C H2_PREPOST

python scripts/run_v6_7_warm_study.py benchmark --run-dir "$RUN" \
  --variants H0 H1 H2 H2_NH H2_L H3 H2_2C H2_PREPOST \
  --tag architecture --repeats 5 --warmups 1 \
  --rhs-counts 1 4 16 64 --regimes warm_multiple multiple
```

Research 기본값: calibration train/tune 각각 63 operators, NN train168,
architecture-validation42, n=15/31/63. 각 variant1120 updates, RHS augmentation4.
H2_L은 coarse level1 phase -> fine level0 phase로 1120 updates를 나누며,
실제로 해당 level까지 도달하는 TRAIN operator만 phase에서 선택한다.
아래 parameter count는 smoother만의 수이며 사용하지 않는 transfer 모델까지 합한 값이 아니다.

| Variant | 의미 | hidden=16 smoother parameters |
|---|---|---:|
| H0 | 기존 compact3 student, 단일 stencil | 6266 |
| H1 | 새 bounded polynomial approximate-inverse-inspired stage | 5 |
| H2 | 독립 CNN stage 2개, updated-residual cascade | 12532 |
| H2_NH | H2 + no-harm=0.1 | 12532 |
| H2_L | 2-stage x 2-level experts + coarse-aware + no-harm | 25064 |
| H3 | 3-stage + no-harm | 18798 |
| H2_2C | 2 classical pre slots -> 1 two-stage application | 12532 |
| H2_PREPOST | pre1/post1에서 two-stage replacement | 12532 |

H1의 다항식은 Weymouth 원 논문 계수식의 복제가 아니다. 5개 학습계수로 local
approximate inverse를 만드는 발상만 채택한 신규 bounded formula다.
H2/H3는 Huang의 nonlinear residual CNN 복제가 아니다. A-only cached linear
operators를 residual-update로 합성한 변형이다. 논문의 속도비를 그대로 기대하면 안 된다.

### Stage의 실제 수학

각 stage에서 `d_j=B_j r_j`, `r_(j+1)=r_j-A d_j`, 최종 correction은 `sum(d_j)`이다.
2-stage이면 `B_eff=B1+B2(I-A B1)`이지 `B2 B1`이 아니다.
마지막 stage 이후에는 다음 residual을 만들 필요가 없어 추가 matvec를 생략한다.
그러나 stage 사이 A matvec와 모든 local apply 비용은 Stats와 시간에 들어간다.
NN은 A별 bank setup에서 실행하고 이후 RHS에서는 재사용한다.

No-harm은 `ReLU(log(h_H)-log(h_C)-log(1+epsilon))^2`의 가중합이다.
C trajectory는 detach되며, 단순 `log(h_H)-log(h_C)`를 loss에 더하는 것과 다르다.
이는 경험적 regularizer이지 OOD에서 해롭지 않다는 정리가 아니다.

Coarse-complement 항은 TRAIN에서 classical coarse space에 투영한 성분을 제거하고,
남은 error를 smoother가 줄이도록 한다. 여기서 exact coarse solve는 학습용 진단이며
deployment에서는 기존 recursive V-cycle을 사용한다. H_P에 필요한 slow-error loss와
방향이 다르다. 본 연구의 H_P 개선안은 별도 문서를 참고한다.

## D. Expert 고정 뒤 실제 policy 데이터 측정

```bash
python scripts/run_v6_7_warm_study.py select --run-dir "$RUN" --tag architecture
# 명시적으로 선택할 때는 위 명령 대신 아래 예처럼 실행한다.
# python scripts/run_v6_7_warm_study.py select --run-dir "$RUN" --tag architecture --variant H2_NH

python scripts/run_v6_7_warm_study.py policy-fit --run-dir "$RUN"
python scripts/run_v6_7_warm_study.py policy-validate --run-dir "$RUN" \
  --probes 0 1 2 --tag policy_validation --repeats 5 --warmups 1 \
  --rhs-counts 1 4 16 64 --regimes warm_multiple multiple
```

새 split은 normalized A digest 기준으로 기존 train/validation/calibration/서로 간에
분리한다. 기본 policy-fit42, policy-tune21, policy-validation42 operators다.
policy-validate는 actual RHS를 독립적으로 생성하고 C*/forced H_S/adaptive를 비교한다.

Policy는 MLP가 아니라 ridge regression이다. log(N), nnz/row, depth, complexity,
RHS 수, bank cache, A의 anisotropy/contrast/orientation, selected strategy를 입력으로
log(T_C/T_H)를 예측한다. TRAIN operator 하나의 전체 가중치는 workload 수에 관계없이
1이다. 반복 측정은 별도 독립표본으로 세지 않는다.

TUNE에서 operator별 최악의 낙관오차를 이용해 경험적 margin을 정하고, H_S가 C 성공을
잃은 전략은 차단한다. 미관측 strategy/cache, RHS 범위 밖, 크기외삽 상한 초과,
불충분한 margin이면 C를 선택한다. n의 정확한 lookup은 없다. **연속 크기 feature와
bounded extrapolation은 일반화 보장이 아니다.** `policy_coverage.json`에서 크기별
실제 H 사용률/abstention reason/classical coverage fallback을 확인한다.

Probe=0이 기본. Probe1/2는 model이 H를 선택한 경우에만 C1/2 cycle로 쉬운 문제인지
검사하는 보수적 veto다. 불확실한 C 결정을 근거 없이 H로 뒤집지 않는다.
Probe 이후 iterate를 재시작하지 않고 original residual threshold/cycle budget을 유지한다.
Probe/decision 비용은 warm timer 안에 포함한다. Residual safety는 속도 우위 증명이 아니다.

### D-1. adaptive → C fast path overhead 진단

최종 expert와 policy를 고정한 뒤, policy가 C를 선택했을 때 wrapper 자체가 strong_C보다
얼마나 느린지 별도의 개발용 microbenchmark로 확인한다. 이 진단은 policy 학습/architecture
선택/final evidence에 절대 사용하지 않는다. C 선택을 강제하고 numerical equality를 먼저
검사한 뒤 warm batch만 측정한다.

```bash
python scripts/run_v6_7_warm_study.py policy-overhead \
  --run-dir "$RUN" \
  --sizes 7 15 31 63 \
  --repeats 20 \
  --rhs-count 4
```

결과는 `$RUN/policy/policy_overhead.json`에 저장된다. 각 size에서
`adaptive_overhead_fraction`, `strong_over_adaptive`, `controller_seconds`,
`policy_batch_decisions`, `numerically_identical`을 확인한다. 목표는 C를 선택했을 때
`adaptive_overhead_fraction`을 가능한 한 0에 가깝게 만드는 것이다. 작은 grid에서는
절대 시간이 매우 짧아 상대 overhead 비율의 noise가 클 수 있으므로 여러 repeat의 median을 본다.

새 runtime은 `solve_many`에서 policy prediction을 batch당 한 번만 수행한다. C가 선택되면
Neural bank나 generic adaptive state를 만들지 않고 동일한 C*(A) hierarchy의 전용 fast path로
직행한다. Frozen expert freshness는 매 solve마다 full weight hash를 계산하지 않고 tensor
revision/device/dtype token을 우선 검사하며, revision이 실제로 바뀐 경우에만 full signature를
재검증한다. Expert 변경이 감지되면 policy는 stale로 처리되어 C로 abstain한다.

## E. 결과 파일과 중단 재개

```
$RUN/experts/<variant>/{candidate.pt,status.json,training.json}
$RUN/expert_selection.json
$RUN/study_split_manifest.json
$RUN/benchmarks/architecture/{comparison.json,comparison.csv,raw_results.json}
$RUN/policy/{policy.json,labels.json,policy_fit/,policy_tune/}
$RUN/policy/{policy_overhead.json,policy_overhead_manifest.json}
$RUN/benchmarks/policy_validation/{comparison.json,policy_coverage.json}
$RUN/policy_validation_evidence.json
```

학습량은 status의 updates/requested_updates/skipped를 확인한다. phase, RHS index,
noharm, coarse_complement, reference_history, gradient_norm은 training.json에 기록한다.
측정은 repeat_records 단위로 저장된다. 중단하면 **같은 명령·같은 옵션에 --resume만 추가**한다.

```bash
python scripts/run_v6_7_warm_study.py train --run-dir "$RUN" --variants H2 H2_NH --resume
python scripts/run_v6_7_warm_study.py policy-fit --run-dir "$RUN" --resume
python scripts/run_v6_7_warm_study.py policy-validate --run-dir "$RUN" \
  --probes 0 1 2 --tag policy_validation --repeats 5 --warmups 1 \
  --rhs-counts 1 4 16 64 --regimes warm_multiple multiple --resume
```

Expert 선택 전에서만 train resume 가능하다. 선택 후에는 expert 변경을 의도적으로 차단한다.
서로 다른 protocol로 benchmark를 수행할 때는 새 tag를 쓴다. Source/환경 변경은 새 RUN 필요.
Policy training은 expert 변경 전 timing evidence를 재사용하지 않는다.

## F. 정확한 classical portfolio 진단

사용자 로컬의 **원래 RUN**에 저장된 정확한 A/RHS를 재생성한다. 새 demo로 대체하지 않는다.

```bash
OLD_RUN="artifacts/실제_이전_research_RUN_이름"
python scripts/run_v6_7_warm_study.py diagnose \
  --source-run "$OLD_RUN" --case validation_channel_n63_1_r0 \
  --rhs-count 64 --repeats 3 \
  --output artifacts/channel_n63_portfolio_diagnostic.json
```

기존 report에서 정확한 실패 RHS index를 안다면 `--rhs-index 17`처럼 추가할 수 있다.
모르면 전체64개를 돌린다. 후보16 x RHS64 x repeat3이므로 계산량이 크다.
원래 manifest에 해당 case가 없으면 거부한다. 진단은 selector를 변경하지 않는다.
다른 후보가 성공하면 selected-parent rescue이며, 모든 테스트 후보가 실패한 경우에만
해당 portfolio/budget의 경험적 성공영역 확대라고 해석한다. 세상의 모든 classical solver를
이겼다는 뜻은 아니다. 원래 사용자 RUN은 이 구현 검증 환경에서 재현 실행하지 않았다.

## G. Final/OOD: 설계 결정을 마친 뒤에만

```bash
python scripts/run_v6_7_warm_study.py freeze --run-dir "$RUN" \
  --probes 0 --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 \
  --regimes warm_multiple multiple

python scripts/run_v6_7_warm_study.py final --run-dir "$RUN"
# 중단된 동일 평가만 재개:
# python scripts/run_v6_7_warm_study.py final --run-dir "$RUN" --resume
```

freeze는 expert/checkpoint/selector/source/config/split plan/실제 architecture 및 독립
policy-validation evidence/policy/probe 수를 고정한다. smoke에서는 불가능하다.
기존 final-grid/topology/coefficient/OOD 구성을 유지하며 실제 final은 single-use claim이다.
테스트에서만 아주 작은 별도 synthetic final plan을 검증한다. 실제 연구 final/OOD는 열지 않았다.

Final 결과를 보고 수정하면 그 set은 개발자료가 된다. 새로운 untouched holdout이 필요하다.
최종 성공 여부는 same tolerance 성공률, paired speedup/CI, 실제 Neural 사용률,
policy coverage와 setup 포함/제외 scope를 함께 봐야 한다. `freeze`/`final completed`가
수학적 정확도 인증이나 속도 우월성 인증은 아니다.
