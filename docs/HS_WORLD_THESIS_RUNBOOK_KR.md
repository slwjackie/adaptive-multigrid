# World-Model-Guided Adaptive Neural Smoothing for Time-Varying Multigrid Solvers

## 0. 이번 주 연구 경로

기준 main: `7ec2a645b2add2a523a755bd5dffad63a6745c81`.
새 주 실행 파일: `scripts/run_v6_7_hs_world_study.py`.

- 메인 정적 비교는 **C_tuned vs C_tuned + H_S**이다.
- C_tuned는 calibration-fit에서 고른 **하나의 전역 classical plan**을 tune에서 검증한 것이다.
  Operator/크기마다 다른 plan을 고르지 않는다. EM, schedule도 선택된 plan의 일부다.
- 88개 후보 기반 PDE-adaptive strong_C는 `strong-audit`를 요청했을 때만 보정한다.
  별도 `strong_C`/robustness-only 결과이며 H_S 학습/선택의 분모가 아니다.
- H_P/H_SP 기존 코드, checkpoint 형식, 테스트는 보존한다. 새 경로에서는 학습/호출하지 않는다.
  Classical P/EM, restriction P^T, Galerkin Ac=P^TAP 자체는 계속 필요하다.
- H0/H1/H2/H2_NH 아키텍처를 유지한다. 네 가지 중 실제 validation으로 후보를 선택한다.
- World Model은 화염장이 아니라 **solver의 행동별 비용·수렴·다음 상태**를 예측한다.
- Synthetic coefficient sequence를 hydrogen CFD라고 부르지 않는다. 실제 OpenFOAM solver plugin,
  chemistry/Navier–Stokes 적분, 연소 물리량 검증은 이 저장소에 구현하지 않았다.

## 1. 비교와 해석

정적 주평가는 같은 A/RHS/x0/tolerance/parent P/coarsening/schedule/backend에서 C와 H_S를 비교한다.
`comparison.json`의 `reference_arm`은 `C_tuned`, 행별 배율은 `speedup_vs_reference`이다.
기존 legacy evaluator의 기본 reference/필드 이름은 그대로 유지했다. 이전 CLI의 `strong_C`를
단순히 C_tuned로 이름만 바꾼 것이 아니라, 전역 plan 선택과 명시적인 reference를 추가한 것이다.

C_tuned 선택은 fit 성공 수를 먼저 보며, 기존 anchor 성공을 잃지 않는 후보의 동일 fit cohort
시간으로 순위를 정한다. Tune은 선택한 하나의 후보를 검증하며, 실패하면 사전 anchor로 돌아간다.
Tune에서 두 번째/세 번째 후보를 다시 골라서 validation을 tuning에 쓰지 않는다. 후보가 충분히
robust하지 않으면 실패 수가 그대로 기록된다. 이 과정은 전역 최적이나 전체 PDE 수렴 보장이 아니다.

H_S 선택은 architecture-validation의 실제 warm 시간과 새 실패를 본다. 느린 H_S도 단순 개발
후보로 선택될 수 있다. `superior_in_observed_gm=false`이면 성능 성공으로 주장하지 않는다.
해당 validation을 보고 아키텍처를 고쳤다면 최종 독립 test와 구분한다.

시간 시퀀스 비교군:

| 이름 | 내용 |
|---|---|
| C_rebuild | 매번 C_tuned hierarchy 재구성 |
| C_tuned_reuse | TUNE에서 고른 비신경망 P 재사용/재구성 정책, 주 분모 |
| HS_rebuild | 매번 같은 C_tuned + H_S 재구성 |
| HS_matched_reuse | C_tuned_reuse와 동일한 heuristic + H_S |
| HS_tuned_reuse | H_S 쪽에서도 별도 TUNE으로 고른 가장 나은 heuristic |
| World_C | Classical-only 행동으로 학습한 recurrent model |
| World_HS | Classical/Neural smoothing을 선택하는 제안 모델 |
| World_HS_horizon1 | 동일한 world 모델의 1-step planning ablation |

`World_C`와 `World_HS`는 각각의 행동/방문 상태에 맞춰 별도로 학습한다. World_C가 H_S의
label을 보고 학습하지 않는다. 결과에는 C_tuned_reuse뿐 아니라 HS_matched_reuse, World_C
대비 paired contrast도 함께 들어간다. 다른 cohort의 평균 배율을 곱해서 결합 이득을 만들지 않는다.

## 2. 행동과 수치 계약

행동을 **P 재사용 여부 × H_S 사용 여부**로 명시했다.

| 행동 | P | S |
|---|---|---|
| REBUILD_C | 현재 A로 classical P hierarchy 재생성 | Classical |
| REUSE_C | 기존 classical P 유지 | Classical |
| REBUILD_HS | 현재 A로 classical P hierarchy 재생성 | 현재 A에서 H_S 생성 |
| REUSE_HS | 기존 classical P 유지 | 현재 실제 hierarchy에서 H_S 생성/캐시 |

같은 A/mesh/boundary/plan/expert라면 P, Ac, LU와 준비된 H_S를 그대로 재사용할 수 있다.
**A 값이 바뀌면 P는 재사용하더라도 모든 실제 Ac와 factor를 다시 구성하고 H_S도 갱신한다.**
예전 A로 만든 smoother 또는 LU를 shape가 같다는 이유로 계속 사용하지 않는다.
C로 전환하면 모든 level의 neural overlay를 사용하지 않는다. H로 전환해도 P는 classical이다.
신경망 P checkpoint나 inherited learned P는 새 backend에서 거부한다.

실패 trial은 마지막 수용 iterate로 rollback하고 현재 A의 C_tuned로 복구한다.
원래 tolerance/initial residual 기준/전체 attempt budget은 유지한다. 복구 비용도 포함한다.
C 자체가 budget 안에 실패하면 실패로 보고한다. 안전장치는 수렴/속도 인증서가 아니다.

## 3. World Model

GRUCell 기반 small ensemble과 action-conditioned 다음 관측 예측을 유지한다.
H_S 활성 상태와 현재 A에 대한 H_S cache-ready 상태를 별도 feature로 추가했다.
새 action/schema와 이전 H_P 지향 world checkpoint는 호환되지 않는다.

학습 때는 같은 이전 bank에서 가능한 행동을 각각 실제 실행해 counterfactual label을 수집한다.
실제 미래 snapshot은 다음 관측 label에만 사용한다. 실행 API `WorldSolver.step(snapshot)`은
현재 snapshot 하나만 받고, horizon2에서는 예측한 다음 관측만 사용한다.

TUNE의 trajectory별 pairwise optimism margin으로 행동을 보수적으로 승인한다.
불확실하면 **TUNE에서 정한 classical reuse heuristic**을 따른다. 이전 world 모델처럼 항상
매번 classical rebuild로 abstain하지 않는다. 정확히 같은 A를 재사용할 수 있어도 모델이 나쁘면
추론 overhead로 느릴 수 있다. 이득은 실제 sequence total로 평가해야 한다.

비신경망 heuristic은 NN-only feature 추출이나 weight hash를 불필요하게 실행하지 않는다.
World Model의 feature/관측/추론 비용은 제안 모델 시간에 포함한다.

## 4. 설치·업데이트

기존 변경을 commit/보관한 뒤 저장소 루트에서 실행한다.

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
python -m pytest tests/test_hs_world_thesis.py -ra
python -m pytest -ra
shasum -a 256 -c HS_WORLD_SOURCE.sha256
```

Native build는 선택 사항이다. 기본 새 config는 CSR이며 calibration 이후 backend/thread/library를
바꾸지 않는다. 코드가 달라졌으므로 **기존 RUN을 resume하지 말고 새 RUN을 생성**한다.

## 5. 첫 smoke: C_tuned 및 H_S

```bash
RUN="artifacts/hs_world_smoke_$(date +%Y%m%d_%H%M%S)"
echo "$RUN"

python scripts/run_v6_7_hs_world_study.py calibrate \
  --config configs/v6_7_hs_world_smoke.json --run-dir "$RUN"

python scripts/run_v6_7_hs_world_study.py train \
  --run-dir "$RUN" --variants H0 H1 H2 H2_NH

python scripts/run_v6_7_hs_world_study.py benchmark \
  --run-dir "$RUN" --variants H0 H1 H2 H2_NH --tag architecture \
  --repeats 3 --warmups 1 --rhs-counts 1 4 \
  --regimes warm_multiple multiple
```

Smoke의 fit/tune은 n=7,15에서 각각4 operator, H_S train/validation도 각각4 operator이다.
각 smoother8 updates이므로 파이프라인 점검이지 성능 일반화 입증이 아니다.

88개 strong_C 보조 비교가 필요하면 **select 전에** 아래를 실행한다.
이 단계는 생략 가능하며, 메인 C_tuned나 H_S weight를 바꾸지 않는다.

```bash
python scripts/run_v6_7_hs_world_study.py strong-audit --run-dir "$RUN"
python scripts/run_v6_7_hs_world_study.py benchmark \
  --run-dir "$RUN" --variants H0 H1 H2 H2_NH --tag architecture_with_audit \
  --repeats 3 --warmups 1 --rhs-counts 1 4 --regimes warm_multiple multiple
```

위 audit를 했다면 이후 select의 tag를 `architecture_with_audit`로 사용한다.

```bash
python scripts/run_v6_7_hs_world_study.py select --run-dir "$RUN" --tag architecture
# 수치 근거를 보고 직접 고를 때:
# python scripts/run_v6_7_hs_world_study.py select --run-dir "$RUN" --tag architecture --variant H2_NH
```

선택 후에는 같은 RUN에서 expert를 재학습하거나 다시 선택하지 못한다.
No-new-failure 후보가 없으면 selection을 거부한다. 성능 악화를 숨기려고 guard를 해제하지 않는다.

## 6. World Model: H_S를 고정하고 시간 정책 학습

```bash
python scripts/run_v6_7_hs_world_study.py world-prepare --run-dir "$RUN"
python scripts/run_v6_7_hs_world_study.py collect --run-dir "$RUN"
python scripts/run_v6_7_hs_world_study.py world-train --run-dir "$RUN"
python scripts/run_v6_7_hs_world_study.py world-validate --run-dir "$RUN" --repeats 3
```

H_S checkpoint는 선택한 것을 자동 연결한다. 별도 랜덤 H_S를 성공 모델처럼 넣지 않는다.
Smoke는 시퀀스당4 systems, train/tune/validation/test=3/2/2/2개 independent case이다.
World_C, World_HS 각각 ensemble2, hidden12, 4epochs를 학습한다.
합성 데이터라는 scope는 manifest와 결과에 계속 표시한다.

## 7. Research 설정

Smoke 후 **새 RUN**에서 config만 변경한다.

```bash
RUN="artifacts/hs_world_research_$(date +%Y%m%d_%H%M%S)"
python scripts/run_v6_7_hs_world_study.py calibrate \
  --config configs/v6_7_hs_world_research.json --run-dir "$RUN"
python scripts/run_v6_7_hs_world_study.py train \
  --run-dir "$RUN" --variants H0 H1 H2 H2_NH
python scripts/run_v6_7_hs_world_study.py benchmark \
  --run-dir "$RUN" --variants H0 H1 H2 H2_NH --tag architecture \
  --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 --regimes warm_multiple multiple
```

Research 기본값: n=15/31/63, classical 후보12개, fit/tune 각각63 operators,
H_S train168/validation42 operators, variant마다1120 updates.
원하면 select 전에 optional strong-audit를 추가한다. Benchmark의 primary reference는 계속 C_tuned다.

이후 select/world-prepare/collect/world-train/world-validate 순서는 smoke와 같다.
Research temporal 기본값:16 systems/case, train/tune/validation/test=20/8/8/8 cases,
ensemble3, hidden24,100epochs. 이것은 검증할 시작 설정이지 추천 속도비/보장값이 아니다.

모든 변화하는-A 시퀀스 시간에는 준비/갱신/feature/GRU/solve/recovery가 포함된다.
Static warm_multiple은 준비를 제외하며 multiple은 준비 한 번 포함한다. 서로 다른 scope를
같은 성능이라고 보고하지 않는다. Disk I/O와 결과 직렬화는 모두 동일하게 제외한다.

## 8. 중단 재개·결과

`calibrate`, `train`, `benchmark`, `strong-audit`, `collect`, `world-validate`, `test`는
같은 설정/프로토콜에 `--resume`으로 재개한다. 완료된 temporal validation/test는 다시 열지 않는다.
World-model fitting은 초기 버전에서 epoch별 resume를 제공하지 않는다. 일부 파일만 남았을 때
무작정 덮어쓰지 말고 별도 RUN에서 다시 수집·학습한다.

```
$RUN/tuned_classical.json
$RUN/selector_rules.json                    # uniform C_tuned rules
$RUN/strong_audit_rules.json                # optional, separate
$RUN/experts/{H0,H1,H2,H2_NH}/status.json
$RUN/benchmarks/architecture/comparison.json
$RUN/expert_selection.json
$RUN/temporal/transitions/{world_C,world_HS}/{train,tune}/
$RUN/temporal/training.json
$RUN/temporal/{world_C,world_HS}.pt
$RUN/temporal/validation/report.json
```

최우선 지표: 성공 case 수, 실제 accepted neural cycles, C_tuned/C_tuned_reuse 대비 paired 시간,
World_C/HS_matched_reuse 대비 추가 이득, World_HS horizon1 vs horizon2.
`neural_systems=0`이면 해당 결과를 Neural smoothing 가속이라고 주장하지 않는다.

## 9. Freeze와 최종 독립 평가

```bash
python scripts/run_v6_7_hs_world_study.py freeze --run-dir "$RUN"
python scripts/run_v6_7_hs_world_study.py test \
  --run-dir "$RUN" --component sequence --repeats 5
python scripts/run_v6_7_hs_world_study.py test \
  --run-dir "$RUN" --component static --repeats 5 --warmups 1 \
  --rhs-counts 1 4 16 64 --regimes warm_multiple multiple
```

독립 temporal validation 이후 checkpoint, rules, tuning, 데이터, evidence를 함께 고정한다.
Static test는 freeze 후에 새 operator를 생성한다. Research static test 크기는127/255이고,
이전 개발에서 해당 resolution을 보지 않았을 때만 grid-OOD라고 부른다.
Temporal test 기본값은 **새 physical trajectories**이며 자동으로 미관측 grid가 되는 것은 아니다.
실제 test 결과를 보고 바꾸면 그 test는 개발자료가 되므로 새 untouched set이 필요하다.

Smoke의 test는 smoke의 작은 synthetic holdout이며 실제 연구 final을 소비하지 않는다.
이번 구현 검증에서만 작은 별도 test 계획을 실행한다. 실제 research final/OOD는 실행하지 않았다.

## 10. 실제 OpenFOAM/GAMG 연결 범위

기존 finalized-LDU recorder/importer를 사용한다. Serial scalar SPD, nested structured2D,
경계/reference 반영 완료, nullspace 없음, native matvec witness와 cell 순서가 필요하다.
임의의3D/unstructured/MPI pressure matrix를 지원한다고 주장하지 않는다.

```bash
python scripts/run_v6_7_hs_world_study.py world-prepare \
  --run-dir "$RUN" --input-ldu pressure_exports
```

`pressure_exports`는 finalized-LDU 형식이지 일반 OpenFOAM case 폴더가 아니다.
하나의 physical case는 시간 순서가 보존된 완전한 하나의 trajectory로 넣는다.
그 case를 train/tune/test로 나누지 않는다. Static calibration/H_S 데이터와의 동일 A도 거부한다.

Native GAMG를 직접 실행하는 plugin은 없다. 외부에서 실제 GAMG를 측정했을 때만:

```bash
python scripts/run_v6_7_hs_world_study.py gamg-compare \
  --run-dir "$RUN" --split validation --reference gamg_sequence_results.json
```

Reference schema는 `openfoam-gamg-sequence-v1`: solver='GAMG', OpenFOAM 버전,
설정해시, imported data_sha256, hardware(machine/cpu_model/affinity_count), execution_threads=1,
time_scope='linear_sequence_setup_solve_recovery', trajectories[].runs[].total_seconds,
systems[]의 step/matrix_digest/rhs_digest/x0_digest/threshold/final_true_residual/success가 필요하다.
Hashes는 본 study의 structured CSR 및 RHS 순서 기준으로 맞춰야 한다.
비교기는 일치하지 않는 데이터·초기값·정지조건·시간범위를 거부한다.

이는 외부 측정 evidence 비교이며 OpenFOAM 플러그인 실행 증거가 아니다. 연소 관련 pressure
system을 풀어도 CFD 전체 시간, 온도/화학종/질량보존까지 검증한 것은 아니다.
`in Hydrogen Combustion CFD`라는 제목 확장은 실제 해당 데이터/연동/물리 검증 완료 범위에 맞춘다.

Python integration은 freeze 후 아래처럼 시작한다.

```python
from adaptive_mg.v67.hs_world.temporal import solver_from_run
solver = solver_from_run(run_dir, mode="World_HS")
x, diagnostics = solver.step(current_verified_snapshot)
# 서로 다른 물리 case를 시작할 때:
solver.reset()
```

정적 benchmark는 기존 PreparedStrongMG의 safeguard를 사용하고, temporal benchmark는
현재 A에 대한 sequence safeguard를 사용한다. 둘은 MG kernel을 공유하지만 시간 정책·복구
시점은 같지 않을 수 있다. Static 속도비와 sequence 속도비를 곱하지 않고 각각 직접 측정한다.
