# EM / schedule / affine H_P 실행 안내

## 구현 범위

기준 main: `aded8e267f247c84df1948cd9dc2334dce606d91`.
새 실행 파일: `scripts/run_v6_7_em_transfer_study.py`.
기존 H0/H1/H2/H2_NH와 multistage smoother 구조는 수정하지 않았다. 공통 sparse 학습
연산과 classical plan 전달은 확장했고 기존 16개 global classifier output 순서는 유지했다.
이 구현은 성능 개선을 검증할 연구 경로이며, 1.10x 향상이나 OOD 수렴을 보장하지 않는다.

## 실제 변경 순서

1. EM-CG5/10과 V(2,2), V(1,2), V(2,1), V(1,1)을 `em_schedule` bank에 추가.
   총 88개의 **명시적 schedule plan**이다. 옛 controlled/all bank는 변경하지 않았다.
   plan 이름 예: `line_alt_energymin_full__em10__v11`.
   MGConfig는 plan의 pre/post/EM budget을 해석한다. C/H/fallback/cache/학습 모두 같은 plan을 쓴다.
2. 행합/injection 전용 EM tangent를 vectorized Helmert basis로 생성한다.
   여러 mode 제약은 기존 AffineSupport의 일반 구현을 유지한다.
   EM budget=10은 최적화 수렴 선언이 아니다. 실제 iterations/converged/energy/setup을 기록한다.
3. 학습 그래프에서 독립 zebra 색을 batch solve하고, 동일 forward의 동일 SparseTensor LU를 재사용.
   symbolic same-colour coupling이 있으면 sequential fallback. 실제 새 Ac의 factor를 항상 재구성.
   `.data`/외부 NumPy view로 tensor를 몰래 바꾸는 사용은 지원하지 않는다.
4. 기존 small GNN + affine head + parent-support-only decoder. NN decoder는 자유도가 있는 F-row의
   허용 edge만 처리한다. graph feature 전체를 완전히 ragged하게 다시 구현한 것은 아니다.
   `transfer_levels: "all"`은 모든 비터미널 level이며 H_S level 설정과 분리된다.
5. Random A-norm probes와 persistent slow probes, raw full-V-cycle bulk/tail/stability loss.
   Numerical validation은 실제 safeguarded NumPy solver의 diverse RHS warm 시간/성공률을 사용.
6. 선택한 H_P를 고정한 뒤 actual multi-RHS fit/tune data로 C/H_P cost policy를 fitting.
   H_S policy는 그대로. 새 H_P policy는 명시적 schedule plan과 frozen rules를 요구한다.

## Frozen parent의 정확한 의미

C*(A)의 전체 hierarchy를 먼저 구성한다. 각 level의 P_C*는 고정 reference이며 NN은
**실제로 변경된 A_l^H**와 P_C*, geometry를 입력받아 delta를 출력한다.

    P_l^H = P_C*,l + projected_delta_theta(A_l^H, P_C*,l)
    A_(l+1)^H = (P_l^H)^T A_l^H P_l^H

단순히 변경된 coarse A에서 SciPy EM을 재실행하고 그 gradient를 몰래 끊는 방식이 아니다.
EM baseline 계산 자체는 고정 reference이므로 미분 대상이 아니며, 실제 learned Galerkin Ac와
smoother에 대한 gradient는 유지된다. Zero head는 전체 classical hierarchy로 정확히 돌아간다.
Support/row-sum/injection/L1 row<=8/parent-relative complexity cap은 학습과 실행이 공유한다.
Dirichlet 경계 제거 후 행합은 **P_C*의 행합**이다. 무조건 1을 강제하지 않는다.
이는 이전 리뷰의 dynamic-EM-parent와 다른 명시적 설계 선택이다.

## 설치

기존 로컬 변경을 먼저 commit/보관한다. repository 루트에서:

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
python -m pytest tests/test_em_transfer_study.py -ra
python -m pytest -ra
shasum -a 256 -c THREE_PILLARS_SOURCE.sha256
shasum -a 256 -c P_HEADROOM_SOURCE.sha256
```

Native build는 선택 사항이며 CSR로도 실행 가능하다. 측정 중 backend/thread/library/source를
변경하지 않는다. **기존 RUN에 새 코드를 resume하지 말고 새로운 RUN을 만든다.**
이전 CLI는 호환성을 위해 남겼으며 새 H_P 학습은 아래 CLI에서만 선택된다.

## Smoke 전체 과정

```bash
RUN="artifacts/em_hp_smoke_$(date +%Y%m%d_%H%M%S)"
echo "$RUN"

python scripts/run_v6_7_em_transfer_study.py calibrate \
  --config configs/v6_7_em_hp_smoke.json --run-dir "$RUN"

python scripts/run_v6_7_em_transfer_study.py train --run-dir "$RUN"

python scripts/run_v6_7_em_transfer_study.py benchmark --run-dir "$RUN" \
  --tag hp_validation --repeats 3 --warmups 1 --rhs-counts 1 4 \
  --regimes warm_multiple multiple

python scripts/run_v6_7_em_transfer_study.py select --run-dir "$RUN" --tag hp_validation
python scripts/run_v6_7_em_transfer_study.py policy-fit --run-dir "$RUN"

python scripts/run_v6_7_em_transfer_study.py policy-validate --run-dir "$RUN" \
  --repeats 3 --warmups 1 --rhs-counts 1 4 --regimes warm_multiple multiple
```

Smoke: n=7,15, train14/validation14, 8 optimizer updates, width8, random2+slow2 probes,
3 raw cycles, numerical selection at 0/4/8 updates. Calibration uses actual distinct RHS
and excludes preparation; 88 plans × train/tune operators are measured.
Policy fit/tune/validation are separate normalized-A-disjoint splits, each4 operators.
Only sufficiently covered classical leaves are admitted; other sizes/leaves use explicit fixed V22.

Smoke failure on a shared C/H problem does not imply code corruption. Inspect raw residuals,
cycle limit and new failures separately. Successful test execution is not a performance result.

## 본 연구

```bash
RUN="artifacts/em_hp_research_$(date +%Y%m%d_%H%M%S)"
echo "$RUN"

python scripts/run_v6_7_em_transfer_study.py calibrate \
  --config configs/v6_7_em_hp_research.json --run-dir "$RUN"

python scripts/run_v6_7_em_transfer_study.py train --run-dir "$RUN"

python scripts/run_v6_7_em_transfer_study.py benchmark --run-dir "$RUN" \
  --tag hp_validation --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 \
  --regimes warm_multiple multiple
```

기본 research: n=15,31,63, NN TRAIN168, architecture validation42, calibration train/tune
각63. H_P width16 / 1120 updates / random8+slow8 probes / 4 cycles / persistent E^3 갱신.
Slow probes는 TRAIN operator별로 보존하고 매번 새 random probes를 섞는다.
80 updates마다 independent development validation의 실제 C/H_P warm timing으로 best를 고른다.
Validation probes/RHS는 training RNG와 분리되며 final/OOD는 열지 않는다.
이 설정은 시작점이지 optimal hyperparameter나 속도 예측이 아니다.

검증 후 expert를 고정한다:

```bash
python scripts/run_v6_7_em_transfer_study.py select --run-dir "$RUN" --tag hp_validation
python scripts/run_v6_7_em_transfer_study.py policy-fit --run-dir "$RUN"
python scripts/run_v6_7_em_transfer_study.py policy-validate --run-dir "$RUN" \
  --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 --regimes warm_multiple multiple
```

`select`는 new failures가 없는 warm evidence를 요구하지만 speedup>1을 강제하지 않는다.
즉 느린 candidate를 고정한 후 policy가 전부 C를 고르는 것도 정상이다.
실제로 빨라졌다는 주장은 paired operator confidence interval, 성공률, real neural usage,
setup scope를 모두 보고 판단한다. `best_step=0`이라면 zero-head reference가 선택된 것이며
학습 가속에 성공했다고 주장하지 않는다.

## 중단과 resume

같은 명령, 같은 옵션에 `--resume`을 추가한다. 학습에는 시험용 `--max-updates`가 있다.

```bash
python scripts/run_v6_7_em_transfer_study.py train --run-dir "$RUN" --max-updates 40
python scripts/run_v6_7_em_transfer_study.py train --run-dir "$RUN" --resume
python scripts/run_v6_7_em_transfer_study.py policy-fit --run-dir "$RUN" --resume
```

resume.pt에는 optimizer, training records, persistent probes, RNG, validation selection state가
함께 저장된다. 중단된 update는 마지막 원자적 저장 이후 다시 실행한다. candidate.pt는 마지막
weights가 아니라 validation-selected weights이며 last.pt와 best.pt를 별도로 보존한다.
Expert 선택 후 재학습, policy-validation을 본 뒤 policy 재튜닝은 같은 RUN에서 금지한다.

## 기록할 파일

```
$RUN/selector_rules.json
$RUN/selector_evidence.json
$RUN/calibration_records/{selector_train,selector_tune}/
$RUN/experts/H_P/{training.json,status.json,best.pt,last.pt,candidate.pt,resume.pt}
$RUN/experts/H_P/validation/step_*.json
$RUN/benchmarks/hp_validation/{comparison.json,comparison.csv,raw_results.json}
$RUN/hp_selection.json
$RUN/hp_policy/{policy.json,labels.json,policy_fit/,policy_tune/}
$RUN/benchmarks/hp_policy_validation/{comparison.json,policy_coverage.json}
$RUN/hp_policy_validation.json
```

Training log의 bulk/tail/stability, gradient norm, level별 relative P delta와 max row L1을 본다.
Setup feasibility-only update는 numerical performance loss로 표시하지 않는다.
Tail rho는 finite probes/finite cycles의 진단이지 정확한 spectral radius가 아니다.
동일 sparse support 자체가 동일 timing을 보장하지 않으므로 warm 시간을 직접 비교한다.

## Final/OOD (설계 결정을 모두 마친 후만)

```bash
python scripts/run_v6_7_em_transfer_study.py freeze --run-dir "$RUN" \
  --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 --regimes warm_multiple multiple
python scripts/run_v6_7_em_transfer_study.py final --run-dir "$RUN"
# 중단된 동일 final만 재개:
# python scripts/run_v6_7_em_transfer_study.py final --run-dir "$RUN" --resume
```

Smoke RUN은 freeze 불가. 실제 final은 기존 single-use claim과 source/config/checkpoint/policy/
독립 validation evidence pinning을 사용한다. Test suite만 아주 작은 synthetic final을 연다.
코드 작성 중 본 연구용 n=127/255 final/OOD나 full1120-update 연구학습은 실행하지 않았다.
연속 크기 policy는 지원 범위 밖에서 abstain할 수 있고 OOD 속도를 보장하지 않는다.

## 이번에 의도적으로 하지 않은 것

- H_S 신규 아키텍처, 강제 S→P/P→S joint training.
- Expanded support, learned non-Galerkin Ac, outer Krylov wrapper.
- 기존 p_headroom의 per-operator direct 탐색을 전역 최적/엄밀한 상한으로 재명명.
- Actual learned Ac에서 EM을 다시 계산하면서 그 gradient를 숨기는 처리.

기존 headroom/factorial CLI는 그대로 남는다. 새로운 all-level trained generator를 검증한 뒤
H_SP를 별도 계획/데이터에서 비교해야 한다. 이번 학습 경로는 H_P만 학습하며 H_S 상태는 보존한다.
