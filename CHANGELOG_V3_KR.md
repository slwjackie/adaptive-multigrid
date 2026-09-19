# Adaptive Neural Multigrid v3 — 전체 변경 내역

이 문서는 v2 코드 검토에서 제기된 문제와 v3 구현 중 추가로 발견한 문제를 하나씩 대응시킨 기록입니다.

## 1. 요청사항 및 지적사항 반영표

| 번호 | 분류 | 지적·요구사항 | v3 최종 반영 | 주요 구현 위치 | 검증 |
|---:|---|---|---|---|---|
| 1 | 학습 구조 | two-grid 학습과 multi-level V-cycle 추론의 불일치 | production과 동일한 재귀적 differentiable full V-cycle을 학습 loss에 사용 | `torch_ops.py`, `train.py` | full-V-cycle parity 및 cross-grid 테스트 |
| 2 | 학습 구조 | 레벨별 미세한 열화가 격자 크기에 따라 누적 | 연구 학습은 최소 transfer 2회, 즉 3-grid 이상을 요구; `n=7` two-grid는 verification profile에서만 명시 허용 | `grid.py`, `train.py`, `train_staged.py` | shallow-training 거부 테스트 |
| 3 | 일반화 | `n=15` 단일 격자 학습 | 기본 train roots를 `15,31,63`으로 설정 | `train.py`, `train_staged.py` | config 및 pool 테스트 |
| 4 | 일반화 | 학습과 같은 `n`에서만 validation | 기본 validation `15,31,63`, 별도 OOD `127` 지원; train에 없는 validation grid가 없으면 오류 | `train.py`, `train_staged.py` | cross-grid config 테스트 |
| 5 | learned P | two-grid 지표는 좋아도 실제 V-cycle에서 손해 | learned-transfer stage 선택과 rollback을 full-V-cycle A-norm validation으로 수행 | `train.py` | verification 학습에서 실제 rollback 확인 |
| 6 | loss | 1-cycle transient 과적합 | 여러 full V-cycle을 이어붙인 geometric A-norm contraction 사용 | `torch_ops.py`, `train.py` | multi-cycle gradient 테스트 |
| 7 | loss | raw NN 발산이 평균 loss에서 가려질 수 있음 | nonfinite/large contraction에 `raw_instability_penalty`, finite clamp 추가 | `train.py` | nonfinite robustness 테스트 |
| 8 | smoother | NN correction scale `0.20`이 사실상 무력화 가능 | additive correction 구조는 유지하고 기본 coefficient scale을 `0.75`로 상향; 안정성은 raw loss·safeguard로 분리 | `models.py`, checkpoint metadata | zero-init 동등성 및 raw/safe 테스트 |
| 9 | validation | 선택은 mean(log ρ), 출력은 mean(ρ) | 선택·출력·저장을 모두 geometric mean으로 통일 | `train.py` | 로그와 `best_validation` 일치 테스트 |
| 10 | validation | `val_rho < best`인데 갱신되지 않는 것처럼 보임 | `val_rho`, `val_objective`, corresponding best를 함께 출력 | `train.py` | verification training log 확인 |
| 11 | validation | stage별 서로 다른 metric을 같은 성능 수열처럼 저장 | `evaluation_mode`, `smooth_error_steps`, objective와 contraction을 분리 저장 | `train.py` | metadata schema 테스트 |
| 12 | validation | 학습 종료 후 동일 조건의 mode 비교가 없음 | 5개 mode를 동일 grid/error/smoothing protocol로 평가한 `common_validation` 추가 | `train.py`, `train_staged.py` | common-validation JSON 생성 테스트 |
| 13 | validation | stage가 validation을 악화해도 마지막 state 사용 가능 | best state 복원, 개선 실패 시 stage entry state로 전체 rollback | `train.py` | rollback 회귀 테스트 |
| 14 | coarse distribution | NN이 coarse Galerkin operator 분포를 충분히 못 봄 | full V-cycle computation graph 안에서 실제 Galerkin operators를 모두 통과하며 학습 | `torch_ops.py`, `train.py` | training hierarchy 테스트 |
| 15 | train/inference parity | 학습 Galerkin에만 작은 diagonal regularization이 있어 추론과 불일치 | 숨은 regularization 제거; 같은 `PᵀAP` 사용 | `torch_ops.py` | float64 parity 상대오차 약 `4e-16` |
| 16 | sparse 학습 | root matrix를 `A.toarray()`로 dense화 | root와 중간 operator를 Torch sparse로 유지 | `torch_ops.py` | source grep·memory profile |
| 17 | sparse 학습 | dynamic line smoother가 전체 coarse matrix를 dense화 | 필요한 1D line block만 sparse slicing 후 local dense solve | `torch_ops.py` | full-matrix dense 경로 제거 검사 |
| 18 | GPU | operator-dependent baseline이 SciPy CPU로 왕복 | Torch sparse approximate-ideal operator interpolation 경로 추가 | `torch_ops.py` | SciPy 기준과 약 `2e-7` 이내 비교 테스트 |
| 19 | learned P 표현력 | 최대 4개 bilinear coarse candidate에 제한 | 4×4 local support, 최대 16개 candidate로 확장 | `transfer.py`, `models.py` | shape, row-sum, rank 테스트 |
| 20 | learned P candidate | operator strength를 반영한 후보 선별 필요 | expanded 16-support 안에서 approximate-ideal weight와 configurable top-k를 동적으로 적용 | `transfer.py`, `torch_ops.py` | operator full-rank/SPD stress |
| 21 | learned P 안정성 | 극단적 OOD weight가 coarse injection까지 바꿔 P rank 훼손 가능 | geometric coarse-node injection row를 classical 값으로 고정; NN은 fine rows만 수정 | `transfer.py` | extreme random decoder rank/SPD 검사 |
| 22 | tied transfer | SPD 단계의 `R=Pᵀ`, Galerkin 구조 유지 | learned P, tied R, `A_H=PᵀAP` 유지 | `transfer.py`, `solver.py`, `torch_ops.py` | symmetry/SPD 테스트 |
| 23 | smoother baseline | 강한 line smoother 필요 | Jacobi, line-x, line-y, line-alt 모두 유지·캐싱 | `smoothers.py`, `strategy.py` | anisotropy baseline 검사 |
| 24 | semicoarsening | line smoother만 있고 semicoarsening 없음 | `full`, `semi_x`, `semi_y`, `alternating` 실제 직사각 hierarchy 구현 | `grid.py`, `transfer.py`, `solver.py`, `torch_ops.py` | rectangular Galerkin/SPD 및 수렴 테스트 |
| 25 | classical 비교 | 한 방법을 모든 문제의 최강 baseline으로 간주 | 통제된 9개 strategy catalog + optional 32-combination exhaustive benchmark | `strategy.py`, `run_baseline_benchmark.py` | 6 PDE baseline CSV |
| 26 | baseline 해석 | 비싼 semicoarsening을 residual만으로 “최강”이라 할 위험 | 최저 residual 표와 residual/work Pareto frontier를 모두 생성 | `run_baseline_benchmark.py` | `pareto_classical_by_case.csv` |
| 27 | 최종 구조 | classical 전략 자체를 상황에 따라 선택해야 함 | 18개 global operator feature → 9개 controlled classical strategy selector | `strategy.py`, `models.py`, `train.py`, `solver.py` | selector I/O 및 integration 테스트 |
| 28 | 최종 구조 | global selector와 local adaptive를 계층적으로 결합 | global classical selection 후 local detector/NN correction 적용 | `solver.py` | end-to-end mode 테스트 |
| 29 | detector 계산 | gate가 0이어도 전체 grid NN을 먼저 실행 | gate threshold 후 active tile+halo에서만 NN forward 실행 | `solver.py` | easy gate에서 smoother/transfer NN call 0 확인 |
| 30 | detector 계산 | selective computation 효과를 정량화할 수 없음 | active patch/cell fraction 및 NN별 call/time 기록 | `solver.py`, `reporting.py` | profile CSV |
| 31 | detector feature | feature 생성 코드가 solver/training/diagnostic에 3중 중복 | 공통 `detector_features.py`로 단일화 | `detector_features.py`, `solver.py`, `torch_ops.py`, `diagnostics.py` | NumPy/Torch 최대오차 회귀 테스트 |
| 32 | detector feature | post probe는 reverse, 학습은 forward | post-smoothing feature가 실제 `reverse=True` correction을 사용 | `solver.py`, `torch_ops.py` | reverse path 테스트 |
| 33 | detector 비용 | feature용 full zebra sweep을 한 번 더 실행 | pre-smoothing에서 계산한 correction/gate를 재사용 | `solver.py` | work/profile 비교 |
| 34 | dense P | `sparse_coo_tensor(...).to_dense()` 경고·비용 | 미분 가능한 `zeros().index_put(..., accumulate=True)`로 dense P assembly | `transfer.py` | warning-free assembly 테스트 |
| 35 | operator P 성능 | CSR 행별 Python loop와 `.toarray()` 병목 | COO key lookup, vectorized gather, row normalization, top-k | `transfer.py` | `n=31/63/127` profile |
| 36 | sparse drop 성능 | `_drop_sparse_relative` 행별 Python loop | `np.repeat` + `np.maximum.at` 기반 vectorization | `transfer.py` | numerical equivalence 테스트 |
| 37 | cache | transfer pattern을 반복 재구성 | shape/coarsening transfer pattern `lru_cache` | `transfer.py` | cache reuse 테스트 |
| 38 | cache | line factors 반복 생성 | level/direction별 sparse LU cache | `smoothers.py` | factor reuse 테스트 |
| 39 | adaptive hierarchy | 매 cycle 모든 learned level 재구축 | `adaptive_refresh=k`로 hierarchy 갱신 주기 제어 | `solver.py`, CLI | refresh profile |
| 40 | safeguard | fixed α grid search가 비싸고 최적 아님 | residual 2-norm exact closed-form line search | `solver.py` | grid-search 대비 비열화 테스트 |
| 41 | safeguard 실험 | safety가 raw NN 성능을 가림 | 모든 neural mode에 raw/safe profile 분리 | `run_ablation.py` | 별도 CSV rows |
| 42 | safeguard 통계 | alpha가 대부분 0인지 확인 불가 | alpha 원자료 JSON, mean/median/p10/p90/zero/active 비율 노출 | `reporting.py`, `run_ablation.py` | output schema 테스트 |
| 43 | safeguard 비용 | fallback 비용이 일반 work에 숨음 | `safety_work_units` 별도 집계 | `solver.py`, `reporting.py` | work counter 테스트 |
| 44 | fallback 보장 | 현재 neural iterate에서 classical 1-cycle과만 비교하면 pure classical trajectory 하한 미보장 | 독립 pure-classical shadow trajectory 유지, cycle count별 residual 하한 보장 | `solver.py` | random neural weights shadow-bound stress |
| 45 | fallback 범위 | adaptive에만 fallback 적용 | `nn_smoother`, `learned_transfer`, `combined`, `adaptive`에 공통 적용 | `solver.py` | mode별 safe 테스트 |
| 46 | CG 조건 | residual-dependent fallback을 CG preconditioner에 사용 가능 | CG에서 variable fallback·비대칭 pre/post·adaptive smoothing을 사전 거부 | `solver.py` | invalid-config 테스트 |
| 47 | FCG | restart 후 direction은 바뀌지만 old `rho=r·z` 사용 | restart direction 기준 `rho=r·p` 재계산 | `krylov.py` | forced-restart 회귀 테스트 |
| 48 | FCG | variable preconditioner에서 단순 recurrence 안정성 부족 | 저장된 directions를 이용한 full A-orthogonalization | `krylov.py` | random SPD integration 테스트 |
| 49 | FGMRES | estimated/true residual history 혼재 | `estimated_residual_history`, `residual_history` 분리 | `krylov.py`, `solver.py` | history semantics 테스트 |
| 50 | FGMRES | true residual 측정 간격이 달라 contraction 왜곡 | `residual_iteration_indices` 저장, iteration gap으로 log-rate 정규화 | `krylov.py`, `solver.py` | restart-gap contraction 테스트 |
| 51 | FGMRES | happy breakdown 후 zero basis 진행 가능 | 즉시 restart update | `krylov.py` | happy-breakdown 테스트 |
| 52 | FGMRES | Hessenberg system singular 가능 | direct solve 실패 시 least-squares fallback | `krylov.py` | singular-path 테스트 |
| 53 | roundoff | `tolerance=0`에서 machine floor 이후 breakdown | requested tolerance와 numerical residual floor를 함께 사용 | `krylov.py`, `solver.py` | zero-tolerance 테스트 |
| 54 | API | `case_suite`가 package root에 export되지 않음 | `adaptive_mg.case_suite` export | `__init__.py` | import 테스트 |
| 55 | grid validation | `n=9` 같은 비중첩 크기에서 깊은 재귀 예외 | root에서 `n=2^L-1` 및 coarsest reachability 명시 검증 | `grid.py`, `transfer.py`, `solver.py` | invalid-size 테스트 |
| 56 | checkpoint | old model 의미가 바뀌어도 로드될 위험 | checkpoint format version 3, dtype·asset status·config metadata, legacy 기본 거부 | `models.py` | compatibility 테스트 |
| 57 | checkpoint | 동봉 checkpoint가 smoke 수준인데 오해 가능 | metadata/README에 verification-only를 강제 표기; paper claim 경고 | README, checkpoint metadata | artifact inspection |
| 58 | checkpoint load | unsafe `torch.load(weights_only=False)` | `weights_only=True` 우선 | `models.py` | load 테스트 |
| 59 | packaging | ZIP에 `.pytest_cache`, `__pycache__` 포함 | packaging 전 cache·compiled artifact 제거, `.gitignore` 갱신 | packaging process | ZIP listing 검사 |
| 60 | work accounting | cycle 수만 비교하면 overhead 은폐 | work units, setup/solve, matvec, hierarchy rebuild, NN call/time, active area 기록 | `solver.py`, `reporting.py` | CSV schema 테스트 |
| 61 | stress | deterministic tests만으로 OOD structural bug를 놓칠 수 있음 | random PDE, strategy, extreme decoder, safeguard stress script 추가 | `run_stress_validation.py` | 12 cases, 204 checks |
| 62 | profiling | 수정 전후 병목이 정량화되지 않음 | operator P scaling, refresh, selective NN, sparse memory profile 추가 | `profile_v3.py` | profile CSV/JSON |

## 2. 구현 중 추가로 발견하여 고친 버그

| 추가 발견 | 위험 | 수정 |
|---|---|---|
| Pure-classical fallback 비교가 현재 neural state에서 시작 | 매 cycle local 비교는 통과해도 장기 trajectory가 pure classical보다 나빠질 수 있음 | 독립 shadow trajectory로 교체 |
| Extreme learned P가 coarse injection row를 변경 | column independence와 Galerkin SPD를 잃을 수 있음 | coarse row를 classical injection으로 고정 |
| Training Galerkin에만 작은 diagonal 보정 | inference와 다른 operator를 최적화 | 보정 제거, exact parity |
| GPU dynamic coarse operator에서 SciPy operator baseline 호출 | CPU roundtrip 및 gradient/device 불일치 | Torch sparse operator-dependent path |
| FGMRES true residual index 미기록 | restart 길이에 따라 asymptotic contraction 왜곡 | iteration index + gap normalization |
| Validation contraction이 nonfinite일 때 비교 불안정 | rollback selection 오류 가능 | finite conversion 및 penalty |
| Semicoarsening 후 square-grid 가정 | shape mismatch/잘못된 line indexing | 모든 level 자료구조를 `(nx,ny)`로 일반화 |
| Classical “최저 residual”만 보고 | work가 수배 큰 전략을 공정한 baseline으로 오인 | Pareto frontier 추가 |

## 3. 최종 검증 요약

- 회귀 테스트: **40 passed**
- random stress: **12 PDE cases, 204 checks, 0 failures**
- learned transfer zero-init/classical equivalence: 통과
- 16-support row-sum, injection, full rank, tied Galerkin SPD: 통과
- semicoarsening rectangular hierarchy: 통과
- raw/safe output 및 alpha statistics: 통과
- FCG/FGMRES edge cases: 통과
- pure-classical shadow bound: random neural weights에서도 통과
- package reinstall 후 test: 최종 ZIP 검증 단계에서 재실행

상세 수치와 한계는 `artifacts/verification_v3/VERIFICATION_V3_KR.md`에 정리했습니다.
