# World-model-guided time-varying Neural MG

## 구현 범위 — 무엇을 실제로 계산하는가

기준 main: `f115ca49f7678ed53ae1b6586c388be565217edd`.
새 CLI: `scripts/run_v6_7_world_study.py`.
기존 EM/schedule, H0/H1/H2/H2_NH, H_P/H_SP training과 static policy는 변경하지 않는다.
새 경로는 시간 순서로 들어오는 **이미 조립된 scalar SPD 선형계**를 실제 기존 MG
kernel로 풀며, 작은 action-conditioned recurrent world model이 hierarchy 작업을 고른다.

**이 패키지는 수소 화염장을 생성하거나 Navier–Stokes/chemistry를 적분하지 않는다.**
실제 OpenFOAM 설치, 원본 H2 pressure-matrix 시퀀스, OpenFOAM runtime solver plugin은
이 구현에 포함되지 않았다. `source_kind=synthetic_elliptic`의 demo를 combustion CFD로
부르면 안 된다. 외부 LDU recorder/importer 및 Python sequential-solver API를 제공한다.
실제 CFD 측의 boundary-finalized export/호출, 물리 정확도와 전체 CFD 시간 검증은 별도다.

## 1. World model의 의미

모델은 화염장 surrogate가 아니라 **solver 상태의 동역학 모델**이다.
작은 GRUCell(hidden24, 기본 ensemble3)가 현재 관측, 이전 행동, 실제 이전 결과를 받아
memory를 갱신하고, 행동별로 다음을 예측한다.

- hierarchy 준비시간, solve시간, 경험적 residual contraction, cycle 수
- solve 성공 확률
- 다음 solver 관측 상태(행렬 변화·age·비용·상태 통계 등)

입력에는 A의 규모/대각/이방성/방향, 직전 A와의 상대변화, P age, 초기 residual,
허용오차, 이전 결과 및 classical plan이 들어간다. 외부 CFD가 제공한 density/temperature
평균은 optional feature이고, 합성 데이터에서 가짜 물리량을 만들어 넣지 않는다.

학습은 실제 각 행동을 동일 이전 hierarchy에서 실행해 counterfactual label을 모은다.
다음 A는 offline dynamics label 생성에만 사용한다. Online에서는 **현재 snapshot 하나만**
받으며 실제 미래 A를 미리 읽지 않는다. Horizon2는 예측된 다음 관측으로 한 단계 더
rollout한다. 1-step cost의 tune-based conservative gate를 통과한 후보들만 미래비용으로
순위를 매긴다. 따라서 현재 비용을 희생해야만 얻는 긴 horizon 이득은 이 초기 버전이
공격적으로 탐색하지 않는다. 이는 Dreamer/PlaNet 재현이나 성능 보장 모델이 아니다.

## 2. 행동 정의 및 수치 계약

| 행동 | 수행하는 작업 |
|---|---|
| REBUILD_C | 현재 A의 frozen selector plan으로 classical hierarchy 재구성 |
| REUSE_P | 모든 level의 P 재사용; A가 바뀌면 Ac와 수치 factor는 재구성 |
| REFRESH_FINE | fine P만 현재 A 기준으로 갱신, 아래 P 유지; 모든 실제 Ac/factor 갱신 |
| REBUILD_H | 현재 A에서 기존 H_S/H_P/H_SP bank builder 호출 |

REBUILD_H는 호환되는 trained expert checkpoint가 있을 때만 후보에 들어간다.
Checkpoint 없이 실행하면 **world-model controller + classical MG** 실험이다. 랜덤
checkpoint를 trained expert처럼 만들거나 benchmark 결과에 넣지 않는다.

정확히 같은 A, mesh/boundary/plan/expert 계약일 때만 REUSE_P가 전체 bank/LU를 재사용한다.
A의 수치값이 바뀌면 항상 `Ac_new = P_old.T @ A_new @ P_old`를 계산하고 line/LU를 새로 만든다.
이 의미의 partial reuse는 기존 classical 연구가 있다. 새 요소는 그 작업의 비용·동역학
예측 및 실행 정책이며, reuse 자체를 새 방법으로 주장하면 안 된다.

Mesh ID, boundary ID, grid shape, fine sparsity pattern, 선택된 classical plan, expert가
다르면 기존 hierarchy 재사용을 허용하지 않는다. 현재 지원은 nested structured 2D이다.
H_S가 들어 있으면 변경된 실제 A에서 smoother bank도 재생성한다. A-conditioned stencil을
stale하게 재사용하지 않는다. REFRESH_FINE의 P reference는 현재 fine classical P이고,
아래는 이전 hierarchy의 P다. 이것은 기존 all-level frozen-parent 재학습의 의미와 다른
**명시적인 temporal update action**이다. REBUILD_H는 기존 계약 그대로다.

Temporal reused/partial hierarchy에는 injection/full-rank, finite, row-L1, 전체 complexity
검사를 한다. Neural fine-update에는 현재 fine baseline-relative cap도 검사한다. Temporal
계층이 항상 '지금 완전히 다시 만든 C*와 동일한 hierarchy'라는 뜻은 아니다.

Residual이 증가/비정상화하면 trial iterate를 버리고 마지막 accepted iterate에서 현재 A의
C*로 복귀한다. Stagnation/예산도 확인한다. 처음의 stopping threshold와 전체 attempt budget은
복귀 후에도 바꾸지 않는다. C* 자체가 budget 내에 실패하면 실패로 보고한다.
Outer Krylov, 정답을 이용한 보정, 미래 operator oracle는 없다.

## 3. 파일

```
src/adaptive_mg/v67/world_model/
    data.py       # 순서/물리 case 분리, snapshot hash, SPD admission, finalized LDU import
    adapter.py    # Python-side finalized-LDU recorder, native 순서로 해 벡터 복원
    backend.py    # 실제 hierarchy reuse/update/rebuild, 기존 MG numerical kernels, recovery
    learning.py   # GRU dynamics ensemble, training, empirical calibration, short planning
    study.py      # data -> collect -> train -> validation -> freeze -> test
scripts/run_v6_7_world_study.py
configs/v6_7_world_model_{smoke,research}.json
tests/test_sequence_world_model.py
```

## 4. 설치 및 기존 코드 보호

저장소 루트에서, 로컬 수정을 먼저 보관한다.

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
python -m pytest tests/test_sequence_world_model.py -ra
python -m pytest -ra
```

이전 EM/warm RUN은 읽기 전용 source로만 쓸 수 있다. 새 world RUN은 따로 만든다.
Source/config/thread/library/rules/expert 변경 후 과거 timing evidence를 resume하지 않는다.

## 5. 바로 실행할 smoke — 합성 시퀀스이며 연소 CFD 아님

```bash
RUN="artifacts/world_smoke_$(date +%Y%m%d_%H%M%S)"
python scripts/run_v6_7_world_study.py prepare --run-dir "$RUN" \
  --config configs/v6_7_world_model_smoke.json
python scripts/run_v6_7_world_study.py collect --run-dir "$RUN"
python scripts/run_v6_7_world_study.py train --run-dir "$RUN"
python scripts/run_v6_7_world_study.py evaluate --run-dir "$RUN" \
  --split validation --repeats 3
```

기본 smoke: n=7/15, trajectory당5개 system, train4/tune3/validation2/test2 trajectories,
ensemble3, GRU hidden24, 12 epochs. Coefficient의 완만한 변화/급변/같은 A의 새 RHS가 들어간다.
원본 source-run을 안 주면 고정 `line_alt_energymin_full__em5__v11` plan을 사용한다.
이 기본 demo를 88개 후보에서 재보정된 strong portfolio라고 주장하지 않는다.

`collect` 중단 시 같은 명령에 `--resume`을 붙이면 완료 trajectory를 재사용한다.
`train`은 고정 budget을 한 번 실행한다. Epoch 중간 resume는 제공하지 않는다.
학습 도중 중단됐다면 world.pt가 생기지 않은 상태에서 같은 train을 다시 시작한다.
Validation/test는 trajectory×repeat×baseline 단위로 저장하며, 중단된 동일 protocol만
`--resume` 가능하다. 완료된 평가를 같은 run에서 반복해 보고 유리한 결과만 고를 수 없다.
Source 변경은 새 RUN을 요구한다.

독립 validation을 검토한 후 다음을 실행한다.

```bash
python scripts/run_v6_7_world_study.py freeze --run-dir "$RUN"
python scripts/run_v6_7_world_study.py evaluate --run-dir "$RUN" --split test --repeats 3
```

여기서 test는 **새 world-sequence 데이터셋의 별도 시퀀스**다. 기존 static MG의 final/OOD를
자동으로 열지 않는다. Synthetic test를 통과해도 hydrogen CFD 검증을 의미하지 않는다.

## 6. 실제로 학습된 Neural MG expert 연결

```bash
SOURCE_RUN="artifacts/실제_em_hp_research_RUN"
RUN="artifacts/world_with_hp_$(date +%Y%m%d_%H%M%S)"
python scripts/run_v6_7_world_study.py prepare --run-dir "$RUN" \
  --config configs/v6_7_world_model_research.json \
  --source-run "$SOURCE_RUN" \
  --expert-checkpoint "$SOURCE_RUN/experts/H_P/candidate.pt"
python scripts/run_v6_7_world_study.py collect --run-dir "$RUN"
python scripts/run_v6_7_world_study.py train --run-dir "$RUN"
python scripts/run_v6_7_world_study.py evaluate --run-dir "$RUN" --split validation --repeats 5
```

source-run에서 solver config와 frozen selector rules를 가져온다. Trained branch와 rules
signature가 checkpoint와 다르면 거부한다. H_S/H_SP는 world config의 `expert_branch`를
그에 맞춰 설정한 **새 run**을 만든다. 어느 branch도 이 workflow에서 몰래 재학습하지 않는다.
Research 기본은 n15/31/63, trajectory당16 systems, train20/tune8/validation8/test8,
100 epochs이다. 이 숫자는 최적 설정이나 예상 speedup이 아니며 실제 CFD 데이터량에 맞춰
별도 development에서 결정해야 한다.

## 7. 비신경망 기준선과 timing

항상 classical rebuild뿐 아니라 always-reuse P, periodic(2/4), drift(.01/.05/.2)를
같은 numerical safeguard 아래에서 TUNE trajectory로 측정한다. C가 성공한 trajectory를
잃지 않는 후보 중 성공률/총시간 순으로 practical reference를 고정한다. Test에서는
사후 최선 baseline을 골라 이름을 바꾸지 않는다. Trained expert가 있으면 항상 H rebuild도
추가 비교한다. No-NN reuse만으로 나온 개선을 world-model 고유 성과로 주장하지 않는다.

각 trajectory는 별도 fresh stream이다. 시퀀스 전체 시간에는 feature/hash/selector,
world inference, hierarchy 준비, numerical refactor, rejected trials, classical recovery가
들어간다. 준비된 행렬 시퀀스의 디스크 읽기와 사후 report 직렬화는 모든 방법에서 제외한다.
학습용 counterfactual actions 실행비용은 offline data-generation 비용이다.
**이 수치는 CFD 전체 wall-clock이 아니다.** 실제 CFD coupling의 변환/통신/전체 반응계산
시간은 아직 측정하지 않았다. Matrix admission의 SPD 검사는 데이터 ingestion 단계에 있고,
solver 단계는 finite/symmetry/shape를 다시 검사한다. SPD admission도 numerical check이며
모든 부동소수점 환경의 엄밀한 고유값 증명은 아니다.

`speedup_vs_tune_selected_baseline = T_reference / T_method`다. Pair가 모두 성공한
trajectory의 repeat-median ratio를 사용하고 trajectory 단위 bootstrap을 한다.
같은 시퀀스의 timesteps/RHS/반복 측정을 독립 표본처럼 늘리지 않는다.

```
$RUN/training.json
$RUN/collection.json
$RUN/transitions/{train,tune}/*.json
$RUN/world.pt
$RUN/validation/report.json
$RUN/test/report.json
```

주요 항목: success/new_failure_trajectories, 전체 시퀀스 시간, setup시간, action counts,
fallback, decision_reason, 실제 learned transfer flag. Calibration margin/ensemble disagreement는
경험적 불확실성 정보일 뿐 no-harm/속도 보장이 아니다. Policy가 모두 C를 고르면 Neural
가속 성공이 아니다. 본인이 검사한 validation을 보며 바꾸면 새 independent test가 필요하다.

## 8. OpenFOAM/CFD 데이터 입력 계약

현재 제공된 것은 **Python 측 recorder/importer와 sequential solve API**다. OpenFOAM의
fvSolution에 없는 `WorldMG` 이름만 써서 호출할 수 있는 C++ plugin은 제공하지 않는다.

OpenFOAM native wrapper에서 FINALIZED scalar system을 넘겨야 한다. Foundation v13
`fvScalarMatrix::solveSegregated()`는 boundary diagonal과 source를 더한 뒤 ldu solver를
호출한다. 그 단계보다 앞의 내부 diag/source만 저장하면 실제 pressure system과 다를 수 있다.
Pressure-reference/nullspace, cyclic/processor coupling까지 처리해야 하므로, 초기 지원은
**anchored SPD, serial, coupled interface 0개, explicit structured2D cell mapping**으로 제한한다.
Pure Neumann, nonsymmetric/transonic, MPI, unstructured3D는 조용히 변환하지 않고 거부한다.

Python-side recorder:

```python
from adaptive_mg.v67.world_model.adapter import FinalizedLduRecorder
recorder = FinalizedLduRecorder("pressure_exports", producer="solver/version/commit", physics="H2-air case description")
# 각 solve에서 native wrapper가 공급한 FINALIZED 배열로 record()를 호출한다.
# 전체 인자와 저장 schema는 adapter.py 및 tests/test_sequence_world_model.py 참고.
```

NPZ 필수 배열:
`diag`, `lower_addr`, `upper_addr`, `lower`, `upper`, `b`, `x0`, `probe_vectors`,
`probe_products`, `structured_to_native`. `A[lower_addr,upper_addr]=upper`, 반대가 lower다.
`probe_products`는 외부 코드의 실제 matrix application에서 계산한 두 개 이상의 독립 witness다.
Exporter가 스스로 동일 CSR로 계산한 값을 넣으면 boundary 누락을 검증하지 못한다.

`structured_to_native[i]`는 structured x-major 위치 i에 해당하는 native cell index다.
Permutation이 수학적으로 유효한지 검사하지만 실제 mesh가 그 tensor-product ordering을
따르는지는 native wrapper가 책임져야 한다. Test에서 Nx×Ny reshape만 해서 임의 mesh를
지원한다고 주장하면 안 된다. `mesh_id`와 `boundary_id`는 좌표/경계/reference 변경을 반영한
안정적인 digest여야 한다. 각 pressure solve의 index는 엄격히 증가, 물리 time은 비감소다.
같은 time의 PIMPLE correctors도 다른 index로 저장한다.

Manifest는 `ldu_sequence.json`이고 schema는 `finalized-ldu-sequence-v1`이다.
Trajectory마다 `case_group`, `split`을 지정한다. 같은 물리 case의 time 구간을 train/test로
임의 분할해서 누출시키지 않는다. 단일 case 하나만으로 독립 case 일반화를 주장할 수 없다.

```bash
RUN="artifacts/world_cfd_$(date +%Y%m%d_%H%M%S)"
python scripts/run_v6_7_world_study.py prepare --run-dir "$RUN" \
  --config configs/v6_7_world_model_research.json \
  --source-run "$SOURCE_RUN" \
  --expert-checkpoint "$SOURCE_RUN/experts/H_P/candidate.pt" \
  --input-ldu pressure_exports
```

이후 collect/train/evaluate 절차는 동일하다. Synthetic에서 학습한 artifact를 외부 CFD에
그대로 적용하려 하면 domain guard가 C로 돌린다. CFD sequence로 별도 학습·보정한다.

## 9. 실제 runtime 결합 API

`WorldMGSolver.step(snapshot)`은 미래 A 없이 현재 system의 x와 기록을 돌려준다.
한 CFD case는 한 solver instance를 사용하며, 독립 case 전환 때 `reset()`한다.
Native order로 돌려놓을 때 `solution_to_native(x, permutation)`을 사용한다.
반환 `success=False`를 OpenFOAM이 성공처럼 처리하면 안 된다. 원 CFD의 reference solver로
복귀하거나 timestep 처리를 실패로 중단해야 한다. 이 native 호출/에러 처리 wiring은 아직
제공하지 않았으므로 실시간 reactingFoam 가속이 구현됐다고 주장하지 않는다.

실제 H2 논문 검증에는 pressure time뿐 아니라 전체 CFD 시간, mass conservation, 온도/종/열방출
변화, timestep/PIMPLE 수, 동일 이산화/chemistry/허용오차 검증이 추가로 필요하다.

## 근거와 novelty 범위

- Demidov (2021), *Partial Reuse AMG Setup Cost Amortization Strategy for the Solution of Non-Steady State Problems*:
  https://arxiv.org/abs/2108.02054 — P 재사용과 current coarse matrices/smoother rebuild의 고전 기준.
- OpenFOAM Foundation v13 `fvScalarMatrix.C`:
  https://cpp.openfoam.org/v13/fvScalarMatrix_8C_source.html — finalized boundary/source 위치.

GRU, partial reuse, ensemble, model-based planning 자체가 새로운 발명이라는 주장은 없다.
이 구현의 검증할 가설은 action-conditioned solver dynamics가 **강한 비신경망 reuse 정책**보다
실제 time-varying CFD 압력계의 전체 sequence 비용을 줄이는가이다. 현재 구현·합성 테스트는
그 가설의 성능 우월성 또는 hydrogen combustion 응용 성공을 입증하지 않는다.
