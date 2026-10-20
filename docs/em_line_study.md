# EM 모방 P → 줄 smoother → 공동 미세조정

브랜치는 **`residual-u-net-teacher`**다. 실행 파일은
`scripts/run_v6_7_em_line_study.py`다. 이전 P-only, sparse smoother, pruning/TTT
실험과 구분되는 **새 run**을 만든다. 이전 run의 manifest나 freeze 기록은 수정하지 않는다.

## 학습과 배포 구조

| 단계 / 평가 이름 | 학습 대상 | 고정된 부분 | 손실 |
| --- | --- | --- | --- |
| `em_P` | P 생성 CNN | Classical `line_alt` | 각 학생 level의 A에 대한 EM5 P 모방 |
| `frozen_P_line` | 줄 smoother 생성 CNN | 앞 단계에서 학습한 P CNN | 전체 V-cycle의 다중-cycle 잔차 손실 |
| `joint_line` | P CNN과 smoother CNN | Full coarsening, Galerkin 공식, 최하위 LU | 같은 전체 V-cycle 손실로 공동 미세조정 |

`train` 한 번으로 이 순서를 수행한다. 두 번째 단계는 첫 번째 checkpoint에서,
세 번째 단계는 두 번째 checkpoint에서 시작한다. 두 번째 단계에서는 P 가중치가
바뀌지 않았는지 검사한다. 각 단계의 마지막 checkpoint를 평가 후보로 보존한다.
세 번째 단계가 앞 단계보다 항상 빠르다고 가정하지 않는다.

첫 단계의 EM5는 **오프라인 정답 생성용 수치해석 알고리즘**이다. 각 level에서
fixed-budget 5회를 사용하되, 수치적으로 이미 정지한 경우의 실제 횟수도 기록한다.
EM 모방 손실은 보정할 수 있는 P 행의 정규화된 제곱 오차를 level별로 평균한다.
제한된 CNN과 P 제약 안에서 모방하므로 EM5 P를 정확히 재현한다는 보장은 없다.

학생 계층은 다음처럼 연결된다.

\[
P_\ell=\Pi_P(P_{\mathrm{bilinear},\ell}+\Delta P_\theta(A_\ell)),
\qquad A_{\ell+1}=P_\ell^T A_\ell P_\ell.
\]

다음 CNN 입력과 EM 정답은 모두 **이 학생이 만든 실제 \(A_{\ell+1}\)** 를 사용한다.
EM 정답을 만드는 수치 최적화는 역전파하지 않지만, 학생의
\(P_\ell\rightarrow A_{\ell+1}\rightarrow P_{\ell+1}\) 연결에는 역전파가 이어진다.
별도로 고정한 EM 계층의 행렬을 학생에게 입력하는 방식이 아니다.

P는 bilinear의 허용 연결, 경계, 행합, coarse 대표점 보존과 행별 크기 제한을 지킨다.
출력은 signed affine 보정이므로 음수 가중치도 표현할 수 있다. Coarsening은 full로
고정하고 \(R=P^T\), \(A_c=P^TAP\), 최하위 내부 격자 3×3 LU를 유지한다.

## CNN이 생성하는 줄 smoother

하나의 작은 CNN이 모든 nonterminal level에서 가중치를 공유한다. 입력은 실제
\(A_\ell\)의 10개 특징과 상대 level 위치, 격자 크기 특징이다. 출력 4개 채널은
x/y 방향별 줄 LDL 분해의 아래쪽 계수와 양수 pivot을 보정한다.
기본 중간 채널 수는 두 생성기 모두 8이다. P CNN은 3,208개, 줄 smoother CNN은
3,244개로, 공동 미세조정하는 두 NN의 파라미터는 합계 6,452개다.

\[
(A_\ell,\text{level 정보})
\xrightarrow{\text{공유 CNN}}
(L_{x,\ell},D_{x,\ell},L_{y,\ell},D_{y,\ell}).
\]

초기값은 실제 A에서 뽑은 줄의 삼중대각 성분으로 계산한다. CNN은 이 초기 계수에
제한된 보정을 더한다. 기본 9점 격자 연산자의 삼중대각 줄 블록에서는 초기 분해가
그 줄 문제의 정확한 풀이에 해당한다. Galerkin 과정에서 더 넓은 같은 줄 연결이
생기면 삼중대각 **근사**가 된다. 이때도 잔차 갱신은 잘라낸 행렬이 아닌 전체 A를 쓴다.

한 sweep은 x 방향 줄 보정 → 실제 A로 잔차 갱신 → y 방향 줄 보정을 수행한다.
필요한 잔차 갱신은 각 줄 그룹 사이에도 적용한다. 서로 결합된 같은 색 줄은
순차 처리한다. Pre/post는 계수를 공유하고 post에서 방향과 줄 그룹 순서를 뒤집는다.
Pre/post 각각 2 sweep으로 시작한다.

현재 온라인 줄 풀이는 FP64 NumPy 배열로 독립적인 줄을 묶어 처리하고,
줄 길이 방향의 전진·후진 대입에는 Python 반복문을 사용한다. 기존 classical의
컴파일된 sparse LU보다 빠르다고 가정할 수 없다. 계수 생성·재사용의 정확성과
실제 실행 속도는 별도로 검증해야 한다.

새 A에서 level마다 CNN을 한 번 실행해 계수를 생성한다. 생성한 P/R/Ac, 줄 분해,
최하위 LU는 동일한 준비 객체의 cycle과 여러 RHS에서 재사용한다. 새 프로세스에서
checkpoint를 읽는 것만으로 이전 A의 계층 캐시가 복원되는 것은 아니다. 다른 A에는
다시 setup이 필요하다. **매 cycle의 NN forward나 test-time training은 없다.**

양수 pivot은 각 줄 근사 행렬의 SPD 구조를 유지하지만 전체 MG 수렴을 보장하지는
않는다. 배포의 fallback 대상은 `line_alt + bilinear + full`이다. 잔차 증가·비정상 값·
learned setup 실패에 대한 기존 복구와 tolerance 근처의 classical 마무리를 유지한다.
속도 평가에는 이 복구 비용도 포함한다.

## 손실과 실제 모델 선택

두 번째·세 번째 단계는 무작위 오차, classical smoothing 뒤 남은 오차, 이전
V-cycle로 줄이기 어려운 오차에서 만든 잔차를 사용한다. 전체 계층을 펼쳐 처음에는
3 cycle, 이후에는 6 cycle의 잔차를 평가한다. 뒤 cycle에 더 큰 가중치를 주는
로그 잔차 손실과 잔차 증가에 대한 항을 사용한다. 학습과 별도 raw 검사에는
fallback을 허용하지 않는다.

이는 PDE의 이산 방정식 \(Ax=b\)를 사용하는 오프라인 물리 기반 학습이다.
좌표에서 해 \(u(x)\)를 직접 예측하는 별도 PINN을 붙이지 않는다. 벽시계 시간이나
정수 cycle 수를 그대로 역전파한다고 주장하지 않는다.

최종 선택 기준은 같은 residual tolerance를 만족할 때의 다음 실측값이다.

\[
T_{\mathrm{setup}}+\sum_{j=1}^{m}T_{\mathrm{solve}}(A,b_j).
\]

Setup에는 classical fallback 계층 준비, CNN 추론, P 제약 적용, Galerkin 구성,
줄 factor 생성·캐시 준비와 최하위 분해 등 배포에 필요한 비용이 포함된다.
EM5 비교군에서는 EM 계산도 setup에 포함된다. Learned 경로에서는 EM 정답 생성과
오프라인 학습 비용이 온라인 setup에 들어가지 않는다.

평가는 아래 5개 비교군을 사용한다.

| 이름 | P | Smoother |
| --- | --- | --- |
| `fixed_C` | Bilinear | Classical `line_alt` |
| `em5_C` | 각 level EM5 | Classical `line_alt` |
| `em_P` | EM 모방 CNN | Classical `line_alt` |
| `frozen_P_line` | 위 P를 고정 | 그 P에 맞춰 학습한 줄 CNN |
| `joint_line` | 공동 미세조정 P | 공동 미세조정 줄 CNN |

기본 `target_rhs_count=1`은 **setup + 하나의 RHS solve**를 선택 기준으로 삼는다.
Cold는 setup+1 RHS, warm은 이미 준비된 계층의 solve, multiple은 setup+여러 RHS다.
RHS 1/4/16/64는 같은 A에 대해 푸는 서로 다른 \(b\)의 개수다. 반복 측정의 case별
중앙값을 구한 뒤 case들에 대한 산술평균 총시간으로 후보를 비교한다.

선택에는 선언된 validation cohort의 모든 case·반복·RHS에서 성공한 후보만
허용한다. 실패한 case를 빼서 빠른 모델로 선택하지 않는다. Eligible learned 후보가
없으면 learned 모델을 선택하지 않는다. Learned 후보와 classical 모두 성공 조건을
만족하지 않으면 배포 추천도 비워 둔다. `em5_C`는 독립 참고
비교군이고 learned 후보 선택에는 넣지 않는다. 선택한 learned 후보보다 `fixed_C`가
빠르고 성공 조건을 만족하면 배포 추천은 `fixed_C`일 수 있다.

`speedup_vs_fixed = fixed_C 시간 / 비교군 시간`이므로 **1보다 커야 빠르다.**
Warm 개선, raw 잔차 개선, cycle 감소, setup 포함 속도 개선을 구분해서 읽는다.
Unit test와 n=7 smoke는 실행 검증이며 연구 규모의 속도 개선 증거가 아니다.

구현 확인용 n=7, 단계당 2-update CLI smoke에서는 모든 비교군이 7/7 성공했다.
RHS=1의 cycle 합계는 fixed_C와 세 learned 후보 모두 115회였고, setup+solve
선택은 fixed_C였다. 이 짧은 학습에서 속도 개선을 확인한 것은 아니다.

구현 시 전체 테스트는 904 passed / 7 skipped / 1 failed였고 새 테스트 63개는
모두 통과했다. 유일한 실패는 `test_headroom_source_manifest`로, 작업 시작 커밋
`1a7d3e1`에서도 `P_HEADROOM_SOURCE.sha256`과 기존 문서 두 개
(`HS_WORLD_THESIS_RUNBOOK_KR.md`, `P_HEADROOM_RUNBOOK_KR.md`)의 해시가 달랐다.
이 과거 체크섬 기록과 해당 문서는 이번 변경에서 수정하지 않았다.

## Mac 설치와 공통 터미널 환경

새 checkout을 만들 때만 다음을 실행한다. 기존 실행 중인 실험의 checkout은
갱신하지 말고 새 폴더를 쓰는 것이 좋다.

```bash
git clone --single-branch --branch residual-u-net-teacher \
  https://github.com/slwjackie/adaptive-multigrid.git \
  "$HOME/Desktop/adaptive-multigrid-em-line"
cd "$HOME/Desktop/adaptive-multigrid-em-line"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

새 터미널마다 다음을 실행한다. 기존 checkout을 쓰면 `cd` 경로를 바꾼다.

```bash
cd "$HOME/Desktop/adaptive-multigrid-em-line"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set -o pipefail
SCRIPT="scripts/run_v6_7_em_line_study.py"
git branch --show-current
git log -1 --oneline
test -f "$SCRIPT" && python "$SCRIPT" --help
```

기존 clone을 갱신하는 경우 `git status --short`에서 로컬 수정을 확인한 뒤
`git pull --ff-only origin residual-u-net-teacher`를 사용한다. 이미 준비한 run을
재개할 때는 소스·패키지·스레드 설정을 유지한다. Manifest를 편집해서 환경 검사를
우회하지 않는다. `caffeinate -i`는 idle sleep을 막지만 덮개를 닫은 상태의 실행을
보장하지 않으므로 장시간 실행 시 전원을 연결하고 덮개를 열어 둔다.

## Smoke 실행

Train/validation 각각 7개 A, n=7, 각 단계 2 updates다. `prepare`가 run 폴더를
생성하므로 prepare 전에 해당 폴더에 `tee` 로그를 만들지 않는다.

```bash
EM_LINE_SMOKE="artifacts/em_line_smoke_$(date +%Y%m%d_%H%M%S)"
caffeinate -i python -u "$SCRIPT" prepare --run-dir "$EM_LINE_SMOKE" \
  --config configs/v6_7_em_line_smoke.json && \
caffeinate -i python -u "$SCRIPT" train --run-dir "$EM_LINE_SMOKE" \
  2>&1 | tee "$EM_LINE_SMOKE/train.log" && \
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$EM_LINE_SMOKE" \
  --repeats 1 --warmups 0 --rhs-counts 1 4 --regimes cold warm multiple \
  2>&1 | tee "$EM_LINE_SMOKE/benchmark.log"
```

## 본 학습과 validation

| 설정 | 예산 |
| --- | --- |
| Train | 7 families × n=15,31,63 × 20 = 420개 A |
| Validation | 7 families × n=15,31,63,127 × 5 = 140개 A |
| P 사전학습 | 4,200 updates, 각 level EM5 정답 생성 |
| P 고정 smoother 학습 | 1,400 updates, 3→6 cycle |
| 공동 미세조정 | 1,400 updates, 3→6 cycle |
| 기본 선택 workload | `multiple`, RHS=1, 같은 tolerance |

업데이트 수는 시작 예산이다. 충분한 학습 여부는 학습 손실뿐 아니라 raw contraction,
cycle 수, 실패 수와 총시간으로 확인한다. Mac M2 Pro의 이 새 실험 전체 시간은
실측 전이므로 확정된 시간을 제시하지 않는다. Stage별로 EM 정답 생성과 다중-cycle
역전파가 있어 이전 P-only 학습 시간과 같다고 가정하면 안 된다.

```bash
EM_LINE_RUN="artifacts/em_line_research_$(date +%Y%m%d_%H%M%S)"
caffeinate -i python -u "$SCRIPT" prepare --run-dir "$EM_LINE_RUN" \
  --config configs/v6_7_em_line_research.json && \
printf '%s\n' "$EM_LINE_RUN" > artifacts/em_line_current.txt && \
caffeinate -i python -u "$SCRIPT" train --run-dir "$EM_LINE_RUN" \
  2>&1 | tee "$EM_LINE_RUN/train.log" && \
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$EM_LINE_RUN" \
  --repeats 3 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple \
  2>&1 | tee "$EM_LINE_RUN/benchmark.log"
```

140개 A, 5 arms, 3 repeats, 위 6개 workload 조건이면 **12,600개 시간 측정**이다.
Warmup과 fallback 없는 raw-cycle 검사는 별도다. Validation을 시작한 뒤 가중치를
변경하지 않는다. 새 설계나 학습 예산을 비교하려면 새 run을 만든다.

## 진행 상태와 중단 후 재개

새 터미널에서 공통 환경 블록을 실행한 다음 run 경로를 복원한다.

```bash
EM_LINE_RUN="$(cat artifacts/em_line_current.txt)"
ps -ax -o pid,stat,pcpu,time,etime,command | grep '[r]un_v6_7_em_line_study.py'
python - "$EM_LINE_RUN" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for rel in (
    "checkpoints/em_P/status.json",
    "checkpoints/frozen_P_line/status.json",
    "checkpoints/joint_line/status.json",
    "benchmarks/validation/progress.json",
):
    path = root / rel
    print(rel, json.loads(path.read_text()) if path.exists() else "아직 생성되지 않음")
PY
```

프로세스가 종료된 것을 확인한 뒤 중단된 학습은 다음으로 재개한다.

```bash
caffeinate -i python -u "$SCRIPT" train --run-dir "$EM_LINE_RUN" --resume \
  2>&1 | tee -a "$EM_LINE_RUN/train.log"
```

완료된 앞 단계는 검증해서 재사용하고 중단된 단계의 optimizer 상태부터 이어간다.
학습이 완료됐고 benchmark가 중단된 경우, **원래와 같은 평가 옵션**으로 실행한다.

```bash
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$EM_LINE_RUN" --resume \
  --repeats 3 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple \
  2>&1 | tee -a "$EM_LINE_RUN/benchmark.log"
```

| Run 내부 경로 | 내용 |
| --- | --- |
| `checkpoints/<stage>/candidate.pt` | 완료된 각 단계 모델 |
| `checkpoints/<stage>/training.json` | 학습 손실·진단 기록 |
| `checkpoints/<stage>/status.json` | 완료 update, 실패/skip 상태 |
| `benchmarks/validation/raw_results.json` | Setup·solve·cycle·fallback의 원 측정 |
| `benchmarks/validation/comparison.csv` | Classical 대비 시간·성공 비교 |
| `benchmarks/validation/cycle_diagnostics.json` | Cycle 수와 복구 사용 여부 |
| `benchmarks/validation/raw_contractions.json` | Fallback 없는 고정-cycle 검사 |
| `benchmarks/validation/selection.json` | 성공 조건과 실측 총시간에 따른 후보 선택 |
| `benchmark_evidence.json` | 선택에 사용한 모델·측정의 출처 |
| `selected/deployment.json` | 배포 추천 arm; classical 또는 선택 불가 상태도 명시 |
| `selected/candidate.pt` | Learned arm이 배포 추천된 경우에만 저장하는 모델 |

## Validation 이후 final/OOD

Smoke는 final에 사용할 수 없다. 연구 run의 학습·validation 선택이 끝난 뒤
모델, 소스, 평가 조건을 freeze하고 보관한 final/OOD에 적용한다. 최종 평가 결과로
다시 모델을 선택하거나 학습하지 않는다. 격자 OOD n=255도 이 단계에 속한다.

```bash
python "$SCRIPT" freeze --run-dir "$EM_LINE_RUN" \
  --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple && \
caffeinate -i python -u "$SCRIPT" final --run-dir "$EM_LINE_RUN" \
  2>&1 | tee "$EM_LINE_RUN/final.log"
```

중단된 final은 소스와 환경을 유지한 채 재개한다.

```bash
caffeinate -i python -u "$SCRIPT" final --run-dir "$EM_LINE_RUN" --resume \
  2>&1 | tee -a "$EM_LINE_RUN/final.log"
```

## 코드 위치와 이전 실험

| 위치 | 역할 |
| --- | --- |
| `src/adaptive_mg/v67/em_line_study.py` | 준비·순차 학습·실측 선택·freeze/final |
| `src/adaptive_mg/v67/em_line_models.py` | Signed affine P CNN과 components |
| `src/adaptive_mg/v67/em_line_training.py` | 실제 학생 계층 EM 모방·전체 V-cycle 목적함수 |
| `src/adaptive_mg/v67/line_smoother.py` | Level 입력, 줄 factor 생성, FP64 줄 적용 |
| `src/adaptive_mg/v67/unroll.py` | 미분 가능한 학생 Galerkin 계층과 V-cycle |
| `src/adaptive_mg/v67/banks.py` | 런타임 계층·연산자 캐시와 smoother 연결 |

이전 [P-only 설계](learned_p_study.md)와 [실행 안내](learned_p_runbook.md)는
`run_v6_7_learned_p_study.py`를 재현하는 문서다. 그 경로의 Frobenius 손실,
독립적인 sparse smoother 실험, pruning/TTT 결과를 새 실험의 결과로 해석하지 않는다.
