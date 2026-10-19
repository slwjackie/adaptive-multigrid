# Line_alt 고정 + learned P

이 실험은 NN smoother를 제거하고 `line_alt` pre/post smoothing을 고정한다.
새 기본 설정과 `run_v6_7_learned_p_study.py`가 이 경로를 사용한다. 이전 joint
실험은 명시적인 이전 설정 파일을 사용할 때만 재현할 수 있다. 기존 run을
새 설정으로 바꾸거나 freeze/hash를 수정하지 않는다.

## 실행하는 연산

| 항목 | 현재 설정 |
| --- | --- |
| Coarsening | full 고정 |
| Pre/post smoother | classical line_alt, 각각 2회 |
| NN smoother | 없음: 파라미터·저장된 smoother 가중치·추론 모두 0 |
| 학습 대상 | width8 residual CNN이 생성하는 P 가중치만 |
| 초기 P·fallback | bilinear |
| Restriction | R=P.T |
| Coarse 행렬 | 실제 learned P로 Ac=P.T@A@P |
| 최하위 풀이 | LU |
| 학습 level | 최하위 직접 풀이를 제외한 모든 level |
| 독립 reference | line_alt + EM5 + full |

CNN은 3,208개 파라미터이며 P의 허용된 16개 후보 슬롯을 출력한다. 활성 연결은
bilinear support로 제한하고, 실제 경계 행합과 coarse 대표점 injection을 보존한다.
Galerkin으로 내려간 실제 Ac를 다음 level CNN에 입력하며 이를 미분한다.
NN smoother 객체 대신 zero-parameter marker를 저장하므로 P-only checkpoint에는
CNN smoother 가중치가 없다. 기존 Components 컨테이너의 비활성 detector/controller는
호환성을 위해 남지만 학습·추론하지 않는다.

## Spectral radius를 목표로 Frobenius 손실을 쓰는 이유

고정 A와 setup에서 생성한 P에 대해 raw V-cycle은 선형이다. 따라서 오차는

\[
e_{k+1}=E_\theta e_k
\]

로 전달된다. 이 선형 반복의 점근 수렴을 지배하는 것은 spectral radius이고,
정확한 행렬 norm 사이에는 다음 관계가 있다.

\[
\rho(E_\theta)\leq\|E_\theta\|_2\leq\|E_\theta\|_F.
\]

Greenfeld et al.과 Luz et al.은 two-grid의 spectral radius를 목표로 하되
실제 학습에는 squared Frobenius norm을 사용했다. 본 구현은 그 목적을 **실제 전체
multilevel V-cycle**의 오차 연산자로 옮긴다. 논문과 동일한 아키텍처/성능이라는
의미는 아니다.

- https://proceedings.mlr.press/v97/greenfeld19a/greenfeld19a.pdf
- https://proceedings.mlr.press/v119/luz20a/luz20a.pdf

하지만 큰 E 행렬을 구성하면 N×N 메모리가 필요하다. 이 구현은 독립적인 ±1
Rademacher 벡터 z의 공분산이 I인 성질을 사용한다.

\[
\mathbb E_z\|E_\theta z\|_2^2=\|E_\theta\|_F^2,\qquad
\widehat L_F=\frac1{qN}\sum_{j=1}^{q}\|E_\theta z_j\|_2^2.
\]

N은 미지수 수이며, /N은 격자마다 손실 크기를 정규화한다. 기본 q는 8이다.
`b=0`, 초기 오차 `x=z`에서 **한 번의 전체 V-cycle**을 실행한 출력이 E z다.
이 경로에서 line_alt, P, Ac, 하위 line solve, 최하위 직접 풀이까지 미분한다.
학습은 항상 한 cycle의 E를 사용한다. 이전 residual 학습의 3→6 cycle curriculum과
시간/SPSA 손실을 이 기본 경로에 섞지 않는다.

Frobenius는 spectral radius의 대리 목표다. Norm bound는 정확한 Frobenius norm에
대한 것이며 유한 표본의 추정값이 매번 spectral radius의 상한이라는 뜻은 아니다.
Frobenius가 줄어도 실제 cycle 수·총시간이 반드시 개선되지는 않는다. 특히 fallback과
classical tail을 포함하는 배포 solver 전체가 하나의 고정 E인 것은 아니다.

## 직접 학습, P 탐색, 증류

`train`은 `learned_P` 하나만 학습한다. `learned_S`, `joint` 단계는 실행하지 않는다.
직접 학습에는 smoother 또는 residual loss가 없고 위 Frobenius 손실만 사용한다.

선택적 `search`도 P의 허용 원소만 추가 최적화한다. 초기값은 직접 학습 P,
bilinear P, 그리고 직접 학습 출력에 작은 변화를 준 후보들이다. Smoother는
모든 후보에서 line_alt 그대로다. 학습 A마다 기본 4 starts×200 updates를 사용한다.
P 갱신 때 실제 Ac와 하위 계층을 다시 구성한다. 후보의 초기/최적화 상태를
같은 held-out probe와 fallback 없는 실제 시간으로 비교한다.

`distill`은 선택된 P oracle을 작은 CNN에 옮긴다. Frobenius task loss에 P 원소
matching과 전체 V-cycle 보정 action loss를 더하고, 이후 P CNN만 task 미세조정한다.
Teacher와 student 어느 쪽에도 NN smoother는 없다. 이는 NN smoother와 P의 공동
학습이 아니다. 유효하지 않은 탐색 후보는 기록하고 label로 쓰지 않는다.

| Arm | P | Smoother |
| --- | --- | --- |
| fixed_C | bilinear | line_alt |
| em5_C | EM5 | line_alt |
| learned_P | Frobenius 직접 학습 | line_alt |
| distilled_P | P-only 탐색 후 증류 | line_alt |

EM은 em5_C에서만 계산한다. 학습 setup이나 fallback 준비에는 EM을 호출하지 않는다.
최종 후보는 동일 tolerance에서 실측 setup+solve 시간으로 선택하고, 실패한 문제를
버려서 성공 cohort만으로 순위를 매기지 않는다.

## 학습 전후 증거와 재사용

`train` 뒤에 `diagnose`를 실행하면 search/distill 없이도 직접 학습의 효과를 확인한다.
TRAIN과 별도 validation의 같은 A·동일한 고정 error tensor·한 cycle에서 초기 P와
학습 P의 Frobenius 추정값을 비교한다. 초기 P는 bilinear이고 smoother도 동일하다.
진단 기본 probe 수는 16이다. 초기/학습 모델의 같은 RHS에서 실제 tolerance까지
setup+solve 시간, cycle, rollback, accepted neural cycle, setup failure도 별도로 측정한다.
`diagnostics/p_frobenius.json`에 저장하며 이 진단으로 checkpoint를 선택하지 않는다.
최종/OOD를 이 과정에서 사용하지 않는다.

여기서 before의 시간은 **초기 CNN이 bilinear P를 생성하는 P-only 경로**의 시간이다.
순수 classical보다 빨라졌는지는 이 before/after 시간만으로 판단하지 않는다.
순수 classical과의 공정한 총시간 비교는 benchmark의 fixed_C와 수행한다.

같은 A에서는 이미 적극적인 정확 재사용을 한다.

1. 각 level의 P를 한 번 생성한다.
2. R, Ac, line factors와 LU를 준비한다.
3. 같은 A의 모든 cycle과 RHS에서 저장한 연산자를 재사용한다.

따라서 같은 PreparedAdaptiveMG에 여러 RHS를 전달하면 매 cycle/RHS CNN을
재실행하지 않는다. `solve_many`는 batch당 freshness 검사도 한 번 수행한다.
한편 benchmark의 cold repeat는 이와 별도로 새 setup부터 다시 측정한다.

A가 바뀌면 현재는 P를 다시 생성하고 모든 수치 계층을 갱신한다. 새 A에 대해
이전 P를 계속 쓰는 방법은 가능하지만 별도 approximate-reuse 정책이 필요하다.
그때도 Ac=P.T@A_new@P, line factor, LU는 새 A에 맞게 갱신해야 한다.
P와 함께 오래된 Ac/factor까지 재사용하면 다른 선형계를 푸는 오류가 된다.
현재 구현은 A 변경을 감지하면 정확하게 재생성하며, stale P 재사용은 추가하지 않았다.

Smoother와 P를 각각 학습한 뒤 공동 미세조정하는 방법은 향후 별도 ablation이 될
수 있다. 다만 먼저 P-only가 line_alt와 잘 결합하는지 확인하고, NN smoother가
line_alt 대비 이득을 보인 뒤에 검토하는 것이 실험 원인을 분리하기 좋다.
현재 기본 경로에 NN smoother나 joint fine-tuning을 다시 넣지 않는다.

## Mac 실행

```bash
git clone --single-branch --branch residual-u-net-teacher \
  https://github.com/slwjackie/adaptive-multigrid.git \
  "$HOME/Desktop/adaptive-multigrid-ponly"
cd "$HOME/Desktop/adaptive-multigrid-ponly"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set -o pipefail
SCRIPT="scripts/run_v6_7_learned_p_study.py"
P_RUN="artifacts/learned_p_smoke_$(date +%Y%m%d_%H%M%S)"
python "$SCRIPT" prepare --run-dir "$P_RUN" \
  --config configs/v6_7_learned_p_smoke.json && \
caffeinate -i python -u "$SCRIPT" train --run-dir "$P_RUN" \
  2>&1 | tee "$P_RUN/train.log" && \
caffeinate -i python -u "$SCRIPT" diagnose --run-dir "$P_RUN" \
  2>&1 | tee "$P_RUN/diagnose.log"
```

먼저 diagnose 결과를 읽는다. P-only 손실이나 cycle이 좋아지지 않으면 곧바로
대규모 search/distill 비용을 쓰기보다 원인을 확인한다. 나머지 전체 pipeline은:

```bash
caffeinate -i python -u "$SCRIPT" search --run-dir "$P_RUN" \
  2>&1 | tee "$P_RUN/search.log" && \
caffeinate -i python -u "$SCRIPT" distill --run-dir "$P_RUN" \
  2>&1 | tee "$P_RUN/distill.log" && \
caffeinate -i python -u "$SCRIPT" diagnose --run-dir "$P_RUN" \
  2>&1 | tee "$P_RUN/diagnose_after_distill.log" && \
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$P_RUN" \
  --repeats 3 --warmups 1 --rhs-counts 1 4 --regimes cold warm multiple \
  2>&1 | tee "$P_RUN/benchmark.log"
```

본 학습은 새 run의 prepare에서 `configs/v6_7_learned_p_research.json`을 사용한다.
Train n=15/31/63의 84개 A, validation n=15/31/63/127의 28개 A, size OOD n=255다.
직접 P 학습 336 updates, P-only search 4×200/A, 증류 336 + P 미세조정 168 updates다.
Smoke는 n=7, train/validation 각 7개 A, 직접 학습 14 updates, 탐색 2×4/A,
증류 14 + 미세조정 7 updates다. 두 업데이트짜리 이전 smoke와 예산이 다르다.

본 run에서 validation 후 freeze/final:

```bash
python "$SCRIPT" freeze --run-dir "$P_RUN" \
  --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple
caffeinate -i python -u "$SCRIPT" final --run-dir "$P_RUN" \
  2>&1 | tee "$P_RUN/final.log"
```

Smoke는 freeze/final에 사용할 수 없다. 중단된 train/search/distill/benchmark/final은
동일 환경·설정으로 `--resume`을 사용한다. `diagnose --resume`은 같은 checkpoint의
기존 진단을 읽고, 증류 후 새 모델까지 진단할 때는 `--resume` 없이 실행한다.
새 터미널에서는 SCRIPT와 P_RUN을 다시 설정해야 한다. 소스가 달라진 이전 run의
hash를 고쳐 재개하지 않는다. 구현·smoke 성공은 연구 규모의 속도 개선 증거가 아니다.
