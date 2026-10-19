# Bilinear 기반 P·pre/post smoother 공동 학습

현재 기본 실험은 [line_alt 고정·P-only Frobenius 학습](learned_p_study.md)으로
변경되었다. 아래는 명시적인 이전 joint 설정 파일을 사용하는 과거 실험 재현용이다.
새 P-only run에는 smoother NN과 공동 학습 단계가 없다.

이 실험은 작은 두 CNN이 각 level의 `P`, `B_pre`, `B_post`를 함께 생성하도록
학습한다. 기존 U-net teacher, pruning, TTT, EM5 기반 sparse smoother 실험과는
별도의 새 run이다. 이 경로에는 TTT나 diffusion이 없다. 구현과 smoke 검증만으로
classical MG보다 빠르다고 결론 내리지 않는다.

## 계산 구조와 기준선

| 항목 | 설정 |
| --- | --- |
| Coarsening | classical full coarsening 고정 |
| 학습·fallback 기준 P | bilinear |
| 학습 대상 | P 생성 CNN, pre/post smoother 생성 CNN |
| Restriction | `R = P.T` |
| Coarse 행렬 | 실제 learned P로 `Ac = P.T @ A @ P` |
| 적용 level | 최하위 직접 풀이를 제외한 모든 level |
| Smoothing | pre 2회, post 2회 |
| 최하위 풀이 | LU |
| 배포 평가의 fallback | 같은 A의 `line_alt + bilinear + full` |
| 독립 classical 비교군 | `line_alt + EM5 + full`; level당 EM 5회 고정 |

NN 경로는 EM을 호출해서 기준 P를 만들지 않는다. EM5 계층은 `em5_C` 비교군을
측정할 때만 별도로 만든다. Neural setup에서 준비하는 classical recovery 계층도
bilinear 기반이다. Fallback은 실패한 학습 연산자의 비용을 없애 주지 않는다.
거부된 trial, rollback, classical recovery의 실행 시간과 cycle도 평가에 포함한다.
설정 파일의 공통 MG 필드에 `em_iterations: 5`가 남아 있어도 bilinear 전략은
그 EM 계산을 호출하지 않는다.

## 두 CNN과 생성되는 연산자

두 모델의 기본 폭은 8이고, 각각 5개 3×3 convolution을 사용하는 residual body다.
P 모델과 smoother 모델 사이에는 body를 공유하지 않는다. Smoother의 pre/post
두 head만 하나의 body를 공유한다. 같은 모델 가중치를 모든 level에서 재사용한다.

입력은 각 level에서 실제 만들어진 A의 10개 공간 특징이다. Diagonal의 상대적인
log 크기와 diagonal로 정규화한 9점 A stencil을 사용한다. CNN의 receptive field와
출력 연산자의 9점 연결 범위는 서로 다른 개념이다. Coarse A에 생긴 먼 연결은
행렬곱과 Galerkin 계산에는 남아 있지만, 이 10채널 입력이 그 모든 연결을 표현하는
것은 아니다.

| 모델 | 출력 | 초기화·제약 |
| --- | --- | --- |
| P 생성기 | 위치별 fine–coarse 가중치 보정, 16개 후보 슬롯 | bilinear의 기존 연결만 허용; 경계의 기준 행합과 coarse 대표점 injection 유지 |
| Smoother 생성기 | pre/post 각각 위치별 9점 방향 + level당 양의 scalar gain | 방향은 위치별 L2 정규화; 두 head 모두 gain 2/3인 damped Jacobi에서 시작 |

P 모델의 16개 출력 슬롯이 모두 활성 연결이 되는 것은 아니다. Bilinear P의
지원 집합을 보존하는 projection이 허용하지 않는 연결을 제거한다. 경계 행합을
무조건 1로 바꾸지 않고 해당 bilinear 행합을 보존한다. Sparse complexity 제한도
검사하며, 위반한 후보를 classical의 성공으로 대신 인정하지 않는다.

Smoother는 정규화된 residual `D^{-1}r`에 생성한 방향과 gain을 적용한다.
이를 하나의 희소 행렬 B로 조립하면 실제 smoothing은 다음과 같다.

```python
r = b - A @ x
x = x + B_pre @ r    # pre phase의 각 sweep

r = b - A @ x
x = x + B_post @ r   # post phase의 각 sweep
```

Pre에서 post head까지 연속으로 적용하지 않는다. 각 smoothing slot은 해당
phase의 B 한 개만 적용한다. Setup에서 CNN을 level당 한 번 실행해 두 B를 생성한
뒤, 같은 A의 모든 cycle과 RHS에서 재사용한다. P도 setup에서 생성하고 캐시한다.
완전히 sparse smoothing으로 대체한 learned level은 사용하지 않을 line factor를
미리 만들지 않는다. Recovery용 classical 계층의 준비 비용은 남는다.

양의 gain이나 `R=P.T`만으로 전체 learned V-cycle의 수렴을 보장하지는 않는다.
실제 tolerance까지의 성공 여부와 fallback 빈도를 별도로 확인한다.

## 직접 공동 학습과 어려운 오차

직접 학습은 동일 초기 모델에서 다음 세 실험을 독립적으로 시작한다.

1. `learned_P`: P CNN만 학습하고 `line_alt` 유지.
2. `learned_S`: bilinear P를 유지하고 smoother CNN만 학습.
3. `joint`: P와 pre/post smoother CNN을 함께 학습.

각 update에서는 전체 multilevel V-cycle을 미분한다. 학습 전반은 3-cycle,
후반은 6-cycle이며, P가 달라질 때 실제 Ac와 그 아래 hierarchy도 다시 만들고
그 영향을 gradient에 포함한다. 고정된 coarse 행렬에 대해서 P만 갱신하지 않는다.

동일한 학습 A에서 다음 residual probe를 섞는다.

- 무작위 오차 e로부터 만든 `r = A @ e`.
- Classical smoothing을 통과한 뒤 남은 오차의 residual.
- 현재 모델의 여러 raw V-cycle을 통과한 뒤 남은 오차의 residual.

Probe 생성은 gradient 밖에서 수행하고, 이후 cycle의 학습은 이 probe를 사용한다.
후반에 남는 어려운 모드를 오프라인에서 학습하는 구성이다. 실제 test 해의 정답이나
최종/OOD 데이터를 학습에 사용하지 않는다.

연속적인 Adam 학습에는 여러 cycle의 log residual과 증가에 대한 안정성 항을
사용한다. 이는 미분 가능한 대리 목적이다. 벽시계 시간이나 tolerance까지 필요한
정수 cycle 수를 직접 미분한다고 주장하지 않는다. 같은 architecture에서 상수인
연산량을 손실에 더해서 추론이 빨라지는 것으로 처리하지도 않는다.

## 학습용 A별 추가 탐색과 증류

`search`는 TRAIN A마다 허용된 P 가중치와 두 smoother의 방향·gain을 직접
최적화한다. 이 큰 자유변수 집합이 온라인 배포 모델은 아니다. 학생과 같은
희소 연결, P projection, 두 9점 smoother 표현을 사용한다.

연구 기본값은 A당 4개 초기값, 각 200 Adam updates다. 초기값은 공동 학습 모델의
출력, bilinear + damped Jacobi, 공동 학습 출력에 작은 변화를 준 두 후보다.
각 후보의 초기 상태와 최적화 후 상태를 모두 보관해, 추가 Adam이 오히려 나빠졌을
때 그 결과를 무조건 채택하지 않는다. P 갱신마다 실제 Galerkin 계층을 갱신한다.

최적화에 사용하지 않은 probe에서 동일 tolerance까지 실제 setup + solve 시간을
측정하고 성공한 후보 중 선택한다. 후보 선택에는 classical fallback을 허용하지
않는다. 실패한 시작점과 어떤 A에서 유효한 후보를 얻지 못했는지도 기록한다.
전역 최적해를 찾았다는 의미의 oracle이 아니다.

`distill`은 성공한 TRAIN oracle을 작은 두 CNN에 옮긴다. 손실에는 개별 P/B의
차이뿐 아니라 같은 residual에 대한 **전체 V-cycle 보정의 A-energy 차이**도 들어간다.
학생과 teacher 각각의 실제 coarse A를 사용한다. 증류 이후에는 teacher 손실 없이
전체 TRAIN에서 P/smoother를 다시 공동 미세조정한다. 일부 A에 대한 탐색이 실패하면
그 A는 증류 label에서 빠지지만, 이후 task 미세조정과 평가에서 사라지지 않는다.

## 기본 데이터와 학습 예산

격자 n은 한 방향 내부 미지수 개수이므로 `n=63`이면 fine unknown은 `63²=3969`개다.
데이터는 저장소의 SPD diffusion 문제 생성기로 만든다. Near-isotropic, aligned
anisotropy, rotated anisotropy, local patch, checkerboard, channel,
variable coefficient의 7개 계수군을 사용한다. Train/validation/final은 같은
A의 RHS만 바꾼 분할이 아니라 operator 단위로 분리한다.

| 항목 | Research 기본값 | Smoke |
| --- | --- | --- |
| Train n | 15, 31, 63 | 7 |
| Train A 수 | 3 × 7 × 4 = 84 | 7 |
| Validation n | 15, 31, 63, 127 | 7 |
| Validation A 수 | 4 × 7 × 1 = 28 | 7 |
| 직접 학습 | P-only, S-only, joint 각각 336 updates | 각각 2 updates |
| 행렬별 탐색 | 4 starts × 200 updates | 2 starts × 2 updates |
| 증류 | 336 updates | 2 updates |
| 증류 후 공동 미세조정 | 168 updates | 2 updates |
| 별도 크기 OOD | n=255 | 성능 결론에 사용하지 않음 |

Research의 추가 탐색만 `84 × 4 × 200 = 67,200` updates다. NN 학습은
`3 × 336 + 336 + 168 = 1,512` updates이며, 각 update는 여러 probe와
여러 level/cycle의 forward/backward를 포함한다. 이 횟수는 시간 예측치가 아니다.
Mac M2 Pro에서 전체 연구 run의 소요 시간을 아직 측정하지 않았으므로 몇 시간이라고
단정하지 않는다. Smoke는 실행 경로 확인용이며 실제 연구 실행시간이나 speedup을
추정하기에는 격자가 너무 작다. 이 오프라인 비용과 최종 온라인 시간을 따로 보고한다.

## 여섯 비교군과 시간 판정

| 출력 이름 | P | Smoother |
| --- | --- | --- |
| `fixed_C` | bilinear | line_alt |
| `em5_C` | EM5 | line_alt |
| `learned_P` | 직접 학습한 P | line_alt |
| `learned_S` | bilinear | 학습한 pre/post |
| `joint` | 공동 학습한 P | 공동 학습한 pre/post |
| `distilled_joint` | 추가 탐색 후 증류·미세조정 | 추가 탐색 후 증류·미세조정 |

`cold`는 A마다 hierarchy를 새로 준비한 뒤 RHS 하나를 푸는 시간이다. `multiple`은
한 번의 setup과 같은 A의 여러 RHS 풀이를 합친다. `warm`은 이미 준비된 계층의
solve만 측정한다. RHS count 1/4/16/64는 b 원소의 값이 아니라 서로 다른 b의 개수다.

Online setup에는 bilinear recovery 계층, CNN 추론, P projection, learned
Galerkin 계층, coarse factorization, sparse B 생성이 포함된다. EM5 arm에서는
그 arm의 EM5 setup이 들어간다. 오프라인 학습·탐색·증류, PDE 조립, checkpoint 파일
로드 시간은 이 online solver timing에 포함하지 않는다.

주 판정은 설정 파일에 미리 선언한 `target_rhs_count`의 `multiple` workload다.
제공한 기본 설정의 `target_rhs_count`는 1이다. 여러 RHS 재사용을 주 목표로 삼으려면
새 run을 준비하기 전에 이 값을 바꾸고, 해당 RHS count를 benchmark에도 포함한다.
각 A의 반복 측정 중앙값을 구한 뒤 전체 A에 대한 평균 setup + solve 시간으로
후보를 선택한다. 모든 A, RHS, 측정 반복에서 성공해야 선택 자격이 있다. 실패한
A를 빼고 남은 쉬운 문제만으로 우승자를 정하지 않는다. `warm`은 원인 분석용
보조 지표다. `em5_C`는 독립 reference이며 학습 checkpoint 선택 후보는 아니다.
선택된 NN도 bilinear classical보다 느리면 배포 권고는 `fixed_C`가 된다.

`cycle_diagnostics.json`에는 실제 실행 cycle 수, accepted neural cycle,
fallback/rollback, setup 실패, setup/solve 시간 및 격자별 비교가 들어간다.
Validation의 `raw_contractions.json`은 별도의 고정-cycle 검사이며 fallback,
rollback, classical 대체를 전혀 사용하지 않는다. Residual 수축 지표와 실제
풀이 시간 순위를 구분해서 읽는다. 최종/OOD 결과로 다시 모델을 고르거나 학습하지 않는다.

## Mac 터미널 실행

실험을 계속 실행 중인 기존 checkout의 소스를 바꾸지 말고 새 checkout에서 시작한다.
아래 `REPO` 경로는 원하는 새 폴더로 바꿀 수 있다. Python 3.12가 설치된 환경 기준이다.

```bash
REPO="$HOME/Desktop/adaptive-multigrid-joint"
git clone --single-branch --branch residual-u-net-teacher \
  https://github.com/slwjackie/adaptive-multigrid.git "$REPO"
cd "$REPO"
git log -1 --oneline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
set -o pipefail
SCRIPT=scripts/run_v6_7_joint_transfer_study.py
```

먼저 작은 smoke run을 validation까지 실행할 수 있다.

```bash
JOINT_SMOKE="artifacts/joint_smoke_$(date +%Y%m%d_%H%M%S)"
python "$SCRIPT" prepare --run-dir "$JOINT_SMOKE" \
  --config configs/v6_7_joint_transfer_smoke.json
caffeinate -i python -u "$SCRIPT" train --run-dir "$JOINT_SMOKE" \
  2>&1 | tee "$JOINT_SMOKE/train.log"
caffeinate -i python -u "$SCRIPT" search --run-dir "$JOINT_SMOKE" \
  2>&1 | tee "$JOINT_SMOKE/search.log"
caffeinate -i python -u "$SCRIPT" distill --run-dir "$JOINT_SMOKE" \
  2>&1 | tee "$JOINT_SMOKE/distill.log"
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$JOINT_SMOKE" \
  --repeats 1 --warmups 0 --rhs-counts 1 4 --regimes cold warm multiple \
  2>&1 | tee "$JOINT_SMOKE/benchmark.log"
```

본 연구 실행은 새 run에서 시작한다.

```bash
JOINT_RUN="artifacts/joint_research_$(date +%Y%m%d_%H%M%S)"
python "$SCRIPT" prepare --run-dir "$JOINT_RUN" \
  --config configs/v6_7_joint_transfer_research.json
printf '%s\n' "$JOINT_RUN" > artifacts/joint_research_current.txt
caffeinate -i python -u "$SCRIPT" train --run-dir "$JOINT_RUN" \
  2>&1 | tee "$JOINT_RUN/train.log"
caffeinate -i python -u "$SCRIPT" search --run-dir "$JOINT_RUN" \
  2>&1 | tee "$JOINT_RUN/search.log"
caffeinate -i python -u "$SCRIPT" distill --run-dir "$JOINT_RUN" \
  2>&1 | tee "$JOINT_RUN/distill.log"
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$JOINT_RUN" \
  --repeats 3 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple \
  2>&1 | tee "$JOINT_RUN/benchmark.log"
```

Validation 결과를 확인한 뒤 모델과 평가 조건을 고정하고 최종/OOD 전체를 실행한다.
`freeze` 후에는 같은 run에서 추가 학습·탐색·증류를 진행할 수 없다.

```bash
python "$SCRIPT" freeze --run-dir "$JOINT_RUN" \
  --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple
caffeinate -i python -u "$SCRIPT" final --run-dir "$JOINT_RUN" \
  2>&1 | tee "$JOINT_RUN/final.log"
```

`final`은 기본 final split과 grid/rotation/anisotropy/contrast/topology OOD를
실행하므로 validation보다 오래 걸릴 수 있다. `caffeinate -i`는 idle sleep을
막지만 덮개를 닫았을 때의 sleep까지 보장하지 않는다. 비교 중에는 다른 학습이나
benchmark를 동시에 실행하지 않는다.

## 중단 후 재개와 결과 확인

새 터미널에서는 같은 checkout, venv, thread 환경을 다시 활성화한다.

```bash
cd "$HOME/Desktop/adaptive-multigrid-joint"
source .venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
set -o pipefail
SCRIPT=scripts/run_v6_7_joint_transfer_study.py
JOINT_RUN="$(cat artifacts/joint_research_current.txt)"
```

중단된 단계 하나를 같은 인자로 재개한다. 아래 예시는 search다.

```bash
caffeinate -i python -u "$SCRIPT" search --run-dir "$JOINT_RUN" --resume \
  2>&1 | tee -a "$JOINT_RUN/search.log"
```

`train`, `distill`, `benchmark`, `final`도 `--resume`을 지원한다. Benchmark는 기존
`--repeats`, `--warmups`, `--rhs-counts`, `--regimes`를 동일하게 유지해야 한다.
`prepare` 재개에는 원래 `--config`도 동일하게 전달한다. `freeze`에는 `--resume`
옵션이 없고, 완료된 freeze 뒤에는 `final --resume`으로 평가를 이어간다.

학습과 탐색의 optimizer 상태도 저장하므로 완료된 update를 다시 시작할 필요가 없다.
소스, 설정, checkpoint, library 버전이나 thread 환경이 달라지면 provenance 검사가
재개를 거부한다. Manifest/hash를 수정해서 검사를 건너뛰지 않는다. 코드나 설정을
변경할 실험은 새 run으로 시작하고 기존 결과를 그대로 보존한다.

| 파일 | 내용 |
| --- | --- |
| `checkpoints/*/status.json` | 학습 단계별 완료 여부·update 수 |
| `search/progress.json` | 완료된 TRAIN A 수 |
| `search/report.json` | 모든 시작점·실패·선택 oracle·coverage |
| `benchmarks/validation/selection.json` | 주 workload의 측정 시간 기반 선택 |
| `benchmarks/validation/cycle_diagnostics.json` | cycle·fallback·시간의 격자별 진단 |
| `benchmarks/validation/raw_contractions.json` | fallback 없는 고정-cycle 잔차 검사 |
| `final/*/progress.json` | 각 final/OOD split의 평가 진행률 |
| `final/*/raw_results.json` | 반복별 원본 측정치 |

이 구현의 성공 기준은 작은 CNN이 생성한 조합을 같은 tolerance에서 실행했을 때
setup을 포함한 실제 총시간이 classical 비교군보다 줄어드는지다. 오프라인 자유변수
후보가 좋았다는 것과, 증류된 학생이 빠르다는 것은 별개의 결과로 보고한다.
