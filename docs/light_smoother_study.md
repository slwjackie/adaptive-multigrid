# 같은 learned P에서 비교하는 경량 줄 smoother

작업 브랜치는 `residual-u-net-teacher`다. 새 실행 경로는
`scripts/run_v6_7_light_smoother_study.py`이며, 기존 EM-line와 P-only 실행 파일을
수정해 이어 붙인 실험이 아니다. 완료된 **EM-line의 `em_P`**를 가져와 P를 고정한 뒤,
각 nonterminal level의 줄 smoother를 조절하는 두 숫자만 예측한다.

- **작은 MLP:** 12개 요약 특징 → 16개 ReLU → x/y 계수 2개, bias 포함 **242개** 파라미터.
- **NN 없는 회귀:** 같은 특징과 같은 탐색 정답으로 ridge regression, **26개** 회귀 계수.
- **선택적 온라인 적응:** MLP 출력의 2개 보정값만 TTT하거나, 회귀 출력 주변을 수치 탐색한다.
  기본값은 off이며 `benchmark --include-online`에서 별도 비교군 두 개를 추가한다.

여기서 “NN 없음”은 **smoother 계수 예측**에 해당한다. 공통 learned P는 계속 같은 CNN이 생성한다.
성공 기준은 동일 tolerance까지의 **실측 setup + solve 시간**이다. 계수 오차가 작거나
cycle 수가 적다는 사실만으로 속도 향상을 주장하지 않는다.

## 1. 고정하는 수치 조건과 P 출처

| 항목 | 설정 |
| --- | --- |
| 문제 | 합성 2차원 SPD 확산 선형계 |
| Coarsening | Full 고정 |
| P | 학습 완료 `em_P` checkpoint; 모든 경량 arm이 같은 P 가중치 사용 |
| Restriction | `R=P.T` |
| Coarse 행렬 | 실제 student P로 `Ac=P.T @ A @ P` 구성 |
| Pre/post | 각각 2회; x/y 계수는 pre/post에서 공유 |
| 최하위 | 기존 직접 LU 풀이 |
| 배포 fallback | `line_alt + bilinear + full` |
| 독립 참고군 | `line_alt + EM5 + full` |
| 온라인 EM | 경량 모델 setup에서는 실행하지 않음 |
| 학습할 대상 | P, Ac, 줄 분해를 고정한 상태의 level별 `omega_x`, `omega_y` |

`prepare --em-run-dir`는 원래 run의 설정, 완료 상태, checkpoint hash, 단계 연결,
P 학습 데이터 목록을 검사한 뒤 필요한 파일을 새 run의 `imported/em_line/`에 복사한다.
과거 소스 fingerprint도 출처로 보존한다. **현재 소스로 과거 run의 manifest를 다시 쓰지 않는다.**
이전 `61c154d` 등에서 완료한 EM checkpoint도 완료 증거를 검증해 가져올 수 있지만,
그 과거 run 자체를 변경된 checkout에서 `--resume`할 수 있다는 뜻은 아니다.

설정의 `include_frozen_line=true`이면 완료된 `frozen_P_line`도 확인한다.
실제 transfer signature가 공통 `em_P`와 같은 경우만 포함한다. Checkpoint가 없거나
미완료이면 사용 불가 이유를 기록하고 출력한다. 다른 모델로 바꿔 넣지 않는다.
`joint_line`은 P가 달라질 수 있으므로 같은-P smoother 비교에 포함하지 않는다.

새 경량 실험의 학습·validation A는 새 seed로 생성하며, imported P의 학습 A와도
겹치지 않게 검사한다. 기본 research는 train 420개(15²·31²·63²), validation
140개(15²·31²·63²·127²)다. 현재 artifacts와 설정의 `historical_roots`에서 찾은 과거
평가 기록도 데이터 분리 검사에 사용한다. 다른 폴더에 보관한 과거 실험은 새 설정의
`historical_roots`에 명시한다. 이미 사용한 final/OOD를 새 학습 데이터로 옮기지 않는다.

## 2. 무엇을 스케일링하는가

`Sx(r)`는 기존 classical 구현의 **x 방향 전체 zebra sweep**을 초기 보정값 0에서
수행한 결과다. Sweep 내부의 줄/색상별 잔차 갱신과 cached compiled line solve는
그대로 사용한다. `Sy`도 같은 뜻이다.

\[
\begin{aligned}
r &= b-Ax,\\
d_x &= S_x(r), & x_1 &= x+\omega_xd_x,\\
r_1 &= b-Ax_1,\\
d_y &= S_y(r_1), & x_2 &= x_1+\omega_yd_y.
\end{aligned}
\]

따라서 계수는 **방향 sweep을 완료한 보정 전체**에 곱한다. 각 줄 갱신마다 계수를
곱하는 line SOR와는 연산이 다르다. Post에서는 y→x 순서와 기존 줄/색상 순서를
모두 반대로 적용한다. `omega=(1,1)`에서는 기존 `line_alt`의 연산 결과와 일치한다.
계수가 1일 때 기존 sweep이 계산한 잔차를 재사용하고, 다른 값이면 실제 전체 A로
스케일된 보정 이후 잔차를 계산한다.

새 Python 삼각풀이 루프로 기존 compiled solve를 대체하지 않는다. 기존
`LineSmootherCache`의 SuperLU factor와, 같은 색상 줄이 독립일 때의 batched solve를
재사용한다. 원래부터 같은 색상 줄이 결합된 넓은 coarse stencil은 기존 순차 경로를
유지한다. 계수를 바꿔도 line factor, P, Ac를 다시 만들지 않는다.

현재 공통 bounds는 설정 파일의 `[0.8,1.2]`다. 이 범위는 탐색·예측 제약이며,
전체 MG 수렴을 증명하는 조건이 아니다. 계수가 1과 달라지면 추가 잔차 matvec이
필요할 수 있으므로, 작은 모델이라는 이유만으로 cycle이 더 저렴해지는 것도 아니다.

기존 EM-line CNN과의 차이는 다음과 같다.

| 구분 | 기존 `frozen_P_line` | 새 MLP / 회귀 |
| --- | --- | --- |
| 입력 | 위치별 연산자 특징과 level 정보 | level 전체의 12개 요약 통계 |
| 예측 | 위치별 x/y LDL 줄 분해 계수 | level마다 x/y 스칼라 2개 |
| Smoother 모델 크기 | 폭 8 CNN, 3,244개 파라미터 | MLP 242개 / 회귀 26개 |
| 적용 연산 | 생성한 줄 factors의 보정 | 기존 classical 방향 sweep의 보정에 스칼라 곱 |
| 줄 풀이 | 기존 EM-line의 생성 factor 적용 경로 | classical cached compiled line solve |
| 추론 빈도 | A마다 nonterminal level당 한 번 | A마다 nonterminal level당 한 번 |
| 매 cycle/RHS 추론 | 없음 | 없음 |

## 3. 12개 특징의 정의와 비용

미지수 `i`의 좌표를 `(ix,iy)=(i//ny,i%ny)`, 대각을 \(d_i=A_{ii}>0\)라 하자.
\(q_i=\sum_{j\ne i}|A_{ij}|/d_i\)다. `qx`는 같은 y 좌표의 x 방향 줄 연결,
`qy`는 같은 x 좌표의 y 방향 줄 연결, `qm`은 두 좌표가 모두 다른 연결을 모은다.
최근접 stencil만 보지 않으며, 실제 Galerkin A에 존재하는 먼 연결도 합산한다.

| 번호 | 특징 | 정의 |
| ---: | --- | --- |
| 1 | x 줄 결합 | \(\operatorname{mean}_i q^x_i\), \(q^x_i=\sum_{j\ne i,j_y=i_y}|A_{ij}|/d_i\) |
| 2 | y 줄 결합 | \(\operatorname{mean}_i q^y_i\), \(q^y_i=\sum_{j\ne i,j_x=i_x}|A_{ij}|/d_i\) |
| 3 | 혼합 결합 | \(\operatorname{mean}_i q^m_i\), 두 좌표 모두 다른 항 |
| 4 | 방향 대비 | \((\overline q^x-\overline q^y)/(\overline q^x+\overline q^y)\); 분모 0이면 0 |
| 5 | 평균 대각 비중 | \(\operatorname{mean}_i 1/(1+q_i)\) |
| 6 | 최소 대각 비중 | \(\min_i 1/(1+q_i)\) |
| 7 | 대각 변동 | \(\operatorname{std}(d)/\operatorname{mean}(d)\) |
| 8 | 행별 off-diagonal 변동 | \(\operatorname{std}(q)/\operatorname{mean}(q)\); 분모 0이면 0 |
| 9 | 양수 off-diagonal 비중 | \(\sum_{i\ne j,A_{ij}>0}|A_{ij}|/\sum_{i\ne j}|A_{ij}|\); 분모 0이면 0 |
| 10 | x 격자 크기 | \(\log_2(n_x+1)\) |
| 11 | y 격자 크기 | \(\log_2(n_y+1)\) |
| 12 | 상대 level | `index/(total_levels-1)`; finest index=0 |

표준편차는 population standard deviation(`ddof=0`)이다. 학습 level 행들에서만
평균과 표준편차를 구하고 `z=(feature-mean)/scale`로 표준화한다. 표준편차가
`1e-12` 이하인 열은 scale=1을 사용한다. 평균·scale·학습 행 수·특징 정의 버전을
각 predictor 파일에 저장한다. Validation/final 통계로 다시 맞추지 않는다.

CSR 희소 항목과 행별 통계를 한 번씩 처리하므로 특징 추출은
\(O(\operatorname{nnz}(A)+N)\) 작업·임시 저장공간을 사용한다. 격자를 dense로
변환하거나 flatten한 원소를 MLP에 넣지 않는다. PDE family, RHS, 정확해를 입력하지
않는다. 매 nonterminal level의 setup에서 한 번 계산하며 **특징 추출 시간도 청구**한다.

## 4. 오프라인 탐색과 두 모델의 학습

`search`는 학습 A마다 모든 계수를 `(1,1)`로 시작한다. 한 level의 `(omega_x,omega_y)`를
`{0.8,1.0,1.2}²`에서 바꾸고, 나머지 level은 현재 선택을 유지한다. 각 후보를
**전체 solve의 같은 tolerance**까지 실행해 비교한다. Fallback과 온라인 적응은 없다.

Research 기본값은 pass 최대 2회, 최적화 RHS 3개, 별도 확인 RHS 3개, 각 측정 3반복,
최소 시간 개선 3%다. 후보 순서는 seed 기반 permutation과 반복별 순환으로
초기/후기 실행 편향을 줄인다. 중간 좌표 단계에서 좋아도 확인 RHS에서 개선이
확인되지 않으면 전체 `(1,1)`을 유지한다. 기본값이 실패했고 후보가 모두 성공했다면
성공 복구로 별도 표시한다. 실패한 후보·문제·걸린 시간도 저장한다.

최종 후보는 확인 RHS뿐 아니라 동일한 전체 계수 조합으로 이미 수행한 최적화 RHS
측정도 모두 성공해야 한다. 모든 후보가 실패하면 `(1,1)`은 단지 보존할 기본값이며,
`label_eligible=false`, `selection_outcome=no_successful_candidate`로 기록한다.
`search/summary.json`의 `failed_cases`에 문제를 남기고 **fit을 중단**한다. 실패한 A를
제거하거나 그 기본값을 성공한 탐색 정답으로 학습하지 않는다. 탐색 범위·예산 등을
변경하려면 기록을 보존한 채 새 설정과 새 run으로 시작한다.

`search/<case>/result.json`에 모든 계수 조합, 성공 여부, cycle 수, 잔차 이력,
측정 시간, 최종 선택과 기본값 대비 비교를 남긴다. `progress.json`은 완료된 **각 측정**을
기록한다. 재개하면 완료된 측정을 다시 재지 않으며, 중단 순간 실행 중이던 측정은
반복할 수 있다. Oracle의 시간은 공통 계층을 재사용한 solve 시간이다. 이를 새 A에
즉시 적용 가능한 배포 시간으로 보고하지 않는다.

이 탐색은 유한한 좌표 후보에서의 결과다. 전체 계수 조합의 전역 최적해가 아니다.
예를 들어 nonterminal level이 3개이면 한 pass에 대략 `3×9`개 후보를 반복 RHS에
평가하므로, 학습 A가 많으면 **모델 fit보다 search가 훨씬 큰 작업**이 될 수 있다.
정확한 작업량은 pass 조기 종료·중복 후보·실패 cycle 수에 따라 달라진다.

`fit`은 두 방법에 **같은 search 정답, 같은 특징, 같은 표준화·bounds**를 사용한다.

\[
\widehat\omega_{\rm MLP}
=\operatorname{clip}_{[l,u]}
\left(\mathbf1+W_2\operatorname{ReLU}(W_1z+b_1)+b_2\right).
\]

MLP parameter 수는 `12×16+16+16×2+2=242`다. 마지막 head를 0으로 초기화하여
초기 출력이 정확히 `(1,1)`이다. Research 기본 fit은 1,500 updates, batch 64,
learning rate 0.003이며, 이는 학습 시작 예산일 뿐 충분한 학습을 보장하지 않는다.

\[
\min_{W,c}\sum_i\left\|Wz_i+c-(\omega_i^*-\mathbf1)\right\|_2^2
+\lambda\|W\|_F^2,
\qquad
\widehat\omega_{\rm ridge}=\operatorname{clip}_{[l,u]}(\mathbf1+Wz+c).
\]

회귀는 `W:2×12`, `c:2`로 총 26개 계수이며 기본 \(\lambda=0.01\)이다. Intercept는
penalty를 적용하지 않는다. Penalized design을 쌓아 `numpy.linalg.lstsq`로 풀며 명시적
역행렬은 계산하지 않는다. 추론은 NumPy 행렬·벡터 곱이다.

P는 두 학습에서 모두 고정한다. 최적화/확인 RHS를 만들 때 해를 제조하는 절차는
학습 A의 테스트 신호 생성에만 사용한다. 이 정확해가 predictor 입력에 들어가는 것은 아니다.

## 5. 온라인 적응의 범위와 한계

`--include-online`은 다음 두 arm을 기본 arm과 **동시에 따로 측정**한다.

| Arm | 계수 생성 | 실행 중 적응 |
| --- | --- | --- |
| `fixed_C` | Bilinear classical | 없음 |
| `em5_C` | EM5 classical | 없음; 독립 참고군 |
| `learned_P_line` | 같은 learned P, 계수 `(1,1)` | 없음 |
| `learned_P_MLP` | MLP | 없음 |
| `learned_P_regression` | Ridge regression | 없음 |
| `learned_P_MLP_TTT` | 같은 MLP | 선택 level의 출력 보정 2개만 gradient update |
| `learned_P_regression_online` | 같은 ridge | 선택 level의 두 계수 주변 소수 후보 탐색 |
| `frozen_P_line` | 기존 CNN 줄 계수 | 없음; 동일 P 검사 통과 시에만 선택적 포함 |

기본 trigger는 최근 2 cycle 평균 감소율이 0.8 이상이고, 잔차가 tolerance의
100배보다 크며, 현재 감소율로 추정한 잔여 작업이 4 cycle 이상인 경우다.
기본 level은 finest(0), A당 호출 최대 1회, numerical 후보 최대 4개, MLP 보정
gradient step 최대 2회다. 값은 설정의 `online` 항목에 명시되어 있다.

MLP TTT는 **MLP 본체를 다시 실행하거나 바꾸지 않는다.** 현재 level 출력에 더할
2개 임시 보정만 실제 고정 계층의 미분 가능한 V-cycle로 갱신한다. 기존 cached LU
solve와 sparse 연산을 이용하며 backward에는 transpose solve를 사용한다. 현재
`x,b`는 detach하고 update 간 그래프를 끊는다. P CNN 재실행·Ac 재구성·factor 재분해는 없다.

NN 없는 온라인 적응은 현재 계수에 ±step을 주는 소수 후보를 직접 실행한다.
두 방법 모두 **같은 현재 x와 b에서** 기본 계수와 후보를 검사하며, 검사에는 fallback을
허용하지 않는다. 기본 설정은 1-cycle 잔차가 최소 2% 더 작을 때 채택한다. 검사 후보가
만든 해 자체를 solve 진전으로 채택하지 않으며, 선택한 계수만 이후 실제 cycle에 사용한다.
이 검사는 이후 모든 오차/RHS에서의 우위나 총시간 절약을 보장하지 않는다.

기본 시간 예산은 A당 0.05초다. 이는 연산 사이에서 확인하는 **협조적 예산**이다.
이미 시작한 sparse solve/forward/backward를 중간에 강제로 끊지 않으므로 실제 비용이
한 작업만큼 넘을 수 있고, 이 시간도 기록·청구한다. 다음 작업을 시작하기 전에 예산 소진을 감지하거나, 비정상 gradient·개선 없음이
확인되면 이전 계수로 복원한다. 마지막 원자 작업에서 예산을 넘긴 경우에도 완료한
검증 결과와 실제 초과 시간은 모두 기록한다. P, predictor checkpoint는 변형하지 않는다. 채택한 계수는 같은 A의
이후 cycle/RHS에서 재사용하되, 잔차 증가 시 classical recovery와 근접 tolerance 마무리를
유지한다. 온라인 학습이 안전장치 자체를 대신하는 것은 아니다.

## 6. 시간과 결과를 읽는 방법

| 조건 | 측정 시간 |
| --- | --- |
| Cold, RHS=1 | 새 constructor/setup + 1 RHS solve |
| Multiple, RHS=m | 새 constructor/setup 한 번 + 같은 A의 서로 다른 m개 RHS solve |
| Warm, RHS=1 | 준비된 계층에서 solve; **그 solve 안의 온라인 적응 비용도 포함** |

`RHS counts=1,4,16,64`는 b의 값이 아니라 **동일 A에 대해 푸는 서로 다른 b의 개수**다.
온라인 적응 arm의 warm priming은 적응을 끈 상태에서 별도 RHS로 수행한다. Timed solve는
원래 predictor 계수와 새 적응 예산에서 시작한다. 이미 적응한 계층을 공짜로 사용하는
warm 결과와 혼합하지 않는다.
Classical 및 기존 CNN 비교군에도 같은 별도 RHS priming 규칙을 적용한다. 기존 실험의
평가 함수는 변경하지 않고, 새 경량 실험의 평가 프로토콜만 `light-smoother-measured-v2`로
구분한다. 따라서 v1 평가 결과에 v2 측정을 resume로 추가할 수 없다.

Fallback용 classical 계층, learned P CNN, 실제 Galerkin 계층, 줄 factor, 특징,
MLP/회귀 예측은 setup 시간에 포함한다. 온라인 adaptation, 후보 검증, 거부한 시도,
잔차 검사와 fallback은 solve timer 안에 있다. `online_adaptation_seconds`는 solve의
**부분집합**이므로 `wall_seconds`에 다시 더하지 않는다. 오프라인 search/fit 시간은
온라인 측정에서 제외하고 별도로 기록한다.

`benchmarks/validation/`에서 `raw_results.json`, `cycle_diagnostics.json`,
`paired_effects.json`, `selection.json`을 확인한다. `paired_effects.json`은 같은-P
기본값 대비 MLP/회귀 효과와, 각 기본 모델 대비 온라인 적응의 추가 효과를 나눠 기록한다.
Predictor 호출·계수 변경·온라인 채택/거부·factor 재생성 등의 상세 counter도 보존한다.
`executed_cycles`는 거부된 live 시도를 포함하는 기존 solve budget이고,
`all_executed_cycles`는 온라인 후보 및 gradient forward V-cycle까지 포함한다.
Backward 계산비용은 시간에 포함하며 별도 V-cycle 수로 환산하지 않는다.
`accepted_cycles`는 실제 해에 채택한 cycle 수다.
도중에 예외가 난 후보·gradient V-cycle도 실행 시도로 센다. 완료 횟수는
`online_probe_completed_cycles`, `online_gradient_completed_vcycles`로 구분하고,
실패한 후보의 오류와 시간도 온라인 event에 보존한다.
`fallback_event_count`는 safety lock 진입(setup 실패·수치적 거절·near-tolerance)과
gate-closed 결정을 센다. `classical_safety_cycles` 및 `fallback_count`는 실제 classical
복구와 이후 lock tail의 cycle 수이며, `fixed_C`/`em5_C`의 정상 계산은 제외한다.
이 정의를 CNN과 경량 비교군에 동일하게 적용한다.

선택은 설정의 `target_rhs_count`(기본 1)에 해당하는 **multiple setup+solve**로 한다.
모든 평가 문제·RHS·반복의 성공을 먼저 요구하며, 실패를 제거한 속도 평균만으로 우승자를
고르지 않는다. `fixed_C`가 더 빠르면 classical을 권고하고, 어떤 후보도 조건을 만족하지
못하면 비선택을 기록한다. Final/OOD에서는 validation 선택을 다시 바꾸지 않는다.
Smoke는 실행 경로 검증이며 속도 향상 증거가 아니다.

## 7. Mac M2 Pro: 새 폴더와 환경

기존 실험 폴더·가상환경은 그대로 둔다. 아래는 Python 3.12가 설치된 Mac에서의 명령이다.
설치되어 있지 않고 Homebrew를 사용하는 환경이면 먼저 `brew install python@3.12`를 실행한다.

```bash
git clone --single-branch --branch residual-u-net-teacher \
  https://github.com/slwjackie/adaptive-multigrid.git \
  "$HOME/Desktop/adaptive-multigrid-light"
cd "$HOME/Desktop/adaptive-multigrid-light"
git log -1 --oneline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set -o pipefail
mkdir -p artifacts
SCRIPT="scripts/run_v6_7_light_smoother_study.py"
```

새 터미널에서는 `source`, thread 환경변수, `SCRIPT`를 다시 설정한다.
`caffeinate -i`는 idle sleep 억제용이다. 노트북 덮개를 닫아도 계속 계산된다는 보장으로
해석하지 말고, 장시간 실행은 전원 연결·덮개를 연 상태에서 진행한다.

## 8. Smoke: 작은 EM checkpoint부터 새 경로 끝까지

Smoke용으로만 작은 EM-line run을 새로 만든다. 기존 `train` CLI는 세 단계를 모두
실행한다. 이 결과는 본 연구용 P로 사용하지 않는다.

```bash
EM_SMOKE="artifacts/em_for_light_smoke_$(date +%Y%m%d_%H%M%S)"
python -u scripts/run_v6_7_em_line_study.py prepare \
  --config configs/v6_7_em_line_smoke.json --run-dir "$EM_SMOKE"
caffeinate -i python -u scripts/run_v6_7_em_line_study.py train \
  --run-dir "$EM_SMOKE" 2>&1 | tee "$EM_SMOKE/train.log"

LIGHT_SMOKE="artifacts/light_smoke_$(date +%Y%m%d_%H%M%S)"
python -u "$SCRIPT" prepare --em-run-dir "$EM_SMOKE" \
  --config configs/v6_7_light_smoother_smoke.json --run-dir "$LIGHT_SMOKE"
printf '%s\n' "$LIGHT_SMOKE" > artifacts/light_smoke_current.txt

caffeinate -i python -u "$SCRIPT" search --run-dir "$LIGHT_SMOKE" \
  2>&1 | tee "$LIGHT_SMOKE/search.log" && \
caffeinate -i python -u "$SCRIPT" fit --run-dir "$LIGHT_SMOKE" \
  2>&1 | tee "$LIGHT_SMOKE/fit.log" && \
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$LIGHT_SMOKE" \
  --include-online --repeats 1 --warmups 0 \
  --rhs-counts 1 4 --regimes cold warm multiple \
  2>&1 | tee "$LIGHT_SMOKE/benchmark.log"
```

`prepare` 전에는 run 폴더 안에 log를 만들지 않는다. `prepare`가 비어 있는 새 경로를
요구하므로 `tee "$LIGHT_SMOKE/prepare.log"`를 동시에 실행하면 충돌할 수 있다.
Prepare 실패 시 이어지는 단계를 실행하지 말고 그 오류부터 해결한다.

## 9. 본 실험: 완료된 연구용 em_P 가져오기

아래 `EM_RUN`은 **실제 완료된 EM-line run 경로로 교체**한다. 저장소만 clone하면
과거 로컬 artifacts가 함께 내려오는 것은 아니다. P-only의 `learned_P` checkpoint나
smoke checkpoint를 연구용 `em_P` 대신 지정하면 안 된다.

```bash
EM_RUN="$HOME/Desktop/adaptive-multigrid-em-line/artifacts/실제_완료된_EM_run"
test -f "$EM_RUN/checkpoints/em_P/candidate.pt"
python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"checkpoints/em_P/status.json"; d=json.loads(p.read_text()); print(d); assert d["status"]=="complete" and d["all_updates_completed"]' "$EM_RUN"
```

연구용 EM checkpoint가 없으면 새 폴더의 현재 소스로 다음을 먼저 실행한다. 이 명령은
P 4,200 updates 외에 smoother 1,400·joint 1,400 updates도 수행하므로 별도 큰 작업이다.
기존 완성 checkpoint가 있다면 생략한다.

```bash
EM_RUN="artifacts/em_for_light_research_$(date +%Y%m%d_%H%M%S)"
python -u scripts/run_v6_7_em_line_study.py prepare \
  --config configs/v6_7_em_line_research.json --run-dir "$EM_RUN"
caffeinate -i python -u scripts/run_v6_7_em_line_study.py train \
  --run-dir "$EM_RUN" 2>&1 | tee "$EM_RUN/train.log"
```

기존 평가 run들이 다른 Desktop 폴더에 있다면, **prepare 이전에** research 설정의
복사본을 만들고 제외할 출처를 추가한다. 아래 예시는 실제 존재하는 폴더만 입력한다.

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("configs/v6_7_light_smoother_research.json")
d = json.loads(p.read_text())
names = ("adaptive-multigrid-ponly", "adaptive-multigrid-joint",
         "adaptive-multigrid-pruning", "adaptive-multigrid-em-line")
d["historical_roots"] = [str(Path.home()/"Desktop"/n/"artifacts")
                         for n in names if (Path.home()/"Desktop"/n/"artifacts").exists()]
Path("artifacts/light_research_local.json").write_text(json.dumps(d, indent=2)+"\n")
PY

LIGHT_RUN="artifacts/light_research_$(date +%Y%m%d_%H%M%S)"
python -u "$SCRIPT" prepare --em-run-dir "$EM_RUN" \
  --config artifacts/light_research_local.json --run-dir "$LIGHT_RUN"
printf '%s\n' "$LIGHT_RUN" > artifacts/light_research_current.txt

caffeinate -i python -u "$SCRIPT" search --run-dir "$LIGHT_RUN" \
  2>&1 | tee "$LIGHT_RUN/search.log" && \
caffeinate -i python -u "$SCRIPT" fit --run-dir "$LIGHT_RUN" \
  2>&1 | tee "$LIGHT_RUN/fit.log" && \
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$LIGHT_RUN" \
  --include-online --repeats 3 --warmups 1 \
  --rhs-counts 1 4 16 64 --regimes cold warm multiple \
  2>&1 | tee "$LIGHT_RUN/benchmark.log"
```

`--include-online`을 생략하면 기본 비교군만 실행한다. 온라인 효과도 보려면 **첫
benchmark부터** 넣는다. 이미 시작한 평가에 나중에 옵션을 붙여 같은 manifest로
재개할 수 없다. 설정·모델·평가 구성을 바꾸려면 새 run을 사용한다.

실측 소요 시간은 A별 수렴 cycle, 줄 factor 밀도, 탐색 후보와 반복 수, 온라인 trigger
발생 여부에 따라 크게 달라진다. M2 Pro에서 검증하지 않은 시간 수치를 제시하지 않는다.
`search/progress.json`의 완료 case 수와 실제 경과 시간을 이용해 같은 실행의 잔여 시간을
추정하되, 해상도별 case 비용 차이를 고려한다.

## 10. 중단 후 재개

동일한 checkout·설정·패키지·스레드로 재개한다. 계산 중이던 프로세스가 남았는지 먼저
확인하고, 같은 run을 동시에 두 개 실행하지 않는다.

```bash
cd "$HOME/Desktop/adaptive-multigrid-light"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set -o pipefail
SCRIPT="scripts/run_v6_7_light_smoother_study.py"
LIGHT_RUN="$(cat artifacts/light_research_current.txt)"
ps -ax -o pid,stat,pcpu,etime,command | grep '[r]un_v6_7_light_smoother_study.py'
```

**중단된 단계에 해당하는 명령만 실행한다.** 아직 시작하지 않은 단계는 `--resume` 없이
실행한다. Search는 완료된 개별 측정, fit은 Adam 상태와 update, benchmark는 완료된
측정을 복구한다. 아래 benchmark 옵션은 원래 실행과 일치해야 한다.

```bash
caffeinate -i python -u "$SCRIPT" search --run-dir "$LIGHT_RUN" --resume \
  2>&1 | tee -a "$LIGHT_RUN/search.log"

caffeinate -i python -u "$SCRIPT" fit --run-dir "$LIGHT_RUN" --resume \
  2>&1 | tee -a "$LIGHT_RUN/fit.log"

caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$LIGHT_RUN" --resume \
  --include-online --repeats 3 --warmups 1 \
  --rhs-counts 1 4 16 64 --regimes cold warm multiple \
  2>&1 | tee -a "$LIGHT_RUN/benchmark.log"
```

소스/환경 fingerprint 불일치는 manifest를 수동 수정해 우회하지 않는다. 이전 checkout과
환경에서 재개하거나 새 run으로 시작한다. Validation을 시작하면 search/fit 변경을 막는다.

## 11. Validation 선택 고정과 final/OOD

Validation 결과를 확인하고 다음 명령으로 최종 평가 조건을 고정한다. `freeze`는
완료된 benchmark의 `include_online` 상태와 선택 결과를 그대로 가져오므로 별도의
`--include-online` 옵션이 없다.

```bash
python -u "$SCRIPT" freeze --run-dir "$LIGHT_RUN" \
  --repeats 3 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple
caffeinate -i python -u "$SCRIPT" final --run-dir "$LIGHT_RUN" \
  2>&1 | tee "$LIGHT_RUN/final.log"
```

중단된 final만 다음처럼 재개한다.

```bash
caffeinate -i python -u "$SCRIPT" final --run-dir "$LIGHT_RUN" --resume \
  2>&1 | tee -a "$LIGHT_RUN/final.log"
```

Freeze 이후 소스·import·계수 모델·설정·선택 증거가 달라지면 중단한다. Final/OOD에서
가장 빠른 방법을 다시 골라 validation 선택을 바꾸지 않는다. 체크섬과 freeze를 수정해
과거 결과를 새 실험으로 재해석하지 않는다.

## 12. 코드와 검증 위치

| 파일 | 역할 |
| --- | --- |
| `src/adaptive_mg/v67/light_smoother_models.py` | 12개 특징, 표준화, MLP·ridge, 저장·복원 |
| `src/adaptive_mg/v67/light_smoother_runtime.py` | 방향 전체 보정, 고정 계층·factor 재사용, 선택적 온라인 적응 |
| `src/adaptive_mg/v67/light_smoother_search.py` | fallback 없는 training-only 좌표 탐색과 측정별 resume |
| `src/adaptive_mg/v67/light_smoother_fit.py` | 동일 탐색 데이터의 두 모델 fit와 Adam resume |
| `src/adaptive_mg/v67/light_smoother_evaluation.py` | 실측 cold/warm/multiple, 비용 분해·선택 |
| `src/adaptive_mg/v67/light_smoother_study.py` | EM import, 출처 검사, CLI 단계, freeze/final |
| `tests/test_light_smoother_*.py` | 수치 일치·계수·재사용·온라인 비용·데이터/상태 검증 |

```bash
python -m pytest -q tests/test_light_smoother_*.py
```

이 문서의 예산은 연구 설계이고 성능 결과가 아니다. 소수 smoke A에서 실행에 성공해도
본 데이터에서의 수렴·속도 이득은 별도 실측해야 한다. 특히 MLP/회귀가 모두 `(1,1)`에
가까우면 예측 overhead만 추가될 수 있고, 온라인 적응은 추가 계산을 회수하지 못해
총시간을 늘릴 수 있다.

2026-10-21 재검증에서는 `n=7`, train/validation 각각 7개, fit 4 updates의
CLI smoke를 실행했다. 동일 P의 기존 CNN을 포함한 **8개 비교군, 224개 workload 측정,
392개 RHS 풀이가 모두 성공**했다. 탐색에서 확인용 RHS까지 통과한 조합은 1/7이었으나,
배포 setup+solve 선택은 **`fixed_C`**였다. 따라서 속도 향상이 확인됐다고 보고하지 않는다.
한 반복의 작은 smoke timing은 통계적 성능 증거가 아니다.

기본 온라인 trigger는 0회였고 계층/factor 재생성도 0회였다. 따라서 이 smoke로 TTT의
추가 효과를 평가할 수 없다. 강제 trigger를 사용한 별도 테스트에서 두 방식의 적응,
거부·실패한 후보의 비용, checkpoint 불변, warm의 graph 생성/forward/backward 비용을
검사했다. Benchmark `--resume`는 기존 224개 측정을 검증해 재사용했다.

배포 setup 중 P/특징/계수 생성이 실패하면 이미 준비한 classical 계층으로 복구하고
실패 비용과 이유를 기록한다. 오프라인 탐색은 이를 허용하지 않으며 setup 실패가 나면
해당 A를 기록하고 중단한다. 그 A를 제거하거나 허위 학습 label로 대체하지 않는다.

최종 경량 실험 테스트는 **87 passed**다. 전체 회귀 확인은 **991 passed, 7 skipped,
1 failed**였고, 이후 최종 비용/counter 보완에 대해 경량 테스트 87개를 다시 통과했다.
실패는 기존 `test_headroom_source_manifest`다. 작업 전 `cf95031`의 git 객체에서도
`P_HEADROOM_SOURCE.sha256`와 두 과거 runbook
(`docs/HS_WORLD_THESIS_RUNBOOK_KR.md`, `docs/P_HEADROOM_RUNBOOK_KR.md`)의 hash가
불일치함을 확인했다. 기존 provenance를 다시 쓰지 않았다. Skipped 항목은 선택적
native backend 관련이다. Linux CPU 환경에서 확인했으며 Mac M2 Pro의 본 탐색/학습,
실제 final/OOD 성능은 실행하지 않았다. Final 봉인·중단·resume·재평가 금지 경로는
별도의 작은 테스트 데이터로 검증했다. Smoke 설정 자체의 final 진입은 의도적으로 금지된다.

이번 후속 수정은 기존 `cf95031` 작업을 이어서 **새 light-smoother 모듈·테스트·이 문서만**
보완했다. 모두 실패한 탐색 label 차단, 실패 시도 집계, warm priming 공정성, 비교군별
파라미터 수와 일관된 fallback 지표가 범위다. 기존 EM-line/P-only solver, 설정, manifest,
freeze hash는 변경하지 않았다. 소스 변경으로 과거 run의 resume fingerprint가 달라지면
원래 checkout을 사용해야 하며, 완료된 em_P는 새 run으로 검증하여 import할 수 있다.
