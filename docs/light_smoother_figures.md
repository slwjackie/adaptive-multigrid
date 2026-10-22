# 학습 완료된 light smoother의 수렴·시간·PDE별 일관성 그림

실행 파일: `scripts/run_v6_7_light_smoother_figures.py`

기존 학습과 validation을 다시 할 필요는 없습니다. `report`는 완료된 validation을
읽어 분석하고, `replay`는 저장된 checkpoint로 일부 validation 문제를 다시 풉니다.
공간 오차에는 최종 해 벡터가 필요하지만 기존 benchmark JSON에는 이 벡터가 없으므로,
그 부분만 새로운 풀이가 필요합니다. 두 명령 모두 모델 선택·재학습·final/OOD 평가를
수행하지 않습니다. 새로운 출력 폴더를 사용하고 기존 run은 읽기만 합니다.

## 포함되는 비교군

| 이름 | 구성 |
|---|---|
| fixed_C | Bilinear P + classical alternating line smoothing + full coarsening |
| em5_C | EM5 P + classical line smoothing; 별도 classical 참고군 |
| learned_P_line | 공통 learned P + omega=(1,1) |
| learned_P_MLP | 공통 learned P + 242-parameter MLP 계수 |
| learned_P_regression | 공통 learned P + 26개 ridge 회귀 계수 |
| learned_P_MLP_TTT | MLP 계수 + 선택적 출력 offset TTT |
| learned_P_regression_online | Ridge 계수 + 온라인 수치 계수 탐색 |
| frozen_P_line | 동일 P 확인을 통과한 기존 CNN line smoother |

이 목록은 **현재 light study에서 함께 평가한 구성**입니다. 과거 U-Net/distillation,
pruning, GRU 또는 P가 다른 joint_line은 이 run의 비교군이 아닙니다.
존재하지 않는 frozen_P_line checkpoint를 다른 모델로 대체하지 않습니다.
가용 비교군과 누락 이유를 manifest에 기록합니다.

## Mac에서 실행

현재 학습을 마친 환경을 그대로 사용합니다. matplotlib만 없으면 설치합니다.

```bash
cd "$HOME/Desktop/adaptive-multigrid-light"
source .venv/bin/activate
git pull --ff-only origin residual-u-net-teacher
python -m pip install matplotlib
```

기존 run과 새로운 그림 폴더를 지정합니다. 아래 marker는 이전에 사용한
`light_subset_20261021_232353` 같은 실제 학습 run을 가리켜야 합니다.

```bash
LIGHT_RUN="$(cat artifacts/light_subset_current.txt)"
FIG_SCRIPT="scripts/run_v6_7_light_smoother_figures.py"
FIG_RUN="artifacts/light_figures_$(date +%Y%m%d_%H%M%S)"
BENCH="$LIGHT_RUN/benchmarks/validation_warm_rhs1_16"
printf '%s\n' "$FIG_RUN" > artifacts/light_figures_current.txt
```

먼저 기존 **전체 140 A, RHS 1·16** 결과를 분석합니다. 추가 solve가 없습니다.
완료되지 않은 `benchmarks/validation` 대신, 완료된 warm benchmark를 지정하세요.

```bash
python -u "$FIG_SCRIPT" report --benchmark-dir "$BENCH" \
  --output "$FIG_RUN/existing"
open "$FIG_RUN/existing"
```

그다음 **각 family의 사전 지정 sample=0, n=63**을 다시 풀어 실제 wall-time
곡선과 오차 분포를 만듭니다. 7 A × 8 arms × 1 repeat = 56개 측정입니다.
Warm 준비용 독립 RHS 풀이 56개가 추가됩니다. 학습은 없습니다.

```bash
caffeinate -i python -u "$FIG_SCRIPT" replay --run-dir "$LIGHT_RUN" \
  --n 63 --regimes warm --repeats 1 --output "$FIG_RUN/replay_n63"
open "$FIG_RUN/replay_n63"
```

`n=31` 그림도 필요하면 별도 폴더에 생성합니다.

```bash
caffeinate -i python -u "$FIG_SCRIPT" replay --run-dir "$LIGHT_RUN" \
  --n 31 --regimes warm --repeats 1 --output "$FIG_RUN/replay_n31"
```

Cold와 warm을 각각 3회 진단하려면 새 출력 폴더를 사용합니다.
7 × 8 × 2 × 3 = 336개 측정이며 warm prime 168개가 별도로 실행됩니다.
그림의 대표 곡선은 사전에 정한 repeat=0이고, 모든 반복은 CSV에 저장합니다.
최고 성능의 repeat를 골라 그림으로 만들지 않습니다.

```bash
caffeinate -i python -u "$FIG_SCRIPT" replay --run-dir "$LIGHT_RUN" \
  --n 63 --regimes cold warm --repeats 3 \
  --output "$FIG_RUN/replay_n63_cold_warm_r3"
```

전체 140 A의 최종 해 벡터가 반드시 필요할 때만 다음을 사용합니다.
Warm 기준 1,120개 측정 + 1,120개 prime 풀이입니다. PDE별 통계는 앞의
`report`가 이미 전체 140 A를 사용하므로, 통계 때문에 이것을 다시 할 필요는 없습니다.

```bash
caffeinate -i python -u "$FIG_SCRIPT" replay --run-dir "$LIGHT_RUN" \
  --all-validation --regimes warm --repeats 1 \
  --output "$FIG_RUN/replay_all_validation"
```

## 중단 후 resume

새 터미널에서는 먼저 같은 저장소·venv에 들어가고 변수를 복구합니다.

```bash
LIGHT_RUN="$(cat artifacts/light_subset_current.txt)"
FIG_RUN="$(cat artifacts/light_figures_current.txt)"
FIG_SCRIPT="scripts/run_v6_7_light_smoother_figures.py"
```

중단했던 명령과 **같은 옵션·출력 폴더**에 `--resume`만 추가합니다.
완료된 측정은 다시 실행하지 않으며, 마지막에 커밋되지 않은 측정만 재시도합니다.

```bash
caffeinate -i python -u "$FIG_SCRIPT" replay --run-dir "$LIGHT_RUN" \
  --n 63 --regimes warm --repeats 1 --output "$FIG_RUN/replay_n63" --resume
```

```bash
cat "$FIG_RUN/replay_n63/progress.json"
```

그림 그리기만 중단된 경우에도 같은 명령을 재개하면 저장된 측정으로 그림을 만듭니다.
기존 결과 분석 역시 같은 `report` 명령에 `--resume`을 붙여 재실행할 수 있습니다.

## 생성 파일과 읽는 방법

모든 그림은 PNG와 SVG 두 형식으로 저장합니다.

| 출력 | 의미 |
|---|---|
| existing/cycles/ | 기존 validation의 승인된 해 갱신 횟수–상대 잔차; 각 구성과 fixed_C, EM5를 겹쳐 표시 |
| existing/families/*success_rate* | 전체 평가 A의 성공/전체 수; 실패를 제외하지 않음 |
| existing/families/*fallback_rate* | Classical safety step을 사용한 A의 비율; near-tolerance tail 포함 |
| existing/families/*paired_geomean_speedup* | fixed_C와 해당 구성 모두 성공한 A의 시간 speedup; 표본 수 표시 |
| existing/families/*common_median_cycles* | 모든 구성이 성공한 동일 A 집합에서 family별 cycle 중앙값 |
| existing/families/*common_median_seconds* | 같은 집합에서 family별 RHS당 시간 중앙값 |
| existing/families/*completion_profile* | 전체 A 중 fixed_C 시간의 배수 이내로 두 방법 모두 성공한 비율 |
| existing/consistency/ | 각 구성 vs fixed_C/EM5, family별 시간·cycle 중앙값 및 A 간 P10–P90 범위 |
| existing/case_metrics.csv | 실패·시간·cycle·오차·fallback·온라인 적응을 포함한 모든 A의 수치 |
| existing/family_metrics.csv | Family별 성공률·시간·cycle·표본 수 |
| existing/robustness_summary.csv | 같은 성공 A 집합에서 7개 family 중앙값의 CV, 최악 family 시간, 최대/최소 비율 |
| replay_n63/cycles/ | 다시 푼 문제의 승인된 해 갱신–상대 잔차 |
| replay_n63/wall/ | 실제 경과 시간–상대 잔차 |
| replay_n63/errors/ | 각 구성의 최종 해와 동일 A,b의 fine-grid LU 해 사이 절대 오차 분포 |
| replay_n63/replay_metrics.csv | 모든 반복의 성공·실행 cycle·시간·오차·fallback·온라인 적응 |
| replay_n63/references/*.npz | 기준 해와 RHS |
| replay_n63/records/*.npz | 각 구성의 최종 해 벡터; 실패 시 비어 있을 수 있음 |
| replay_n63/records/*.json | 잔차·실제 시각·분기·안전 복구·적응 이력과 해 파일 hash |
| manifest.json, progress.json, README_KR.md | 입력/프로토콜 provenance, 진행 상태, 해석 안내 |

현재 모든 조건의 입력 격자는 동일하므로 `n31`, `n63`을 나누어 비교합니다.
RHS 1과 16, cold와 warm도 섞지 않습니다. 하나의 A에서 반복한 RHS들을 서로 다른
독립 PDE 표본처럼 세지 않습니다. 대표 곡선은 기본적으로 sample index=0, RHS
index=0, repeat=0이며 결과를 보고 좋은 사례로 바꾸지 않습니다.

## “느리지만 일관적”이라는 주장을 검토하는 기준

각 family별 같은 A에서 측정한 지표를 비교해야 합니다. 모델 구조/가중치를 공유한다는
사실 자체가 성능 일관성의 증거는 아닙니다. fixed_C 역시 모든 PDE에 같은
line_alt + bilinear + full 설정을 사용합니다.

1. 전체 성공률, 최악 family 성공률을 먼저 봅니다.
2. `consistency/`에서 cycle과 실제 시간의 family 간 차이와 family 내부 P10–P90 폭을 봅니다.
3. 같은 성공 A 집합에서 `robustness_summary.csv`의 CV와 최악 family 시간을 봅니다.
4. `fallback_rate`를 같이 봅니다. 성공이 classical recovery에 의존하면 배포 결합 방식의
   안정성으로 설명해야 하며, NN 단독의 강건성으로 설명하면 안 됩니다.

CV는 family 중앙값 \(m_1,\dots,m_7\)에 대해
\(\sqrt{\frac17\sum_f(m_f-\bar m)^2}/\bar m\)입니다.
낮을수록 family 중앙값의 상대 산포가 작습니다. 그러나 느린 방법이 단지 모든
family에서 비슷하게 느린 경우에도 CV가 낮을 수 있으므로 평균 수준·실패·tail과
함께 제시합니다. 모든 구성이 성공한 동일 A가 하나도 남지 않은 family가 있으면
7-family CV를 계산하지 않고 빈 값으로 남깁니다. 이 표본 제외는 **조건부 산포 계산만**의
공통 기준이며, 전체 평가 실패는 별도 표와 그림에서 모두 유지됩니다.

P10–P90은 operator 간 관측 범위이며 신뢰구간이 아닙니다. 기존 warm 실험은
측정 반복이 1회이므로 작은 시간 차이나 안정성 우열을 통계적으로 확정할 수 없습니다.
speedup은 \(t_{fixed}/t_{method}\)이고 1보다 크면 빠릅니다. Completion profile은
\(t_{method}/t_{fixed}\)가 가로축이며, 실패한 쌍은 전체 A의 분모에 남고 완주자로 세지 않습니다.

## 시간 축과 수렴의 정확한 의미

수렴 곡선은 \(\|b-Au_k\|_2/\|b\|_2\)를 사용합니다. Tolerance 선은 기존 solver의
\(\max(atol,rtol\|b\|_2)/\|b\|_2\)입니다. 초기값은 0입니다.
Cycle 그림의 가로축은 **승인된 해 갱신 횟수**입니다. 거절된 trial, recovery,
온라인 probe/gradient V-cycle은 해당 CSV의 별도 실행 횟수로 남기며, 시간에는 포함합니다.
곡선의 네모는 NN 비교군에서 classical safety step으로 갱신한 지점입니다.
끝점의 X는 해당 RHS 실패입니다. Warm prime 실패로 전체 workload만 실패한 경우도
CSV에 기록합니다.

기존 기록의 `cycle_seconds` 합으로 정확한 전체 경과 시간 곡선을 재구성할 수는 없습니다.
따라서 해당 합을 wall time으로 표시하거나, 전체 시간을 cycle 수로 나누어 가상의
시간 축을 만들지 않습니다. `replay`는 기존 solver의 clock 호출을 관측하여 실제
승인 잔차와 절대 시각을 연결합니다. 수치 경로는 그대로 사용하며 source 파일을 바꾸지
않습니다. 관측 함수의 overhead는 측정 시간에 포함되어 `observer_seconds`로도 기록합니다.
시간 예산을 갖는 온라인 적응은 이 작은 관측 overhead의 영향도 받을 수 있으므로,
공식적인 속도 우열에는 비계측 validation을 사용합니다.

- **Cold**: fresh constructor + solve. Classical fallback 계층, learned P/Ac, factor,
  특징, 계수 예측, lazy CNN setup, 온라인 적응·거절·fallback 모두 wall interval 안에 포함됩니다.
- **Warm**: constructor와 독립 RHS prime을 제외한 실제 solve. Prime 동안 온라인
  적응을 끄고, 측정 전에 원래 계수와 적응 예산을 복원합니다. 측정 도중 TTT/수치 적응은 포함합니다.
- 모델 파일 로딩·데이터 복구·정답 LU·그림/파일 저장은 비교 시간 밖입니다.
  Cold는 프로세스 시작이나 OS cache까지 초기화한 시간이라는 뜻이 아닙니다.
- `online_adaptation_seconds`는 wall 시간의 부분집합입니다. 다시 더하지 않습니다.
- `--rhs-index 15`는 기존 RHS 생성 규칙의 16번째 RHS를 fresh 상태에서 **하나만** 푸는
  진단입니다. 앞선 15개 RHS에서 적응한 배치의 상태를 재현한다고 주장하지 않습니다.
- LU와의 오차는 이산 선형계 오차입니다. 연속 PDE에 대한 격자 이산화 오차 인증은 아닙니다.

## 입력 provenance와 검증

`report`는 완료 상태, 모든 선언된 A/arm/RHS/repeat, 원래 repeat ledger의 hash,
같은 P signature를 검사합니다. `replay`는 기존 light/subset loader로 데이터 분할,
checkpoint, scaler, 학습 정답 정책, source/environment provenance를 검증합니다.
Run의 환경/source mismatch를 무시하거나 기존 manifest hash를 바꾸지 않습니다.
추가된 파일은 새 실행 파일·테스트·문서뿐이며 기존 수치 src와 원래 실행 파일은 수정하지 않았습니다.

검증에는 세 solver 경로(classical, light, CNN)의 수치 일치, warm 적응/거절 비용,
module clock 복원, 원본 파일 불변, 실패 보존, 불완전/변조 기록 거절,
측정 직후 중단 및 resume, 모델·설정 변경 resume 거절을 포함합니다.

사용자 Mac의 실제 checkpoint와 결과를 가져온 것은 아니므로, 이 코드의 smoke 결과를
실제 연구 모델의 수렴·속도·강건성 개선 증거로 해석하지 않습니다.
