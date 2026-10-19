# Direct CNN + 학습한 희소 smoother

기존 direct CNN이 만드는 P는 가져오되, 학습한 계층의 `line_alt` smoothing을
**캐시한 희소 행렬 B(A)의 곱**으로 바꾸는 독립 실험이다. Teacher, distillation,
pruning 및 TTT 실험의 checkpoint와 결과 파일은 수정하지 않는다.

## 실제 바뀐 계산

각 nonterminal level에서 setup 때 한 번 B를 생성하고, pre/post 각각 2회
`r = b - A @ x; x += B @ r`를 수행한다. Coarse solve와 잔차 검사는 유지한다.
매 cycle에 CNN을 다시 실행하거나 line block을 푸는 방식이 아니다.
Sparse branch의 사용하지 않는 learned-level line factorization도 생략한다.
Classical recovery 계층은 기존 방식대로 준비한다.

`d_i=A_ii`, `s=max(abs(A_ij), i!=j)`로 두고 다음 5개 계수를 전체 level에서 공유한다.
Off-diagonal이 모두 0이면 s는 최대 diagonal이다.

```text
fd(q) = c0 + c1*q + c2*q*q
fo(q) = c3*q + c4*q*q
B_ii  = fd(A_ii/s) / A_ii
B_ij  = fo(A_ij/s) / (A_ii + A_jj), i != j
```

B는 실제 A의 희소 연결을 사용한다. Galerkin coarse A의 먼 연결을 3×3로 잘라내지 않는다.
양의 스케일 변화에 대해 `B(alpha*A)=B(A)/alpha`이고 대칭 A에서 B도 대칭이다.
양의 정부호나 수렴까지 보장하는 제약은 아니다. 초기값은 `(0.72,0,0,0,0)`으로
damped Jacobi이며, 전체 V-cycle 손실을 미분하여 계수를 학습한다.

[Weymouth 논문](https://arxiv.org/abs/2110.11029)의 저렴한 학습형 희소 approximate
inverse에서 착안했다. **논문 전체 재현은 아니다.** 이 저장소는 SPD 부호의 A를 사용하며,
여기서는 pre와 post 모두 학습한 B를 사용한다. 논문의 Jacobi pre + 학습한 post,
고정 기하학적 transfer 대신 기존 학습형 P와 EM5 기준을 유지한다.

## 비교와 학습

| Arm | Transfer | Smoother |
| --- | --- | --- |
| `fixed_C` | EM5 기준 P | line_alt, pre/post 2/2 |
| `direct_CNN` | 기존 direct checkpoint | line_alt, pre/post 2/2 |
| `direct_CNN_sparse` | 기존 direct P의 NN 가중치 고정 | 새 TRAIN으로 학습한 희소 B |
| `direct_CNN_sparse_joint` | 위 단계에서 P도 추가 미세조정 | P와 함께 추가 미세조정한 희소 B |

연구 기본 설정은 S-only 140 updates(lr=1e-3), P+S 공동 140 updates(lr=1e-4)이다.
둘 다 fallback 없는 learned-prefix 3 V-cycle 손실과 안정성/no-harm 항을 사용한다.
Transfer가 희소성 제약을 위반한 경우 repair loss는 따로 기록하며, classical로 대신
계산한 값을 성공한 learned task update로 인정하지 않는다. 모든 update가 완료되고
feasible learned task update가 있어야 benchmark에 진입한다. Teacher는 사용하지 않는다.

Train 168개와 validation 42개 operator, 내부격자 n=15,31,63을 사용한다. 기존 source와
이전 실험의 operator 중복을 제외한 새 데이터를 만든다. Full coarsening, EM 반복 5회,
상대 잔차 tolerance 1e-8, fallback의 고정 classical 정책은 그대로다.
S-only와 original direct 비교는 P 고정 상태에서 smoothing 변경을 평가한다.
Joint arm은 추가 P/S 학습 효과를 포함하므로 smoothing만의 효과라고 해석하지 않는다.

`cold`는 EM classical 계층 + CNN/P projection + learned Galerkin 계층 + B 생성 + solve를
포함한다. `multiple`은 이 setup을 A마다 한 번 포함한다. `warm`은 준비된 계층의 solve만
측정한다. 오프라인 140+140 update 학습, 디스크 checkpoint 로드와 PDE 조립은 제외한다.
TTT는 이 실험에 없다. Setup 비용과 learned cycle 실패 후 rollback/fallback 시간은 숨기지 않는다.

## Mac 터미널: 새 checkout과 실행

기존 frozen run에서 pull하거나 소스·manifest를 덮어쓰지 않는다. 아래 새 폴더가 이미
존재한다면 다른 새 이름을 사용한다. 동시 학습/benchmark 없이, 전원을 연결하고 덮개를 연다.
`caffeinate -i`는 덮개를 닫았을 때의 sleep까지 보장하지 않는다.

```bash
git clone --single-branch --branch residual-u-net-teacher \
  https://github.com/slwjackie/adaptive-multigrid.git \
  "$HOME/Desktop/adaptive-multigrid-sparse-smoother"
cd "$HOME/Desktop/adaptive-multigrid-sparse-smoother"
git log -1 --oneline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
set -o pipefail

SOURCE_RUN="$(python - <<'PY'
from pathlib import Path
root = Path.home() / 'Desktop/adaptive-multigrid-unet'
run = Path((root / 'artifacts/unet_research_current.txt').read_text().strip())
print((run if run.is_absolute() else root / run).resolve())
PY
)"
SCRIPT=scripts/run_v6_7_sparse_smoother_study.py
python -m pytest tests/test_sparse_smoother.py tests/test_sparse_smoother_study.py -q
```

`SOURCE_RUN`은 original U-net study의 run이다. Pruning study run을 넣지 않는다.
Prepare는 기존 direct checkpoint와 그 학습 데이터 계보를 검증한 후 별도로 복사한다.

선택적으로 n=7, 단계당 2 updates로 전체 경로만 확인한다. 성능 결론용 학습이 아니다.

```bash
SPARSE_SMOKE="artifacts/sparse_smoke_$(date +%Y%m%d_%H%M%S)"
python "$SCRIPT" prepare --source-run "$SOURCE_RUN" --run-dir "$SPARSE_SMOKE" \
  --config configs/v6_7_sparse_smoother_smoke.json
caffeinate -i python -u "$SCRIPT" train --run-dir "$SPARSE_SMOKE" \
  2>&1 | tee "$SPARSE_SMOKE/train.log"
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$SPARSE_SMOKE" \
  --repeats 1 --warmups 0 --rhs-counts 1 4 --regimes cold warm multiple \
  2>&1 | tee "$SPARSE_SMOKE/benchmark.log"
```

본 학습과 validation:

```bash
SPARSE_RUN="artifacts/sparse_smoother_$(date +%Y%m%d_%H%M%S)"
python "$SCRIPT" prepare --source-run "$SOURCE_RUN" --run-dir "$SPARSE_RUN" \
  --config configs/v6_7_sparse_smoother_research.json
printf '%s\n' "$SPARSE_RUN" > artifacts/sparse_smoother_current.txt
caffeinate -i python -u "$SCRIPT" train --run-dir "$SPARSE_RUN" \
  2>&1 | tee "$SPARSE_RUN/train.log"
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$SPARSE_RUN" \
  --repeats 3 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple \
  2>&1 | tee "$SPARSE_RUN/benchmark.log"
```

Validation 비교가 끝나고 평가 프로토콜을 확정한 뒤 final/OOD를 실행한다.
Full run은 기본 final 56개와 OOD를 포함한 총 126개 operator이고 grid OOD는 n=255이다.
기본 final만 원하면 freeze 명령에 `--splits final`을 추가한다.

```bash
python "$SCRIPT" freeze --run-dir "$SPARSE_RUN" \
  --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple
caffeinate -i python -u "$SCRIPT" final --run-dir "$SPARSE_RUN" \
  2>&1 | tee "$SPARSE_RUN/final.log"
```

중단 후에는 원래 Python 프로세스가 끝났는지 확인하고, 환경과 변수부터 복원한다.

```bash
cd "$HOME/Desktop/adaptive-multigrid-sparse-smoother"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
set -o pipefail
SPARSE_RUN="$(cat artifacts/sparse_smoother_current.txt)"
SCRIPT=scripts/run_v6_7_sparse_smoother_study.py
ps -ax -o pid,stat,pcpu,etime,command | grep '[r]un_v6_7_sparse_smoother_study.py'
```

다음 중 **중단된 단계 하나**만 실행한다. Train 재개는 완료된 S-only를 검증해서 읽고
joint 단계만 이어갈 수 있다. Freeze 후에는 train/validation을 다시 실행하지 않는다.

```bash
caffeinate -i python -u "$SCRIPT" train --run-dir "$SPARSE_RUN" --resume \
  2>&1 | tee -a "$SPARSE_RUN/train.log"

caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$SPARSE_RUN" --resume \
  --repeats 3 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple \
  2>&1 | tee -a "$SPARSE_RUN/benchmark.log"

caffeinate -i python -u "$SCRIPT" final --run-dir "$SPARSE_RUN" --resume \
  2>&1 | tee -a "$SPARSE_RUN/final.log"
```

## 결과 판독

`benchmarks/validation/comparison.csv`는 시간·정확도·성공/실패를 기록한다.
`cycle_diagnostics.json`은 같은 성공 cohort에서 classical보다 cycle이 적은/같은/많은
case 수와 합계, case별 accepted learned cycles, rollback, fallback, setup 실패를 기록한다.
Multiple의 cycle은 여러 RHS의 **합계**이다. 실패 case는 따로 명시하므로 성공 cohort만의
빠른 시간으로 전체 성공을 주장하지 않는다.

`raw_contractions.json`은 validation RHS probe 2개에서 **fallback 없이 3 cycles**의
잔차 이력을 기록한다. Runtime의 near-tolerance classical 마무리도 이 검사에는 없다.
이 진단은 시간 측정 밖에서 실행하며 학습 데이터를 추가하는 절차가 아니다.
Raw 검사 실패는 그대로 실패로 기록한다. Final에서는 동결된 설정의 safeguarded
time-to-tolerance와 cycle 진단을 기록하며, 추가 학습·모델 선택을 수행하지 않는다.

희소 곱 한 번이 싸더라도 cycle 증가나 fallback이 절감분을 넘어설 수 있다.
특히 강한 이방성에서 pointwise sparse smoother가 line_alt보다 약할 수 있다.
Setup 포함 총시간, 동일 tolerance와 정확도, failure 수를 함께 확인해야 한다.
본 구현 자체나 소규모 smoke 통과는 속도 개선의 증거가 아니다.

## 구현 검증 기록

Linux CPU에서 전체 회귀 검사 **783 passed, 7 skipped**를 확인했다. Skip은 선택적인
native backend 관련이다. 새 검사에는 실제 sparse/runtime 대 미분 unroll의 다중 level
일치, 유한차분 gradient, 희소 연결/스케일/대칭, 고정 P 불변성, 공동 학습, B 캐시,
실패 시 고정 classical 복구, immutable source import, benchmark resume가 포함된다.

실제 CLI smoke는 n=7의 7개 새 validation operator, 단계별 2 updates,
3 repeats/1 warmup에서 아래와 같았다. 모든 arm은 7/7 성공했지만 속도 향상은 없었다.
이는 Mac의 본 학습 checkpoint 성능이나 큰 격자의 결과가 아니다.

| Arm | Cold RHS1 cycle 합 | Cold speedup vs fixed | Warm speedup vs fixed | Multiple RHS4 speedup vs fixed |
| --- | ---: | ---: | ---: | ---: |
| fixed_C | 60 | 1.000 | 1.000 | 1.000 |
| direct_CNN | 60 | 0.598 | 0.599 | 0.724 |
| direct_CNN_sparse | 203 | 0.491 | 0.493 | 0.563 |
| direct_CNN_sparse_joint | 203 | 0.504 | 0.475 | 0.587 |

Speedup은 fixed 시간 / candidate 시간이며 1 미만이면 더 느리다. 이 smoke에서 sparse
두 arm은 cold RHS1에서 accepted learned cycles가 각각 193개였고 rollback/fallback은
0이었다. 성공 결과가 전부 classical 대체로 나온 것은 아니지만, cycle 증가로 총시간이
나빠졌다. Fallback 없는 raw 검사도 별도로 생성되었다. 본 연구 설정의 충분한 학습과
동일 정확도의 총시간 비교가 끝나기 전에는 개선되었다고 해석하지 않는다.
