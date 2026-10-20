# Direct CNN pruning + test-time training

> **과거 실험·구현 기록.** 아래의 기본값·명령·결과는 이 문서가 다루는 경로에 한정된다.
> 현재 `residual-u-net-teacher`의 주 실험은 [line_alt 고정 + P-only 학습](learned_p_study.md)이다.
> 현재 실행은 [Mac runbook](learned_p_runbook.md), 경로별 구분은 [문서 목록](README.md)을 따른다.

Teacher는 그대로 두고, 기존 `residual-u-net-teacher` 실험의 **direct CNN만** 가져오는 독립 실험이다. 기존 frozen run의 코드·파일·모델을 덮어쓰지 않는다. 이 실험은 빨라졌다고 가정하지 않으며, 실패와 느려진 결과도 그대로 기록한다.

## 비교 대상과 기본 설정

| Arm | 동작 |
| --- | --- |
| `fixed_C` | `line_alt + energy minimization(각 level 정확히 5회) + full coarsening`, pre/post 각 2회 |
| `direct_CNN` | 기존 run의 검증된 direct checkpoint 그대로 사용 |
| `pruned_CNN` | direct CNN의 채널을 실제 삭제하고 새 TRAIN 데이터에서 MG 손실로 미세조정 |
| `pruned_CNN_TTT` | 동일한 pruned checkpoint에서 각 A에 대해 제한된 온라인 적응 후 풀이 |

기본 채널 폭은 8→6이고 transfer 파라미터 수는 **3,208→1,978(38.34% 감소)**이다. 모든 hidden convolution과 residual 연결에 동일한 채널 집합을 사용한다. 중요도는 연결된 가중치 절댓값·bias의 합이며 동점 처리는 결정적이다. 0 mask만 넣는 방식이 아니라 작은 dense convolution으로 재구성한다. 10개 입력, 16개 출력, 희소 P 연결 및 기존 보간 제약은 유지한다. `--hidden 4`이면 1,036개지만 성능은 별도 검증해야 한다.

연구 설정의 pruning 후 미세조정은 **280회**, train/validation 격자는 **15,31,63**이다. 기존 학습 데이터와 중복되지 않는 새 operator를 사용한다. 따라서 direct 대비 결과는 **pruning + 추가 미세조정의 효과**이며, pruning 단독의 인과 효과를 분리한 비교는 아니다. Pruning과 TTT의 비교는 정확히 같은 pruned checkpoint에서 시작한다. Teacher·distillation은 사용하지 않는다.

TTT는 NN의 마지막 `transfer.head`만 최대 **2회 Adam(lr=1e-4)** 갱신한다. 현재 A에서 만든 정규화 잔차 probe 2개로 1 raw V-cycle 후 상대 잔차의 log 평균을 줄이고, 시작 가중치와의 거리 정규화를 적용한다. 별도 guard probe 2개에서 모두 악화되지 않고 기하평균이 개선되어야 채택한다. Probe 생성·학습·guard에는 평가 RHS나 정답 해가 전달되지 않는다. Guard는 classical fallback을 허용하지 않는다. 불가능하거나 악화된 후보는 원래 pruned 모델로 되돌린다. 실제 solve의 기존 classical fallback은 유지된다.

TTT마다 기존 classical EM 계층을 미분 그래프의 기준으로 재사용한다. 갱신한 P에 맞춰 Galerkin 계층도 갱신한다. 같은 A의 여러 RHS에서는 적응을 한 번만 수행한다. 다른 A 또는 새 반복 측정에서는 항상 동일한 시작 checkpoint로 초기화한다. CPU 단일 thread, NN FP32 및 수치 풀이 FP64 설정을 사용한다.

## Setup과 cold/warm의 의미

이 코드의 cold는 **A, RHS, NN checkpoint가 이미 메모리에 준비된 상태에서 새 MG를 구성하는 비용부터** 측정한다. 디스크 checkpoint 로드·오프라인 학습·PDE 행렬 조립·제조해 RHS 생성은 포함하지 않는다.

| 비용 | 포함되는 작업 |
| --- | --- |
| Classical setup | A 검증/CSR 준비, full-coarsening 계층, EM5 P, R/Ac, smoother와 coarse 풀이 factorization, cache key |
| Neural setup | A 특징, CNN 실행, P 제약 projection/희소 조립, 학습한 P의 Galerkin 계층·factorization, feasibility 검사 |
| TTT setup | 사본 생성, probe, 미분 그래프, forward/backward/Adam, raw guard, 채택/복구 및 최종 학습 계층 생성 |
| Solve | 실제 V-cycle, 잔차 검사, freshness 검사, 필요 시 rollback/classical fallback |

- `cold`: setup + RHS 1개 solve. 첫 A를 실제 처리하는 비용이다.
- `warm`: 준비/prime 후 같은 A에서 RHS 1개 solve. 준비 비용은 별도 보고한다.
- `multiple`: setup 한 번 + 같은 A의 서로 다른 RHS 여러 개 solve. 유한 횟수 재사용의 총비용이다.
- `warm_multiple`: 준비 및 독립 prime RHS 제외 후 여러 RHS solve. 필요하면 추가 측정할 수 있다.

Warm은 **준비 완료 후 풀이 속도**를 비교하는 유효한 지표이다. 하지만 warm에서는 cached P를 사용하므로 CNN을 다시 실행하지 않는다. Pruning이 줄이는 생성 시간과 TTT의 적응 시간은 warm 측정 밖이다. 전체 속도 향상은 cold/multiple로 판단하고, warm만 빠른 경우에는 그 범위만 주장한다. 새 A나 바뀐 A에 대한 setup을 무료로 취급하지 않는다.

`ttt_seconds`는 cold/multiple에 포함된 추가 적응 비용이고 `warm_ttt_seconds`는 warm에서 제외된 적응 비용이다. `ttt_preparation_seconds`는 어느 regime에서든 실제 발생한 적응 시간을 표시한다. `total_setup_seconds`에는 constructor 비용이 이미 들어 있으므로 TTT를 다시 더하면 안 된다. `ttt_operator_generation_seconds`는 TTT의 runtime bank 생성 부분이며 전체 적응 시간과 중첩되는 진단값이다. 학습용 forward/backward는 `ttt_seconds` 안에 있다. Raw 결과의 `ttt_report`는 채택 여부, 수행/유지 update 수, 전후 guard 잔차 및 실패 이유를 기록한다.

## Mac에서 실행

기존 final/OOD가 실행 중이면 **기존 checkout에서 pull하거나 source를 변경하지 않는다.** 아래처럼 새 checkout을 만든다. 기존 평가가 끝난 뒤 새 학습/benchmark를 시작해야 두 작업의 CPU 경쟁으로 시간 측정이 오염되지 않는다. 새 환경은 기존 가상환경과 분리한다.

```bash
git clone --single-branch --branch residual-u-net-teacher \
  https://github.com/slwjackie/adaptive-multigrid.git \
  "$HOME/Desktop/adaptive-multigrid-pruning"
cd "$HOME/Desktop/adaptive-multigrid-pruning"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
set -o pipefail

SOURCE_RUN="$(python - <<'PY'
from pathlib import Path
root = Path.home() / 'Desktop/adaptive-multigrid-unet'
run = Path((root / 'artifacts/unet_research_current.txt').read_text().strip())
print((run if run.is_absolute() else root / run).resolve())
PY
)"
SCRIPT=scripts/run_v6_7_direct_pruning_study.py
```

기존 폴더가 이미 있으면 재clone하지 말고, 해당 폴더의 실험이 실행 중이지 않은 상태에서 `git pull --ff-only`를 사용한다. 동결한 실험의 정확한 resume에는 당시 source와 환경이 필요하므로 코드 변경 후에는 새 run을 만든다.

### 선택: 작은 실행 확인

본 학습 전 n=7에서 import, pruning, TTT, 측정 경로를 확인한다. 이 결과로 연구 성능을 주장하지 않는다.

```bash
PRUNE_SMOKE="artifacts/direct_pruning_smoke_$(date +%Y%m%d_%H%M%S)"
python "$SCRIPT" prepare --source-run "$SOURCE_RUN" --run-dir "$PRUNE_SMOKE" \
  --config configs/v6_7_direct_pruning_smoke.json
caffeinate -i python -u "$SCRIPT" train --run-dir "$PRUNE_SMOKE" \
  2>&1 | tee "$PRUNE_SMOKE/train.log"
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$PRUNE_SMOKE" \
  --repeats 1 --warmups 0 --rhs-counts 1 4 --regimes cold warm multiple \
  2>&1 | tee "$PRUNE_SMOKE/benchmark.log"
```

### 본 실험

```bash
PRUNE_RUN="artifacts/direct_pruning_$(date +%Y%m%d_%H%M%S)"
python "$SCRIPT" prepare --source-run "$SOURCE_RUN" --run-dir "$PRUNE_RUN" \
  --config configs/v6_7_direct_pruning_research.json
printf '%s\n' "$PRUNE_RUN" > artifacts/direct_pruning_current.txt

caffeinate -i python -u "$SCRIPT" train --run-dir "$PRUNE_RUN" \
  2>&1 | tee "$PRUNE_RUN/train.log"

caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$PRUNE_RUN" \
  --repeats 3 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple \
  2>&1 | tee "$PRUNE_RUN/benchmark.log"

python "$SCRIPT" freeze --run-dir "$PRUNE_RUN" \
  --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple

caffeinate -i python -u "$SCRIPT" final --run-dir "$PRUNE_RUN" \
  2>&1 | tee "$PRUNE_RUN/final.log"
```

마지막 두 명령은 프로토콜 확정 후 실행한다. 전체 final/OOD는 기본 final 56개(n=15,31,63,127), grid OOD n=255를 포함해 총 126개 operator이다. 계산량을 제한하려면 **freeze 시점에** `--splits final`을 붙여 기본 final만 평가할 수 있다. 이 경우 전체 OOD 평가 완료로 해석하지 않는다. Freeze 이후 같은 run의 split 목록이나 TTT 예산을 바꿀 수 없다.

학습·측정은 Mac 덮개를 열고 전원을 연결한 상태에서 실행한다. `caffeinate -i`는 idle sleep을 억제하지만 덮개를 닫아 발생하는 sleep까지 보장하지 않는다. 시간은 기존 학습 로그와 이번 validation 진행으로 판단한다. 본 학습 checkpoint가 없는 환경에서 만든 smoke 시간으로 M2 Pro의 전체 실행 시간을 단정하지 않는다.

### 중단 후 재개

먼저 기존 프로세스가 끝났는지 확인하고 **중단된 단계 하나만** 실행한다. 완료된 benchmark를 frozen run에서 다시 실행하지 않는다.

```bash
cd "$HOME/Desktop/adaptive-multigrid-pruning"
source .venv/bin/activate
set -o pipefail
PRUNE_RUN="$(cat artifacts/direct_pruning_current.txt)"
SCRIPT=scripts/run_v6_7_direct_pruning_study.py
```

학습 단계에서 중단됐으면:

```bash
caffeinate -i python -u "$SCRIPT" train --run-dir "$PRUNE_RUN" --resume \
  2>&1 | tee -a "$PRUNE_RUN/train.log"
```

Validation benchmark에서 중단됐으면:

```bash
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$PRUNE_RUN" --resume \
  --repeats 3 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple \
  2>&1 | tee -a "$PRUNE_RUN/benchmark.log"
```

Final/OOD에서 중단됐으면:

```bash
caffeinate -i python -u "$SCRIPT" final --run-dir "$PRUNE_RUN" --resume \
  2>&1 | tee -a "$PRUNE_RUN/final.log"
```

## 결과 읽기

- `imported/pruning_report.json`: 실제 선택 채널, 파라미터 수, pruning 직후 서명. 미세조정 이후 checkpoint 서명은 `checkpoints/pruned/status.json`에 있다.
- `checkpoints/pruned/training.json`: 오프라인 미세조정 기록.
- `benchmarks/validation/comparison.csv`: 모든 arm의 시간·cycle·오차·fallback·TTT 진단.
- `benchmarks/validation/paired_pruning_ttt.json`: `pruned_CNN_vs_direct_CNN`, `pruned_CNN_TTT_vs_pruned_CNN`의 대응 성능비. **1보다 크면 candidate가 빠르다.** 실패 사례는 speedup에서 제외하되 별도 보고한다.
- `benchmarks/validation/raw_results.json`: 반복별 TTT 채택/거부, guard 잔차, 각 RHS 결과.
- `final/<split>/`: 고정 프로토콜의 final/OOD 결과. `progress.json`으로 진행률을 확인한다.

Pruning 후 같은 tolerance에서 cycle이 늘 수 있고, TTT 후 잔차가 개선되어도 학습 비용 때문에 전체 시간이 느려질 수 있다. Warm 시간, cold/multiple 시간, 성공률 및 정확도 진단을 함께 읽는다. Negative result도 정상적인 결과이며 자동으로 성공이라고 선언하지 않는다.

## 구현 검증 기록

Linux CPU/Python 3.12/PyTorch 2.14.1 환경의 전체 테스트: **775 passed, 7 skipped**. 물리적인 채널 삭제와 masked 원본의 함수 일치, checkpoint 왕복, head gradient, 원본 모델 불변, EM 재계산 방지, guard의 fallback 금지, TTT 비용의 중복 계상 방지, resume·freeze·final 무결성을 포함한다.

별도 CLI smoke는 n=7의 validation operator 7개, 3 repeats/1 warmup, RHS 1·4로 실행했다. 시작 direct는 n=7·15에서 8회만 학습한 smoke 모델이고 pruning 후 미세조정은 2회였다. 실제 사용자의 1,120회 학습 checkpoint 또는 Mac 측정이 아니다.

| 조건 | Pruned / direct 속도비 | TTT / pruned 속도비 |
| --- | ---: | ---: |
| cold RHS=1 | 0.982 | 0.175 |
| warm RHS=1 | 0.975 | 1.070 |
| multiple RHS=4 | 0.985 | 0.264 |

표의 속도비는 reference 시간/candidate 시간의 기하평균이며 1보다 커야 candidate가 빠르다. TTT는 7개 중 2개 A에서 guard를 통과했지만 cold RHS=1의 solve cycle 수는 모든 A에서 pruned와 같았다. 따라서 warm의 작은 차이는 검증된 수렴 개선으로 해석하지 않는다. 이 기록의 목적은 실행과 비용 측정 확인이며, 연구 성능 판정에는 본 설정과 학습 완료 checkpoint가 필요하다. Smoke run의 final freeze는 의도적으로 금지된다.
