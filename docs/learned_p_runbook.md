# Mac: learned P 학습·평가 실행 안내

브랜치는 계속 **`residual-u-net-teacher`**다. 현재 경로는 `line_alt`를 고정하고 P만
학습한다. [설계](learned_p_study.md) · [문서 목록](README.md)

명령은 저장소 루트의 macOS 터미널에서 실행한다. 기존 clone이 있으면 그 폴더를
사용하면 된다. 아래 예시의 `adaptive-multigrid-ponly`는 폴더명이며 브랜치명이 아니다.
확대 학습은 새 run을 만든다. 이미 준비한 run의 설정·manifest를 바꾸지 않는다.

## 설치와 공통 환경

새로 clone할 때만:

```bash
git clone --single-branch --branch residual-u-net-teacher \
  https://github.com/slwjackie/adaptive-multigrid.git \
  "$HOME/Desktop/adaptive-multigrid-ponly"
cd "$HOME/Desktop/adaptive-multigrid-ponly"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

기존 설치는 아래 첫 줄을 실제 clone 위치로 바꾼다. 이후 새 터미널을 열 때도 같은
환경을 설정한다. 실행 중인 run을 재개할 때는 소스·패키지·스레드 설정을 유지한다.
새 실험 전에 저장소를 갱신한다면 `git pull --ff-only origin residual-u-net-teacher`를
사용하되, 먼저 `git status --short`에서 로컬 변경 여부를 확인한다.

```bash
cd "$HOME/Desktop/adaptive-multigrid-ponly"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set -o pipefail
SCRIPT="scripts/run_v6_7_learned_p_study.py"
git branch --show-current
git log -1 --oneline
test -f "$SCRIPT" && python "$SCRIPT" --help
```

`caffeinate -i`는 idle sleep을 막는다. 덮개를 닫아도 실행을 유지한다는 보장은 없으므로
장시간 실행 중에는 전원을 연결하고 덮개를 열어 둔다. 폴더를 새로 여는 VS Code 창보다
별도 macOS Terminal 창을 사용하는 편이 실행 창을 관리하기 쉽다.

## 선택 사항: 작은 smoke

이미 같은 버전의 P-only smoke를 마쳤다면 반복할 필요 없다. 처음 설치했을 때만
작은 실행으로 경로·의존성을 확인한다. n=7, train/validation 각 7개, 14 updates다.

```bash
SMOKE_RUN="artifacts/learned_p_smoke_$(date +%Y%m%d_%H%M%S)"
caffeinate -i python -u "$SCRIPT" prepare --run-dir "$SMOKE_RUN" \
  --config configs/v6_7_learned_p_smoke.json && \
caffeinate -i python -u "$SCRIPT" train --run-dir "$SMOKE_RUN" \
  2>&1 | tee "$SMOKE_RUN/train.log" && \
caffeinate -i python -u "$SCRIPT" diagnose --run-dir "$SMOKE_RUN" \
  2>&1 | tee "$SMOKE_RUN/diagnose.log"
```

## 확대 학습: 420개 A, 4,200 updates

저장소 research JSON은 예비 예산인 84개 A / 336 updates다. 아래는 별도 JSON을
만들어 데이터와 학습량만 늘리는 실행안이다. Loss, 모델 폭, smoother는 유지한다.

| 항목 | 확대 설정 |
| --- | --- |
| Train | 7 families × n=15,31,63 × 20개 = 420개 A |
| Validation | 7 families × n=15,31,63,127 × 5개 = 140개 A |
| 직접 학습 | 4,200 updates, 전체 train을 약 10회 사용 |
| 목적함수 | 1 raw V-cycle, update당 Rademacher probe 8개 |
| 적용 모델 | Width-8 P CNN, classical line_alt 고정 |

4,200은 시작 예산이지 충분한 학습의 보장은 아니다. `updates`는 A 하나로 한 번
갱신하는 횟수다. Mac에서 이 예산의 실측 소요 시간은 아직 확정하지 않았다.

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("configs/v6_7_learned_p_research.json").read_text())
config["training_sizes"] = [15, 31, 63]
config["validation_sizes"] = [15, 31, 63, 127]
config["training_per_family"] = 20
config["validation_per_family"] = 5
config["training"]["updates"] = 4200
path = Path("configs/v6_7_learned_p_420.json")
path.write_text(json.dumps(config, indent=2) + "\n")
print("Train=420, validation=140, updates=4200")
PY
P_RUN="artifacts/learned_p_420_$(date +%Y%m%d_%H%M%S)"
caffeinate -i python -u "$SCRIPT" prepare --run-dir "$P_RUN" \
  --config configs/v6_7_learned_p_420.json && \
printf '%s\n' "$P_RUN" > artifacts/learned_p_420_current.txt && \
caffeinate -i python -u "$SCRIPT" train --run-dir "$P_RUN" \
  2>&1 | tee "$P_RUN/train.log" && \
caffeinate -i python -u "$SCRIPT" diagnose --run-dir "$P_RUN" \
  2>&1 | tee "$P_RUN/diagnose.log"
```

`prepare`가 run 폴더를 만든다. Prepare 전에 그 폴더에 로그를 생성하면 새 run의
비어 있음 검사와 충돌할 수 있다. `&&`와 `pipefail`은 앞 단계가 실패하면 다음 단계로
넘어가지 않게 한다. 로그를 보고 싶다면 새 터미널에서 저장한 run 경로를 복원한다.

## 직접 P의 세 비교군 평가

현재 일반 `benchmark` CLI는 `distilled_P` checkpoint까지 요구한다. 아직 직접 P만
학습했으면 아래 **기존 평가 API** 호출을 사용한다. 새 CLI 옵션을 가정하거나 증류
checkpoint를 복사해서 만드는 방식이 아니다. 이 코드는 현재 버전의 내부 API에
의존하므로 실행 코드를 갱신하면 이 안내와 함께 확인한다.

비교군은 bilinear classical, EM5 classical, 직접 학습 P다. 모두 `line_alt`를 쓴다.
Cold는 setup+1 RHS, warm은 준비된 계층의 1 RHS, multiple은 setup+여러 RHS를 잰다.
RHS count는 같은 A에 대한 서로 다른 b의 개수이며 b의 값이 아니다.

```bash
caffeinate -i python -u - "$P_RUN" <<'PY' 2>&1 | tee -a "$P_RUN/benchmark_direct.log"
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from adaptive_mg.v67 import joint_transfer_study as s

out, settings, cfg, baseline = s.load(sys.argv[1])
s._ensure_development_open(out)
_, validation, _ = s._development(out, settings, baseline)
model = s._checkpoint(out, "learned_P", baseline)
s.require_p_only(model, cfg)
emcfg = s.replace(cfg, mg=s.replace(cfg.mg,
    strategy_name=s.FIXED_STRATEGY_NAME,
    em_iterations=5, em_fixed_iterations=True, em_rtol=1e-8))
arms = {
    "fixed_C": dict(model=None, branch="C", selector=False),
    "em5_C": dict(model=None, branch="C", selector=False, reference_only=True,
                  config=emcfg, rules=s.FixedClassicalBaseline.from_config(emcfg)),
    "learned_P": dict(model=model, branch="H_P", selector=False),
}
target = out / "benchmarks/direct_validation"
print(f"Validation {len(validation)}개 평가 시작", flush=True)
result = s.evaluate_research(validation, arms, cfg, baseline, target,
    reference_arm="fixed_C", resume=(target / "run_manifest.json").exists(),
    repeats=3, warmups=1, rhs_counts=[1, 4, 16], regimes=["cold", "warm", "multiple"])
s.report(result, target)
print("Fallback 없는 고정-cycle 검사 시작", flush=True)
s.write_json(target / "raw_contractions.json",
    s.raw_contractions(validation, arms, cfg,
        cycles=settings["raw_cycles"], probes=settings["raw_probes"]))
print(f"평가 완료: {target}", flush=True)
PY
```

140개 A라면 3 arms × 3 repeats × 5 workload 조건 × 140 = **6,300개 측정**이며
warmup과 마지막 raw 검사는 별도다. 동일한 블록을 다시 실행하면 저장된 시간 측정은
resume한다. Raw contraction은 별도 부분 재개가 없어서 다시 계산한다.

이 결과는 development validation이다. `benchmark_evidence.json`을 만드는 정식
네 비교군 CLI 실행을 대신하지 않으므로 이 단계만으로 기존 `freeze/final`을 실행할 수 없다.

## 결과와 진행 상태

| 경로: P_RUN 내부 | 내용 |
| --- | --- |
| `checkpoints/learned_P/status.json` | 학습 update 완료·skip 여부 |
| `diagnostics/p_frobenius.json` | 같은 A·고정 probe에서 초기/학습 P 손실, cycle, 시간 |
| `benchmarks/direct_validation/progress.json` | 시간 측정 진행률 |
| `benchmarks/direct_validation/comparison.csv` | Classical 대비 시간·성공 여부 |
| `benchmarks/direct_validation/cycle_diagnostics.json` | Cycle 증감, rollback 등 |
| `benchmarks/direct_validation/raw_contractions.json` | Fallback 없는 고정-cycle 잔차 감소 |

속도비 `speedup_vs_fixed`는 fixed_C 시간 / 해당 방법 시간이다. **1보다 커야 빠르다.**
실패 수, cycle 수, 총시간을 함께 보고, warm 개선만으로 setup 포함 개선을 주장하지 않는다.
진단의 before는 초기 CNN 경로이므로 순수 classical 비교군과 다르다. 손실만 소폭
감소하거나 cycle이 같다면, 그것만으로 실용적인 속도 개선을 확인한 것은 아니다.

새 터미널에서 공통 환경 블록을 실행한 뒤:

```bash
P_RUN="$(cat artifacts/learned_p_420_current.txt)"
python - "$P_RUN" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for rel in ("checkpoints/learned_P/status.json",
            "benchmarks/direct_validation/progress.json"):
    path = root / rel
    print(rel, json.loads(path.read_text()) if path.exists() else "아직 생성되지 않음")
PY
```

## 중단 후 재개

실행이 실제로 끝났는지 먼저 확인하고 같은 run을 두 프로세스에서 동시에 실행하지 않는다.
새 터미널에서 공통 환경 블록과 위 `P_RUN` 복원을 실행한다. 중단된 학습은:

```bash
caffeinate -i python -u "$SCRIPT" train --run-dir "$P_RUN" --resume \
  2>&1 | tee -a "$P_RUN/train.log" && \
caffeinate -i python -u "$SCRIPT" diagnose --run-dir "$P_RUN" --resume \
  2>&1 | tee -a "$P_RUN/diagnose.log"
```

학습은 저장된 checkpoint부터 재개한다. `diagnose --resume`은 완성된 같은 모델의
진단 JSON이 있으면 재사용하며, 중간 사례별 재개는 아니다. 모델을 갱신한 후 새로
진단할 때는 `--resume` 없이 실행한다. 직접 P 평가는 앞의 Python 블록을 다시 실행한다.
환경 hash가 다르면 실제 차이를 확인하고 같은 환경으로 복원한다. Manifest의 기록을
수정해 검사를 우회하지 않는다.

## 선택적 후속 단계: P 탐색·증류와 최종 평가

직접 P 평가를 확인한 뒤 선택한다. 모두 P-only이고 smoother 공동 학습이 아니다.
확대 설정은 search/distillation 예산을 바꾸지 않았으므로 탐색은 학습 A마다
4 starts × 200 updates, 증류 336 + 미세조정 168 updates다. 420개 A 탐색은 총
336,000 optimizer updates가 되어 직접 학습보다 훨씬 큰 별도 실험이다.

```bash
caffeinate -i python -u "$SCRIPT" search --run-dir "$P_RUN" \
  2>&1 | tee "$P_RUN/search.log" && \
caffeinate -i python -u "$SCRIPT" distill --run-dir "$P_RUN" \
  2>&1 | tee "$P_RUN/distill.log" && \
caffeinate -i python -u "$SCRIPT" diagnose --run-dir "$P_RUN" \
  2>&1 | tee "$P_RUN/diagnose_after_distill.log" && \
caffeinate -i python -u "$SCRIPT" benchmark --run-dir "$P_RUN" \
  --repeats 3 --warmups 1 --rhs-counts 1 4 16 --regimes cold warm multiple \
  2>&1 | tee "$P_RUN/benchmark.log"
```

연구 run의 validation 선택이 끝나면, 모델·소스·평가 조건을 freeze하고 final/OOD를
한 번 평가한다. Freeze 이후 해당 run은 개발 학습에 다시 사용하지 않는다. Smoke는
final 평가에 사용할 수 없다. 격자 OOD n=255도 이 단계에서만 사용한다.

```bash
python "$SCRIPT" freeze --run-dir "$P_RUN" \
  --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple && \
caffeinate -i python -u "$SCRIPT" final --run-dir "$P_RUN" \
  2>&1 | tee "$P_RUN/final.log"
```

중단된 정식 benchmark/final도 같은 조건에 `--resume`을 추가한다. 최종 평가까지
완료하기 전에는 연구 규모의 성공률·속도 개선이 입증됐다고 표시하지 않는다.
