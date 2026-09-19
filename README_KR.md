# Adaptive Neural Multigrid v6.7

**기준 코드:** `adaptive_neural_multigrid_v6_6_final.zip`  
**새 production API:** `adaptive_mg.v67.PreparedAdaptiveMG`  
**비교:** standalone Classical MG vs standalone Adaptive Neural MG. 외부 Krylov solver를 사용하지 않습니다.

## Strong Classical 기준의 새 연구 경로

기존 deterministic selector를 동결한 상태에서 새 CNN/D4/O(2) smoother,
graph-aware P, training-only teacher/KD를 비교하는 경로가 추가되었습니다.
[새 실행 안내](STRONG_AWARE_README_KR.md),
[설치 후 검증 결과](docs/STRONG_AWARE_COMPLETION_KR.md),
[아키텍처·수학적 제약](docs/STRONG_AWARE_ARCHITECTURE_KR.md),
[변경 기록](CHANGELOG_STRONG_AWARE_KR.md)을 참고하세요.
기존 audit와 이미 확인한 14개 문제는 새 final test로 재사용하지 않습니다.
아래 기존 구현·검증 설명과 결과는 해당 시점의 기록으로 보존합니다.

## 현재 전달물의 상태

이 ZIP은 **실행 코드, 학습 코드, 재개 기능, 결과 산출 스크립트가 포함된 구현본**입니다. `configs/v6_7_research.json`의 224개 operator 학습 및 35개 독립 audit 전체를 이 환경에서 끝낸 논문 성능 검증본은 아닙니다. 최종 성능을 나타내는 값은 직접 실행한 결과로만 생성됩니다. 작은 smoke run은 파이프라인 실행 확인용이며, 큰 데이터셋의 속도 우위를 뜻하지 않습니다.

초기/미인증 모델은 기본 `production` 모드에서 classical만 사용합니다. `research`는 미인증 controller 동작을 조사하고, `hybrid`는 강제로 replacement branch를 실행하는 ablation입니다. 미인증 모델을 classical로 실행한 결과를 NN 가속으로 계산하지 않습니다.

실제 검사 결과: **133개 테스트 통과, 6단계 smoke 완료, n=127 양쪽 generator gradient 확인, n=255 포함 runtime 계약 10/10 통과**. 상세 범위는 `artifacts/verification_v6_7/VALIDATION_REPORT_KR.md`를 참고하세요.

## 1. 바로 실행

```bash
unzip adaptive_neural_multigrid_v6_7_final.zip
cd adaptive_neural_multigrid_v6_7
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/build_native_stencil.py
pytest -ra
```

Native 빌드가 어려운 환경에서는 `stencil_backend="csr"`로 실행할 수 있습니다. `auto`는 빌드된 native backend가 있고 격자가 충분히 클 때 사용하며, solve 도중 컴파일하지 않습니다. OpenMP를 사용할 수 없으면 빌드 스크립트가 serial C++로 재시도하고 실제 OpenMP 사용 여부를 출력합니다.

### 먼저 전체 파이프라인 실행 여부 확인

```bash
python scripts/run_v6_7_pipeline.py \
  --config configs/v6_7_smoke.json \
  --output-dir artifacts/my_smoke
```

이 설정은 소수 operator/짧은 epoch로 **6단계 학습, 후보 채택/거부, audit, JSON/CSV 생성**을 확인합니다. 성능 연구용 데이터가 아닙니다.

### v6.6 가중치에서 연구용 전체 학습

```bash
python scripts/run_v6_7_pipeline.py \
  --config configs/v6_7_research.json \
  --initial-checkpoint checkpoints/from_v6_6_operators.pt \
  --output-dir artifacts/my_v6_7_research
```

기본 연구 설정:

| 구분 | 설정 |
|---|---|
| Train | n=15,31,63,127, 크기당 56개 → **224 operators** |
| Tune | n=15,31,63,127, 크기당 7개 → 28 operators |
| Audit | n=15,31,63,127,255, 크기당 7개 → 35 operators |
| NN 숫자 형식 | FP32 |
| 수치 해법 | CPU FP64 |
| Unroll | learned prefix 4 cycle + classical tail 2 cycle |
| 기본 classical | line-alt + bilinear P + full coarsening, pre/post 각각 2회 |
| 후보의 시간 이득 문턱 | stage 기본 3%, deployment 기본 5% time reduction |

이는 **설정된 작업량**입니다. 전체 학습시간이나 성공률을 측정한 값은 아닙니다. 지원되는 classical 전략은 `adaptive_mg.strategy`에 있으며, 한 고정 line-alt 조합을 모든 PDE에서 최강이라고 주장하지 않습니다. 다른 baseline을 고르면 양쪽 solver에 동일하게 적용하고 controller를 해당 설정으로 재학습/검증해야 합니다.

### 중단 후 재개

위 명령에 `--resume`을 붙입니다.

```bash
python scripts/run_v6_7_pipeline.py \
  --config configs/v6_7_research.json \
  --initial-checkpoint checkpoints/from_v6_6_operators.pt \
  --output-dir artifacts/my_v6_7_research \
  --resume
```

가중치·optimizer·epoch·다음 operator 위치는 `resume.pt`, 단계별 채택된 모델은 `current.pt`, 완료 단계는 `pipeline_progress.json`에 저장합니다. 임시 파일을 쓴 뒤 rename하므로 저장 도중 기존 완료 checkpoint를 먼저 삭제하지 않습니다. 완료된 단계는 다시 학습하지 않습니다. 재개 시 config가 바뀌면 오류로 중단합니다. 완료 run에 `--resume`을 주면 저장된 최종 결과를 읽으며 audit 데이터를 추가 튜닝에 사용하지 않습니다.

## 2. 아키텍처

```text
A ──→ Classical hierarchy eager cache
                  │
        Cheap temporal controller
                  │
      ┌───────────┼────────────────┐
 CLASSICAL       HYBRID      CLASSICAL_LOCK
      │            │                 │
 no detector   lazy learned bank     │
 no NN            │                 │
      │       learned P/R/Ac         │
      │       replaces coarse path  │
      │            │                 │
      │       occasional block      │
      │       spatial detector      │
      │            │                 │
      │       cached NN stencil     │
      │       replaces sweeps       │
      └───────────┴─────────────────┘
                 FP64 true residual
```

Heavy networks는 **A-dependent operator generator**입니다. 선택한 각 level에서 coefficient/P를 생성한 뒤 cache합니다. 다층 hierarchy에서 “A당 한 번”은 **bank 생성 한 번/선택 level당 각 generator 한 번**을 뜻하며, 모든 level을 한 번의 CNN forward로 계산한다는 의미는 아닙니다. 한 번 준비한 bank를 cycle 및 동일 A의 여러 RHS에서 재사용합니다.

### Replacement

Classical이 pre 2 / coarse correction 1 / post 2라면 기본 Hybrid는 pre 중 1회를 NN stencil로 교체하고, coarse path 하나를 learned P로 교체하고, post 2회는 classical을 유지합니다. classical cycle과 learned coarse cycle을 중복 실행하지 않습니다. `application="additive"`는 비교용으로만 남겨두고 production에서는 거부합니다.

### Learned hierarchy

`P_theta`, `R_theta=P_theta.T`, `A_c=P_theta.T @ A @ P_theta`를 별도 bank에 저장합니다. 상위 P가 바뀌면 **실제 새 coarse A에 맞춰** 하위 classical 연산자·line factors를 다시 준비합니다. 이전 hierarchy의 다른 coarse A에 대한 factor를 재사용하지 않습니다. 이 준비는 lazy bank를 만들 때 한 번 수행하고 residual 때문에 반복하지 않습니다.

### 스케일과 precision

학습된 방향은 정규화하며 level scalar gain과 분리합니다. 기존 v6.6의 입력 convention인 `C(D^-1 r)`를 유지하되, runtime에서는 source-column diagonal inverse와 gain을 stencil coefficient에 접어 넣어 raw residual에 적용합니다. `C*r`라는 표기를 사용할 때 이 scaling을 누락하지 않습니다.

NN 생성은 기본 FP32이며 출력된 delta/coefficient는 CPU로 옮긴 후 FP64로 변환합니다. P는 baseline FP64 + NN이 만든 FP32 delta로 조립하여 zero-delta 초기화가 classical interpolation과 일치하도록 하고, injection 및 row sum도 다시 맞춥니다. 해, residual, sparse operator application, LU solve, 정지 판정은 FP64입니다.

**FP64 residual 검사는 동일 discrete equation의 허용오차를 확인하는 것이지, 임의 문제의 수렴 또는 PDE의 이산화 오차를 자동 보증하지는 않습니다.**

## 3. Temporal / spatial 역할

### Temporal

`CLASSICAL ↔ HYBRID → CLASSICAL_LOCK`을 지원합니다. 초기에는 실제 해를 개선하는 classical cycle에서 속도/수렴 관측을 얻습니다. 단순 비교를 위한 shadow classical cycle을 매번 추가하지 않습니다. Hybrid의 아직 관측하지 못한 성능은 offline paired rollout으로 학습한 예측을 사용하므로 **추정치이지 현재 상태의 실측 counterfactual이 아닙니다**.

Controller는 남은 log residual 감소량, classical throughput, 예상 hybrid throughput, 아직 지불하지 않은 NN setup 비용을 이용합니다.

```text
classical로 남은 시간
  > lazy setup + hybrid burst 시간 + classical tail 시간
```

에 충분한 margin이 있을 때만 Hybrid에 들어갑니다. 현재 solve에서 이미 setup을 지불했으면 setup remaining은 0입니다. 최소 dwell, hysteresis, entry budget, 반복 underperformance lock을 둡니다. 시간이 변동되거나 error spectrum이 바뀌므로 예측은 항상 정확하다는 보장을 하지 않습니다.

### Spatial

최대 16×16 block feature에서 작은 MLP를 실행하며, score 계산은 NumPy로 compile한 FP32 micro-model입니다. Heavy PyTorch NN을 cycle 안에서 호출하지 않습니다. 기본 refresh 간격 4 cycle, 재진입 시 갱신, hysteresis를 제공합니다.

`detector_features="current"`는 refresh 시 현재 residual을 읽습니다. `restriction_lagged`는 **restriction과 block statistics를 같은 native fine-row pass에서 계산**하고 다음 refresh에서 재사용합니다. 이 선택은 이전 restriction 시점의 residual을 사용한다는 명시적 trade-off가 있으며, 현재 residual과 동일하다고 주장하지 않습니다.

낮은 활성 비율에서는 selected-row kernel, 높은 비율에서는 full-grid kernel 후 동일 semantic mask를 적용합니다. Backend를 바꾼다는 이유로 비활성 영역을 몰래 NN으로 변경하지 않습니다. `active_skip`은 알고리즘을 바꾸므로 단순 microbenchmark만으로 조정하지 않고 solver 검증이 필요합니다.

### Line smoother의 중요한 제약

Line relaxation은 격자점별 독립 연산이 아닙니다. 부분 line을 잘라 계산한 뒤 원래 line solve와 같다고 할 수 없습니다. 이 구현은 **전체 line이 선택된 경우만 line solve를 NN으로 교체**하고 나머지 line은 원래 factor solve와 zebra 순서로 처리합니다. 작은 사각 patch는 replaceable line을 거의 만들지 못할 수 있으며, 이 경우 공간 선택의 시간 이득은 제한됩니다. Chebyshev의 부분 mask도 full polynomial + masked NN을 replacement라고 위장하지 않고 classical로 처리합니다.

## 4. 정확히 보장하는 계약과 보장하지 않는 것

* `mode="classical"`, 또는 시작부터 모든 learned 기능을 비활성화한 경우 기존 classical V-cycle 함수와 동일합니다.
* Global gate-off는 **smoother뿐 아니라 hierarchy도 classical로 선택**합니다. Learned P를 쓰면서 smoother gate만 0이면 전체 cycle은 classical과 같지 않습니다. P-only ablation은 의도적으로 이 구분을 노출합니다.
* 모든 gate가 닫히면 해당 hybrid 시도는 전체 classical bank로 우회합니다. detector/setup을 먼저 실행했다면 이미 쓴 시간은 그대로 기록합니다.
* 위험한 hybrid candidate는 x를 덮어쓰기 전에 검증합니다. NaN/Inf 또는 허용 범위 밖 residual 증가면 폐기하고 직전 accepted x에서 classical recovery 후 lock합니다.
* **폐기한 시도도 max cycle budget에 포함**합니다. 남은 예산이 없으면 숨겨진 추가 classical cycle을 실행하지 않습니다.
* Lock 이후 같은 시작 상태에 대한 subsequent update는 classical 연산입니다. NN을 사용하기 전의 최초 x0에서 출발한 pure-classical trajectory와 같아진다는 뜻은 아닙니다.
* 임의 SPD 문제에서 “동일 budget에서 절대로 classical보다 뒤처지지 않는다”는 수학적 보장을 새로 증명하지 않았습니다. Shadow trajectory 없이 그 보장을 암묵적으로 주장하지 않습니다. 경험적 non-regression과 동일 true-residual tolerance를 독립 audit으로 검사합니다.

## 5. 학습 6단계

1. Temporal smoother pretraining: 실제 replacement prefix + classical tail.
2. Temporal learned-P pretraining: Galerkin 및 coarse solve를 통한 gradient.
3. Smoother+P 공동 학습: 여러 cycle을 같은 learned bank로 unroll.
4. Spatial detector: 같은 residual에서 실제 classical/NN smoothing branch를 비교한 block utility. 실제 양성만 oversample하고 balanced BCE/ranking으로 학습.
5. Temporal/break-even controller: 동일 상태에서 C와 H-burst+classical-tail을 실제 실행. `J=T/Tc + lambda_I*instability + lambda_eta*max(0,1-etaH/etaC)`와 throughput regression으로 학습.
6. Hierarchical fine-tuning: frozen controller의 C/H 실행 일정을 replay하고, 별도로 표시된 offline exploration으로 H 상태를 학습. product-level STE를 사용하며 최종 채택은 실제 research controller 실행시간으로 판단. 가중치가 채택되면 spatial/controller label을 다시 생성하고 갱신.

고정된 kernel/sparsity/schedule의 compute proxy는 가중치에 대해 **상수**일 수 있습니다. 이를 미분 가능한 wall-clock이라고 부르지 않습니다. Gate-dependent expected work에는 gradient를 줄 수 있고, discrete mode/budget 선택은 실측 cost supervision으로 학습합니다.

Sparse autograd는 SciPy sparse product/LU의 forward와 **fixed sparsity value에 대한 adjoint derivative**를 구현합니다. 전체 NxN dense gradient를 만들지 않습니다. 확장된 P의 초기 zero entries는 symbolic pattern에 남겨 gradient를 보존합니다. Classical approximate-ideal interpolation의 support/drop 선택은 각 forward에서 재계산하지만 그 선택 자체의 미분은 고정한 surrogate를 사용합니다. 직접적인 runtime forward parity와 작은 gradient check가 테스트에 포함되어 있습니다.

## 6. Benchmark와 ablation

```bash
python scripts/benchmark_v6_7.py \
  --checkpoint artifacts/my_v6_7_research/checkpoint_v6_7_final.pt \
  --sizes 15,31,63,127,255 \
  --strategy line_alt_bilinear_full \
  --repeats 5 \
  --output-dir artifacts/my_benchmark

python scripts/benchmark_v6_7.py \
  --checkpoint artifacts/my_v6_7_research/checkpoint_v6_7_final.pt \
  --sizes 15,31,63,127,255 \
  --all-ablations \
  --output-dir artifacts/my_ablations
```

A–J와 additive/replacement, pre/post 비대칭, FP32/FP64 generator, fine/first-two/all levels, detector refresh 비교를 생성합니다. `--regime warm`은 양쪽 solver를 명시적으로 준비한 뒤 warm solve만 측정합니다. 같은 A/RHS batch의 cache 재사용은 별도 계약 스크립트로 검사합니다.

```bash
python scripts/verify_v6_7.py \
  --checkpoint artifacts/my_v6_7_research/checkpoint_v6_7_final.pt \
  --sizes 15,31,63,127,255 \
  --output-dir artifacts/my_contracts
```

Cold timing은 **constructor + solve**이며 classical hierarchy, 필요한 lazy NN bank, 생성/전송/검사/실패 trial 비용을 포함합니다. Python interpreter startup, checkpoint 파일 읽기, offline 학습 및 결과 파일 쓰기는 제외합니다. Warm timing은 constructor와 이미 준비된 bank를 제외한다고 명시합니다. 모든 run의 raw JSONL/JSON과 CSV를 보존합니다.

실패한 solve의 짧은 runtime을 speedup으로 취급하지 않습니다. 공통 성공 문제의 speedup은 성공률·새 실패 수와 함께 보고합니다. 최종 인증은 최소 audit 수, 독립 operator, NN의 실제 사용, 새 실패 없음, 충분한 speedup 신뢰하한, model/hardware/config scope를 모두 확인합니다.

### M2 Pro / A100

`inference_device`를 `mps` 또는 `cuda`로 바꾸면 NN operator generation을 해당 장치에서 FP32로 수행할 수 있습니다. CPU FP64 numerical solver는 그대로입니다. MPS에 FP64 tensor를 보내지 않도록 CPU 전송 후 승격합니다. 장치 synchronize와 CPU/device 전송 시간을 따로 계측합니다.

**이 환경의 실제 측정은 Linux CPU입니다. M2 Pro와 A100 수치는 제공하지 않습니다.** CUDA generation 지원이 V-cycle 전체 CUDA 포팅을 뜻하지 않습니다. FP64 NN ablation은 CPU/CUDA 대상으로만 실행하며 MPS에서는 제외합니다.

## 7. 파일 안내

* `src/adaptive_mg/v67/`: 새 production/학습/benchmark 구현.
* `src/adaptive_mg/{grid,pde,transfer,smoothers,hierarchy,...}.py`: v6.6 수치 구성요소. Native stencil에 selected-row/fused-statistics API 추가.
* `src/adaptive_mg/{solver,learning,train_switch,...}.py`: 기존 exact-K 경로. Legacy 호환과 이전 회귀 테스트용이며 새 CLI에서 호출하지 않음.
* `checkpoints/from_v6_6_operators.pt`: 실제 v6.6 최종 smoother만 명시적으로 migration. Learned P/micro-controllers 새 초기화, 기존 인증 폐기.
* `checkpoints/initial_v6_7.pt`: 작은 새 초기화 모델. 학습 완료 모델이 아님.
* `artifacts/verification_v6_7/`: 이 전달물에서 실제 실행한 검사 결과. 전체 연구용 결과와 구분.
* `docs/ARCHITECTURE_KR.md`, `docs/REQUIREMENTS_COVERAGE_KR.md`: 경로·계약·남은 성능 검증 범위.

기존 소스의 API 이름 `PreparedTemporalMGSolver`는 one-way v6.6 비교용입니다. **새 코드에서는 `PreparedAdaptiveMG`를 사용하세요.**

```python
from adaptive_mg.v67 import AdaptiveConfig, Components, PreparedAdaptiveMG
from adaptive_mg import MGConfig

model = Components.load("checkpoints/from_v6_6_operators.pt")
config = AdaptiveConfig(
    mg=MGConfig(mode="classical", strategy_name="line_alt_bilinear_full",
                pre_steps=2, post_steps=2),
    mode="research", inference_device="cpu",
)
prepared = PreparedAdaptiveMG(A, n, model, config)
result = prepared.solve(b)
print(result.to_dict())
# Same operator, several RHS: shape (batch, n*n)
results = prepared.solve_many(rhs_batch)
```
