# Adaptive Neural Multigrid v6.6
## Learned one-way temporal switching — standalone solver

v6.5의 공간별 selective NN을 제거하고, **한 solve의 처음 `K_NN`개 cycle만 NN을 사용한 뒤 classical MG로 영구 전환**하도록 바꾼 코드입니다. NN 사용 구간이 끝난 뒤에는 detector, predictor, NN 적용, hierarchy 재구축을 하지 않습니다. 단, 새 RHS를 풀 때는 별개의 solve이므로 초기 정책을 다시 결정합니다.

이 버전에서 실제로 학습하는 선택은 **공간별 gate가 아니라 정수 `K_NN` 하나**입니다. `K_NN=0`도 정당한 선택입니다. 시간 이득이 없는 NN을 강제로 사용하지 않습니다.

```text
A + 명시적으로 지정한 classical strategy
       ↓
classical P / R / Ac / coarse LU / line factorization 준비·캐시
       ↓
solve 시작: A 요약 + 초기 residual + tolerance → 작은 MLP 1회
       ↓
K_NN ∈ {0,1,...,Kmax}
       ↓
초기 K_NN cycle: 동일 classical hierarchy + NN smoother increment
       ↓                  ↘ 성장/비정상 값/정체이면 조기 전환
최대 한 번 NEURAL → CLASSICAL
       ↓
cached classical V-cycle만 반복 → FP64 true residual 정지 검사
```

외부 Krylov solver, 공간별 detector, gate map, patch 추출, residual-dependent Galerkin 재구축, global strategy selector, 과거 router는 실행 경로에 없습니다. **Classical strategy는 설정으로 명시합니다.** 기본값 `line_alt_bilinear_full`은 하나의 비교 기준이지 모든 PDE에서 가장 좋은 전략이라는 뜻은 아닙니다. Line-x/y/alternating/45°, Chebyshev, operator-dependent interpolation, semicoarsening 후보는 그대로 지원합니다.

## 1. 정확한 수치 구조

같은 classical hierarchy를 초기와 후기 모두 사용합니다. 각 smoothing step의 초기 neural 경로는 다음과 같습니다.

```math
r=b-Ax,\qquad x_C=x+B_Cr,
\qquad x_{new}=x_C+\gamma\,g_\theta(A)\widetilde C_\theta(A)D^{-1}(b-Ax_C).
```

여기서 `B_C`는 지정한 classical smoother이고, 마지막 항만 NN increment입니다. Classical tail은 마지막 NN 항을 실행하지 않습니다. Restriction, prolongation, coarse operator, coarse LU는 처음 구성한 것을 그대로 씁니다. **NN이 classical smoother 자체를 대체하지 않습니다.**

`nn_levels=2`가 기본값으로, 맨 위 두 nonterminal level에만 NN increment를 적용합니다. 이는 잔차와 무관한 정적인 level 설정이며 공간적 selective NN이 아닙니다. `nn_levels=-1`이면 모든 nonterminal level에 적용합니다. 각 level의 CNN 방향·gain은 해당 A와 모델 버전에서 처음 한 번만 계산하며, 이후 cycle에서는 캐시한 FP64 stencil을 적용합니다.

### 이전 learned transfer 처리

**v6.5의 별도 learned-transfer auxiliary coarse cycle은 새 production 경로에서 제외했습니다.** 공유 classical hierarchy 위에서 NN smoother increment만 추가하는 구조를 구현하기 위한 명시적인 변경입니다. 기존 `P_theta`를 잔차에 맞춰 혼합하거나 별도 coarse hierarchy를 매번 만드는 방식을 몰래 유지하지 않습니다. Migration은 기존 smoother의 방향·gain 가중치만 가져옵니다. Detector·selector·router·learned transfer·기존 인증은 가져오지 않습니다.

### `K_NN=0` 계약

처음부터 K=0이면 순수 classical과 **같은 `classical_cycle()` 함수**를 실행합니다. 해와 residual history가 동일함을 16개 strategy에서 검사합니다. NN coefficient 생성도 하지 않습니다. 다만 실제 MLP를 실행한 결과 K=0이 나온 경우에는 이미 소비한 일회성 정책 판단 비용이 남습니다. 이를 시간에서 빼지 않습니다.

## 2. 한 방향 상태 전환과 안전장치

정상적으로는 K회의 neural prefix가 끝나면 영구 전환합니다. 다음 상황은 더 일찍 전환합니다.

| 상황 | 처리 |
|---|---|
| NN candidate가 NaN/Inf를 만듦 | candidate 폐기 → 직전 정상 해에서 classical 1회 → 이후 classical만 사용 |
| NN candidate의 true residual이 증가 | 같은 rollback 및 영구 전환 |
| NN 감소율이 `stagnation_rho` 이상인 상태가 연속 발생 | 유한한 candidate는 유지하되 다음 cycle부터 classical |
| residual이 tolerance의 `near_tolerance_factor` 배 이내 | NN 준비/호출 없이 classical로 전환 |
| 예측 K 소진 | 다음 cycle부터 classical |
| NN 구간에서 이미 정지 허용오차 만족 | 즉시 종료; 불필요한 마무리 cycle을 강제로 실행하지 않음 |

기본 성장 판정은 `rho > 1.0*(1+1e-12)`, 정체 판정은 `rho >= 0.98`이 2회 연속입니다. 최소 한 번의 classical tail을 위한 예산을 남기되, 이미 수렴했다면 tail은 필요 없습니다. 정체·near-tolerance·성장 기준은 모두 설정 가능하고 training cost table에도 동일한 값을 사용합니다.

**수렴 보장은 조건부입니다.** 고정 classical error-propagation 연산자 `E_C`가 `rho(E_C)<1`이고 전환 시 iterate가 유한하면, 이후 error는 `E_C^j e_switch`로 수렴합니다. 이 코드가 임의의 A에 대해 classical MG의 수렴을 증명하지는 않습니다. Classical 자체가 정체하거나 발산하면 그대로 failure로 기록합니다. 같은 iteration 한도 안에 classical보다 항상 먼저 끝난다는 보장도 없습니다.

또한 선형 문제에서 residual의 절대 크기가 작다는 이유만으로 NN이 반드시 불리해지는 것은 아닙니다. Error mode, A, 초기 residual, tolerance, setup 비용에 따라 유리한 구간이 달라지므로 실제 완주 시간을 측정해 K를 학습합니다.

## 3. 학습 목표: 실제 T + 불안정성

각 training operator와 초기 RHS에서 모든 후보 K를 **끝까지 standalone으로** 실행합니다. 정지 조건, baseline, safety, thread 수는 모든 K에서 같습니다. Warm-up 후 후보 순서를 섞어 반복 측정합니다.

```math
J_i(K) = T_i(K)/T_i(0) + \lambda I_i(K).
```

이는 각 문제에서 `(T_i(K)+lambda*T_i(0)*I_i(K))/T_i(0)`와 같습니다. 문제 크기에 따른 시간 단위 차이를 정규화한 것이며, 시간 대리값 대신 실제 측정 시간을 사용합니다.

`I`에는 실패 100, classical 성공인데 candidate 실패한 경우 추가 25, 비정상 NN 시도당 2, rollback당 0.25, residual 성장의 양의 log 합이 들어갑니다. 기본 `lambda=2`입니다. 실패는 유한한 time-to-tolerance를 가진 것으로 취급하지 않습니다. 파일에는 `time_to_tolerance_seconds=null`과 관측한 중단 시점까지의 시간이 분리됩니다. 모든 후보가 실패한 문제는 K=0 abstention을 학습하며, 빨리 실패하는 NN을 보상하지 않습니다.

MLP는 `softmax(logits)`를 이용해 다음 경험적 목적함수를 학습합니다.

```math
L = mean_i sum_K p_phi(K|features_i) J_i(K).
```

즉 **실측 T와 불안정성으로 만든 후보별 cost table을 통해 학습**합니다. Clock이나 정수 iteration 수를 autograd로 직접 미분했다고 주장하지 않습니다. 모든 후보의 비용을 보여주므로 닫힌 spatial gate에 의해 gradient가 사라지는 문제도 없습니다. 실제 실행에서는 soft mixture가 아니라 정수 K 한 개를 선택합니다.

MLP 입력은 A의 18개 전역 요약, classical strategy, 초기 residual의 방향별 roughness·상관도·분포 요약, tolerance까지의 거리, NN bank cache 상태입니다. A 요약은 prepared solver에 캐시하고, RHS 정보는 solve 시작 시 한 번만 계산합니다. 모델은 작은 tanh MLP이며 CPU 추론은 동일 가중치의 NumPy 연산으로 실행해 PyTorch dispatch를 피합니다. NumPy/PyTorch 출력을 별도 검증합니다.

### 학습 단계와 검증 분리

1. 기존 v6.5 smoother 가중치를 명시적으로 가져옵니다. 동봉 검증에서는 smoother를 다시 바꾸지 않았습니다.
2. 선택 사항으로 neural-prefix + classical-tail residual rollout을 이용한 smoother 사전학습을 지원합니다. 이것은 **초기화용 surrogate**이고 실제 시간 loss와 구분합니다. 활성화하면 모든 K timing label을 그 새 모델로 다시 만듭니다.
3. Train cost table로 K predictor를 학습하고, 별도 tune table로 epoch를 선택합니다.
4. 추가 audit operator에서 실제 predictor 비용까지 포함한 complete solve를 재측정합니다. Audit 결과로 가중치나 K를 다시 맞추지 않습니다.
5. 정확도·시간 검증을 통과해야 배포 정책으로 인증합니다.

Split은 seed만 다르게 하지 않습니다. A를 공통 scale로 정규화한 fingerprint까지 비교해, 단순히 A와 b에 같은 상수를 곱한 문제가 서로 다른 split에 들어가는 것을 방지합니다. Isotropic training family도 약간 다른 실제 tensor를 사용합니다. Benchmark의 정형 Poisson은 별도입니다.

## 4. 실행 모드

| mode | 의미 |
|---|---|
| `classical` | 순수 classical standalone MG. Predictor/NN 없음 |
| `fixed` | 사용자가 지정한 K. 고정-prefix ablation. 숫자 안전장치는 그대로 유지 |
| `predicted` | 학습된 K를 실제 사용. 경험적 성능 인증을 우회하는 연구용 진단 모드이며 numerical safeguards는 유지 |
| `temporal` | 기본 배포 모드. 독립 audit를 통과한 policy만 사용; 인증 실패/환경 변경/설정 범위 밖이면 K=0 |

**동봉 smoke 모델은 predictor 학습을 완료했지만 정책 시간 인증은 통과하지 못했습니다.** 따라서 `temporal`은 K=0으로 돌아갑니다. `predicted`로 학습된 one-way 행동을 검사할 수 있습니다. 이를 성능이 인증된 경로와 혼동하지 마십시오.

인증은 현재 smoother/predictor 가중치, CPU·라이브러리·thread 환경, tolerance, cycle 예산, classical strategy, kernel 설정, 검증한 격자 범위와 cache regime에 묶입니다. CPU나 solver 설정이 바뀌면 새 데이터로 재검증해야 합니다. 기존 `fast_path_quality_certified`는 새로운 temporal 정책 판단에 사용하지 않으며, 핵심은 `metadata.temporal_certificate.validated`입니다.

모든 감사 case의 성공, 새로운 수렴 실패 없음, 실제 NN 사용, common-success geometric speedup 기준, paired case bootstrap 하한 >1을 요구합니다. 이는 유한 held-out 표본에서의 경험적 인증이지 모든 PDE에 대한 증명은 아닙니다. 실패를 제외한 speedup만으로 전체 우위라고 주장하지 않습니다.

## 5. 설치·실행

실제 검증 환경은 Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.10.0+cpu입니다. 런타임은 SciPy/FP64 CPU이며, A100 또는 MPS 실행 성능은 검증하지 않았습니다.

```bash
python -m pip install -e ".[dev]"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/build_native_stencil.py
pytest -ra
```

Native compiler가 없으면 `stencil_backend="csr"` 또는 자동 CSR fallback을 사용할 수 있습니다. Solve 중에는 컴파일하지 않습니다. OpenMP가 없으면 명시적 build가 serial C++로 재시도합니다. GPU에 CNN만 올리고 SciPy와 매 cycle 왕복하는 경로는 제공하지 않습니다.

### 학습된 전환 trace 확인

```bash
python scripts/demo_temporal.py \
  --checkpoint artifacts/verification_v6_6/training/checkpoint_v6_6_final.pt \
  --n 63 --case poisson --mode predicted \
  --output artifacts/my_temporal_trace.json
```

`predicted`는 미인증 정책의 동작을 확인하기 위한 명시적 선택입니다. 기본 `--mode temporal`은 현재 동봉 모델에서 K=0으로 돌아갑니다.

### 동일 조건 비교

```bash
python scripts/benchmark_temporal.py \
  --checkpoint artifacts/verification_v6_6/training/checkpoint_v6_6_final.pt \
  --sizes 15,31,63 \
  --modes classical,temporal,predicted,fixed2 \
  --repeats 3 \
  --output-dir artifacts/my_temporal_benchmark
```

큰 격자의 full time-to-tolerance 실험은 `--sizes 15,31,63,127,255`로 실행합니다. 동봉 검증의 큰 격자 테스트는 별도 report에 적힌 6-cycle 계약 검증이며, 255까지의 전체 시간 우위를 주장하지 않습니다.

### 학습부터 다시 실행

```bash
python scripts/train_temporal.py \
  --config configs/v6_6_verification.json \
  --initial-checkpoint checkpoints/initial_temporal.pt \
  --output-dir artifacts/my_temporal_training
```

연구 설정은 `configs/v6_6_research.json`입니다. Training 15/31/63, tune 31/63/127, audit 63/127/255, 후보 K=0..12, cold/warm regime과 더 많은 반복을 포함합니다. **이 긴 설정은 동봉 smoke 검증에서 실행하지 않았습니다.** 해당 장기 설정은 optional smoother surrogate 사전학습도 켭니다.

```bash
python scripts/run_v6_6_pipeline.py \
  --config configs/v6_6_research.json \
  --initial-checkpoint checkpoints/initial_temporal.pt \
  --sizes 15,31,63,127,255 \
  --output-dir artifacts/my_v6_6_research
```

연구용 긴 설정에서 case 수 × K 후보 수 × 반복 수만큼 full solves가 필요합니다. 이 offline label 생성비용은 기록하지만, inference speedup에 포함시키거나 숨겨서 섞지 않습니다.

## 6. API와 동일 A 재사용

```python
import numpy as np
from adaptive_mg import (
    MGConfig, TemporalComponents, PreparedTemporalMGSolver,
    DiffusionCase, assemble_stiffness,
)

model = TemporalComponents.load(
    "artifacts/verification_v6_6/training/checkpoint_v6_6_final.pt"
)
a = assemble_stiffness(DiffusionCase(n=63))
config = MGConfig(mode="temporal")
prepared = PreparedTemporalMGSolver(a, 63, model, config)
bs = np.random.default_rng(7).normal(size=(6, a.shape[0]))
results = prepared.solve_many(bs)
```

Hierarchy는 한 번 생성합니다. NN이 필요했던 경우에만 coefficient bank를 생성하며 다음 RHS에도 재사용합니다. 각 RHS의 residual, prefix counter, state와 switching event는 초기화합니다. Batch 진입 시 freshness 검사는 한 번입니다. A가 조금이라도 바뀌면 원래 hierarchy를 그대로 재사용하지 않습니다. NN 가중치만 바뀌면 classical hierarchy는 유지하고 NN bank만 무효화합니다. 한 인스턴스를 여러 thread에서 동시에 호출하지 마십시오.

## 7. 계측 해석

- `mg_cycles`: 수용된 update slot 수. 이 중 `classical_recovery` slot에는 폐기한 NN 시도와 classical cycle이 모두 존재할 수 있습니다.
- `executed_mg_cycles`: 실제 top-level V-cycle 호출 수 = attempted neural cycles + classical cycles. 폐기한 NN 시도도 포함합니다.
- `cycle_seconds`/`wall_seconds`: 폐기 시도, rollback 후 classical 연산, residual 검사, lazy NN setup을 포함합니다.
- `setup_seconds`: classical hierarchy·factorization·정책 준비 등 생성 비용. `cold` 시간은 constructor+solve 전체 외부 wall clock으로 측정합니다.
- `predictor_calls`: solve당 최대 1. 이미 수렴했거나 고정 K, 인증 거부, NN을 쓸 level이 없는 경우 0.
- `detector_calls`, `spatial_gate_calls`, `residual_hierarchy_rebuilds`, `switch_hierarchy_builds`: 새 구조에서 0임을 검사하는 계약 counter입니다.
- `work_flops`: 수치 적용 단계의 근사 proxy이며 CNN forward/factorization의 정확한 FLOP 총계가 아닙니다. 최종 목적함수는 이 proxy가 아니라 측정 시간입니다.

## 8. 파일 구성

`solver.py`는 state machine, `hierarchy.py`는 공유 classical/NN cycle, `policy.py`는 일회성 feature·K 판단, `measurement.py`는 full-solve cost table, `train_switch.py`는 실측 risk 학습/인증입니다. `learning.py`는 선택적 smoother 초기화용이며 production 판단과 분리됩니다.

검증 결과와 한계는 `artifacts/verification_v6_6/VALIDATION_REPORT_KR.md`를 확인하십시오. 소스·패키지 해시는 각각 `SOURCE_MANIFEST.sha256`, `PACKAGE_MANIFEST.sha256`에 있습니다.
