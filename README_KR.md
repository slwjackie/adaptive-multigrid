# Adaptive Neural Multigrid v3

구조화 격자의 2차원 scalar SPD 이방성·불균질 elliptic diffusion 문제

\[
-\nabla\cdot(K(x,y)\nabla u)=f,\qquad u|_{\partial\Omega}=0
\]

를 대상으로 하는 졸업논문 연구 코드입니다. v3의 핵심은 다음 두 단계입니다.

\[
\boxed{\text{Global classical-strategy selection}}
\quad\longrightarrow\quad
\boxed{\text{Local selective neural correction}}
\]

즉, 문제 전체의 operator context를 보고 먼저 적합한 classical MG 전략을 선택하고, 그 classical solver가 충분하지 않은 공간 patch에서만 NN smoother 및 learned transfer correction을 계산합니다.

> 동봉된 checkpoint는 구조·학습·교차격자 검증용 짧은 자산입니다. 강한 classical baseline보다 NN이 우월하다는 논문 성능을 주장하기 위한 최종 학습본이 아닙니다. raw/safeguarded ablation과 충분한 다중 격자 학습이 반드시 필요합니다.

---

## 1. 최종 구조

```text
                         PDE / sparse A
                               │
                               ▼
                  Global classical selector
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
       Jacobi              line-x/y/alt       semicoarsening
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
              bilinear P / operator-dependent P
                               │
                               ▼
                       Classical MG hierarchy
                               │
                               ▼
                 Local residual-based detector
                      ┌────────┴────────┐
                      │                 │
                   easy patch        hard patch
                   classical       NN correction 실행
                                      │
                         ┌────────────┴────────────┐
                         │                         │
              additive NN smoother        learned P correction
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                           R=Pᵀ, A_H=PᵀAP
                                      │
                                      ▼
                    optional analytical safeguards
```

Global selector는 classical 방법을 고르고, local detector는 그 선택된 classical hierarchy 안에서 NN을 어디에 적용할지를 고릅니다.

---

## 2. Model PDE와 문제군

`adaptive_mg.pde.case_suite`는 다음 structured-grid 문제를 제공합니다.

- Poisson
- grid-aligned anisotropy
- 45° rotated anisotropy
- local anisotropic patch
- checkerboard coefficient jump
- channel coefficient field

모든 문제는 scalar SPD이므로 tied transfer와 CG 계열 비교의 수학적 조건을 분리해 검증할 수 있습니다.

---

## 3. 강한 classical baseline

### 3.1 Smoother

- `jacobi`: weighted Jacobi
- `line_x`: x방향 zebra block-line relaxation
- `line_y`: y방향 zebra block-line relaxation
- `line_alt`: x/y alternating line relaxation

Line block의 sparse LU factorization은 level별로 캐싱합니다. 대칭 V-cycle을 위해 post-smoothing은 역순 sweep을 사용합니다.

### 3.2 Coarsening

- `full`: x, y 동시 coarsening
- `semi_x`: x방향 우선 semicoarsening
- `semi_y`: y방향 우선 semicoarsening
- `alternating`: level마다 x/y를 교대로 coarsening

내부 grid 표현을 `(nx, ny)`로 확장하여 semicoarsening이 실제 직사각 hierarchy를 구성하도록 구현했습니다.

### 3.3 Transfer

- `bilinear`: geometric bilinear/linear interpolation
- `operator`: sparse approximate-ideal operator-dependent interpolation

Operator-dependent baseline은

\[
A_{FF}P_F\approx-A_{FC}
\]

를 sparse Jacobi sweep으로 근사합니다. 행별 Python CSR indexing과 dense conversion을 제거하고, vectorized sparse candidate lookup·relative dropping·top-k를 사용합니다. 이는 de Zeeuw 알고리즘의 문자 그대로의 복제는 아니며, structured-grid 논문 비교용 안정화 approximate-ideal baseline입니다.

### 3.4 통제된 strategy catalog

Global selector가 고르는 후보는 무제한 hyperparameter 조합이 아니라 다음 9개로 제한됩니다.

1. `jacobi_bilinear_full`
2. `line_x_bilinear_full`
3. `line_y_bilinear_full`
4. `line_alt_bilinear_full`
5. `jacobi_operator_full`
6. `line_alt_operator_full`
7. `line_x_operator_semi_y`
8. `line_y_operator_semi_x`
9. `line_alt_operator_alternating`

이 제한은 selector가 의미 없는 조합을 고르는 것을 막고, ablation을 해석 가능하게 유지합니다.

---

## 4. Learned transfer: 16-candidate tied encoder–decoder

기존 최대 4개 bilinear coarse node support를 **4×4, 최대 16개 후보**로 확장했습니다. NN은 expanded support에서 row-local interpolation correction을 예측합니다.

\[
P=P_\theta,\qquad R=P_\theta^T,\qquad A_H=P_\theta^TAP_\theta.
\]

- Decoder: learned prolongation `Pθ`
- Encoder: tied restriction `R=Pθᵀ`
- Coarse operator: Galerkin product
- classical row sum 보존
- geometric coarse node의 injection row는 classical 값으로 고정

마지막 항목은 OOD neural output이 coarse degree of freedom 자체를 훼손하여 `P` rank를 잃는 문제를 방지합니다. Operator-dependent branch에서는 16개 후보 안에서 approximate-ideal strength와 top-k를 동적으로 반영합니다.

---

## 5. Additive NN smoother

NN smoother는 classical smoother를 대체하지 않습니다.

\[
\delta u
=
\delta u_{\mathrm{classical}}
+
\Delta u_{\mathrm{NN}}.
\]

따라서 NN head가 zero-initialized되면 Jacobi, line-x/y/alt 중 무엇을 선택해도 학습 전 결과가 정확히 classical baseline과 일치합니다. v3에서는 correction scale을 지나치게 억제하지 않도록 기본 coefficient scale을 `0.75`로 조정하고, raw instability penalty와 safeguard를 별도로 둡니다.

---

## 6. 실제 patch-selective NN computation

이전 방식은 전체 grid에서 NN을 계산한 뒤 gate를 곱했기 때문에 계산량을 줄이지 못했습니다. v3은 다음 순서를 사용합니다.

```text
Detector 실행
  → threshold를 넘는 patch 식별
  → active patch + halo만 추출
  → 해당 patch에서만 smoother/transfer NN 실행
  → overlap 영역 조립
```

결과에는 다음이 기록됩니다.

- `active_patch_fraction`
- `active_cell_fraction`
- `smoother_nn_calls`
- `transfer_nn_calls`
- `detector_nn_calls`
- `nn_seconds`

따라서 “공간적으로 선택적”이라는 주장과 실제 NN 계산 절감 여부를 분리해 검증할 수 있습니다.

---

## 7. Detector feature 단일화

실행 solver, differentiable training, diagnostic plot이 모두 `detector_features.py`의 동일 정의를 사용합니다.

입력은 다음 13개 channel입니다.

1. log diagonal + normalized 9-point operator stencil: 10
2. normalized residual: 1
3. 선택된 classical smoother 전후 local energy ratio: 1
4. residual roughness: 1

Post-smoothing feature는 실제 post sweep과 동일한 `reverse=True` correction을 사용합니다. 세 구현이 조용히 달라지는 문제를 막는 일치 회귀 테스트도 포함했습니다.

---

## 8. Full-V-cycle, cross-grid differentiable training

v2의 핵심 결함은 two-grid에서 학습한 작은 오차가 full V-cycle의 여러 level에서 누적될 수 있다는 점이었습니다. v3은 production solver와 동일한 Galerkin hierarchy를 미분 가능하게 구성하고 **full V-cycle을 직접 학습**합니다.

기본 연구 설정은 다음과 같습니다.

- train roots: `n=15,31,63`
- in-distribution validation: `n=15,31,63`
- OOD validation: `n=127`
- 최소 세 grid level, 즉 transfer 2회 이상
- sample당 multi-cycle A-norm contraction

\[
\rho_A
=
\left(\frac{\|e_m\|_A}{\|e_0\|_A}\right)^{1/m},
\qquad
\|e\|_A=\sqrt{e^TAe}.
\]

Tiny two-grid `n=7`은 `--verification-profile`에서만 명시적으로 허용됩니다.

### Sparse training path

Root PDE matrix를 dense로 바꾸지 않습니다. Sparse Torch matvec, sparse Galerkin product, local line block extraction을 사용합니다. `to_dense()`는 terminal coarse direct solve처럼 크기가 작은 명시적 지점에만 남겨 두었습니다. Operator-dependent transfer도 GPU에서 SciPy로 왕복하지 않는 Torch sparse 경로를 가집니다.

---

## 9. Validation, rollback, 공통 평가

### 9.1 기하평균 통일

Stage 선택 기준과 출력되는 `val_rho` 모두 log contraction의 평균, 즉 geometric mean을 사용합니다.

\[
\rho_{\mathrm{val}}
=
\exp\left(\frac1N\sum_i\log\rho_i\right).
\]

따라서 로그에서 `val_rho`가 더 작은데 `best`가 갱신되지 않는 모순이 없습니다.

### 9.2 Stage별 의미 명시

각 `StageSummary`는 다음을 별도로 저장합니다.

- `evaluation_mode`
- `smooth_error_steps`
- `entry_objective`, `best_objective`
- `entry_validation`, `best_validation`
- `best_epoch`, `accepted`

Stage마다 목적이 다르므로 이 값들을 직접 성능 시계열로 해석하지 않습니다.

### 9.3 동일 protocol 공통 평가

학습 종료 후 5개 mode를 같은 grid, 같은 error sample, 같은 smoothing protocol로 다시 평가해 `common_validation`에 저장합니다.

- classical
- nn_smoother
- learned_transfer
- combined
- adaptive

Learned transfer stage가 full-V-cycle validation을 악화하면 자동 rollback됩니다.

---

## 10. Global classical-strategy selector

`GlobalSelectorNet`은 sparse operator에서 계산한 18개 global context feature를 사용해 9개 classical strategy 중 하나를 고릅니다.

- normalized 9-point stencil magnitude
- log diagonal의 mean/std/min/max
- x/y/diagonal directional coupling share
- grid scale

Selector label은 production classical solver를 실제 한 cycle 실행한 뒤 A-norm contraction과 work penalty를 결합하여 생성합니다. 전체 solver configuration을 무제한으로 예측하지 않고, 통제된 catalog에서만 선택합니다.

---

## 11. Analytical safeguards와 raw/safe 분리

### 11.1 Smoother exact line search

Classical correction `c`와 neural delta `d`에 대해

\[
s=r-Ac,\qquad w=Ad,
\]

\[
\alpha^*
=
\operatorname{clip}_{[0,1]}
\frac{s^Tw}{w^Tw}
\]

를 사용합니다. Residual 2-norm 기준의 닫힌형 최소화이며 grid search보다 싸고 정확합니다.

### 11.2 Pure-classical shadow trajectory

각 neural cycle을 “현재 neural 상태에서 classical 한 cycle”과 비교하는 것만으로는 여러 cycle 뒤 pure classical trajectory 하한을 보장하지 못합니다. v3은 독립적인 classical shadow trajectory를 유지하고, 매 cycle 같은 cycle count의 shadow보다 residual이 나쁘면 shadow 상태로 fallback합니다.

### 11.3 논문용 분리

`run_ablation.py`는 모든 neural mode를 다음 두 profile로 저장합니다.

- `raw`: `smoother_safeguard=False`, `safety_fallback=False`
- `safe`: exact alpha + pure-classical shadow fallback

추가 기록:

- `fallback_count`
- alpha 원자료 JSON
- `alpha_mean`, median, p10, p90
- `alpha_zero_fraction`, `alpha_active_fraction`
- `safety_work_units`

Raw NN 성능과 배포 안전성 주장을 섞지 않도록 강제합니다.

---

## 12. Krylov acceleration

- `cg`: 고정·대칭 SPD MG preconditioner용 PCG
- `fcg`: iteration별로 달라질 수 있는 variable preconditioner용 Flexible CG
- `fgmres`: nonlinear/variable/non-symmetric 효과를 더 일반적으로 허용하는 restarted Flexible GMRES

수정 사항:

- FCG restart 시 새 direction에 맞춰 `rho=r·p` 재계산
- full A-orthogonalization
- FGMRES estimated residual과 true residual을 분리
- true residual이 측정된 Krylov iteration index 저장
- restart 간격이 다른 residual reduction을 interval-normalized contraction으로 계산
- happy breakdown 및 least-squares fallback

CG는 residual-dependent fallback이나 비대칭 pre/post smoothing과 함께 사용하지 못하도록 사전 검증합니다.

---

## 13. 성능과 work accounting

기록 항목:

- `work_units`: finest-grid sparse matvec 환산
- `safety_work_units`
- `matvec_count`
- setup/solve/elapsed seconds
- NN별 호출 수와 `nn_seconds`
- hierarchy rebuild 수
- active patch/cell 비율

`adaptive_refresh=k`는 learned hierarchy를 매 cycle이 아니라 k cycle마다 갱신합니다. Detector용 classical probe는 직전 pre-smoothing correction을 재사용합니다.

Classical baseline 결과는 단순 “최저 residual”뿐 아니라 residual/work Pareto frontier도 저장합니다. 매우 비싼 semicoarsening 조합을 비용 설명 없이 단일 최강 baseline으로 제시하지 않기 위함입니다.

---

## 14. 설치

```bash
unzip adaptive_neural_multigrid_v3_final.zip
cd adaptive_neural_multigrid_v3
python -m pip install -e .
pytest -q
```

최종본은 **40개 회귀 테스트**를 포함합니다.

---

## 15. 주요 실행 명령

### 15.1 강한 classical baseline

```bash
python scripts/run_baseline_benchmark.py \
  --sizes 31,63 \
  --cycles 20 \
  --output-dir artifacts/baseline_v3
```

32개 smoother/transfer/coarsening 조합 전체를 시험하려면 `--exhaustive`를 추가합니다.

### 15.2 논문용 cross-grid staged training

```bash
python scripts/train_staged.py \
  --output artifacts/checkpoint_v3.pt \
  --train-sizes 15,31,63 \
  --validation-sizes 15,31,63 \
  --ood-validation-sizes 127 \
  --cycles-per-sample 2 \
  --strategy-source selector
```

### 15.3 빠른 구조 검증 학습

```bash
python scripts/train_staged.py \
  --verification-profile \
  --strategy-source fixed \
  --fixed-strategy jacobi_bilinear_full \
  --output artifacts/checkpoint_v3_verification.pt
```

이 profile의 checkpoint는 논문 성능본이 아닙니다.

### 15.4 Raw/safe 전체 ablation

```bash
python scripts/run_ablation.py \
  --checkpoint artifacts/checkpoint_v3.pt \
  --sizes 15,31,63 \
  --output-dir artifacts/ablation_v3
```

### 15.5 Profiling

```bash
python scripts/profile_v3.py \
  --output-dir artifacts/profile_v3
```

### 15.6 Random stress validation

```bash
python scripts/run_stress_validation.py \
  --cases 12 \
  --output-dir artifacts/stress_v3
```

---

## 16. 동봉된 검증 자산의 해석

동봉된 짧은 verification checkpoint에서:

- full-V-cycle common validation에서는 NN smoother/combined가 classical보다 낮은 평균 A-norm contraction을 보였습니다.
- learned transfer stage는 validation 개선에 실패해 rollback됐습니다.
- fixed hard-case cross-grid 평가에서 learned transfer는 `n=15→63`으로 갈수록 열화 비율이 폭증하지 않고 classical과 거의 동일하게 유지됐습니다.
- 같은 짧은 checkpoint의 raw NN smoother는 targeted PDE benchmark에서 classical보다 나빴습니다.
- detector gate가 threshold 0.5를 넘지 않아 targeted adaptive 실행은 classical branch만 사용했습니다.

따라서 v3은 **요청한 구조·학습·안전·프로파일링 결함을 수정한 최종 연구 코드**이지만, NN의 최종 우월성을 입증한 충분히 학습된 결과물은 아닙니다. 최종 논문 주장은 반드시 장기 multi-grid training과 raw/safe OOD ablation 결과에 근거해야 합니다.

상세 수치는 `artifacts/verification_v3/VERIFICATION_V3_KR.md`를 참조하십시오.
