# H_P headroom-first 연구 경로

이 경로는 `947e4f5` 이후의 main에 **추가**하는 development 전용 진단이다.
기존 H0/H1/H2/H2_NH 학습, continuous-size policy, classical fast path와 P2를 변경하지 않는다.
목적은 먼저 P 개선 여지와 고전 보간법을 측정한 뒤 NN/LWLS/공동학습 투자 여부를 판단하는 것이다.
새 learned-LWLS, non-Galerkin Ac, S/P 교대학습을 이미 구현/검증했다는 뜻은 아니다.

## 포함한 것 / 포함하지 않은 것

| 포함 | 범위 |
|---|---|
| P2 계약 검사 | source RUN이 baseline-support, parent-relative cap이어야 시작 |
| Constrained energy interpolation | 지정 support, injection, 정확한 PBc=Bf 제약 아래 trace(P^TAP) 최소화 |
| Plain LS / energy-weighted LS | 고전 test-vector 생성 비용 포함, 한 번의 constrained LS; full BAMG 재현 아님 |
| Direct P 진단 | 기존 projection의 logits를 operator별 직접 최적화; 실제 multilevel V-cycle loss |
| Dense coarse-space headroom | 실제 pre/post error maps, A-whitened SVD, adjoint 검증, 메모리 제한 |
| 기존 NN P 대조 | 명시적으로 제공된 동일 rules의 support-preserving checkpoint만 사용 |
| Warm / setup-included timing | 실제 서로 다른 RHS, 독립 prime, 동일 초기값, 동일 C* 복구 |
| 2x2 factorial | frozen S/P와 같은 selector arm에서 C,H_P,H_S,H_SP 비교 |
| Gate report | 진단상 권고만; 자동 학습/승격/연구중단 없음 |

미포함: learned-LWLS 신경망 학습, P support 확대, coarse sparsification, 새 S/P alternating trainer,
외부 Krylov wrapper, 실제 final/OOD 평가. 이들은 headroom과 독립 development evidence가 확인된 뒤의 단계다.

## 1. 기본 실행

기존 수정 사항을 먼저 보관한 뒤 저장소 루트에서 실행한다.

```bash
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
source .venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python -m pytest tests/test_p_headroom_math.py tests/test_p_headroom_integration.py -ra
```

원래 RUN은 읽기만 한다. 기존 timing을 새 소스로 resume하지 않는다. 별도 PRUN을 만든다.
새 경로는 원래 RUN의 configuration, frozen rules, development manifest를 복사해 참조한다.
A, 원래 RHS, exact solution 및 selector decision의 digest가 동일하게 재구성되지 않으면 거부한다.

```bash
SOURCE_RUN="artifacts/실제_three_pillars_또는_warm_study_RUN"
PRUN="artifacts/p_headroom_$(date +%Y%m%d_%H%M%S)"

python scripts/run_v6_7_p_headroom.py plan \
  --source-run "$SOURCE_RUN" --out "$PRUN" \
  --split train --limit 7 --sizes 15 \
  --repeats 3 --rhs-count 4 --direct-steps 20

python scripts/run_v6_7_p_headroom.py run --run-dir "$PRUN"
python scripts/run_v6_7_p_headroom.py report --run-dir "$PRUN"
```

처음에는 작은 표본으로 통합 실행을 확인한다. 7개 사례의 결과를 전체 PDE 일반화 주장으로 쓰지 않는다.
기존 source가 smoke(n7/15)라면 `--sizes 7 --limit 2 --direct-steps 2 --repeats 1`로 줄일 수 있다.
자료를 새로 생성하지 않으므로 source에 없는 크기는 요청할 수 없다.
`--split validation`도 허용하지만, operator별 직접최적화를 수행한 자료는 더 이상 untouched validation이 아니다.
TRAIN을 기본값으로 권장한다. Final/OOD 입력은 허용하지 않는다.

### n31이나 더 깊은 level 확인

기본 dense/direct 한도는 N=225(전체 미지수), 즉 n15이다. n31은 N=961이다.
명시적으로 예산을 올린 새 PRUN에서 실행한다.

```bash
PRUN31="artifacts/p_headroom_n31_$(date +%Y%m%d_%H%M%S)"
python scripts/run_v6_7_p_headroom.py plan \
  --source-run "$SOURCE_RUN" --out "$PRUN31" --split train \
  --sizes 31 --limit 3 --level 0 --repeats 3 --rhs-count 4 \
  --dense-max-dofs 1024 --direct-max-dofs 1024 --direct-steps 20
python scripts/run_v6_7_p_headroom.py run --run-dir "$PRUN31"
```

`--level 1`은 기존 hierarchy의 두 번째 transfer를 바꾼다. 위쪽 P/A는 보존하고, 해당 P 아래의
실제 Ac와 factorization을 재구성한다. `--level 1`과 `--sizes 63`을 결합해 coarse-only 실험 가능.
단 direct-max-dofs는 **전체 fine operator**에 적용된다. Full V-cycle 역전파의 비용이기 때문이다.
Dense-max-dofs는 진단 대상 level의 N에 적용된다. n63 전체 dense SVD는 자동 실행하지 않는다.
예상 workspace=20*8*N^2와 max-bytes(기본512MiB)도 검사한다. 이 추정은 peak RSS 인증이 아니다.
예산을 넘기면 `skipped_budget`이며, 최적값 0이나 가짜 upper bound를 기록하지 않는다.

### 기존 H_P checkpoint 포함

```bash
python scripts/run_v6_7_p_headroom.py plan \
  --source-run "$SOURCE_RUN" --out artifacts/p_headroom_with_nn \
  --split train --sizes 15 --limit 7 \
  --methods classical energy_min ls_uniform ls_energy direct nn \
  --p-checkpoint "$SOURCE_RUN/checkpoints/H_P/candidate.pt"
python scripts/run_v6_7_p_headroom.py run --run-dir artifacts/p_headroom_with_nn
```

실제 path가 다르면 해당 학습 checkpoint 경로를 명시한다. 새 random NN을 기존 trained P로 대체하지 않는다.
Checkpoint의 training_rules_digest와 support schema가 source와 다르면 거부한다.

## 2. 수학적 비교의 정확한 의미

에너지 최소화는 고정 support에서 P=particular+Zq로 표현한다. 행별 null-space Z로 PBc=Bf를
정확히 유지하고 coarse injection 행은 움직이지 않는다. 기본 Bc=1, Bf=P_C 1이다.
**Homogeneous Dirichlet 경계 제거 뒤의 행까지 P1=1을 강제하지 않는다.**
일반 near-nullspace mode를 지정하는 core API도 있으나, 해당 support에서 불가능하면 오류를 낸다.
고정 LS weight와 energy-based weight를 별도 비교한다. Soft regularization으로 hard constraint를 대체하지 않는다.
Projected CG는 P를 만드는 setup 알고리즘이다. 실제 PDE solve를 CG로 감싸지 않는다.
Energy minimizer가 반복 상한에서 멈추면 `optimizer_converged=false`로 표시한다.

Direct P는 per-operator projection logits를 역전파로 학습한다. 같은 classical smoother, level 수,
coarsening, 실제 recursive V-cycle을 사용한다. 학습 random errors와 별도 held-out error probes를 둔다.
기존 projection/support 제약에서 찾은 best-found candidate이지 sparsity 제약 아래의 전역최적 상한 증명이 아니다.
실제 guard 분기를 미분하는 것이 아니라 raw m-cycle energy 감소를 학습하고, runtime guarded solve는 별도로 측정한다.
Coarse operator-dependent P의 discrete dropping/support 결정은 기존 unroll처럼 각 forward에서 재계산하며
그 결정 자체를 미분하지 않는다.

### Norm와 norm 제곱을 혼동하지 않기

W^T W=A, S=W E_pre W^-1로 놓는다. Rank nc의 자유로운 A-coarse space가 주어지면
one-sided `(I-Q)S`의 최소 2-norm은 `sigma[nc]`(0-based)다.
**실제 post가 pre의 A-adjoint일 때만** 전체 symmetric two-grid `S^T(I-Q)S`의 최소 norm은
`sigma[nc]^2`이다. 즉 one-sided norm의 제곱과 전체 symmetric cycle norm을 구분한다.
소스의 pre/post sweep 전체를 basis vectors에 적용해 error maps를 만들고 adjoint defect를 검사한다.
불일치하면 `deployed_optimality_applicable=false`다. 임의로 대칭 smoother를 대체해 배치 알고리즘의 상한이라 하지 않는다.

이 SVD 진단은 exact coarse solve와 희소성/injection 제약 없는 coarse space다.
해당 공간을 structured-grid coarse PDE인 것처럼 다음 V-cycle에 배치하지 않는다.
`Vcycle_error`는 실제 sparse hierarchy의 독립 random-error 감쇠이고, exact worst-case bound가 아니다.

## 3. 비용과 성공률

`warm_multiple`: 준비와 timed RHS와 다른 prime을 제외. 실제 K개의 다른 RHS를 zero initial guess로 푼다.
`multiple`: fresh constructor + P 생성/test-vector relaxation/LS/새 Ac factorization을 한 번 포함한다.
Dual hierarchy의 추가 setup은 실제 안전한 C* fallback bank를 보유하는 현재 구현 비용이다.

P2/row magnitude/전체 hierarchy complexity 검사가 실패하면 원래 selected C*로 복구한다.
그 결과가 수렴해도 candidate_applied=false이면 그 P의 speedup 근거로 쓰지 않는다.
EM/LS는 classical_interpolation, direct는 offline_fitted_P, 기존 NN은 NN_P로 기록한다.
기존 solver 내부의 hybrid/learned-transfer counter는 경로 구분일 뿐 EM을 NN이라고 부르는 근거가 아니다.

`economics`의 break-even은 같은 operator의 관측 batch-average warm 시간과 측정 setup 차이로 계산한
선형 추정이다. 다른 K나 다른 RHS 분포에서 검증된 crossover가 아니다. H_S 숫자를 H_P에 대신 넣지 않는다.
Direct optimization 시간은 별도 기록하며, 저장된 per-A P의 solve time은 offline oracle 참고치다.
이를 일반화된 NN deployment 시간으로 집계하지 않는다.

## 4. 결과 파일 / resume

```
$PRUN/plan.json
$PRUN/case_results/<operator_digest>.json
$PRUN/P_<operator_digest>_<method>.npz
$PRUN/report.json
$PRUN/progress.json
```

```bash
python scripts/run_v6_7_p_headroom.py run --run-dir "$PRUN" --resume
```

완료된 operator 기록과 저장 P를 검증하고 재사용한다. 중단된 operator 한 건은 처음부터 다시 계산한다.
소스, hardware/thread/library, source manifest/config/rules, checkpoint, 저장 P의 변경을 검출한다.
RUN 기록은 덮어써서 다른 protocol과 섞지 않는다. 가정/예산을 바꿀 때는 새 PRUN을 만든다.

Gate는 `little_two_grid_headroom_only`, `headroom_remains_compare_classical_then_learned`,
`inspect_candidate_constraints_or_optimizer_before_generator_claim` 등을 권고한다. 연구 중단/모델 승격을 자동화하지 않는다.
Energy-min이 충분히 빠르면 먼저 독립 자료에서 그 classical 후보를 검증한다. 그렇다고 NN setup 대체의 가치까지
논리적으로 없다는 뜻은 아니다. 상한 gap이 작아도 cycle 비용 감소/다른 S 아래 상호작용의 여지는 별도다.

## 5. H_SP: 같은 frozen weights로 2x2

Best H_S를 warm-study에서 select한 뒤 그 expert_selection.json을 제공한다.
Headroom source와 S selection의 frozen rules/기본 numerical config가 같아야 한다.

```bash
python scripts/run_v6_7_p_headroom.py factorial \
  --run-dir "$PRUN" \
  --s-selection "$SOURCE_RUN/expert_selection.json" \
  --p-kind energy_min
```

`--p-kind ls_uniform`, `ls_energy`, `direct`, `nn`도 가능하다(먼저 그 P 측정을 완료해야 함).
C=(S_C,P_C), H_P=(S_C,P_new), H_S=(S_new,P_C), H_SP=(S_new,P_new).
각 반복에서 같은 P digest인지 검증하고 S checkpoint를 수정하지 않는다.
아래 Ac가 바뀌면 해당 actual operator에 맞춘 S stencil/factor를 재생성한다. 이는 같은 생성기 weights의 비교다.

P의 conditional gains와 `log(T_HS/T_HSP)-log(T_C/T_HP)`를 기록한다.
모든 네 조합이 성공한 공통 operator에서만 interaction speedup을 계산한다.
`HSP_speedup_vs_best_other`는 C/H_P/H_S 중 최선에 비해 H_SP가 이기는지 보여준다.
Standalone H_P가 약하다고 조건부 시너지가 논리적으로 불가능한 것은 아니므로, 2x2 **진단**은 막지 않는다.
이 경로가 alternating/joint training을 자동으로 시작하거나 final을 여는 일은 없다.

## 연구 근거 (직접 재현과 구분)

- Xu & Zikatanov, Algebraic Multigrid Methods: https://arxiv.org/abs/1611.01917
- Garcia Ramos & Nabben, On Optimal Algebraic Multigrid Methods: https://arxiv.org/abs/1906.01381
- Olson, Schroder & Tuminaro, constrained energy interpolation: https://doi.org/10.1137/100803031
- Brandt et al., Bootstrap AMG: https://doi.org/10.1137/090752973
- Katrutsa et al., Deep Multigrid: https://arxiv.org/abs/1711.03825

이번 구현은 이론에 맞춘 진단 및 고전 baseline/실제 V-cycle 실험 경로다. 논문 benchmark의 완전 재현이나
learned-LWLS 성능 개선, 전 grid 일반화 또는 모든 classical solver 대비 우월성을 주장하지 않는다.
