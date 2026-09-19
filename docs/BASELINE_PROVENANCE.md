# Numerical baseline provenance

Source: user-supplied `adaptive_neural_multigrid_v6_5_final.zip`.

| Component | Treatment |
|---|---|
| grid.py | Retained grid-shape / nesting / semicoarsening logic |
| pde.py | Retained structured P1 diffusion assembly |
| smoothers.py | Retained classical smoother arithmetic; diagonal helper moved out of detector module |
| transfer.py | Retained classical interpolation, patterns and Galerkin arithmetic |
| strategy.py | Retained explicit strategy registry and operator-summary helper; no global learned selector instantiated |
| native/stencil9.cpp | Retained native stencil implementation; new runtime calls it with no spatial gate |
| NativeStencil | Retained optional C++/OpenMP loading, native/CSR fallback |
| NeuralSmootherNet | Retained operator-conditioned directions and scalar gain state layout |
| classical_cycle | Retained baseline arithmetic and reverse post-smoothing order |
| New solver/policy | Replaces spatial runtime with a one-way temporal state machine and measured full-solve cost learning |

The new API does not pretend to be drop-in compatible with the previous spatial `mode="adaptive"`. Such a mode now raises a configuration error rather than silently interpreting old options differently.


## 실제 v6.5 소스 대조

`scripts/verify_legacy_baseline.py`를 원본 v6.5 소스에 별도 PYTHONPATH로 실행했습니다. 15×15의 이방성 operator에서 16개 classical strategy, 각 3 V-cycle의 최종 해가 비트 단위로 동일했습니다. Residual norm은 v6.6에서 overflow-resistant `stable_norm()`으로 바뀌어 마지막 자리 반올림 차이가 있습니다. 최대 상대 차이 2.42e-16이며, 이전 버전과 residual 기록까지 비트 동일하다고 주장하지 않습니다. 원자료: `artifacts/verification_v6_6/legacy_baseline_parity.json`.

새 v6.6 내부의 K=0 vs classical은 두 경로가 동일 norm 함수를 사용하므로 해와 residual history 모두 비트 동일합니다.
