# 구현 위치와 데이터 흐름

| 기능 | 구현 |
|---|---|
| 정확한 numerical/config 계약 | `v67/config.py` |
| FP32 A→normalized S/gain/P, tiny controller | `v67/models.py` |
| Lazy learned hierarchy, one coarse path, replacement | `v67/banks.py` |
| Selected rows / whole-line replacement | `v67/banks.py:replacement_step` |
| C/H/LOCK + end-to-end break-even | `v67/controller.py` |
| Rollback·cycle budget·FP64 residual·same-A cache | `v67/solver.py` |
| Coarse block detector, refresh/hysteresis | `v67/spatial.py` |
| Indexed native stencil / fused Rr+statistics | `native/stencil9.cpp`, `v67/native.py` |
| Sparse values autograd | `v67/autograd_sparse.py` |
| Cached S/P multi-cycle unroll | `v67/unroll.py` |
| Six stages, measured validation, atomic resume | `v67/training.py` |
| Paired cold/warm runtime & failure-aware results | `v67/evaluation.py` |
| 7 PDE families & split manifest | `v67/data.py` |

## Cache ownership

Prepared solver owns a copy of canonical FP64 A, an eager immutable classical bank and optionally a lazy learned bank. Freshness is checked once per public `solve`, or once per `solve_many` batch. A/model/config changes invalidate the appropriate bank. Nothing hashes full model weights at every cycle. Residual-dependent masks live in per-solve SpatialState and do not mutate P or Ac.

A model with a different inference precision can require one bank rebuild. CPU/MPS/CUDA transfer or conversion occurs during lazy preparation, not numerical cycle execution. The tiny controller/detector are compiled NumPy arrays.

## Spatial semantics

Full-grid NN replacement means a designated classical sweep is not computed. Selected-row Jacobi replacement computes only the classical complement. Coupled line relaxation preserves whole-block selection and exact block ordering; it does not promise independence of neighboring rows. The selected bank's post-smoother acts on its own actual coarse A.

A all-zero root gate routes the cycle to the complete classical bank. A P-only ablation intentionally bypasses smoother selection and still uses learned transfer. After an earlier accepted H update, switching banks does not undo that historical update; only a rejected candidate is rolled back.

## Stability scope

Runtime rejects nonfinite/growing hybrid candidates and permanently locks to the cached classical bank. Standalone classical MG itself may fail on some anisotropic problems. Neither FP64 arithmetic nor a static learned P guarantees convergence on every operator; the code reports these failures and never reports a shorter failing runtime as speedup.

No full classical shadow trajectory is maintained. Thus the code cannot promise universal per-budget domination of the original classical trajectory. Independent audit certificates are empirical and hardware/config/model scoped.
