# Adaptive Neural Multigrid v3

Research code for hierarchical adaptive neural multigrid on structured-grid, scalar SPD heterogeneous/anisotropic diffusion problems.

The v3 architecture is:

```text
sparse PDE operator
  -> controlled global classical-strategy selector
  -> classical MG (Jacobi/line relaxation, bilinear/operator transfer,
                   full/semicoarsening)
  -> local residual detector
  -> patch-selective additive neural smoother and learned transfer correction
  -> optional analytical safeguards
```

Key properties:

- differentiable **full V-cycle** training, not a two-grid surrogate;
- default train roots `15,31,63` and OOD validation at `127`;
- 16-candidate learned prolongation with tied `R=P.T` and Galerkin `Ac=P.T A P`;
- actual patch-selective NN execution;
- controlled global strategy selection;
- line relaxation and semicoarsening baselines;
- sparse/vectorized operator-dependent interpolation;
- raw versus safeguarded ablation, alpha distributions, and work accounting;
- CG, FCG, and FGMRES integrations;
- 40 regression tests and randomized stress validation.

The bundled checkpoint is a short verification asset, **not a paper-performance model**. The included targeted evaluation shows that raw neural smoothing still requires more training and that learned transfer was rolled back when full-V-cycle validation did not improve.

## Install

```bash
unzip adaptive_neural_multigrid_v3_final.zip
cd adaptive_neural_multigrid_v3
python -m pip install -e .
pytest -q
```

## Full cross-grid training

```bash
python scripts/train_staged.py \
  --output artifacts/checkpoint_v3.pt \
  --train-sizes 15,31,63 \
  --validation-sizes 15,31,63 \
  --ood-validation-sizes 127 \
  --strategy-source selector
```

## Raw/safe ablation

```bash
python scripts/run_ablation.py \
  --checkpoint artifacts/checkpoint_v3.pt \
  --sizes 15,31,63 \
  --output-dir artifacts/ablation_v3
```

## Classical baseline catalog

```bash
python scripts/run_baseline_benchmark.py \
  --sizes 31,63 \
  --cycles 20 \
  --output-dir artifacts/baseline_v3
```

See `README_KR.md`, `CHANGELOG_V3_KR.md`, and `artifacts/verification_v3/VERIFICATION_V3_KR.md` for the complete implementation and validation record.
