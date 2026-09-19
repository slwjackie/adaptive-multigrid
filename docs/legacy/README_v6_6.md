# Adaptive Neural Multigrid v6.6

Standalone multigrid with a learned **one-way temporal switch**. Predict a finite neural prefix K once, use the same classical hierarchy with additive NN smoother corrections for that prefix, then irreversibly run only cached classical V-cycles. K=0 is a legitimate action. No spatial detector, local masks, auxiliary learned-transfer hierarchy, or outer iterative accelerator is present in the runtime.

See **README_KR.md** for implementation contracts, commands, exact objective, migration, measurement scopes and qualifications. The bundled policy is a small empirical training/verification artifact, not a validated universal speedup claim. Default `temporal` mode abstains when its independently audited performance certificate is absent; `predicted` explicitly exposes the learned decisions for research diagnostics while preserving numerical safeguards.

```bash
python -m pip install -e ".[dev]"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/build_native_stencil.py
pytest -ra
python scripts/demo_temporal.py --checkpoint artifacts/verification_v6_6/training/checkpoint_v6_6_final.pt --mode predicted
```

The v6.5 numerical grid/PDE/transfer/smoother primitives were retained. The new runtime shares only the classical hierarchy; migration retains the legacy trained smoother, not its spatial policy or learned-transfer auxiliary cycle. CPU/FP64 runtime only; optional CUDA offline smoother pretraining is not GPU solver support.
