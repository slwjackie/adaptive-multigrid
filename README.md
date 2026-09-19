# Adaptive Neural Multigrid v6.7

Standalone multigrid with **cached operator-generating neural networks**, a lightweight temporal break-even controller, optional block spatial selection, and a classical lock/recovery path.

The new production API is `adaptive_mg.v67.PreparedAdaptiveMG`. The v6.6 exact-K API is retained only for legacy comparison/regression tests. No external Krylov solver is used by the new production path.

See [README_KR.md](README_KR.md) for setup, the six-stage resumable training pipeline, diagnostics, precision contracts, and the exact scope of stability/performance claims.

```bash
pip install -e '.[dev]'
python scripts/build_native_stencil.py
pytest -ra
python scripts/run_v6_7_pipeline.py --config configs/v6_7_smoke.json --output-dir artifacts/my_smoke
```

The included smoke checks do **not** establish speedup over a tuned classical MG solver. Full research training and independent performance auditing are runnable, but not represented as completed unless their generated records say so.
