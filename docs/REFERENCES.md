# Implementation references

The numerical baseline is the user-supplied `adaptive_neural_multigrid_v6_6_final.zip`. The old API and numerical regression tests are retained; no previous unverified v6.7 work or results were assumed to exist.

Official PyTorch references consulted for backend behavior:

- https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html
- https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html
- https://docs.pytorch.org/docs/stable/sparse.html

The new sparse training backend does not assume that every native sparse×sparse operation supports the needed autograd in every installed PyTorch version. It implements explicit sparse-value first-order adjoints and tests them against finite differences. Source code and exact local test results, rather than external package version assumptions, determine what this ZIP has verified.
