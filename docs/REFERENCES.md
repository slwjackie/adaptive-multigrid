# Implementation references

> **과거 실험·구현 기록.** 아래의 기본값·명령·결과는 이 문서가 다루는 경로에 한정된다.
> 현재 `residual-u-net-teacher`의 주 실험은 [line_alt 고정 + P-only 학습](learned_p_study.md)이다.
> 현재 실행은 [Mac runbook](learned_p_runbook.md), 경로별 구분은 [문서 목록](README.md)을 따른다.

The numerical baseline is the user-supplied `adaptive_neural_multigrid_v6_6_final.zip`. The old API and numerical regression tests are retained; no previous unverified v6.7 work or results were assumed to exist.

Official PyTorch references consulted for backend behavior:

- https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html
- https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html
- https://docs.pytorch.org/docs/stable/sparse.html

The new sparse training backend does not assume that every native sparse×sparse operation supports the needed autograd in every installed PyTorch version. It implements explicit sparse-value first-order adjoints and tests them against finite differences. Source code and exact local test results, rather than external package version assumptions, determine what this ZIP has verified.
