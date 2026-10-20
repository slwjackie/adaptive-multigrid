# Current P-only algorithm and sources

This document describes the `residual-u-net-teacher` branch's current fixed-line,
learned-P workflow. See [the design](learned_p_study.md), [runbook](learned_p_runbook.md)
and [historical document index](README.md) for experimental scope.

## Why learn P?

For an SPD A and full-column-rank P, an exact two-grid coarse correction has error
operator C = I − P(PᵀAP)⁻¹PᵀA. It removes errors in range(P). A smoother should
reduce the complementary components; a good P represents errors that relaxation
leaves behind. More accurate coarse solves alone cannot remove a component e with
PᵀAe=0. In the actual multilevel method, the coarse inverse is approximated by a
recursive cycle until the terminal LU solve.

The current n=7 smoke has only a 7×7 fine grid and a 3×3 terminal direct solve.
It cannot establish that an intermediate coarse smoother is a performance bottleneck.
The full experiment has additional nonterminal levels and should be evaluated there.

## Actual loss and performance objective

For a fixed generated hierarchy, one raw cycle is a linear error operator E. The
training loss is `(1 / (q*N)) * sum_j ||E z_j||_2^2`, with independent Rademacher
probes z_j and q=8 by default. Its expectation is `||E||_F^2 / N`. No dense N×N E
is assembled. Autograd includes the actual Galerkin hierarchy and classical line
solves; only P-generator parameters are optimized.

The exact norms satisfy ρ(E) ≤ ||E||₂ ≤ ||E||F. A finite stochastic estimate is not
itself a guaranteed upper bound. Lower Frobenius loss need not imply fewer cycles
or less wall time. Wall-clock measurements, integer stopping and runtime rollback
are not directly differentiated by this loss. Cold and multiple-RHS evaluation
include setup; warm timings are additional evidence.

## Runtime safeguards and conditional convergence

Current explicit H_P execution rejects nonfinite/growing candidates, handles learned
setup failures and switches to classical near the stopping threshold. It does not
use a learned prefix policy, classical shadow trajectory or measured speed-based
switching. See [the architecture](ARCHITECTURE_KR.md) for conditions and counters.

If the solver switches to the fixed classical iteration at a finite accepted state,
then subsequent errors satisfy e_j=E_C^j e_switch. With unlimited iterations this
converges when ρ(E_C)<1. The current learned path is not forced to switch merely
because progress is very slow; a finite attempt budget also does not guarantee
success. SPD input or observed residual decrease alone does not prove unconditional
convergence or speedup for the complete guarded solver.

For a fixed linear iteration, E(αe)=αEe. Small error amplitude by itself does not
make the relative contraction slower. Late iterations can instead be dominated by
hard-to-remove modes, and finite precision can eventually matter.

## Primary literature

- [Greenfeld et al., Learning to Optimize Multigrid PDE Solvers, ICML 2019](https://proceedings.mlr.press/v97/greenfeld19a/greenfeld19a.pdf): fixed Gauss–Seidel and learned prolongation; spectral-radius motivation with squared Frobenius surrogate. Uses two-grid/block-Fourier training, not the identical line_alt full-cycle implementation here.
- [Luz et al., Learning Algebraic Multigrid Using Graph Neural Networks, ICML 2020](https://proceedings.mlr.press/v119/luz20a/luz20a.pdf): GNN-generated P with a two-level Frobenius objective. It is not the architecture currently used in this branch's primary experiment.
- [Huang, Li and Xi, Learning Optimal Multigrid Smoothers via Neural Networks](https://arxiv.org/abs/2102.12071): learned smoothers trained with coarse correction in mind. Its comparisons against Jacobi/Gauss–Seidel do not establish an advantage over this implementation's line_alt.

These sources motivate experiments; their published results are not evidence that
this repository's trained models outperform its fixed classical references.
[Baseline provenance](BASELINE_PROVENANCE.md) and [older implementation references](REFERENCES.md)
remain historical records, not the current training/evaluation contract.
