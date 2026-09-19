# Algorithm and sources

## Source basis

The requested one-way prefix design and the supplied v6.5 code are the implementation basis. Reused numerical primitives are recorded in `BASELINE_PROVENANCE.md`. No literature result is treated as evidence that this particular checkpoint is faster.

## Conditional convergence argument

After a finite neural prefix and any rejected-neural rollback, the accepted state is finite. From the irreversible transition onward the method is exactly the fixed classical MG iteration. For a linear fixed classical iteration with error matrix E_C, the tail error is E_C^j e_switch. It tends to zero if the spectral radius of E_C is less than one. Neither SPD of the input alone nor a decreasing scalar residual during the neural prefix proves that condition for arbitrary MG components. A finite max-cycle budget also does not give an unconditional success guarantee.

A small absolute residual does not intrinsically change the contraction of a fixed linear iteration. The rationale for a finite neural prefix is therefore empirical and depends on the error-mode distribution, target tolerance, costs and numerical precision, not solely on residual magnitude.

## Measured policy training

For each operator/RHS the complete guarded solver is evaluated at a finite set of prefix lengths including zero. Complete time contains all setup, checks, rejected candidates and recovery calls. A measured per-action loss is provided to a softmax policy; minimizing its expected loss gives differentiable policy training without pretending to differentiate a wall clock or an integer stopping time. The oracle is the best *tested* prefix under the budget, not a global optimum over all possible adaptive solvers.

Failures have censored time-to-tolerance. They are not treated as quick successful runs. The diagnostic oracle may identify a time saving, but predictor execution, setup and generalization must be evaluated separately on held-out data before deployment.

## External documentation consulted (2026-09-08)

- PyTorch benchmark utilities: https://docs.pytorch.org/docs/stable/benchmark_utils.html — warmups, repetition, thread control and consistent timing.
- PyTorch inference mode: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html — operator-only inference without autograd overhead.
- PETSc PCMG: https://petsc.org/release/manualpages/PC/PCMG/ — distinctions between multigrid components/roles. PETSc is not a dependency of this package.

The timings in the bundled report come from the provided CPU implementation, not from these documentation pages.
