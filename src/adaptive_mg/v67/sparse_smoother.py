"""Weymouth-inspired, five-parameter sparse approximate inverse for SPD A.

The coefficient functions are learned offline through complete V-cycles. Only
their CSR matrix B(A) is used online: x <- x + B(A)(b-Ax). This is not an
exact reproduction of Weymouth's negative-Laplacian/Jacobi-pre schedule.
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


class SparsePolynomialSmoother(nn.Module):
    training_only = False

    def __init__(self, initial_omega=.72):
        super().__init__()
        if not np.isfinite(initial_omega) or not 0 < initial_omega < 2:
            raise ValueError('initial_omega must lie in (0, 2)')
        self.initial_omega = float(initial_omega)
        self.coefficients = nn.Parameter(torch.tensor([initial_omega, 0., 0., 0., 0.]))

    def sparse_smoother_spec(self):
        return dict(initial_omega=self.initial_omega)

    def sparse_values(self, a):
        """Graph-preserving values on the actual A support, including coarse fill.

        fd(q)=c0+c1*q+c2*q^2, fo(q)=c3*q+c4*q^2. Thus fo(0)=0,
        B is symmetric for symmetric A, and B(alpha*A)=B(A)/alpha.
        Symmetry does not imply positive definiteness or convergence.
        """
        d = a.diagonal()
        if not bool(torch.isfinite(a.values).all()) or not bool((d > 0).all()):
            raise ValueError('Sparse smoother requires finite A with positive diagonal')
        rows = torch.as_tensor(a.rows, device=a.values.device)
        cols = torch.as_tensor(a.cols, device=a.values.device)
        off = rows != cols
        scale = a.values[off].abs().max() if bool(off.any()) else d.max()
        scale = torch.where(scale > 0, scale, d.max())
        q = a.values / scale
        c = self.coefficients.to(a.values)
        fd = c[0] + q * (c[1] + c[2] * q)
        fo = q * (c[3] + c[4] * q)
        return torch.where(off, fo / (d[rows] + d[cols]), fd / d[rows])

    def sparse_matrix(self, a):
        """CPU setup only; no neural network or autograd in a solve cycle."""
        a = sp.csr_matrix(a, dtype=np.float64, copy=True)
        a.sum_duplicates(); a.sort_indices(); a.eliminate_zeros()
        d = a.diagonal()
        if not np.isfinite(a.data).all() or not np.all(d > 0):
            raise ValueError('Sparse smoother requires finite A with positive diagonal')
        rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
        cols = a.indices
        off = rows != cols
        scale = np.max(np.abs(a.data[off]), initial=0.) or np.max(d)
        q = a.data / scale
        c = self.coefficients.detach().cpu().double().numpy()
        fd = c[0] + q * (c[1] + c[2] * q)
        fo = q * (c[3] + c[4] * q)
        values = np.where(off, fo / (d[rows] + d[cols]), fd / d[rows])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite sparse smoother proposal')
        result = sp.csr_matrix((values, cols.copy(), a.indptr.copy()), shape=a.shape)
        result.eliminate_zeros()
        return result
