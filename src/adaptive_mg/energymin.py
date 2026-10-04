"""Fixed-support energy interpolation with exact row-sum/injection constraints.

Projected CG is an OFFLINE/SETUP calculation, never an outer solver. The
unconverged finite-budget candidate is named EM-CGk, not a certified minimizer.
The fast path has only row-sum constraints; general mode constraints retain the
independent p_headroom implementation.
"""
from __future__ import annotations
from inspect import signature
from time import perf_counter
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def row_sum_basis(base, fixed_rows):
    """Orthonormal Helmert tangent blocks without a Python loop over fine rows."""
    base=base.tocsr();counts=np.diff(base.indptr)
    free=counts.copy();free[np.asarray(fixed_rows,dtype=np.int64)]=0
    dofs=np.maximum(free-1,0);starts=np.r_[0,np.cumsum(dofs)[:-1]]
    rr=[];cc=[];vv=[]
    for k in np.unique(free):
        if k<2:continue
        rows=np.flatnonzero(free==k)
        offsets=base.indptr[rows]
        for j in range(1,int(k)):
            r=offsets[:,None]+np.arange(j+1)
            c=np.broadcast_to((starts[rows]+j-1)[:,None],r.shape)
            v=np.ones(j+1)/np.sqrt(j*(j+1.));v[-1]=-j/np.sqrt(j*(j+1.))
            rr.append(r.ravel());cc.append(c.ravel());vv.append(np.broadcast_to(v,r.shape).ravel())
    if not rr:return sp.csr_matrix((base.nnz,int(dofs.sum())))
    return sp.csr_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(base.nnz,int(dofs.sum())))


def energy_weights(a,pattern,*,maxiter=10,rtol=1e-8):
    from .transfer import scipy_prolongation_from_weights,coarse_fine_indices,weights_from_sparse_matrix
    if isinstance(maxiter,bool) or not isinstance(maxiter,int) or maxiter<1 or not np.isfinite(rtol) or not 0<rtol<1:
        raise ValueError('invalid EM controls')
    start=perf_counter();a=a.tocsr().astype(np.float64)
    if not np.isfinite(a.data).all() or np.any(a.diagonal()<=0) or sla.norm(a-a.T)>1e-10*max(sla.norm(a),1e-300):
        raise ValueError('EM requires finite symmetric positive-diagonal A')
    p0=scipy_prolongation_from_weights(pattern,pattern.bilinear_weights)
    rows=np.repeat(np.arange(p0.shape[0]),np.diff(p0.indptr));cols=p0.indices
    z=row_sum_basis(p0,coarse_fine_indices(pattern))
    build_seconds=perf_counter()-start;iterations=0
    def make(data):return sp.csr_matrix((data,p0.indices,p0.indptr),shape=p0.shape)
    def gather(p):return np.asarray(p.tocsr()[rows,cols]).ravel()
    rhs=-np.asarray(z.T@gather(a@p0)).ravel()
    if z.shape[1] and np.linalg.norm(rhs)>0:
        def action(q):return np.asarray(z.T@gather(a@make(z@q))).ravel()
        operator=sla.LinearOperator((z.shape[1],)*2,matvec=action,dtype=np.float64)
        def count(_):
            nonlocal iterations
            iterations+=1
        key='rtol' if 'rtol' in signature(sla.cg).parameters else 'tol'
        q,info=sla.cg(operator,rhs,maxiter=maxiter,atol=0.,callback=count,**{key:rtol})
        if info<0 or not np.isfinite(q).all():raise ValueError('EM projected CG failed')
        result=make(p0.data+z@q)
        stationarity=float(np.linalg.norm(action(q)-rhs)/np.linalg.norm(rhs))
    else:result=p0;info=0;stationarity=0.
    before=float(p0.multiply(a@p0).sum());after=float(result.multiply(a@result).sum())
    if not np.isfinite(after) or after>before+1e-9*max(1.,abs(before)):
        raise ValueError('EM failed energy nonincrease check')
    weights=weights_from_sparse_matrix(pattern,result)
    mask=pattern.bilinear_weights!=0;weights[~mask]=0.
    weights+=(pattern.bilinear_weights.sum(1)-weights.sum(1))[:,None]*mask/np.maximum(mask.sum(1),1)[:,None]
    fixed=coarse_fine_indices(pattern);weights[fixed]=pattern.bilinear_weights[fixed]
    if np.max(np.abs(weights).sum(1))>8.:raise ValueError('EM row magnitude safety limit exceeded')
    return weights,dict(method=f'EM-CG{maxiter}',iterations=iterations,optimizer_converged=info==0,
        energy_before=before,energy_after=after,relative_gradient=stationarity,
        constraint_setup_seconds=build_seconds,setup_seconds=perf_counter()-start,
        scope='fixed support row-sum/injection energy candidate, not convergence certification')
