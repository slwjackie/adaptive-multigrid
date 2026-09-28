"""Small, independent numerical diagnostics for interpolation headroom.

This module does not train a deployable NN, choose a runtime strategy, or certify
speed/generalization. Dense SVDs are diagnostic-only and resource-guarded. Sparse
energy/LS candidates preserve declared support, injection and feasible equality
constraints; for eliminated Dirichlet rows the default target is P_C @ 1, NOT 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import math
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla


class DiagnosticBudgetError(ValueError):
    """A requested offline diagnostic exceeds its explicitly declared budget."""


def checked_csr(value, *, square=False):
    if not sp.issparse(value):
        raise TypeError('expected a sparse matrix')
    result = value.tocsr().astype(np.float64, copy=True)
    result.sum_duplicates(); result.eliminate_zeros(); result.sort_indices()
    if not np.isfinite(result.data).all() or min(result.shape) < 1:
        raise ValueError('matrix must be finite and nonempty')
    if square and result.shape[0] != result.shape[1]:
        raise ValueError('square operator required')
    return result


def symmetry_error(a):
    return float(sla.norm(a-a.T) / max(sla.norm(a), np.finfo(float).tiny))


@dataclass
class AffineSupport:
    """P(q)=particular+Zq on the P_C sparsity, with row-wise exact constraints.

    Z has orthonormal per-row null-space blocks. Injection rows are fixed to
    P_C, not penalized. Optional B_f and B_c must be mutually feasible on EVERY
    row; infeasible targets fail explicitly instead of silently being relaxed.
    """
    base: object
    rows: np.ndarray
    cols: np.ndarray
    particular: np.ndarray
    z: object
    coarse_modes: np.ndarray
    fine_targets: np.ndarray
    fixed_rows: np.ndarray

    @classmethod
    def build(cls, base, fixed_rows=(), coarse_modes=None, fine_targets=None, tol=1e-10):
        base = checked_csr(base)
        n, nc = base.shape
        if not 0 < tol < 1:
            raise ValueError('invalid constraint tolerance')
        fixed = np.asarray(fixed_rows, dtype=np.int64)
        if len(set(fixed.tolist())) != len(fixed) or (fixed.size and (fixed.min()<0 or fixed.max()>=n)):
            raise ValueError('invalid fixed rows')
        bc = np.ones((nc,1)) if coarse_modes is None else np.asarray(coarse_modes,np.float64)
        bf = np.asarray(base@bc) if fine_targets is None else np.asarray(fine_targets,np.float64)
        if bc.ndim!=2 or bf.ndim!=2 or bc.shape[0]!=nc or bf.shape!=(n,bc.shape[1]) or bc.shape[1]<1:
            raise ValueError('mode/target shape mismatch')
        if not np.isfinite(bc).all() or not np.isfinite(bf).all():
            raise ValueError('nonfinite modes')
        fixed_set=set(fixed.tolist()); particular=base.data.copy()
        rows=np.repeat(np.arange(n),np.diff(base.indptr)); cols=base.indices.copy()
        zr,zc,zv=[],[],[]; offset=0
        for i in range(n):
            lo,hi=base.indptr[i:i+2]; index=cols[lo:hi]; c=bc[index].T
            delta=bf[i]-c@particular[lo:hi]
            if i in fixed_set:
                if la.norm(delta)>tol*(1+la.norm(bf[i])):
                    raise ValueError('infeasible fixed-row mode constraint')
                continue
            if hi==lo:
                if la.norm(delta)>tol:raise ValueError('empty support cannot reproduce target')
                continue
            repair=la.lstsq(c,delta,cond=tol)[0]
            if la.norm(c@repair-delta)>tol*(1+la.norm(bf[i])):
                raise ValueError('infeasible mode reproduction on baseline support')
            particular[lo:hi]+=repair
            zi=la.null_space(c,rcond=tol)
            rr,cc=np.nonzero(zi)
            zr.extend((lo+rr).tolist());zc.extend((offset+cc).tolist());zv.extend(zi[rr,cc].tolist())
            offset+=zi.shape[1]
        z=sp.csr_matrix((zv,(zr,zc)),shape=(base.nnz,offset))
        obj=cls(base,rows,cols,particular,z,bc.copy(),bf.copy(),fixed.copy())
        obj.validate(obj.matrix())
        return obj

    @property
    def dofs(self): return self.z.shape[1]

    def matrix(self, q=None):
        values=self.particular.copy()
        if q is not None:
            q=np.asarray(q,np.float64)
            if q.shape!=(self.dofs,) or not np.isfinite(q).all():raise ValueError('invalid free weights')
            values+=np.asarray(self.z@q).ravel()
        result=sp.csr_matrix((values,self.base.indices.copy(),self.base.indptr.copy()),shape=self.base.shape)
        result.eliminate_zeros()
        return result

    def tangent(self, q):
        return sp.csr_matrix((np.asarray(self.z@q).ravel(),self.base.indices,self.base.indptr),shape=self.base.shape)

    def gather(self,p):
        return np.asarray(p.tocsr()[self.rows,self.cols]).ravel()

    def validate(self,p,tol=1e-8):
        p=checked_csr(p)
        if p.shape!=self.base.shape:raise ValueError('P shape changed')
        mask=self.base.copy();mask.data[:]=1
        external=p-p.multiply(mask)
        if external.nnz and np.any(external.data!=0):raise ValueError('P created external support')
        defect=la.norm(p@self.coarse_modes-self.fine_targets)
        if defect>tol*(1+la.norm(self.fine_targets)):raise ValueError('P violates exact mode constraints')
        if self.fixed_rows.size and sla.norm(p[self.fixed_rows]-self.base[self.fixed_rows])>tol:
            raise ValueError('P violates fixed injection')
        return dict(constraint_residual=float(defect),free_parameters=self.dofs,
                    parent_nnz=self.base.nnz,p_nnz=p.nnz,external_support=0)


def energy_minimize(a, constraints, *, maxiter=150, rtol=1e-8):
    """Minimize trace(P.T A P) in a fixed affine support using projected CG.

    This Krylov solve is ONLY an interpolation setup algorithm, not an outer
    Krylov wrapper around the deployment MG solver. CG convergence is reported;
    a finite iterate is not called a globally converged minimizer without it.
    """
    a=checked_csr(a,square=True)
    if a.shape[0]!=constraints.base.shape[0] or symmetry_error(a)>1e-10:
        raise ValueError('symmetric A of matching size required')
    if maxiter<1 or rtol<=0 or not np.isfinite(rtol):raise ValueError('invalid energy solver controls')
    start=perf_counter();p0=constraints.matrix();iterations=0
    energy=lambda p:float(p.multiply(a@p).sum())
    e0=energy(p0)
    if constraints.dofs:
        def action(q):return np.asarray(constraints.z.T@constraints.gather(a@constraints.tangent(q))).ravel()
        rhs=-np.asarray(constraints.z.T@constraints.gather(a@p0)).ravel()
        operator=sla.LinearOperator((constraints.dofs,)*2,matvec=action,dtype=np.float64)
        def count(_):
            nonlocal iterations
            iterations+=1
        q,info=sla.cg(operator,rhs,rtol=rtol,atol=0.,maxiter=maxiter,callback=count)
        if not np.isfinite(q).all() or info<0:raise ValueError('projected energy CG failed')
        result=constraints.matrix(q);stationarity=float(la.norm(action(q)-rhs)/max(la.norm(rhs),1e-300))
    else:
        result=p0;info=0;stationarity=0.
    if energy(result)>e0+1e-9*max(abs(e0),1.):raise ValueError('energy solver increased objective')
    check=constraints.validate(result)
    return result,dict(method='constrained_energy',energy_before=e0,energy_after=energy(result),
        setup_seconds=perf_counter()-start,iterations=iterations,optimizer_converged=info==0,
        projected_relative_gradient=stationarity,**check)


def slow_vectors(a, smooth_error, *, count=8, sweeps=4, seed=1):
    """Algebraically slow TRAIN/setup errors; not assumed geometrically smooth."""
    a=checked_csr(a,square=True)
    if count<1 or sweeps<0:raise ValueError('invalid test-vector controls')
    start=perf_counter();v=np.random.default_rng(seed).normal(size=(a.shape[0],count))
    for _ in range(sweeps):
        for j in range(count):v[:,j]=smooth_error(v[:,j])
    norms=np.linalg.norm(v,axis=0)
    if not np.isfinite(v).all() or np.any(norms<1e-100):raise ValueError('degenerate slow vectors')
    v/=norms
    energies=np.sum(v*(a@v),axis=0)
    if np.any(energies<=0) or not np.isfinite(energies).all():raise ValueError('positive error energy required')
    return v,dict(test_vector_seconds=perf_counter()-start,smoothing_applications=count*sweeps,
                  vector_energies=energies.tolist(),seed=seed)


def least_squares(a,constraints,vectors,coarse_rows,*,weighting='uniform',ridge=1e-8):
    """One-pass constrained LS; not a full BAMG bootstrap-cycle reproduction."""
    a=checked_csr(a,square=True);v=np.asarray(vectors,np.float64)
    coarse_rows=np.asarray(coarse_rows,np.int64);n,nc=constraints.base.shape
    if v.ndim!=2 or v.shape[0]!=n or coarse_rows.shape!=(nc,) or len(set(coarse_rows.tolist()))!=nc:
        raise ValueError('invalid vectors/coarse-row map')
    if coarse_rows.min()<0 or coarse_rows.max()>=n or not np.isfinite(v).all():raise ValueError('invalid coarse samples')
    if weighting not in ('uniform','energy') or ridge<0 or not np.isfinite(ridge):raise ValueError('invalid LS settings')
    start=perf_counter();energies=np.sum(v*(a@v),axis=0)
    if np.any(energies<=0):raise ValueError('test-vector energies must be positive')
    w=np.ones(v.shape[1]) if weighting=='uniform' else 1/np.maximum(energies,1e-100)
    w/=np.mean(w);weights=np.sqrt(w);values=constraints.particular.copy();vc=v[coarse_rows]
    fixed=set(constraints.fixed_rows.tolist());rank_deficient=0
    for i in range(n):
        lo,hi=constraints.base.indptr[i:i+2]
        if i in fixed or hi==lo:continue
        cols=constraints.cols[lo:hi]
        zi=la.null_space(constraints.coarse_modes[cols].T,rcond=1e-10)
        if zi.shape[1]==0:continue
        design=(vc[cols].T@zi)*weights[:,None]
        target=(v[i]-values[lo:hi]@vc[cols])*weights
        if np.linalg.matrix_rank(design)<zi.shape[1]:rank_deficient+=1
        if ridge:
            design=np.vstack((design,np.sqrt(ridge)*np.eye(zi.shape[1])))
            target=np.r_[target,np.zeros(zi.shape[1])]
        q=la.lstsq(design,target)[0];values[lo:hi]+=zi@q
    result=sp.csr_matrix((values,constraints.base.indices.copy(),constraints.base.indptr.copy()),shape=constraints.base.shape)
    result.eliminate_zeros();check=constraints.validate(result)
    return result,dict(method='ls_'+weighting,ls_seconds=perf_counter()-start,
        weighted_fit_error=float(np.sum((v-result@vc)**2*w)),rank_deficient_rows=rank_deficient,
        weights=w.tolist(),**check)


def dense_budget(n,*,max_dofs=1024,max_bytes=536870912):
    # Conservative declared workspace estimate, not a measured peak-RSS bound.
    estimated=20*8*int(n)*int(n)
    if n>max_dofs or estimated>max_bytes:
        raise DiagnosticBudgetError(f'dense diagnostic refused: N={n}, estimated_bytes={estimated}, max_dofs={max_dofs}, max_bytes={max_bytes}')
    return estimated


def spectral_headroom(a,pre,post,p_candidates,*,max_dofs=1024,max_bytes=536870912,tol=1e-8):
    """A-whitened SVD headroom with norm and squared-norm explicitly separated.

    With W.T W=A and S=W*pre*W^-1, an unrestricted rank-nc A-coarse
    correction Q gives min ||(I-Q)S||_2=sigma_(nc+1)(S).
    ONLY IF W*post*W^-1=S.T, the full symmetric two-grid optimum is its
    square: min ||S.T(I-Q)S||_2=sigma_(nc+1)^2. U[:, :nc] defines Q.
    This is an exact-coarse-solve diagnostic, NOT a V-cycle/time bound.
    """
    a=checked_csr(a,square=True);n=a.shape[0];estimated=dense_budget(n,max_dofs=max_dofs,max_bytes=max_bytes)
    if symmetry_error(a)>tol:raise ValueError('spectral diagnostic requires symmetric A')
    pre,post=np.asarray(pre,np.float64),np.asarray(post,np.float64)
    if pre.shape!=(n,n) or post.shape!=(n,n) or not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise ValueError('invalid error-propagation maps')
    if not p_candidates:raise ValueError('at least one P required')
    dims={p.shape[1] for p in p_candidates.values()}
    if len(dims)!=1:raise ValueError('coarse dimension must be identical')
    nc=dims.pop()
    if not 0<nc<n:raise ValueError('invalid coarse dimension')
    w=la.cholesky(a.toarray(),lower=False)
    def whiten(s):return la.solve_triangular(w.T,(w@s).T,lower=True).T
    prew,postw=whiten(pre),whiten(post)
    defect=float(la.norm(postw-prew.T)/max(la.norm(prew),1e-300));adjoint=defect<=tol
    u,s,_=la.svd(prew,full_matrices=False)
    one=float(s[nc]);qopt=u[:,:nc]
    def measure(q):
        one_map=prew-q@(q.T@prew)
        full_map=postw@one_map
        return dict(one_sided_A_norm=float(la.svdvals(one_map)[0]),
                    deployed_two_grid_A_norm=float(la.svdvals(full_map)[0]))
    values={}
    for name,p in p_candidates.items():
        p=checked_csr(p)
        if p.shape!=(n,nc):raise ValueError('P shape mismatch')
        q,r=la.qr(w@p.toarray(),mode='economic')
        if np.linalg.matrix_rank(r)!=nc:raise ValueError('rank-deficient P')
        values[name]=measure(q)
    optimal=measure(qopt)
    valid=adjoint and np.isclose(optimal['deployed_two_grid_A_norm'],one**2,rtol=1e-6,atol=1e-10)
    return dict(status='computed',diagnostic_only=True,coarse_dimension=nc,estimated_workspace_bytes=estimated,
        adjoint_defect=defect,adjoint_verified=adjoint,one_sided_optimal_A_norm=one,
        one_sided_optimal_A_norm_squared=one**2,
        symmetric_two_grid_optimal_A_norm=one**2 if valid else None,
        deployed_optimality_applicable=bool(valid),unrestricted_space_measured=optimal,candidates=values,
        identity='q_one=sigma[nc]; q_symmetric_two_grid=q_one**2 only for verified adjoint post',
        scope='fixed complete pre/post schedule; exact coarse solve; arbitrary rank-nc space, no sparsity/injection constraints; not a time or multilevel bound')


def paired_speedups(reference,candidate,*,seed=1,bootstrap=2000):
    """One time per unique operator (already reduced over repeats/RHS)."""
    keys=sorted(set(reference)&set(candidate))
    common=[k for k in keys if reference[k]['success'] and candidate[k]['success']]
    new=[k for k in keys if reference[k]['success'] and not candidate[k]['success']]
    rescue=[k for k in keys if not reference[k]['success'] and candidate[k]['success']]
    if not common:return dict(common=[],geometric_speedup=None,ci95=None,new_failures=new,rescues=rescue)
    ratios=np.asarray([reference[k]['seconds']/candidate[k]['seconds'] for k in common])
    if not np.isfinite(ratios).all() or np.any(ratios<=0):raise ValueError('invalid successful times')
    log=np.log(ratios);boot=np.random.default_rng(seed).choice(log,size=(bootstrap,len(log))).mean(1)
    return dict(common=common,geometric_speedup=float(np.exp(log.mean())),ci95=np.exp(np.quantile(boot,[.025,.975])).tolist(),
                new_failures=new,rescues=rescue,scope='operator bootstrap; not timing-noise coverage or a final certificate')


def factorial_interaction(t00,t01,t10,t11):
    times=np.asarray([t00,t01,t10,t11],np.float64)
    if not np.isfinite(times).all() or np.any(times<=0):raise ValueError('positive common-success times required')
    pc=t00/t01;pn=t10/t11
    return dict(P_gain_under_C=pc,P_gain_under_N=pn,log_interaction=float(np.log(pn)-np.log(pc)),
                HSP_speedup_vs_best_other=float(min(t00,t01,t10)/t11),
                scope='same operator/RHS/coarsening/frozen S and P; positive log interaction means complementary time gains')


def measured_break_even(extra_setup,per_rhs_saved):
    if not np.isfinite(extra_setup+per_rhs_saved):raise ValueError('nonfinite timing')
    if per_rhs_saved<=0:return None
    return max(1,int(math.floor(max(extra_setup,0.)/per_rhs_saved))+1)
