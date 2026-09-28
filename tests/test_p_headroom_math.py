import numpy as np
import pytest
import scipy.sparse as sp

from adaptive_mg.v67.p_headroom import (
    AffineSupport, energy_minimize, least_squares, slow_vectors,
    spectral_headroom, DiagnosticBudgetError, paired_speedups,
    factorial_interaction, measured_break_even,
)


def problem():
    a=sp.diags([-np.ones(4),2*np.ones(5),-np.ones(4)],[-1,0,1],format='csr')
    p=sp.csr_matrix([[.5,0],[1.,0],[.5,.5],[0,1],[0,.5]])
    return a,p


def test_dirichlet_parent_rows_not_forced_to_constant_one():
    a,p=problem();c=AffineSupport.build(p,[1,3])
    q=np.ones(c.dofs)*.12;v=c.matrix(q);report=c.validate(v)
    np.testing.assert_allclose(np.asarray(v.sum(1)).ravel(),[.5,1,1,1,.5])
    assert report['external_support']==0 and c.dofs==1
    with pytest.raises(ValueError,match='fixed-row'):
        AffineSupport.build(p,[1,3],fine_targets=np.full((5,1),2.))


def test_modes_are_hard_constraints_and_infeasible_support_rejected():
    _,p=problem();bc=np.array([[1.,0],[1.,1]])
    c=AffineSupport.build(p,[1,3],bc,p@bc)
    assert c.dofs==0
    np.testing.assert_allclose(c.matrix().toarray(),p.toarray())
    bad=np.asarray(p@bc);bad[0,1]=1
    with pytest.raises(ValueError,match='infeasible'):AffineSupport.build(p,[1,3],bc,bad)


def test_sparse_energy_minimization_decreases_trace_and_preserves_injection():
    a,p=problem();p[2,0]=.9;p[2,1]=.1
    c=AffineSupport.build(p,[1,3]);opt,report=energy_minimize(a,c)
    assert report['optimizer_converged'] and report['energy_after']<report['energy_before']
    np.testing.assert_allclose(opt.toarray()[2],[.5,.5],atol=1e-8)
    c.validate(opt)


@pytest.mark.parametrize('weighting',['uniform','energy'])
def test_ls_is_constrained_and_records_vector_generation_cost(weighting):
    a,p=problem();c=AffineSupport.build(p,[1,3])
    v,st=slow_vectors(a,lambda e:e-.3*(a@e),count=5,sweeps=3,seed=82)
    result,report=least_squares(a,c,v,[1,3],weighting=weighting)
    c.validate(result);assert st['smoothing_applications']==15
    assert st['test_vector_seconds']>=0 and report['ls_seconds']>=0
    v2,_=slow_vectors(a,lambda e:e-.3*(a@e),count=5,sweeps=3,seed=82)
    np.testing.assert_array_equal(v,v2)


def test_spectral_optimum_separates_one_sided_norm_and_full_symmetric_norm():
    a=sp.diags([1.,2.,3.,4.],format='csr');s=np.diag([.9,.8,.2,.1])
    pc=sp.csr_matrix(np.eye(4)[:,2:3]);po=sp.csr_matrix(np.eye(4)[:,:1])
    r=spectral_headroom(a,s,s,{'classical':pc,'optimal':po})
    assert r['adjoint_verified'] and r['deployed_optimality_applicable']
    assert r['one_sided_optimal_A_norm']==pytest.approx(.8)
    assert r['symmetric_two_grid_optimal_A_norm']==pytest.approx(.64)
    assert r['candidates']['classical']['deployed_two_grid_A_norm']==pytest.approx(.81)
    assert r['candidates']['optimal']['deployed_two_grid_A_norm']==pytest.approx(.64)


def test_nonnormal_adjoint_pair_and_no_false_bound_when_post_differs():
    a=sp.diags([2.,3.,4.],format='csr');pre=np.array([[.1,.3,0],[0,.2,.2],[0,0,.4]])
    post=np.linalg.solve(a.toarray(),pre.T@a.toarray());p=sp.csr_matrix(np.eye(3)[:,:1])
    r=spectral_headroom(a,pre,post,{'P':p})
    assert r['deployed_optimality_applicable']
    r=spectral_headroom(a,pre,pre,{'P':p})
    assert not r['adjoint_verified'] and r['symmetric_two_grid_optimal_A_norm'] is None


def test_dense_oracle_budget_and_rank_checks():
    a=sp.eye(5,format='csr');p=sp.csr_matrix(np.eye(5)[:,:2]);s=.5*np.eye(5)
    with pytest.raises(DiagnosticBudgetError):spectral_headroom(a,s,s,{'P':p},max_dofs=4)
    with pytest.raises(DiagnosticBudgetError):spectral_headroom(a,s,s,{'P':p},max_bytes=1)
    with pytest.raises(ValueError,match='rank'):
        spectral_headroom(a,s,s,{'P':sp.csr_matrix(np.ones((5,2)))})
    with pytest.raises(Exception):spectral_headroom(-a,s,s,{'P':p})


def test_no_cherry_picking_failed_operator_and_factorial_interaction():
    c={'a':dict(success=True,seconds=2.),'b':dict(success=False,seconds=.001)}
    h={'a':dict(success=True,seconds=1.),'b':dict(success=True,seconds=.1)}
    r=paired_speedups(c,h,bootstrap=50)
    assert r['geometric_speedup']==2. and r['common']==['a'] and r['rescues']==['b']
    f=factorial_interaction(10,8,7,4)
    assert f['log_interaction']==pytest.approx(np.log(1.4))
    assert f['HSP_speedup_vs_best_other']==pytest.approx(1.75)
    assert measured_break_even(1.2,.5)==3 and measured_break_even(1.2,-.5) is None


def test_headroom_source_manifest():
    from pathlib import Path
    import hashlib
    root=Path(__file__).resolve().parents[1]
    for line in (root/'P_HEADROOM_SOURCE.sha256').read_text().splitlines():
        expected,path=line.split('  ',1)
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected,path
