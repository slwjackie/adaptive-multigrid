import numpy as np
import pytest
from adaptive_mg.native_stencil import native_available,NativeStencil
from adaptive_mg.hierarchy import stencil_from_coefficients
from adaptive_mg import *


@pytest.mark.parametrize('shape',[(7,7),(15,7),(31,31),(63,63)])
def test_native_csr_equivalence(shape):
    if not native_available(): pytest.skip('optional native library not built')
    rng=np.random.default_rng(1);d=rng.normal(size=(1,9,*shape));diag=rng.uniform(.1,2.,size=shape[0]*shape[1]);r=rng.normal(size=diag.size);gain=np.array([.35])
    kernel=NativeStencil.from_directions(d,diagonal=diag,gains=gain)
    csr=stencil_from_coefficients(d[0],gain[0],diag,shape)
    np.testing.assert_allclose(kernel.apply(r)[:,0],csr@r,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('kind',['dense','wrong_shape','nan','negative_diagonal','nonsymmetric'])
def test_invalid_matrix(problem,kind):
    a,b,n=problem
    if kind=='dense':a=a.toarray()
    if kind=='wrong_shape':a=a[:4,:4]
    if kind=='nan':a.data[0]=float('nan')
    if kind=='negative_diagonal':a.setdiag(-1.)
    if kind=='nonsymmetric':a=a.tolil();a[0,1]=500;a=a.tocsr()
    with pytest.raises((ValueError,TypeError)):PreparedTemporalMGSolver(a,n,config=MGConfig(mode='classical'))


def test_zero_operator_root_size_and_empty_batch(model):
    a=assemble_stiffness(DiffusionCase(n=3,epsilon=1.));p=PreparedTemporalMGSolver(a,3,model,MGConfig(mode='fixed',fixed_k=2))
    r=p.solve(np.ones(9));assert r.converged
    assert p.solve_many(np.empty((0,9)))==[]
