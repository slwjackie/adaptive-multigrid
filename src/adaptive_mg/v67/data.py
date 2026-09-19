"""Seven families; data manifest groups uniform-scaled operators together."""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from ..data import Example,make_examples,split_manifest
from ..pde import DiffusionCase,assemble_stiffness
from ..provenance import operator_digest

@dataclass(frozen=True)
class VariableCase:
    n:int
    epsilon:float
    angle_deg:float
    amplitude:float
    frequency:float
    phase:float
    pattern:str='variable_coefficient'


def variable_stiffness(case):
    """P1 FE assembly of -div(a(x,y)*K*grad u), a>0, same mesh/boundary."""
    n=case.n;i,j=np.meshgrid(np.arange(n+1),np.arange(n+1),indexing='ij');i=i.ravel();j=j.ravel()
    angle=np.deg2rad(case.angle_deg);r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    k=r@np.diag([1.,case.epsilon])@r.T
    rows=[];cols=[];vals=[]
    for offsets,grads in ((np.array([[0,0],[1,0],[1,1]]),np.array([[-1,0],[1,-1],[0,1.]])),
                          (np.array([[0,0],[1,1],[0,1]]),np.array([[0,-1],[1,0],[-1,1.]]))):
        x=(i+offsets[:,0].mean())/(n+1);y=(j+offsets[:,1].mean())/(n+1)
        coeff=np.exp(case.amplitude*np.sin(2*np.pi*case.frequency*x+case.phase)*np.cos(2*np.pi*y-case.phase))
        local=.5*coeff[:,None,None]*(grads@k@grads.T)[None,:,:]
        gi=i[:,None]+offsets[None,:,0];gj=j[:,None]+offsets[None,:,1]
        valid=(gi>=1)&(gi<=n)&(gj>=1)&(gj<=n);idx=(gi-1)*n+(gj-1)
        mask=valid[:,:,None]&valid[:,None,:]
        rr=np.broadcast_to(idx[:,:,None],local.shape);cc=np.broadcast_to(idx[:,None,:],local.shape)
        rows.append(rr[mask]);cols.append(cc[mask]);vals.append(local[mask])
    a=sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(n*n,n*n)).tocsr()
    a=(.5*(a+a.T)).tocsr();a.eliminate_zeros();return a


def dataset(sizes,per_size=56,seed=71):
    """Default research train: 4 sizes x 56 = 224 distinct operators."""
    if per_size<7: # smoke allows fewer; never calls it full research coverage
        return make_examples(sizes,per_size=per_size,seed=seed)
    rng=np.random.default_rng(seed);out=[]
    for n in sizes:
        standard=iter(make_examples([n],per_size=per_size+6,seed=seed+100*n))
        for j in range(per_size):
            if j%7!=6:
                e=next(standard);e.name=f'{e.name}_seed{seed}';out.append(e);continue
            case=VariableCase(n,float(10**rng.uniform(-2,0)),float(rng.uniform(0,90)),float(rng.uniform(.3,3)),float(rng.uniform(.7,3)),float(rng.uniform(0,6.28)))
            a=variable_stiffness(case);scale=float(10**rng.uniform(-1,1));a*=scale
            co=np.arange(1,n+1)/(n+1);x,y=np.meshgrid(co,co,indexing='ij')
            exact=(np.sin(np.pi*x)*np.sin(np.pi*y)+rng.uniform(.1,.3)*np.sin(7*np.pi*x)*np.sin(3*np.pi*y)).ravel()
            out.append(Example(f'variable_{n}_{j}_seed{seed}',case,a,np.asarray(a@exact),exact,
                operator_digest(a),operator_digest(a,scale_invariant=True),seed,scale))
    return out
