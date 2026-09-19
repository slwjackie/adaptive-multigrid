"""CPU FP64 sparse autograd with sparse-value derivatives (no dense N x N grads).

SciPy does the numerical products/factorizations. PyTorch differentiates the
VALUES on a fixed symbolic sparsity pattern. Symbolic zeros are retained so a
zero-initialized expanded-support learned P still has meaningful gradients.
This is a first-order differentiation backend; higher-order grads are not used.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch


def _sample(a,pattern):
    rows=np.repeat(np.arange(pattern.shape[0]),np.diff(pattern.indptr))
    return np.asarray(a.tocsr()[rows,pattern.indices]).ravel()

def _csr(values,pattern):
    return sp.csr_matrix((values.detach().cpu().double().numpy(),pattern.indices,pattern.indptr),shape=pattern.shape)

class _Apply(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,x,pattern):
        a=_csr(values,pattern);ctx.a=a;ctx.pattern=pattern;ctx.save_for_backward(values,x)
        out=np.asarray(a@x.detach().cpu().double().numpy())
        return torch.tensor(out,dtype=x.dtype,device=x.device)
    @staticmethod
    def backward(ctx,g):
        values,x=ctx.saved_tensors;p=ctx.pattern
        rows=np.repeat(np.arange(p.shape[0]),np.diff(p.indptr));cols=p.indices
        gn=g.detach().cpu().double().numpy();xn=x.detach().cpu().double().numpy()
        gv=gn[rows]*xn[cols]
        if gv.ndim>1:gv=gv.sum(axis=1)
        gx=np.asarray(ctx.a.T@gn)
        return torch.tensor(gv,dtype=values.dtype,device=values.device),torch.tensor(gx,dtype=x.dtype,device=x.device),None

class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,av,bv,ap,bp,cp):
        a=_csr(av,ap);b=_csr(bv,bp);ctx.a=a;ctx.b=b;ctx.ap=ap;ctx.bp=bp;ctx.cp=cp
        ctx.save_for_backward(av,bv)
        return torch.tensor(_sample(a@b,cp),dtype=av.dtype,device=av.device)
    @staticmethod
    def backward(ctx,g):
        av,bv=ctx.saved_tensors;gg=_csr(g,ctx.cp)
        da=_sample(gg@ctx.b.T,ctx.ap);db=_sample(ctx.a.T@gg,ctx.bp)
        return torch.tensor(da,dtype=av.dtype,device=av.device),torch.tensor(db,dtype=bv.dtype,device=bv.device),None,None,None

class _Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,b,pattern):
        a=_csr(values,pattern);lu=sla.splu(a.tocsc());x=lu.solve(b.detach().cpu().double().numpy())
        ctx.lu=lu;ctx.x=x;ctx.pattern=pattern;ctx.save_for_backward(values,b)
        return torch.tensor(x,dtype=b.dtype,device=b.device)
    @staticmethod
    def backward(ctx,g):
        values,b=ctx.saved_tensors;p=ctx.pattern
        gb=ctx.lu.solve(g.detach().cpu().double().numpy(),trans='T')
        rows=np.repeat(np.arange(p.shape[0]),np.diff(p.indptr));gv=-gb[rows]*ctx.x[p.indices]
        if gv.ndim>1:gv=gv.sum(1)
        return torch.tensor(gv,dtype=values.dtype,device=values.device),torch.tensor(gb,dtype=b.dtype,device=b.device),None

@dataclass
class SparseTensor:
    pattern: sp.csr_matrix
    values: torch.Tensor
    cache: dict=field(default_factory=dict)
    @classmethod
    def from_scipy(cls,a,requires_grad=False):
        a=a.tocsr(copy=True);a.sum_duplicates();a.sort_indices()
        values=torch.tensor(a.data,dtype=torch.float64,requires_grad=requires_grad)
        p=a.copy();p.data=np.ones_like(a.data)
        return cls(p,values)
    @classmethod
    def from_coo(cls,rows,cols,values,shape):
        key=np.asarray(rows,dtype=np.int64)*shape[1]+np.asarray(cols,dtype=np.int64)
        unique,inverse=np.unique(key,return_inverse=True)
        p=sp.coo_matrix((np.ones(len(unique)),(unique//shape[1],unique%shape[1])),shape=shape).tocsr()
        v=torch.zeros(len(unique),dtype=values.dtype,device=values.device).index_add(0,torch.tensor(inverse,device=values.device),values)
        return cls(p,v)
    @property
    def shape(self):return self.pattern.shape
    @property
    def rows(self):return np.repeat(np.arange(self.shape[0]),np.diff(self.pattern.indptr))
    @property
    def cols(self):return self.pattern.indices
    def numpy(self):return _csr(self.values,self.pattern)
    def apply(self,x):return _Apply.apply(self.values,x,self.pattern)
    def solve(self,b):return _Solve.apply(self.values,b,self.pattern)
    def transpose(self):return SparseTensor.from_coo(self.cols,self.rows,self.values,(self.shape[1],self.shape[0]))
    def product(self,other):
        pattern=(self.pattern@other.pattern).tocsr();pattern.sort_indices();pattern.data[:]=1
        v=_Product.apply(self.values,other.values,self.pattern,other.pattern,pattern)
        return SparseTensor(pattern,v)
    def symmetrize(self):
        return SparseTensor.from_coo(np.r_[self.rows,self.cols],np.r_[self.cols,self.rows],
            torch.cat((self.values*.5,self.values*.5)),self.shape)
    def diagonal(self):
        mask=self.rows==self.cols
        return torch.zeros(self.shape[0],dtype=self.values.dtype).index_add(0,torch.tensor(self.rows[mask]),self.values[torch.tensor(mask)])
    def slice(self,rows,cols):
        """Sparse submatrix; index_select connects values to the original graph."""
        rows=np.asarray(rows,np.int64);cols=np.asarray(cols,np.int64)
        rr=np.full(self.shape[0],-1,np.int64);cc=np.full(self.shape[1],-1,np.int64)
        rr[rows]=np.arange(len(rows));cc[cols]=np.arange(len(cols))
        r=rr[self.rows];c=cc[self.cols];sel=(r>=0)&(c>=0)
        return SparseTensor.from_coo(r[sel],c[sel],self.values[torch.tensor(sel)],(len(rows),len(cols)))
