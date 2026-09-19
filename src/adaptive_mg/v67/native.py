"""Optional native selected rows and fused restriction/block statistics."""
import ctypes as ct
import numpy as np
from ..native_stencil import native_available,load_native

_DP=ct.POINTER(ct.c_double); _IP=ct.POINTER(ct.c_int64)
def _d(a): return None if a is None else a.ctypes.data_as(_DP)
def _i(a): return None if a is None else a.ctypes.data_as(_IP)

def apply_rows(native, residual, rows):
    lib=native._library
    f=lib.anmg_stencil9_rows
    f.argtypes=[_DP,_DP,_IP,_DP,ct.c_int64]+[ct.c_int]*4; f.restype=ct.c_int
    r=np.ascontiguousarray(residual,np.float64)
    rows=np.ascontiguousarray(rows,np.int64)
    out=np.empty(len(rows),np.float64)
    _,_,nx,ny=native.coefficients.shape
    code=f(_d(native.coefficients),_d(r),_i(rows),_d(out),len(rows),nx,ny,native.threads,native.parallel_min)
    if code: raise ValueError(f'selected stencil failed: {code}')
    return out

def block_statistics(r,shape,max_side,p=None):
    nx,ny=shape; bx,by=min(nx,max_side),min(ny,max_side)
    r=np.ascontiguousarray(r,np.float64).reshape(-1)
    if not np.isfinite(r).all(): raise ValueError('nonfinite detector input')
    coarse=None if p is None else np.empty(p.shape[1],np.float64)
    feat=np.zeros((bx*by,6),np.float64)
    if native_available():
        f=load_native().anmg_restrict_features
        f.argtypes=[_IP,_IP,_DP,_DP,_DP,_DP]+[ct.c_int]*5;f.restype=ct.c_int
        ip=None if p is None else p.indptr.astype(np.int64,copy=False)
        col=None if p is None else p.indices.astype(np.int64,copy=False)
        val=None if p is None else np.ascontiguousarray(p.data,np.float64)
        code=f(_i(ip),_i(col),_d(val),_d(r),_d(coarse),_d(feat),nx,ny,0 if p is None else p.shape[1],bx,by)
        if code: raise ValueError('native restriction/features failure')
    else:
        grid=r.reshape(shape)
        ids=(np.arange(nx)[:,None]*bx//nx)*by+(np.arange(ny)[None,:]*by//ny)
        ids=ids.ravel()
        dx=np.zeros(shape);dy=np.zeros(shape)
        dx[:-1]=(grid[:-1]-grid[1:])**2;dy[:,:-1]=(grid[:,:-1]-grid[:,1:])**2
        for k,v in enumerate((r*r,np.abs(r),np.abs(r),dx.ravel(),dy.ravel(),np.ones_like(r))):
            if k==2: np.maximum.at(feat[:,k],ids,v)
            else: feat[:,k]=np.bincount(ids,weights=v,minlength=bx*by)
        if p is not None: coarse[:]=p.T@r
    return coarse,feat,(bx,by)
