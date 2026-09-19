"""Reproducible diffusion families and scale-invariant operator-disjoint splits."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from .pde import DiffusionCase, assemble_stiffness, case_suite
from .provenance import operator_digest


@dataclass
class Example:
    name: str
    case: DiffusionCase
    a: object
    b: np.ndarray
    exact: np.ndarray
    digest: str
    group_digest: str
    seed: int
    scale: float

    @property
    def n(self): return self.case.n

    def manifest(self):
        return dict(name=self.name,case=asdict(self.case),operator_digest=self.digest,
                    normalized_operator_digest=self.group_digest,seed=self.seed,scale=self.scale)


def make_examples(sizes, *, per_size=6, seed=7):
    rng=np.random.default_rng(seed); examples=[]
    if per_size < 1: raise ValueError('per_size must be positive')
    for n in sizes:
        for j in range(per_size):
            family=j%6
            name=case_suite(n)[family][0]
            eps=float(10**rng.uniform(-3,-.5))
            angle=float(rng.uniform(20,65)); contrast=float(10**rng.uniform(1,3))
            if family==0:
                # Vary the normalized operator as well as its scale; identical
                # Poisson templates must not masquerade as unseen operators.
                case=DiffusionCase(n=n,epsilon=float(rng.uniform(.82,1.)),angle_deg=angle,contrast=1.)
                name='near_isotropic'
            elif family==1:
                case=DiffusionCase(n=n,epsilon=eps,angle_deg=float(rng.choice([0,90])),contrast=1.)
            elif family==2:
                case=DiffusionCase(n=n,epsilon=eps,angle_deg=angle,contrast=1.)
            else:
                center=float(rng.uniform(.38,.62)); width=float(rng.uniform(.13,.24))
                case=DiffusionCase(n=n,epsilon=eps,angle_deg=angle,contrast=contrast,
                    pattern={3:'local_patch',4:'checkerboard',5:'channel'}[family],
                    patch_bounds=(center-width,center+width,.25,.75),
                    checker_cells=int(rng.choice([3,4,5])),channel_width=float(rng.uniform(.06,.13)))
            scale=float(10**rng.uniform(-.5,.5)); a=scale*assemble_stiffness(case)
            c=np.arange(1,n+1)/(n+1); x,y=np.meshgrid(c,c,indexing='ij')
            exact=np.sin(np.pi*x)*np.sin(np.pi*y)
            # Mix frequencies so the initial residual is not just one smooth mode.
            for _ in range(4):
                k,l=rng.integers(1,min(n,12)+1,size=2)
                exact+=rng.uniform(-.3,.3)*np.sin(k*np.pi*x)*np.sin(l*np.pi*y)
            exact=exact.ravel(); b=np.asarray(a@exact)
            examples.append(Example(f'{name}_{n}_{j}',case,a,b,exact,operator_digest(a),operator_digest(a,scale_invariant=True),seed,scale))
    return examples


def fixed_benchmark_examples(sizes):
    examples=[]
    for n in sizes:
        c=np.arange(1,n+1)/(n+1); x,y=np.meshgrid(c,c,indexing='ij')
        exact=(np.sin(np.pi*x)*np.sin(np.pi*y)+.2*np.sin(5*np.pi*x)*np.sin(3*np.pi*y)).ravel()
        for name,case in case_suite(n):
            a=assemble_stiffness(case)
            examples.append(Example(name,case,a,np.asarray(a@exact),exact.copy(),operator_digest(a),operator_digest(a,scale_invariant=True),-1,1.))
    return examples


def split_manifest(**splits):
    seen={}
    for split,examples in splits.items():
        for e in examples:
            if e.group_digest in seen and seen[e.group_digest]!=split:
                raise ValueError(f'normalized-operator leakage: {split} / {seen[e.group_digest]}')
            seen[e.group_digest]=split
    return dict(normalized_operator_disjoint=True,splits={s:[e.manifest() for e in es] for s,es in splits.items()},unique_operator_groups=len(seen))
