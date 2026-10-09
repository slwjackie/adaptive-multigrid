"""Factorial temporal actions: classical P reuse/rebuild x classical/learned S.

Never constructs learned P. H_P/H_SP checkpoints and inherited neural transfer
are rejected. All actions solve the CURRENT A, with fresh Ac/LU when A changes.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
import numpy as np

from ...hierarchy import build_fixed_hierarchy
from ...strategy import get_strategy
from ..banks import Stats, prepare_smoother_bank
from ..strong import StrongSelection
from ..world_model.backend import SequenceBackend, Bank, levels

ACTIONS = ('REBUILD_C','REUSE_C','REBUILD_HS','REUSE_HS')
NEURAL_ACTIONS = (2,3)


@dataclass(frozen=True)
class SmoothingBank(Bank):
    # Both roots have EXACTLY the same P/Ac/factors; only the S overlay differs.
    classical_root: object = None
    smoother_root: object = None


class HSSmoothingBackend(SequenceBackend):
    actions = ACTIONS

    def __init__(self,cfg,rules,expert=None,*,max_complexity=8.):
        if cfg.use_transfer:raise ValueError('learned transfer is excluded from H_S thesis')
        if expert is not None and (expert.metadata.get('training_branch')!='H_S' or
                expert.metadata.get('optimizer_updates',0)<1):
            raise ValueError('genuinely trained H_S checkpoint required')
        super().__init__(cfg,rules,expert,expert_branch='H_S',max_complexity=max_complexity)
        plans=set(dict(rules.strategy_by_rule).values())|{rules.fallback_strategy_name}
        self.fixed_plan=next(iter(plans)) if len(plans)==1 else None
        self.fixed_rules_digest=rules.digest()

    def select(self,s):
        if self.fixed_plan is None:
            return super().select(s)  # optional robustness audit only
        # A fixed classical plan does not require a classifier/feature scan.
        chosen=replace(self.cfg,mg=replace(self.cfg.mg,strategy_name=self.fixed_plan))
        selection=StrongSelection(self.fixed_plan,'fixed_tuned',{},
            {'fallback_for_coverage':False,'fixed_global_plan':True},0.,self.fixed_rules_digest)
        return selection,chosen

    def compatible(self,s,bank,plan):
        return (isinstance(bank,SmoothingBank) and not bank.neural_transfer and
                super().compatible(s,bank,plan))

    def available(self,s,bank,selection=None):
        selection=selection or self.select(s)[0]
        compatible=self.compatible(s,bank,selection.strategy_name)
        return np.array([True,compatible,self.expert is not None,compatible and self.expert is not None],bool)

    def build(self,s,old,action,selection,cfg):
        if action not in ACTIONS:raise ValueError('unknown H_S-only action')
        reuse=action in ('REUSE_C','REUSE_HS');neural=action in ('REBUILD_HS','REUSE_HS')
        if neural and self.expert is None:raise ValueError('no trained smoother')
        if reuse and not self.compatible(s,old,selection.strategy_name):raise ValueError('cannot reuse incompatible P')
        if neural and self.expert.signature()!=self.expert_signature:
            raise ValueError('expert changed after temporal collection')
        stats=Stats();same=reuse and old.matrix_digest==s.matrix_digest
        if same:
            base=old.classical_root;smoothed=old.smoother_root
        elif reuse:
            # _updated(refresh=False) copies P, recomputes every actual Ac,
            # and factorizes current blocks; it never calls a transfer network.
            base=self._updated(s,old,cfg,stats,False);smoothed=None
        else:
            base=build_fixed_hierarchy(s.a,s.shape,get_strategy(selection.strategy_name),cfg.mg,stats)
            smoothed=None
        if any(getattr(l,'learned_transfer',False) for l in levels(base)):
            raise ValueError('learned P is forbidden in this workflow')
        if neural and smoothed is None:
            hcfg=replace(cfg,mode='research',branch='H_S',use_transfer=False,use_smoother=True,
                         spatial=False,gate_mode='open')
            smoothed=prepare_smoother_bank(base,self.expert,hcfg,stats)
        root=smoothed if neural else base
        complexity=sum(l.a.nnz for l in levels(root))/s.a.nnz
        if not np.isfinite(complexity) or complexity>self.max_complexity:
            raise ValueError('hierarchy complexity safety cap')
        bank=SmoothingBank(root,s.matrix_digest,s.topology_key,selection.strategy_name,self.contract,
                           old.p_age+1 if reuse else 0,False,base,smoothed)
        return bank,dict(stats.to_dict(),operator_complexity=complexity,exact_matrix_cache_hit=bool(same),
             numeric_refactorized=not same,classical_P_only=True,smoother_enabled=neural,
             temporal_scope='current A; classical interpolation only')
