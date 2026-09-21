"""Existing CNN-P architecture under the same new sparse budget as graph-P."""
from dataclasses import asdict
from .models import TransferNet
from .research_transfer import TransferComplexityCaps


class ControlledTransferCNN(TransferNet):
    training_only=False
    def __init__(self, hidden=16, body_kind='residual5', support='standard', complexity_caps=None, projection_version=None):
        if support not in ('standard','expanded','support_preserving'):raise ValueError('Unknown support')
        super().__init__(hidden=hidden,candidates=36 if support=='expanded' else 16,body_kind=body_kind)
        self.support=support
        self.projection_version=2 if support=='support_preserving' else 1
        if projection_version is not None and projection_version != self.projection_version:
            raise ValueError('projection version differs from checkpoint support')
        self.complexity_caps=asdict(TransferComplexityCaps(**complexity_caps) if complexity_caps else TransferComplexityCaps())
    def research_transfer_spec(self):
        return dict(hidden=self.hidden,body_kind=self.body_kind,support=self.support,
                    complexity_caps=dict(self.complexity_caps),projection_version=self.projection_version)
