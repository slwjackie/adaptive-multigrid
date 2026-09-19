"""Existing CNN-P architecture under the same new sparse budget as graph-P."""
from dataclasses import asdict
from .models import TransferNet
from .research_transfer import TransferComplexityCaps


class ControlledTransferCNN(TransferNet):
    training_only=False
    def __init__(self, hidden=16, body_kind='residual5', support='standard', complexity_caps=None):
        if support not in ('standard','expanded'):raise ValueError('Unknown support')
        super().__init__(hidden=hidden,candidates=16 if support=='standard' else 36,body_kind=body_kind)
        self.support=support
        self.complexity_caps=asdict(TransferComplexityCaps(**complexity_caps) if complexity_caps else TransferComplexityCaps())
    def research_transfer_spec(self):
        return dict(hidden=self.hidden,body_kind=self.body_kind,support=self.support,
                    complexity_caps=dict(self.complexity_caps))
