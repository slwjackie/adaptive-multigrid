"""Small deployable generators for EM imitation and learned line relaxation.

The transfer head uses signed affine corrections on the bilinear support.
EM targets can have negative weights, unlike a probability-valued head.
Online generation never invokes energy minimization.
"""
from dataclasses import asdict

import torch

from .models import Components, TransferNet, TinyNet, CONTRACT, DETECTOR_INPUTS, CONTROLLER_INPUTS
from .research_transfer import TransferComplexityCaps


class AffineTransferCNN(TransferNet):
    training_only = False
    support = 'support_preserving'
    parameterization = 'affine'
    reference = 'actual'

    def __init__(self, hidden=8, body_kind='residual5', complexity_caps=None, version=1):
        if version != 1 or type(hidden) is not int or hidden < 1:
            raise ValueError('Invalid EM affine transfer specification')
        super().__init__(hidden=hidden, candidates=16, body_kind=body_kind)
        self.complexity_caps = asdict(TransferComplexityCaps(**complexity_caps)
                                     if complexity_caps else TransferComplexityCaps())

    def em_affine_transfer_spec(self):
        return dict(hidden=self.hidden, body_kind=self.body_kind,
                    complexity_caps=dict(self.complexity_caps), version=1)


def make_em_line_components(*, hidden=8, seed=20261021,
                            transfer_body_kind='residual5', smoother_body_kind='residual5'):
    from .line_smoother import LineSmootherCNN
    torch.manual_seed(seed)
    model = Components(LineSmootherCNN(hidden=hidden, body_kind=smoother_body_kind),
                       AffineTransferCNN(hidden=hidden, body_kind=transfer_body_kind),
                       TinyNet(DETECTOR_INPUTS, 1), TinyNet(CONTROLLER_INPUTS, 4))
    model.metadata = dict(contract=CONTRACT, role='student', em_line_study=True,
        smoother_trained=False, transfer_trained=False, detector_trained=False,
        controller_trained=False, certificate=dict(validated=False), final_test_seen=False,
        online_em=False, transfer_reference='actual_student_A',
        smoother='level-conditioned shared CNN line factors; shared pre/post coefficients')
    return model.eval()
