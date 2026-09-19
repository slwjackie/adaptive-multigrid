import numpy as np
import pytest
import torch
from adaptive_mg import *

@pytest.fixture(autouse=True, scope='session')
def single_thread():
    torch.set_num_threads(1)

@pytest.fixture
def problem():
    a=assemble_stiffness(DiffusionCase(n=15,epsilon=1.,angle_deg=0.,contrast=1.))
    b=np.random.default_rng(7).standard_normal(a.shape[0])
    return a,b,15

@pytest.fixture
def model():
    torch.manual_seed(7)
    return TemporalComponents.create(hidden=8).eval()
