"""Small, operator-only predictors of two directional relaxation coefficients.

The existing EM-line CNN uses ten *spatial* stencil channels.  Here the twelve
inputs are deliberately different: sparse reductions of the complete actual
level operator, including couplings outside its nearest-neighbour stencil.
They do not consume a right-hand side, an exact solution, or a PDE family label.

For row i let d_i=A_ii>0, q_i=sum_{j!=i}|A_ij|/d_i.  With coordinates
(ix,iy)=(i//ny,i%ny), qx and qy retain couplings on the same x/y line,
respectively; qm retains couplings changing both coordinates.  Extraction is
O(nnz(A)+N) time and temporary storage.  Except for the two grid-size inputs,
the features are invariant under a positive scalar rescaling of A.

Predictors are fit only to training rows.  Their serialized standardizer pins
the feature version/order, training row count, mean and population deviation.
The regression prediction path uses NumPy only; importing this module also
imports Torch for the independent MLP class, but does not run a Torch model.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


FEATURE_VERSION = 'actual-level-summary-12-v1'
MODEL_VERSION = 'light-smoother-predictor-v1'
FEATURE_NAMES = (
    'mean_x_line_coupling', 'mean_y_line_coupling', 'mean_mixed_coupling',
    'directional_contrast', 'mean_diagonal_share', 'min_diagonal_share',
    'diagonal_cv', 'offdiag_row_cv', 'positive_offdiag_fraction',
    'log2_nx_plus_one', 'log2_ny_plus_one', 'relative_level',
)
FEATURE_DEFINITIONS = (
    'mean_i qx_i, qx_i=sum_{j!=i, jy=iy}|A_ij|/d_i',
    'mean_i qy_i, qy_i=sum_{j!=i, jx=ix}|A_ij|/d_i',
    'mean_i qm_i, qm_i=sum_{jx!=ix, jy!=iy}|A_ij|/d_i',
    '(mean(qx)-mean(qy))/(mean(qx)+mean(qy)); 0 if denominator=0',
    'mean_i 1/(1+q_i), q_i=sum_{j!=i}|A_ij|/d_i',
    'min_i 1/(1+q_i)',
    'population_std(d)/mean(d)',
    'population_std(q)/mean(q); 0 if mean(q)=0',
    'sum_{i!=j,A_ij>0}|A_ij| / sum_{i!=j}|A_ij|; 0 if denominator=0',
    'log2(nx+1)', 'log2(ny+1)', 'index/(total_levels-1)',
)


def feature_spec():
    return dict(version=FEATURE_VERSION, names=list(FEATURE_NAMES),
                definitions=list(FEATURE_DEFINITIONS),
                cost='O(nnz(A)+N) time and temporary storage; once per actual nonterminal level',
                standardization='training-only population mean/std; std<=1e-12 replaced by 1')


def _bounds(bounds):
    value = np.asarray(bounds, dtype=np.float64)
    if (value.shape != (2,) or not np.isfinite(value).all()
            or not 0 < value[0] <= 1 <= value[1] or value[0] >= value[1]):
        raise ValueError('Coefficient bounds must satisfy 0 < lower <= 1 <= upper and lower < upper')
    return tuple(float(v) for v in value)


def _matrix(values, columns, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] != columns or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a nonempty finite (rows,{columns}) matrix')
    return result


def extract_features(a, shape, index, total_levels):
    """Return twelve float64 summaries of the *actual* nonterminal A_l.

    Same-line coupling includes all within-line entries, rather than dropping
    wider Galerkin couplings.  No sparse matrix is densified.
    """
    if isinstance(shape, (int, np.integer)):
        shape = (int(shape), int(shape))
    if (len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer))
                               or n < 1 for n in shape)):
        raise ValueError('Expected positive integer grid dimensions')
    nx, ny = map(int, shape)
    if (isinstance(index, bool) or isinstance(total_levels, bool)
            or not isinstance(index, (int, np.integer))
            or not isinstance(total_levels, (int, np.integer))
            or total_levels < 2 or not 0 <= index < total_levels-1):
        raise ValueError('Features require a valid nonterminal level index')
    if not sp.issparse(a) or a.shape != (nx*ny, nx*ny):
        raise ValueError('Actual level matrix must be sparse and match the grid')
    a = a.tocsr(copy=True)
    a.sum_duplicates()
    if not np.isfinite(a.data).all():
        raise ValueError('Nonfinite actual level matrix')
    diagonal = np.asarray(a.diagonal(), dtype=np.float64)
    if not np.all(diagonal > 0):
        raise ValueError('Light smoother features require a positive diagonal')
    rows = np.repeat(np.arange(a.shape[0], dtype=np.int64), np.diff(a.indptr))
    off = rows != a.indices
    rows, cols, entries = rows[off], a.indices[off], a.data[off]
    absolute = np.abs(entries)
    normalized = absolute / diagonal[rows]
    same_y, same_x = cols % ny == rows % ny, cols // ny == rows // ny
    means = [float(normalized[mask].sum() / a.shape[0])
             for mask in (same_y, same_x, ~(same_x | same_y))]
    q = np.bincount(rows, weights=normalized, minlength=a.shape[0])
    share = 1. / (1. + q)
    directional_total = means[0] + means[1]
    off_total = float(absolute.sum())
    q_mean = float(q.mean())
    result = np.array([
        *means,
        (means[0]-means[1])/directional_total if directional_total else 0.,
        float(share.mean()), float(share.min()),
        float(diagonal.std()/diagonal.mean()), float(q.std()/q_mean) if q_mean else 0.,
        float(absolute[entries > 0].sum()/off_total) if off_total else 0.,
        np.log2(nx+1.), np.log2(ny+1.), index/(total_levels-1),
    ], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite light smoother feature summary')
    return result


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    count: int
    training_split: str = 'train'

    def __post_init__(self):
        mean, scale = np.asarray(self.mean, np.float64), np.asarray(self.scale, np.float64)
        if (mean.shape != (12,) or scale.shape != (12,) or not np.isfinite(mean).all()
                or not np.isfinite(scale).all() or not np.all(scale > 0)
                or type(self.count) is not int or self.count < 1 or self.training_split != 'train'):
            raise ValueError('Invalid training-only feature standardizer')
        # Own immutable arrays so a caller cannot silently mutate model inputs.
        mean, scale = mean.copy(), scale.copy()
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, 'mean', mean)
        object.__setattr__(self, 'scale', scale)

    @classmethod
    def fit(cls, features, *, split='train'):
        if split != 'train':
            raise ValueError('Standardization must use the training split only')
        x = _matrix(features, 12, 'Training features')
        scale = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(scale > 1e-12, scale, 1.), len(x))

    def transform(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != 12 or not np.isfinite(x).all():
            raise ValueError('Expected finite (...,12) feature rows')
        result = (x-self.mean)/self.scale
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite standardized features')
        return result

    def to_dict(self):
        return dict(feature_version=FEATURE_VERSION, feature_names=list(FEATURE_NAMES),
                    mean=self.mean.tolist(), scale=self.scale.tolist(), count=self.count,
                    training_split=self.training_split)

    @classmethod
    def from_dict(cls, value):
        if (set(value) != {'feature_version', 'feature_names', 'mean', 'scale', 'count', 'training_split'}
                or value['feature_version'] != FEATURE_VERSION
                or value['feature_names'] != list(FEATURE_NAMES)):
            raise ValueError('Feature version/order or standardizer schema changed')
        return cls(value['mean'], value['scale'], value['count'], value['training_split'])


class TinyMLP(nn.Module):
    """242 parameters; input is standardized z, output is bounded omega."""
    def __init__(self, bounds=(.5, 1.5)):
        super().__init__()
        self.bounds = _bounds(bounds)
        self.hidden = nn.Linear(12, 16, dtype=torch.float64)
        self.output = nn.Linear(16, 2, dtype=torch.float64)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, z):
        if z.ndim not in (1, 2) or z.shape[-1] != 12:
            raise ValueError('TinyMLP expects standardized (...,12) inputs')
        return torch.clamp(1. + self.output(torch.relu(self.hidden(z))), *self.bounds)


class MLPPredictor:
    kind = 'mlp'
    parameter_count = 242

    def __init__(self, model, scaler, metadata=None):
        if not isinstance(model, TinyMLP) or not isinstance(scaler, Standardizer):
            raise TypeError('MLP predictor requires TinyMLP and training Standardizer')
        self.model, self.scaler = model, scaler
        self.bounds, self.metadata = model.bounds, dict(metadata or {})

    def predict(self, features):
        z = self.scaler.transform(features)
        parameter = next(self.model.parameters())
        with torch.inference_mode():
            result = self.model(torch.as_tensor(z, dtype=parameter.dtype, device=parameter.device))
        values = result.detach().cpu().numpy().astype(np.float64, copy=True)
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite MLP coefficient prediction')
        return values

    def to_dict(self):
        return dict(version=MODEL_VERSION, kind=self.kind, feature_spec=feature_spec(),
                    parameter_count=self.parameter_count, bounds=list(self.bounds),
                    scaler=self.scaler.to_dict(), metadata=self.metadata,
                    state={k: v.detach().cpu().tolist() for k, v in self.model.state_dict().items()})

    def save(self, path):
        save_predictor(self, path)


class RidgePredictor:
    kind = 'regression'
    parameter_count = 26

    def __init__(self, W, c, scaler, bounds=(.5, 1.5), regularization=.01, metadata=None):
        self.W, self.c = np.asarray(W, np.float64).copy(), np.asarray(c, np.float64).copy()
        self.bounds = _bounds(bounds)
        if (self.W.shape != (2, 12) or self.c.shape != (2,)
                or not np.isfinite(self.W).all() or not np.isfinite(self.c).all()
                or not isinstance(scaler, Standardizer)
                or not np.isfinite(regularization) or regularization < 0):
            raise ValueError('Invalid ridge coefficient predictor')
        self.scaler, self.regularization = scaler, float(regularization)
        self.metadata = dict(metadata or {})

    def predict(self, features):
        z = self.scaler.transform(features)
        values = np.clip(1. + z @ self.W.T + self.c, *self.bounds)
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite regression coefficient prediction')
        return values

    def to_dict(self):
        return dict(version=MODEL_VERSION, kind=self.kind, feature_spec=feature_spec(),
                    parameter_count=self.parameter_count, bounds=list(self.bounds),
                    scaler=self.scaler.to_dict(), metadata=self.metadata,
                    W=self.W.tolist(), c=self.c.tolist(), regularization=self.regularization)

    def save(self, path):
        save_predictor(self, path)


def fit_ridge(features, targets, *, bounds=(.5, 1.5), regularization=.01,
              scaler=None, metadata=None, split='train'):
    """Minimize sum squared target error + lambda ||W||_F^2 with QR/SVD LS.

    The unpenalized final design column is the intercept.  The same scaler and
    labels can be passed to TinyMLP training; neither path uses validation rows.
    """
    if split != 'train':
        raise ValueError('Ridge fitting must use training examples only')
    x, y = _matrix(features, 12, 'Training features'), _matrix(targets, 2, 'Training targets')
    bounds = _bounds(bounds)
    if (len(x) != len(y) or not np.isfinite(regularization) or regularization < 0
            or np.any(y < bounds[0]) or np.any(y > bounds[1])):
        raise ValueError('Invalid ridge data, targets, or regularization')
    scaler = Standardizer.fit(x) if scaler is None else scaler
    if not isinstance(scaler, Standardizer):
        raise TypeError('Require a training-only Standardizer')
    z = scaler.transform(x)
    design = np.column_stack((z, np.ones(len(x))))
    penalty = np.column_stack((np.sqrt(regularization)*np.eye(12), np.zeros(12)))
    coefficients = np.linalg.lstsq(np.vstack((design, penalty)),
                                  np.vstack((y-1., np.zeros((12, 2)))), rcond=None)[0]
    return RidgePredictor(coefficients[:12].T, coefficients[12], scaler, bounds,
                          regularization, metadata)


def save_predictor(predictor, path):
    if not isinstance(predictor, (MLPPredictor, RidgePredictor)):
        raise TypeError('Unsupported light smoother predictor')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(predictor.to_dict(), indent=2, sort_keys=True,
                                    allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def predictor_from_dict(value):
    if value.get('version') != MODEL_VERSION or value.get('feature_spec') != feature_spec():
        raise ValueError('Light smoother model/feature definition changed')
    scaler = Standardizer.from_dict(value['scaler'])
    if value.get('kind') == 'mlp':
        if value.get('parameter_count') != 242:
            raise ValueError('Invalid MLP parameter count')
        model = TinyMLP(value['bounds'])
        state = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in value['state'].items()}
        if not all(bool(torch.isfinite(v).all()) for v in state.values()):
            raise ValueError('Nonfinite MLP weights')
        model.load_state_dict(state, strict=True)
        return MLPPredictor(model.eval(), scaler, value.get('metadata'))
    if value.get('kind') == 'regression':
        if value.get('parameter_count') != 26:
            raise ValueError('Invalid regression coefficient count')
        return RidgePredictor(value['W'], value['c'], scaler, value['bounds'],
                              value['regularization'], value.get('metadata'))
    raise ValueError('Unknown light smoother predictor kind')


def load_predictor(path):
    return predictor_from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
