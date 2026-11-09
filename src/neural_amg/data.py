"""Seven diffusion families assembled exclusively on unstructured P1 meshes."""
import numpy as np

from .fem import DISCRETIZATION, MESH_VERSION, generate_mesh, assemble_p1

FAMILIES = ('near_isotropic', 'aligned_anisotropy', 'rotated_anisotropy',
            'local_patch', 'checkerboard', 'channel', 'variable_coefficient')
TRAIN_ANGLES = (15., 30., 45., 60., 75.)
OOD_ANGLES = (7.5, 22.5, 37.5, 52.5, 67.5, 82.5)


def validate_data_config(data):
    if data.get('kind') != DISCRETIZATION or data.get('discretization') != DISCRETIZATION:
        raise ValueError('Only p1_unstructured FEM data are supported; create a new P1 run')
    if data.get('forcing', 'constant_one') != 'constant_one':
        raise ValueError('Research data use the assembled physical source f=1')
    if data.get('boundary', 'homogeneous_dirichlet') != 'homogeneous_dirichlet':
        raise ValueError('Research data use homogeneous Dirichlet boundary values')
    if not isinstance(data.get('seed'), int) or data['seed'] < 0:
        raise ValueError('Data seed must be a nonnegative integer')
    if set(data) - {'kind', 'discretization', 'seed', 'families', 'mesh', 'splits', 'forcing', 'boundary'}:
        raise ValueError('Unknown P1 data configuration entry')
    if not {'train', 'validation', 'final'} <= data.get('splits', {}).keys():
        raise ValueError('Nonempty train, validation and final specifications required')
    for split, spec in data['splits'].items():
        if set(spec) - {'sizes', 'per_family', 'families', 'mesh'}:
            raise ValueError('Unknown P1 split setting; other discretizations are not supported')
        families = spec.get('families', data.get('families', list(FAMILIES)))
        if not families or len(set(families)) != len(families) or set(families)-set(FAMILIES):
            raise ValueError('Expected distinct supported diffusion families')
        sizes = spec.get('sizes', [])
        if not sizes or len(set(sizes)) != len(sizes) or any(type(n) is not int or n < 2 for n in sizes):
            raise ValueError('P1 mesh sizes must be distinct integers >= 2')
        if type(spec.get('per_family')) is not int or spec['per_family'] < 1:
            raise ValueError('per_family must be a positive integer')
        settings = dict(data.get('mesh', {}), **spec.get('mesh', {}))
        if set(settings)-{'jitter', 'grading'}:
            raise ValueError('Only jitter and grading are supported mesh settings')
        jitter, grading = settings.get('jitter', .3), settings.get('grading', 1.)
        if not np.isfinite(jitter) or not 0 < jitter <= .45:
            raise ValueError('Unstructured mesh requires 0 < jitter <= .45')
        if not np.isfinite(grading) or not .7 <= grading <= 2.:
            raise ValueError('Mesh grading must lie in [.7, 2]')


def specification(split, data, split_index, n, family, index):
    seeds = np.random.SeedSequence([data['seed'], split_index, n, FAMILIES.index(family), index]).generate_state(2, dtype=np.uint64)
    sample_seed, mesh_seed = map(int, seeds)
    rng = np.random.default_rng(sample_seed)
    epsilon = float(10**rng.uniform(-3., -1.))
    angle = float(rng.choice(OOD_ANGLES if split == 'ood_rotation' else TRAIN_ANGLES))
    contrast = float(10**rng.uniform(1., 3.))
    if split == 'ood_anisotropy':
        epsilon = float(10**rng.uniform(-5., -3.1))
    if split == 'ood_contrast':
        contrast = float(10**rng.uniform(3.2, 5.))
    if family == 'near_isotropic':
        epsilon, contrast = float(rng.uniform(.8, 1.)), 1.
    elif family == 'aligned_anisotropy':
        angle, contrast = float(rng.choice([0., 90.])), 1.
    elif family == 'rotated_anisotropy':
        contrast = 1.
    center_x, center_y = rng.uniform(.35, .65, 2)
    width_x, width_y = rng.uniform(.13, .24, 2)
    field = dict(family=family, epsilon=epsilon, angle_deg=angle, contrast=contrast,
        topology='novel' if split == 'ood_topology' else 'standard',
        patch_bounds=list(map(float, (center_x-width_x, center_x+width_x, center_y-width_y, center_y+width_y))),
        checker_cells=int(rng.choice([3, 4, 5])), channel_width=float(rng.uniform(.06, .13)),
        amplitude=float(np.log(contrast)/2), frequency=float(rng.uniform(.7, 2.5)),
        phase=float(rng.uniform(0, 2*np.pi)), geometry_phase=float(rng.uniform(0, 2*np.pi)),
        scale=float(10**rng.uniform(-.5, .5)))
    mesh = dict(jitter=.3, grading=1.)
    mesh.update(data.get('mesh', {})); mesh.update(data['splits'][split].get('mesh', {}))
    mesh.update(n=n, seed=mesh_seed, version=MESH_VERSION)
    return dict(name=f'{split}_{family}_n{n}_{index}', split=split, family=family,
                seed=sample_seed, discretization=DISCRETIZATION, mesh=mesh, field=field,
                forcing='constant_one', boundary='homogeneous_dirichlet')


def diffusion_field(parameters):
    p = parameters
    if p['family'] not in FAMILIES or p['epsilon'] <= 0 or p['contrast'] <= 0 or p['scale'] <= 0:
        raise ValueError('Invalid SPD diffusion parameters')
    angle = np.deg2rad(p['angle_deg'])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    tensor = rotation @ np.diag([1., p['epsilon']]) @ rotation.T
    family, novel = p['family'], p['topology'] == 'novel'

    def field(x, y):
        if family in FAMILIES[:3]:
            return p['scale'] * tensor
        if family == 'variable_coefficient':
            if novel:
                exponent = p['amplitude']*(.65*np.sin(2*np.pi*p['frequency']*x+p['phase'])*np.cos(4*np.pi*y)
                    + .35*np.cos(6*np.pi*x-2*np.pi*y+p['geometry_phase']))
            else:
                exponent = p['amplitude']*np.sin(2*np.pi*p['frequency']*x+p['phase'])*np.cos(2*np.pi*y-p['phase'])
            return p['scale']*np.exp(exponent)[..., None, None]*tensor
        if family == 'local_patch':
            if novel:
                shift = .04*np.sin(p['geometry_phase'])
                hard = (((x-.30-shift)/.20)**2+((y-.36)/.13)**2 < 1) | (((x-.72)/.12)**2+((y-.65-shift)/.23)**2 < 1)
            else:
                xmin, xmax, ymin, ymax = p['patch_bounds']
                hard = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        elif family == 'checkerboard':
            cells = p['checker_cells']
            if novel:
                row = np.floor(y*(cells+1)).astype(int)
                col = np.floor(x*cells+.5*(row % 2)+p['geometry_phase']/(2*np.pi)).astype(int)
            else:
                row = np.minimum((y*cells).astype(int), cells-1)
                col = np.minimum((x*cells).astype(int), cells-1)
            hard = (row+col) % 2 == 0
        else:
            if novel:
                curve = .20*np.sin(2*np.pi*x+p['geometry_phase'])
                hard = (np.abs(y-.32-curve) < p['channel_width']) | (np.abs(y-.74+.5*curve) < .65*p['channel_width'])
            else:
                hard = np.abs(y-x) <= p['channel_width']
        return p['scale']*np.where(hard[..., None, None], p['contrast']*tensor, np.eye(2))

    return field


def materialize(spec):
    if spec.get('discretization') != DISCRETIZATION or spec['mesh'].get('version') != MESH_VERSION:
        raise ValueError('Unsupported discretization or P1 mesh version')
    if spec.get('forcing') != 'constant_one' or spec.get('boundary') != 'homogeneous_dirichlet':
        raise ValueError('Unsupported physical source or boundary specification')
    mesh = generate_mesh(**{k: v for k, v in spec['mesh'].items() if k != 'version'})
    return assemble_p1(mesh, diffusion_field(spec['field']), forcing=1., dirichlet=0.)
