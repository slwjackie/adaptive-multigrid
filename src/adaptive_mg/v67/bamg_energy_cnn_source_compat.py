"""Read frozen BAMG sources without changing recorded numerical-study files.

The original source reader used the numerical study's spaced JSON digest, while
research_data.freeze_research writes a compact JSON digest. This additive reader
keeps all checkpoint/metadata checks and never rewrites a freeze or a manifest.
The CLI installs it only for its invocation; the old consumer module remains
byte-identical so already-completed numerical teacher manifests still verify.
New CNN study manifests include this module and their updated CLI scripts.
"""
from contextlib import contextmanager
from pathlib import Path

from . import bamg_energy_cnn_study as paired
from . import bamg_energy_study as numerical
from .bamg_study import _file_hash, _initialize_weight, _read
from .bamg_transfer import WeightCNN
from .research_data import _hash as research_config_digest


def source_inputs(source_study):
    """Verify a migrated package without following its obsolete absolute paths."""
    root = Path(source_study).resolve()
    frozen = _read(root / 'data/research_freeze.json')
    # Real research freezes use compact JSON; early importer fixtures used
    # spaced JSON. Both digests bind the complete, unmodified configuration.
    if frozen['config_digest'] not in (
            research_config_digest(frozen['config']),
            numerical.digest(frozen['config'])):
        raise ValueError('Source freeze configuration hash changed')
    selection = frozen['config']['selection']['selected_learned_model']
    choices = {
        'bootstrap_weight_cnn_finetuned_uniform': 'deployment_uniform.pt',
        'bootstrap_weight_cnn_finetuned_terminal': 'deployment_terminal.pt',
        'bootstrap_weight_cnn_supervised': 'supervised_models.pt',
    }
    if selection not in choices:
        raise ValueError('Source selection is not a compatible fixed-support Weight CNN')
    hashes = {}
    for name in choices.values():
        hashes[name] = _file_hash(root / name)
        if hashes[name] != frozen['files']['checkpoint:' + name]['sha256']:
            raise ValueError('Source checkpoint differs from frozen original: ' + name)
    # These metadata files are necessary to exclude old training and test A's.
    for relative, key in [('data/dataset_plan.json', 'plan'),
                          ('data/development_manifest.json', 'development'),
                          ('fixed_classical.json', 'baseline' if 'baseline' in frozen['files'] else 'rules')]:
        actual = _file_hash(root / relative)
        if actual != frozen['files'][key]['sha256']:
            raise ValueError('Source metadata differs from frozen original: ' + relative)
        hashes[relative] = actual
    for relative in ['configuration.json', 'study_manifest.json',
                     'data/research_freeze.json', 'data/final_claim.json',
                     'data/final_data_manifest.json']:
        if (root / relative).exists():
            hashes[relative] = _file_hash(root / relative)
    settings = _read(root / 'configuration.json')
    if settings != frozen['config']['settings']:
        raise ValueError('Source configuration differs from its research freeze')
    if (root / 'data/final_claim.json').exists() and not (root / 'data/final_data_manifest.json').exists():
        raise ValueError('Copy final_data_manifest.json as well to exclude previously exposed final operators')
    if (root / 'data/final_claim.json').exists():
        claim = _read(root / 'data/final_claim.json')
        final = _read(root / 'data/final_data_manifest.json')
        if claim['freeze_sha256'] != _file_hash(root / 'data/research_freeze.json'):
            raise ValueError('Old final claim does not match its research freeze')
        if final['claim_fingerprint'] != claim['fingerprint']:
            raise ValueError('Old final metadata does not match its claim')
        if claim.get('final_data_sha256') is not None and claim['final_data_sha256'] != _file_hash(root / 'data/final_data_manifest.json'):
            raise ValueError('Old final metadata hash changed')
    if settings.get('support_mode') != 'fixed_bilinear':
        raise ValueError('Source must use fixed bilinear support')
    model = WeightCNN(**settings['models']['weight']).double()
    checkpoint = root / choices[selection]
    info = _initialize_weight(model, checkpoint)
    return dict(path=str(root), selected_model=selection, checkpoint=choices[selection],
                hashes=hashes, weight_spec=info['weight_spec'],
                init_weight_sha256=info['sha256']), settings


@contextmanager
def source_reader():
    """Scope the compatibility reader to this single-process CLI invocation."""
    original = paired.source_inputs
    paired.source_inputs = source_inputs
    try:
        yield
    finally:
        paired.source_inputs = original


def main(*, routing=False):
    """Run either supported CLI with producer-compatible source verification."""
    if routing:
        from . import bamg_energy_cnn_routing_study as entrypoint
    else:
        entrypoint = paired
    with source_reader():
        return entrypoint.main()
