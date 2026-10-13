"""Generator math and orchestration, not native OpenFOAM validation."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from adaptive_mg.v67.h2_fixed_p.fullrun import read_internal_field, parse_solver_log
from adaptive_mg.v67.h2_fixed_p.data import validate_case_contract
from adaptive_mg.v67.smoother_reuse.physics import shear_metrics
from adaptive_mg.v67.world_model.data import write_json

REPO = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO/path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module('shear_pipeline','scripts/run_v6_7_shear_pipeline.py')
generator = module('shear_generator','integrations/openfoam13/case/generate_shear_layer.py')


def args(path, *extra):
    return generator.parser().parse_args(['--output',str(path),'--case-id','unit-shear',
        '--case-group','test-only','--split','train',*extra])


def test_shear_geometry_composition_and_actual_field_profiles(tmp_path):
    a = args(tmp_path/'case')
    root = generator.generate(a)
    contract = validate_case_contract(runner.read(root/'contract.json'))
    assert contract['flow_family'] == 'planar_shear_mixing_layer'
    assert contract['physics']['reacting'] is False and contract['reaction_count'] == 0
    mesh = (root/'system/blockMeshDict').read_text()
    assert 'inlet { type patch; faces ((0 4 7 3)); }' in mesh
    assert 'outlet { type patch; faces ((1 2 6 5)); }' in mesh
    assert 'lateral { type patch; faces ((0 1 5 4) (7 6 2 3)); }' in mesh
    assert 'fuel {' not in mesh and 'air {' not in mesh
    n = a.nx*a.ny
    he = read_internal_field(root/'0/He',n)
    n2 = read_internal_field(root/'0/N2',n)
    velocity = read_internal_field(root/'0/U',n,3)
    np.testing.assert_allclose(he+n2,1.)
    assert np.all(velocity[:,0]>0) and np.max(np.abs(velocity[:,1]))>0
    assert he.reshape(a.ny,a.nx)[0,0] > he.reshape(a.ny,a.nx)[-1,0]
    assert velocity.reshape(a.ny,a.nx,3)[0,0,0] > velocity.reshape(a.ny,a.nx,3)[-1,0,0]
    profile = shear_metrics({'U':velocity[None], 'He':he[None]},contract,[0.])
    assert profile['samples'][0]['integrated_enstrophy']>0
    assert not profile['three_dimensional_turbulence_validated']
    assert all(s['composition_width_m']>0 for s in profile['samples'][0]['stations'])
    assert 'name shearU_' in (root/'0/U').read_text()
    assert 'value nonuniform List<vector>' in (root/'0/U').read_text()
    assert 'codedFixedValue' not in (root/'0/He').read_text()
    with pytest.raises(FileExistsError):
        generator.generate(a)


def test_forcing_has_quiet_intervals_and_bounded_complementary_species(tmp_path):
    a = args(tmp_path/'case','--composition-amplitude','.08')
    quiet = [.01,.10,.5,.95]
    for s in quiet:
        assert generator.forcing(a,s*a.end_time) == 0
    q = [generator.forcing(a,t) for t in np.linspace(0,a.end_time,401)]
    assert max(q)>.1 and min(q)<-.1 and max(abs(x) for x in q)<=1
    for time in np.linspace(0,a.end_time,101):
        for y in (-a.height/2,0.,a.height/2):
            u,he = generator.inlet(a,time,y)
            assert u>0 and 0<=he<=1
            assert he+(1-he)==1.
    root = generator.generate(a)
    assert all('codedFixedValue' in (root/'0'/name).read_text() for name in ('U','He','N2'))
    control = args(tmp_path/'control','--forcing-amplitude','0')
    path = generator.generate(control)
    assert not runner.read(path/'contract.json')['coded_inlet']
    assert generator.inlet(control,0.,-.005) == generator.inlet(control,.02,-.005)
    assert all('codedFixedValue' not in (path/'0'/name).read_text() for name in ('U','He','N2'))


def test_emitted_cpp_forcing_matches_python_schedule(tmp_path):
    # Compiles only the scalar formula with std::cmath. This is not an
    # OpenFOAM API/build check; that requires the target Foundation 13 host.
    compiler=shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    a=args(tmp_path/'unused','--phase','2.8','--pulse-shift','.045','--frequency-ratio','.73')
    formula=generator.code_parameters(a).split('const vectorField&',1)[0].replace('this->db().time().value()','time')
    source=tmp_path/'formula.cpp'
    source.write_text('#include <cmath>\n#include <iostream>\n#include <iomanip>\n'
        'using scalar=double; namespace Foam {using std::sin; double sqr(double x){return x*x;}}\n'
        'double q(double time){'+formula+' return q;}\n'
        'int main(){double t;while(std::cin>>t)std::cout<<std::setprecision(17)<<q(t)<<"\\n";}\n')
    exe=tmp_path/'formula'
    subprocess.run([compiler,'-std=c++17',str(source),'-o',str(exe)],check=True,capture_output=True)
    times=np.linspace(0,a.end_time,121)
    output=subprocess.run([str(exe)],input='\n'.join(map(str,times)),text=True,capture_output=True,check=True).stdout
    np.testing.assert_allclose(np.fromstring(output,sep=' '),[generator.forcing(a,t) for t in times],atol=2e-14)


@pytest.mark.parametrize('case',runner.CASES,ids=lambda x:x['name'])
def test_every_declared_case_reaches_native_preparation_with_valid_contract(tmp_path,monkeypatch,case):
    a=argparse.Namespace(config=REPO/'configs/v6_7_shear_temporal.json',n=31,end_time=.04,delta_t=1e-4,timeout=60)
    (tmp_path/'logs').mkdir()
    contracts=[]
    class BeforeOpenFOAM(Exception):
        pass
    def command(argv,log,timeout):
        if Path(argv[1])==runner.GENERATOR:
            parsed=generator.parser().parse_args([str(v) for v in argv[2:]])
            output=generator.generate(parsed)
            contracts.append(validate_case_contract(runner.read(output/'contract.json')))
        else:
            assert '--prepare-only' in argv
            raise BeforeOpenFOAM
    monkeypatch.setattr(runner.shared,'run_logged',command)
    with pytest.raises(BeforeOpenFOAM):
        runner.collect(a,tmp_path,case)
    c=contracts[0]
    assert c['split']==case['split'] and c['n_steps']==400
    assert c['forcing']['schedule_known_to_controller'] is False
    assert c['benchmark_role']==('control' if case['amplitude']==0 else 'primary')


@pytest.mark.parametrize('extra',[
    ('--forcing-amplitude','1'),('--forcing-amplitude','nan'),('--composition-amplitude','.4'),
    ('--frequency','1000'),('--shear-thickness','.00001'),('--nx','127','--ny','127'),
])
def test_invalid_or_underresolved_forcing_rejected(tmp_path,extra):
    with pytest.raises(ValueError):
        generator.generate(args(tmp_path/'bad',*extra))
    assert not (tmp_path/'bad').exists()


def fake_pipeline(tmp_path,monkeypatch,failed=None,stop='test'):
    from adaptive_mg.v67 import benchmark_profile
    from adaptive_mg.v67.smoother_reuse import native_profile
    a=argparse.Namespace(output=tmp_path/'study',config=REPO/'configs/v6_7_shear_temporal.json',
        n=31,end_time=.04,delta_t=1e-4,repeats=1,timeout=60,stop_after=stop)
    calls=[]
    monkeypatch.setattr(runner,'collect',lambda args,root,case,**kw:root/'recordings'/case['name'])
    monkeypatch.setattr(benchmark_profile,'profile_dataset',lambda root,**kw:dict(splits=kw['splits']))
    monkeypatch.setattr(native_profile,'profile',lambda *args,**kw:dict(success=failed!='native'))
    def command(argv,log,timeout):
        temporal=Path(argv[1])==runner.REUSE_DRIVER
        command=argv[2]
        split=argv[argv.index('--split')+1] if '--split' in argv else None
        calls.append((temporal,command,split))
        root=a.output/'study'
        if temporal and command=='collect':
            write_json(root/'reuse/collection.json',{'probe':{'world_model_candidate':False}})
        elif temporal and command=='train':
            write_json(root/'reuse/training.json',{'models':{},'selected':'reuse'})
        elif command=='evaluate':
            path=root/'reuse'/split/'report.json' if temporal else root/split/'replay.json'
            write_json(path,{'success':failed!='replay'})
        elif command=='coupled':
            output=Path(argv[argv.index('--output')+1])
            write_json(output/'report.json',{'success':failed!='coupled','full_cfd_speedup':.8})
    monkeypatch.setattr(runner,'run_logged',command)
    return a,calls


def test_pipeline_freezes_both_models_before_test_and_keeps_slow_results(tmp_path,monkeypatch):
    a,calls=fake_pipeline(tmp_path,monkeypatch)
    result=runner.run_study(a)
    assert result['test_passed'] and result['completed_through']=='test'
    assert not result['hydrogen_validation_completed']
    assert calls.index((False,'freeze',None))<calls.index((True,'freeze',None))<calls.index((False,'evaluate','test'))
    assert result['validation']['validation_forced']['selected']['full_cfd_speedup']==.8
    assert runner.read(a.output/'dataset_profile.json')['splits']==['train','tune','validation']


@pytest.mark.parametrize('failed',['native','replay','coupled'])
def test_failed_validation_preserves_evidence_and_never_opens_test(tmp_path,monkeypatch,failed):
    a,calls=fake_pipeline(tmp_path,monkeypatch,failed)
    with pytest.raises(RuntimeError,match='checks failed'):
        runner.run_study(a)
    assert all(cmd!='freeze' and split!='test' for _,cmd,split in calls)
    assert runner.read(a.output/'summary.json')['status']=='failed'
    assert runner.read(a.output/'pipeline_status.json')['results_preserved']


def test_probe_stop_has_no_model_fit_or_validation(tmp_path,monkeypatch):
    a,calls=fake_pipeline(tmp_path,monkeypatch,stop='probe')
    assert runner.run_study(a)['completed_through']=='probe'
    assert (True,'train',None) not in calls
    assert all(cmd!='evaluate' for _,cmd,_ in calls)


def test_native_mode_pressure_log_is_parsed_without_service():
    from test_h2_fixed_p_fullrun import log_text
    for mode in ('native_pcg','native_gamg'):
        report=parse_solver_log(log_text().replace('classical',mode),{'end_time':.02,'delta_t':.01},1e-6)
        assert report['pressure_seconds']==.005
        assert all(row['mode']==mode for row in report['pressure_rows'])


@pytest.mark.parametrize('fail_h2',[False,True])
def test_hydrogen_is_a_separate_study_and_failure_never_looks_complete(tmp_path,monkeypatch,fail_h2):
    runs=[]
    def run(args,*,hydrogen=False):
        runs.append((args.output,args.config,args.end_time,hydrogen))
        args.output.mkdir(parents=True)
        if hydrogen and fail_h2:
            raise RuntimeError('fixture H2 field mismatch')
        result=dict(status='complete',completed_through='validation',output=str(args.output),
                    hydrogen_validation_completed=hydrogen,world_model_benefit_established=False)
        write_json(args.output/'summary.json',result)
        return result
    monkeypatch.setenv('WM_PROJECT_VERSION','13')
    monkeypatch.setattr(runner.shutil,'which',lambda name:'/mock/'+name)
    monkeypatch.setattr(runner.importlib.util,'find_spec',lambda name:object())
    monkeypatch.setattr(runner,'run_study',run)
    args=['--output',str(tmp_path/'primary'),'--with-h2']
    if fail_h2:
        with pytest.raises(SystemExit) as stopped:
            runner.main(args)
        assert stopped.value.code==1
    else:
        runner.main(args)
    assert len(runs)==2 and runs[0][1]!=runs[1][1]
    assert runs[1][0]==runs[0][0]/'hydrogen_validation'
    assert runs[0][2]==.04 and runs[1][2]==.0002
    summary=runner.read(tmp_path/'primary/summary.json')
    assert summary['status']==('failed' if fail_h2 else 'complete')
    assert summary['hydrogen_validation_completed'] is (not fail_h2)
