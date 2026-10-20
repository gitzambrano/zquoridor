"""Verify portable training defaults and explicit network width selection."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.run_local_training import resolve
from training.target_options import target_settings, matches_targets


@pytest.mark.parametrize('width',[512,768,1024])
def test_temporal_width_options_keep_qat_and_auxiliary_policy(width):
    _,config=resolve(['--hidden',str(width),'--dry-run'])
    assert config['hidden']==width
    assert config['epochs']==100 and config['min_lr']==1e-7
    assert config['qat'] and config['aux_policy_enabled']


def test_policy_is_independent_of_the_target_method():
    _,config=resolve(['--policy-value-fraction','0.1','--target-mode','current'])
    assert config['epochs']==100 and config['policy_value_fraction']==.1
    assert config['value_target_settings']['horizons']==[0]


def test_explicit_training_parameters_override_mode_defaults():
    _,config=resolve(['--policy-value-fraction','0.1','--epochs','30','--lr','0.00001','--min-lr','0.0000001'])
    assert config['epochs']==30 and config['lr']==1e-5 and config['min_lr']==1e-7


def test_future_horizons_and_their_weights_are_explicit_training_options():
    _,config=resolve(['--target-mode','temporal','--future-horizons','[0,2,6]',
        '--future-weights','[0.5,0.25,0.125]','--result-weight','0.2'])
    assert config['value_target_settings']['horizons']==[0,2,6]
    assert config['value_target_settings']['horizon_weights']==[.5,.25,.125]
    assert config['value_target_settings']['outcome_weight']==.2


@pytest.mark.parametrize('kwargs',[dict(mode='current',horizons=[0,4]),
    dict(mode='temporal',horizons=[4,0]),dict(mode='temporal',horizons=[0,4],weights=[1]),
    dict(mode='temporal',outcome_weight=1.1)])
def test_invalid_target_options_fail(kwargs):
    with pytest.raises(ValueError):target_settings(**kwargs)


def test_future_targets_are_not_silently_reused_for_current_targets():
    settings=target_settings('temporal')
    assert matches_targets({'config':settings},settings)
    assert not matches_targets({'config':settings},target_settings('current'))


def test_configured_future_targets_use_the_real_trajectory_function():
    import ast
    source=Path(__file__).resolve().parents[1]/'training/prepare_temporal_search.py'
    tree=ast.parse(source.read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='temporal_target')
    namespace={};exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    settings=target_settings('temporal',[0,4],[.5,.25],.1)
    value,future=namespace['temporal_target'](0,0,{0:(0,.2),4:(1,-.8)},10,1,
        settings['horizons'],settings['horizon_weights'],settings['outcome_weight'])
    assert value==pytest.approx(.9*(.5*.2+.25*.8)/.75+.1)
    assert future


def test_changed_targets_rebuild_from_raw_and_then_reuse_the_matching_cache(tmp_path,monkeypatch):
    import json
    from training.run_local_training import prepare_data
    base=tmp_path/'base';base.mkdir()
    (base/'dataset.manifest.json').write_text(json.dumps({'complete':True,'config':target_settings('current')}))
    ledger=tmp_path/'downloads.jsonl';ledger.write_text('')
    initial=tmp_path/'initial.bin';initial.write_bytes(b'weights')
    local=dict(data=str(base),downloads=str(ledger),data_cache=str(tmp_path/'cache'),policy_value_fraction=0)
    settings=target_settings('temporal',[0,2,6],[.5,.25,.125],.2)
    config=dict(value_target_settings=settings,init_from=str(initial),init_hidden=512,init_architecture='multipath_phase_contact_bucketed')
    calls=[]
    def fake_run(command,**kwargs):
        calls.append(command)
        output=Path(command[command.index('--output')+1]);output.mkdir(parents=True)
        assert json.loads(command[command.index('--horizons')+1])==[0,2,6]
        assert json.loads(command[command.index('--horizon-weights')+1])==[.5,.25,.125]
        (output/'dataset.manifest.json').write_text(json.dumps({'complete':True,'config':settings}))
    monkeypatch.setattr('training.run_local_training.ensure_encoder',lambda:None)
    monkeypatch.setattr('training.run_local_training.subprocess.run',fake_run)
    first=prepare_data(local,config);second=prepare_data(local,config)
    assert first==second and len(calls)==1
