"""Verify canonical aggregation, cross-source splits, and value-only masks."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.run_policy_mix_campaign import assemble, reconcile_groups, masked_loader, sha
from training import run_experiment as trainer
from training.student_model import Student


def dataset(pawns, values, validation):
    n=len(pawns)
    data={key:np.full(n,value,dtype=np.uint64 if key.startswith('walls_') else np.int64)
          for key,value in {'opp_pawn':76,'walls_h':0,'walls_v':0,
              'walls_left_own':10,'walls_left_opp':10,'own_dist':8,'opp_dist':8}.items()}
    data.update(own_pawn=np.array(pawns),value=np.array(values,dtype=np.float32),
                is_val=np.array(validation),weight=np.ones(n,dtype=np.float32))
    for key in ('value_sample_weight','policy_sample_weight','aux_sample_weight'):
        data[key]=np.ones(n,dtype=np.float32)
    data['policy']=np.eye(209,dtype=np.float32)[[13]*n]
    data['aux_policy_target']=data['policy'].copy()
    return data


def test_group_overlap_excludes_the_entire_connected_component():
    keep,val=reconcile_groups([b'a',b'b'],[False,True],
                              [b'a',b'x',b'x',b'b',b'y'],['g1','g1','g2','g2','g3'],[False]*5)
    np.testing.assert_array_equal(keep,[False,False,False,False,True])
    assert not val[-1]


def test_overlap_assigns_all_game_rows_to_the_existing_validation_side():
    keep,val=reconcile_groups([b'a'],[True],[b'a',b'x'],['game','game'],[False,False])
    assert keep.all() and val.all()


def test_aggregation_preserves_policy_and_limits_policy_value_mass(tmp_path):
    base=dataset([4,5],[.2,-.2],[False,True])
    policy=dataset([4,6,6],[.8,.4,.6],[False]*3)
    policy['group_id']=np.array(['a','b','b'])
    arrays,info=assemble(base,policy,np.ones(3),.1,tmp_path)
    assert info['samples']==3
    # Two unique policy states receive 0.05 each before denominator scaling.
    assert arrays['value'][0]==pytest.approx((.2*.9+.8*.05)/.95)
    assert arrays['value'][2]==pytest.approx(.5)
    np.testing.assert_array_equal(arrays['policy'][:2],base['policy'])
    np.testing.assert_array_equal(arrays['aux_policy_target'][:2],base['aux_policy_target'])
    np.testing.assert_array_equal(arrays['policy_available'],[True,True,False])
    assert arrays['value_sample_weight'][[0,2]].sum()==pytest.approx(2)
    assert arrays['policy_sample_weight'][0]==pytest.approx(2)


def test_value_only_sample_has_no_main_or_auxiliary_policy_update(monkeypatch):
    data=dataset([4],[.8],[False]);data['policy_available']=np.array([False])
    monkeypatch.setattr('training.run_policy_mix_campaign._original_loader',lambda path:data)
    data=masked_loader(Path('unused'))
    assert data['policy_sample_weight'][0]==0 and data['aux_sample_weight'][0]==0
    torch.set_num_threads(1)
    model=Student('multipath_phase_contact_bucketed',128,qat=False,aux_policy=True)
    main=model.policy.weight.detach().clone(); auxiliary=model.aux_policy.weight.detach().clone()
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    trainer._epoch(model,data,np.array([0]),dict(trainer.CONFIG,
        architecture=model.architecture,batch_size=1,mirror_h=False,aux_policy_enabled=True),
        'cpu',optimizer=optimizer,rng=np.random.default_rng(1))
    torch.testing.assert_close(main,model.policy.weight,rtol=0,atol=0)
    torch.testing.assert_close(auxiliary,model.aux_policy.weight,rtol=0,atol=0)


def test_real_manifest_loader_accepts_placeholders_and_applies_exact_masks(tmp_path):
    import json
    base=dataset([4,5],[.2,-.2],[False,True])
    policy=dataset([6],[.6],[False]);policy['group_id']=np.array(['g'])
    assemble(base,policy,np.ones(1),.1,tmp_path)
    entries={path.stem:{'filename':str(path.relative_to(tmp_path)),'sha256':sha(path)}
             for path in (tmp_path/'fields').glob('*.npy')}
    (tmp_path/'dataset.manifest.json').write_text(json.dumps(dict(complete=True,
        group_separation_verified=True,arrays=entries)))
    data=masked_loader(tmp_path)
    assert data['policy_sample_weight'][-1]==0
    assert data['aux_sample_weight'][-1]==0
    assert data['value_sample_weight'][-1]>0
