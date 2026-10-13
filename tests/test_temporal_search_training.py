"""Check temporal perspectives and independent auxiliary policy supervision."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from prepare_temporal_search import temporal_target, canonical_policy
from mirror_augmentation import stochastic_mirror_dict_h, ACTION_MIRROR_H
from run_experiment import _epoch, CONFIG
from student_model import Student


def test_future_values_change_perspective_and_end_at_the_result():
    target, future = temporal_target(0,0,{0:(0,.2),1:(1,-.6)},2,1,[0,1,4],[.5,.25,.125],.1)
    assert target == pytest.approx(.9*(.1+.15+.125)/.875+.1)
    assert future
    missing, future = temporal_target(0,0,{0:(0,.2)},20,-1,[0,4],[.5,.25],.1)
    assert missing == pytest.approx(.08)
    assert not future


def test_current_policy_rotation_and_auxiliary_horizontal_mirror():
    visits = np.zeros(209);visits[4]=2;visits[81]=1
    policy = canonical_policy({'visit_counts_global':visits.tolist()},1)
    assert policy[76] == pytest.approx(2/3)
    assert policy[137] == pytest.approx(1/3)
    main = np.eye(209,dtype=np.float32)[[13]]
    auxiliary = np.eye(209,dtype=np.float32)[[12]]
    mirrored = stochastic_mirror_dict_h({'policy':main,'aux_policy_target':auxiliary},np.array([0]),np.array([True]))
    np.testing.assert_array_equal(mirrored['policy'],main[:,ACTION_MIRROR_H])
    np.testing.assert_array_equal(mirrored['aux_policy_target'],auxiliary[:,ACTION_MIRROR_H])


def test_per_head_weight_changes_only_the_requested_loss_component():
    torch.set_num_threads(1)
    data = {k:np.array([v,v]) for k,v in {'own_pawn':4,'opp_pawn':76,'walls_h':0,'walls_v':0,
            'own_dist':8,'opp_dist':8,'walls_left_own':10,'walls_left_opp':10}.items()}
    data['walls_h']=data['walls_h'].astype(np.uint64);data['walls_v']=data['walls_v'].astype(np.uint64)
    data.update(policy=np.eye(209,dtype=np.float32)[[13,13]],value=np.array([.2,.2],dtype=np.float32),weight=np.ones(2,dtype=np.float32))
    cfg=dict(CONFIG,architecture='multipath_phase_contact_bucketed',batch_size=2,policy_weight=0,aux_policy_enabled=False)
    model=Student(cfg['architecture'],128,qat=True)
    before=_epoch(model,data,np.arange(2),cfg,'cpu')
    data['value_sample_weight']=np.full(2,2,dtype=np.float32)
    after=_epoch(model,data,np.arange(2),cfg,'cpu')
    assert after['loss'] == pytest.approx(2*before['loss'])
    assert after['policy_kl'] == pytest.approx(before['policy_kl'])
