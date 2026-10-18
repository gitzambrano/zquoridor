"""Verify transfer, topology, quality targets, and resumable experimental training."""
import json
from pathlib import Path
import sys
import shutil
import subprocess

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"training"))
from architecture_models import BASE, SPECS, encode, make_model, topology, load_model
from architecture_models import export as export_model
from architecture_data import prepare_rich, ensure_encoder, inventory, source_settings
from architecture_campaign import DEFAULTS, selected_experiments, training_config
from mix_experimental_datasets import mix_datasets
from run_experiment import CONFIG, train, load_dataset
from student_model import Student, export


def corpus():
    return dict(own_pawn=np.array([4,13,22,31]), opp_pawn=np.full(4,76),
                walls_h=np.zeros(4,np.uint64), walls_v=np.zeros(4,np.uint64),
                walls_left_own=np.full(4,10), walls_left_opp=np.full(4,10),
                own_dist=np.array([8,7,6,5]), opp_dist=np.full(4,8),
                policy=np.eye(209,dtype=np.float32)[[13,22,31,40]], value=np.array([.1,.2,.3,.4],np.float32),
                weight=np.ones(4,np.float32), is_val=np.array([False,False,False,True]),
                group_id=np.array([b'a',b'b',b'c',b'd']))


@pytest.mark.parametrize("variant", list(SPECS))
def test_all_models_forward_backward_and_exact_transfer(variant):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    data = corpus()
    model = make_model(variant,128,qat=True)
    old = Student(BASE,128,qat=True)
    if SPECS[variant]["transfer"] == "exact":
        model.warm_start(old)
    x = torch.from_numpy(encode(data,np.arange(4),variant))
    value, policy = model(x)
    assert value.shape == (4,) and policy.shape == (4,209)
    assert torch.isfinite(value).all() and torch.isfinite(policy).all()
    if SPECS[variant]["transfer"] == "exact":
        expected_value, expected_policy = old(x[:,:858])
        torch.testing.assert_close(value,expected_value,atol=2e-6,rtol=0)
        torch.testing.assert_close(policy,expected_policy,atol=2e-6,rtol=0)
    loss = value.square().mean() + policy.square().mean() + model.additional_loss(x).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    if variant == "tactical_aux":
        altered=x.clone();altered[:,858:]=1-altered[:,858:]
        torch.testing.assert_close(model(altered)[0],value)


def test_topology_removes_both_edges_of_a_wall():
    neighbors, distances = topology(1,0)
    assert neighbors[0,1] == -1 and neighbors[1,1] == -1
    assert neighbors[9,0] == -1 and neighbors[10,0] == -1
    assert distances[0,0] > 8


def test_independent_target_quality_survives_deduplication(tmp_path):
    a,b=corpus(),corpus()
    a['value'][:]=-1;b['value'][:]=1
    a['policy'][:]=np.eye(209)[0];b['policy'][:]=np.eye(209)[1]
    a['aux_policy_target']=np.eye(209,dtype=np.float32)[[2]*4]
    b['aux_policy_target']=np.eye(209,dtype=np.float32)[[3]*4]
    np.savez(tmp_path/'a.npz',**a);np.savez(tmp_path/'b.npz',**b)
    result=mix_datasets([
        dict(name='sparse',path=str(tmp_path/'a.npz'),fraction=.5,policy_quality=1,value_quality=1),
        dict(name='rich',path=str(tmp_path/'b.npz'),fraction=.5,policy_quality=4,value_quality=3)],tmp_path/'mix')
    data=load_dataset(tmp_path/'mix')
    np.testing.assert_allclose(data['value'],.5)
    np.testing.assert_allclose(data['policy'][:,1],.8)
    np.testing.assert_allclose(data['policy_sample_weight'],2.5)
    np.testing.assert_allclose(data['value_sample_weight'],2)
    np.testing.assert_allclose(data['aux_policy_target'][:,2:4],.5)
    assert result['deduplication']['per_head_targets']=='independent_quality_weighted_means'


@pytest.mark.parametrize("variant",list(SPECS))
def test_one_epoch_export_reload_and_resume(tmp_path,variant):
    data=corpus();np.savez(tmp_path/'data.npz',**data)
    export(Student(BASE,128,qat=True),tmp_path/'teacher.bin')
    cfg=dict(CONFIG,data=str(tmp_path/'data.npz'),out_dir=str(tmp_path/variant),
             architecture=variant,hidden=128,init_architecture=BASE,init_hidden=128,
             init_from=str(tmp_path/'teacher.bin'),from_scratch=SPECS[variant]['transfer']=='distillation',
             device='cpu',cpu_threads=1,epochs=1,batch_size=2,build=False,patience=0,
             warmup_epochs=0,aux_policy_enabled=True,distillation_epochs=1,
             distillation_weight=.5 if SPECS[variant]['transfer']=='distillation' else 0)
    result=train(cfg)
    assert result['epochs']==1 and result['promoted'] is False
    assert train(cfg)['fingerprint']==result['fingerprint']
    folder=tmp_path/variant
    if variant=='tactical_aux':
        assert (folder/'student_int8.bin').exists()
    else:
        assert not (folder/'student_int8.bin').exists()
        restored=load_model(folder/'student.pt')
        with torch.inference_mode():
            output=restored(torch.from_numpy(encode(data,np.arange(4),variant)))
        assert torch.isfinite(output[0]).all()


def test_suite_contains_eight_distinct_models_and_a_control():
    jobs=selected_experiments(DEFAULTS)
    assert len(jobs)==16 and len({j['architecture'] for j in jobs})==9
    for job in jobs:
        cfg=training_config(DEFAULTS,job,Path('dataset'))
        assert not cfg['build']
        if job['architecture'] in ('graph','transformer'):
            assert cfg['from_scratch'] and cfg['distillation_weight']>0


def test_recipe_ablations_change_only_the_declared_factor():
    jobs={job['id']:job for job in selected_experiments(DEFAULTS)}
    for job in jobs.values():
        if 'comparison_to' not in job:
            continue
        baseline=training_config(DEFAULTS,jobs[job['comparison_to']],Path('dataset'))
        candidate=training_config(DEFAULTS,job,Path('dataset'))
        differences={key for key in baseline if baseline[key]!=candidate[key]}-{'out_dir'}
        assert differences=={job['changed_factor']}
    from run_experiment import learning_rate
    baseline=training_config(DEFAULTS,jobs['00_control'],Path('dataset'))
    assert baseline['epochs']==40
    assert learning_rate(baseline,0)==pytest.approx(1e-4)
    assert learning_rate(baseline,39)==pytest.approx(1e-7)
    assert training_config(DEFAULTS,jobs['07_graph'],Path('dataset'))['epochs']==100
    assert training_config(DEFAULTS,jobs['08_transformer'],Path('dataset'))['epochs']==120


def test_reference_surprise_preserves_each_head_training_mass(tmp_path):
    from architecture_recipes import apply_reference_surprise
    data=corpus()
    data['policy_sample_weight']=np.array([1.,2.,3.,4.],np.float32)
    data['aux_sample_weight']=np.array([3.,2.,1.,5.],np.float32)
    data['value_sample_weight']=np.array([2.,3.,4.,5.],np.float32)
    original={k:np.array(v,copy=True) for k,v in data.items()}
    np.savez(tmp_path/'data.npz',**data)
    export(Student(BASE,128,qat=True),tmp_path/'reference.bin')
    report=apply_reference_surprise(data,tmp_path/'data.npz',tmp_path/'reference.bin',tmp_path/'cache',BASE,128,cap=100)
    train_rows=~data['is_val']
    for field in ('policy_sample_weight','aux_sample_weight'):
        assert (data['weight'][train_rows]*data[field][train_rows]).sum()==pytest.approx(
            (original['weight'][train_rows]*original[field][train_rows]).sum())
    np.testing.assert_array_equal(data['value_sample_weight'],original['value_sample_weight'])
    np.testing.assert_array_equal(data['weight'],original['weight'])
    assert report['normalization']=='training_only_per_head_evidence_mass'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA is required for measured capacity')
@pytest.mark.parametrize('variant',[BASE,'graph','transformer'])
def test_cuda_capacity_probe_restores_weights_and_rng(tmp_path,variant):
    from batch_capacity import calibrate
    from run_experiment import _epoch
    torch.set_num_threads(2)
    data=corpus()
    model=make_model(variant,128,qat=True)
    model.enable_aux_policy()
    model.cuda()
    saved={key:value.detach().clone() for key,value in model.state_dict().items()}
    cpu_rng, gpu_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
    config=dict(CONFIG,architecture=variant,hidden=128,auto_batch_size=True,
                batch_size_min=2,batch_size_max=4,aux_policy_enabled=True)
    def probe(batch):
        optimizer=torch.optim.AdamW(model.parameters())
        _epoch(model,data,np.arange(batch),dict(config,batch_size=batch),'cuda',
               optimizer,np.random.default_rng(3))
    result=calibrate(model,config,'cuda',tmp_path/'job',4,probe)
    assert result['batch_size']==4 and result['attempts'][-1]['peak_bytes']>0
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,saved[key],rtol=0,atol=0)
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert torch.equal(gpu_rng,torch.cuda.get_rng_state())
    assert calibrate(model,config,'cuda',tmp_path/'other_recipe',4,probe)==result


def test_batch_capacity_search_stops_before_memory_failure():
    from batch_capacity import largest_batch
    selected, attempts = largest_batch(256,65536,lambda batch:(batch<=16384,batch*1000))
    assert selected==16384 and attempts[-1]['batch_size']==32768
    selected, _ = largest_batch(256,700,lambda batch:(True,batch))
    assert selected==700
    with pytest.raises(RuntimeError,match='minimum training batch'):
        largest_batch(256,1024,lambda batch:(False,0))


def test_local_temporal_ledger_checks_frozen_shard_hashes(tmp_path):
    from architecture_data import local_temporal_ledger
    from mix_experimental_datasets import _sha256
    from read_selfplay import SAMPLE_DTYPE
    from prepare_replay import META_DTYPE
    binary, metadata = tmp_path/'sample.bin', tmp_path/'sample.meta'
    np.zeros(2, dtype=SAMPLE_DTYPE).tofile(binary)
    np.zeros(2, dtype=META_DTYPE).tofile(metadata)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(complete=True, accepted_shards=[dict(
        v3=binary.name, meta_path=metadata.name, bin_sha256=_sha256(binary),
        meta_sha256=_sha256(metadata))])))
    ledger = local_temporal_ledger([manifest], tmp_path/'local.jsonl')
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 2 and rows[1]['path'] == rows[0]['path'][:-4]+'.meta'
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA mismatch'):
        local_temporal_ledger([manifest], ledger)


def test_rich_labels_preserve_dense_policy_and_join_by_id(tmp_path):
    ensure_encoder()
    positions=[dict(id='a',history=[],side_to_move=0,source='gumbel_sims'),
               dict(id='b',history=['e2'],side_to_move=1,source='gumbel_sims')]
    p=np.arange(1,210,dtype=float);p/=p.sum()
    labels=[dict(id='b',target=dict(policy=p.tolist(),value=-.5),budget=dict(move_time_ms=20)),
            dict(id='a',target=dict(policy=p.tolist(),value=.2),budget=dict(move_time_ms=15))]
    (tmp_path/'positions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in positions))
    (tmp_path/'targets.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in labels))
    report=prepare_rich(tmp_path/'positions.jsonl',tmp_path/'targets.jsonl',tmp_path/'dataset.npz')
    with np.load(tmp_path/'dataset.npz') as data:
        np.testing.assert_allclose(data['policy'],np.stack([p,p]),rtol=1e-6)
        assert data['own_pawn'].tolist()==[4,4]
        assert data['opp_pawn'].tolist()==[76,67]
        assert len(set(data['group_id']))==1
        assert data['search_budget_ms'].tolist()==[15,20]
    assert report['samples']==2


def test_training_only_auxiliary_export_matches_native_production_layout(tmp_path):
    reference=Student(BASE,512,qat=True)
    reference.load_float(ROOT/'data/nnue/nnue_weights.bin')
    candidate=make_model('tactical_aux',512,qat=True)
    candidate.warm_start(reference)
    manifest=export_model(candidate,tmp_path/'student.bin')
    compiler=shutil.which('g++') or 'C:/mingw64/bin/g++.exe'
    executable=tmp_path/'probe.exe'
    subprocess.run([compiler,'-O2','-std=c++17','-I'+str(ROOT/'src'),*manifest['cpp_flags'],
                    str(ROOT/'tools/teacher/student_probe.cpp'),'-o',str(executable)],check=True)
    rows=[json.loads(line) for line in subprocess.check_output([str(executable),str(tmp_path/'student_int8.bin')],text=True).splitlines()
          if line.startswith('{')]
    data={key:np.array([row[key] for row in rows],dtype=np.uint64 if key in ('walls_h','walls_v') else np.int64)
          for key in ('own_pawn','opp_pawn','walls_h','walls_v','own_dist','opp_dist','walls_left_own','walls_left_opp')}
    with torch.inference_mode():
        values,policies=candidate(torch.from_numpy(encode(data,np.arange(len(rows)),'tactical_aux')))
    np.testing.assert_allclose(values.numpy(),[row['value'] for row in rows],atol=.002,rtol=0)
    expected=np.array([row['policy'] for row in rows]);valid=np.ones_like(expected,dtype=bool)
    valid[data['walls_left_own']==0,81:]=False
    np.testing.assert_allclose(policies.numpy()[valid],expected[valid],atol=.002,rtol=0)


def test_repeated_rich_ids_preserve_each_search_annotation(tmp_path):
    ensure_encoder()
    position=dict(id='same',history=[],side_to_move=0,source='wall_drops')
    labels=[dict(id='same',target=dict(policy=np.eye(209)[i].tolist(),value=v)) for i,v in ((13,-.5),(22,.5))]
    (tmp_path/'positions.jsonl').write_text((json.dumps(position)+'\n')*2)
    (tmp_path/'targets.jsonl').write_text(''.join(json.dumps(label)+'\n' for label in labels))
    report=prepare_rich(tmp_path/'positions.jsonl',tmp_path/'targets.jsonl',tmp_path/'dataset.npz')
    with np.load(tmp_path/'dataset.npz') as data:
        np.testing.assert_array_equal(data['value'],[-.5,.5])
        np.testing.assert_array_equal(data['policy'].argmax(1),[13,22])
    assert report['unique_history_ids']==1


@pytest.mark.parametrize('module_name,function_name',[
    ('tools.generate_wall_drops','run_generation'),('tools.generate_gumbel_sims','run_gumbel_generation')])
def test_generators_resume_nested_targets_without_zero_policy(tmp_path,monkeypatch,module_name,function_name):
    import importlib
    module=importlib.import_module(module_name)
    executable=tmp_path/'fake.exe';executable.write_bytes(b'fixture')
    position=dict(id='fixture',history=[],side_to_move=0)
    label=dict(id='fixture',target=dict(policy=np.eye(209)[13].tolist(),value=.75),metadata=dict(aux=[8,8,10,10,0,0]))
    (tmp_path/'positions.jsonl').write_text(json.dumps(position)+'\n')
    (tmp_path/'targets.jsonl').write_text(json.dumps(label)+'\n')
    observed=[];normalize=module.normalize_policy
    def record(policy):
        result=normalize(policy);observed.append(result);return result
    monkeypatch.setattr(module,'normalize_policy',record)
    if hasattr(module,'generate_candidate_positions'):
        monkeypatch.setattr(module,'generate_candidate_positions',lambda *args:[[]])
    cfg=dict(module.CONFIG,output=str(tmp_path),zq_executable=str(executable),count=1)
    result=getattr(module,function_name)(cfg)
    assert result['generated']==1 and observed[0][13]==1
