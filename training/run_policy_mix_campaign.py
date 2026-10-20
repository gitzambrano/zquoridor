#!/usr/bin/env python3
"""Wait for a width screen, then compare mixed policy teaching with replay alone."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import run_experiment as trainer
from training.finetune_external_value import prepare, CONFIG as EXTERNAL_CONFIG
from training.run_temporal_campaign import parity
from training.student_model import Student, encode_features
from tools import match_finalists, run_benchmark

CONFIG = {
    'dependency': str(ROOT/'experimental/experiments/width-100ep-20261013'),
    'dataset': str(ROOT/'data/teaching/drive-temporal-100ep/dataset'),
    'archive': str(ROOT/'data/teaching/external-value/titanium_mix_v2.zip'),
    'output': str(ROOT/'experimental/experiments/policy-mix-after-width-20261013'),
    'mixed_dataset': str(ROOT/'data/teaching/policy-mix-after-width-20261013'),
    'policy_value_fraction': 0.1, 'epochs': 10, 'lr': 5e-6, 'min_lr': 1e-6,
    'pairs': 100, 'movetime_ms': 200, 'workers': 3, 'seed': 20261019,
}
STATE_FIELDS = ('own_pawn', 'opp_pawn', 'walls_h', 'walls_v',
                'walls_left_own', 'walls_left_opp', 'own_dist', 'opp_dist')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_keys(data):
    return [struct.pack('<BBQQbb', *(int(data[k][i]) for k in STATE_FIELDS[:6]))
            for i in range(len(data['value']))]


def reconcile_groups(base_keys, base_val, policy_keys, policy_groups, policy_val):
    """Keep canonical states and connected policy games on one validation side."""
    parents = {}
    def find(group):
        parents.setdefault(group, group)
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    seen = {}
    for key, group in zip(policy_keys, policy_groups):
        group = str(group)
        if key in seen:
            a, b = find(group), find(seen[key])
            if a != b:
                parents[max(a,b)] = min(a,b)
        seen[key] = group
    base = dict(zip(base_keys, np.asarray(base_val, dtype=bool)))
    overlaps, defaults = {}, {}
    for key, group, val in zip(policy_keys, policy_groups, policy_val):
        group = find(str(group))
        defaults.setdefault(group, bool(val))
        if key in base:
            overlaps.setdefault(group, set()).add(bool(base[key]))
    keep, validation = [], []
    for group in policy_groups:
        group = find(str(group)); sides = overlaps.get(group, set())
        keep.append(len(sides) < 2)
        validation.append(next(iter(sides)) if len(sides) == 1 else defaults[group])
    return np.array(keep), np.array(validation)


def assemble(base, policy, policy_weights, fraction, out):
    """Merge value evidence and retain policy supervision only from replay."""
    if not 0 < fraction < 1:
        raise ValueError('The policy value fraction must be between zero and one.')
    base_keys, policy_keys = state_keys(base), state_keys(policy)
    if len(set(base_keys)) != len(base_keys):
        raise ValueError('The replay dataset contains duplicate canonical states.')
    keep, policy_val = reconcile_groups(base_keys, base['is_val'], policy_keys,
                                      policy['group_id'], policy['is_val'])
    base_mass = (1-fraction)*np.asarray(base['weight'])*np.asarray(base['value_sample_weight'])
    policy_weights = np.asarray(policy_weights, dtype=np.float64)
    counts = collections.Counter(policy_keys[i] for i in np.flatnonzero(keep))
    policy_weights = np.array([weight/counts[key] if counts[key] else 0
                            for key,weight in zip(policy_keys,policy_weights)])
    eligible = keep & ~policy_val
    if not eligible.any():
        raise ValueError('No policy training groups remain after split reconciliation.')
    scale = fraction/(1-fraction)*base_mass[~base['is_val']].sum()/policy_weights[eligible].sum()
    policy_mass = policy_weights*scale
    locations = {key:i for i,key in enumerate(base_keys)}
    new_keys = []
    for key in (key for key, allowed in zip(policy_keys,keep) if allowed):
        if key not in locations:
            locations[key] = len(base_keys)+len(new_keys); new_keys.append(key)
    n, old_n = len(locations), len(base_keys)
    fields = out/'fields'; fields.mkdir(parents=True,exist_ok=True)
    arrays = {}
    for key in STATE_FIELDS:
        arrays[key] = np.lib.format.open_memmap(fields/(key+'.npy'), mode='w+',
                                               dtype=base[key].dtype, shape=(n,))
    for key in ('value','weight','value_sample_weight','policy_sample_weight','aux_sample_weight'):
        arrays[key] = np.lib.format.open_memmap(fields/(key+'.npy'), mode='w+',dtype=np.float32,shape=(n,))
    for key in ('is_val','policy_available'):
        arrays[key] = np.lib.format.open_memmap(fields/(key+'.npy'), mode='w+',dtype=bool,shape=(n,))
    for key in ('policy','aux_policy_target'):
        arrays[key] = np.lib.format.open_memmap(fields/(key+'.npy'), mode='w+',dtype=np.float32,shape=(n,209))
        arrays[key][old_n:] = 1/209
        arrays[key][:old_n] = base[key]
    for key in STATE_FIELDS:
        arrays[key][:old_n] = base[key]
    arrays['value'][:] = 0
    arrays['value'][:old_n] = base['value']
    arrays['is_val'][:old_n] = base['is_val']
    arrays['policy_available'][:] = np.arange(n) < old_n
    arrays['weight'][:] = 1
    arrays['value_sample_weight'][:old_n] = base_mass
    arrays['value_sample_weight'][old_n:] = 0
    for key in ('policy_sample_weight','aux_sample_weight'):
        arrays[key][:old_n] = np.asarray(base['weight'])*np.asarray(base[key])
        # Positive placeholders pass the original loader. The scoped loader
        # applies policy_available before any loss or optimizer step.
        arrays[key][old_n:] = 1
    merged = 0
    for i in np.flatnonzero(keep):
        destination = locations[policy_keys[i]]
        previous = float(arrays['value_sample_weight'][destination])
        if previous:
            merged += 1
        else:
            for key in STATE_FIELDS:
                arrays[key][destination] = policy[key][i]
            arrays['is_val'][destination] = policy_val[i]
        mass = policy_mass[i]
        arrays['value'][destination] = (float(arrays['value'][destination])*previous+float(policy['value'][i])*mass)/(previous+mass) if previous else policy['value'][i]
        arrays['value_sample_weight'][destination] = previous+mass
    # Equalize the global denominator so policy and replay-value losses keep
    # their original mean scale despite the additional value-only examples.
    normalization = np.count_nonzero(~arrays['is_val'])/np.count_nonzero(~base['is_val'])
    for key in ('value_sample_weight','policy_sample_weight','aux_sample_weight'):
        arrays[key][:] *= normalization
    for array in arrays.values():
        array.flush()
    info = dict(samples=n,train_samples=int((~arrays['is_val']).sum()),
                validation_samples=int(arrays['is_val'].sum()),policy_rows_accepted=int(keep.sum()),
                policy_rows_excluded_for_group_leakage=int((~keep).sum()),merged_value_rows=merged,
                added_canonical_states=len(new_keys),policy_training_value_fraction=fraction,
                policy_supervision='Replay only; value-only rows have exactly zero main and auxiliary policy loss.')
    return arrays, info


def prepare_mix(cfg, reference):
    out = Path(cfg['mixed_dataset']);out.mkdir(parents=True,exist_ok=True)
    identity = dict(base_manifest_sha256=sha(Path(cfg['dataset'])/'dataset.manifest.json'),
                    archive_sha256=sha(cfg['archive']),reference_sha256=sha(reference/'student.bin'),
                    external_preparer_sha256=sha(ROOT/'training/finetune_external_value.py'),
                    config=cfg,runner_sha256=sha(__file__))
    manifest_path = out/'dataset.manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('identity') != identity:
            raise ValueError('The mixed dataset identity changed; use a new output directory.')
        return manifest
    external = out/'external';external.mkdir(exist_ok=True)
    policy, external_info = prepare(dict(EXTERNAL_CONFIG,archive=cfg['archive'],seed=cfg['seed']),external)
    base = trainer.load_dataset(Path(cfg['dataset']))
    architecture = json.loads((reference/'student.architecture.json').read_text())
    torch.set_num_threads(cfg['workers'])
    model = Student(architecture['architecture'],architecture['hidden'],qat=True)
    model.load_float(reference/'student.bin')
    weights = np.empty(len(policy['value']),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(weights),4096):
            indices = np.arange(start,min(start+4096,len(weights)))
            value,_ = model(torch.from_numpy(encode_features(policy,indices,model.architecture)))
            weights[indices] = 1+np.minimum(np.abs(value.sigmoid().numpy()-(policy['value'][indices]+1)/2),.5)
    _, info = assemble(base,policy,weights,cfg['policy_value_fraction'],out)
    manifest = dict(schema='zquoridor.policy_mix_dataset.v1',complete=True,
                    group_separation_verified=True,identity=identity,external_info=external_info,**info,arrays={})
    for path in sorted((out/'fields').glob('*.npy')):
        manifest['arrays'][path.stem] = dict(filename=str(path.relative_to(out)),sha256=sha(path))
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


_original_loader = trainer.load_dataset
def masked_loader(path):
    """Apply exact policy masks without changing the active width trainer."""
    data = _original_loader(path)
    if 'policy_available' in data:
        mask = np.asarray(data['policy_available'])
        if mask.shape != data['value'].shape or mask.dtype != np.bool_:
            raise ValueError('The policy availability mask is invalid.')
        for key in ('policy_sample_weight','aux_sample_weight'):
            data[key] = np.asarray(data[key])*mask
    return data


def replay_validation(folder, config, data):
    """Report real replay targets without placeholder policy diagnostics."""
    model = Student(config['architecture'],config['hidden'],qat=True)
    if config['aux_policy_enabled']:
        model.enable_aux_policy()
    checkpoint = torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    metrics = trainer._epoch(model,data,np.flatnonzero(data['is_val']),config,'cpu')
    (folder/'replay_validation.json').write_text(json.dumps(metrics,indent=2)+'\n')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key,value in CONFIG.items():
        parser.add_argument('--'+key.replace('_','-'),type=type(value),default=value)
    cfg = vars(parser.parse_args());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    dependency = Path(cfg['dependency'])
    def status(phase,**extra):
        payload = dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),promoted=False,**extra)
        temporary=out/'status.tmp';temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(out/'status.json')
        print(json.dumps(payload),flush=True)
    (out/'campaign_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    try:
        status('waiting_width_screen',dependency=str(dependency))
        while True:
            state = json.loads((dependency/'status.json').read_text())
            if state['phase']=='failed':
                raise RuntimeError('The dependency failed: '+state.get('error','unknown error'))
            if state['phase']=='complete':
                break
            time.sleep(15)
        dependency_report = json.loads((dependency/'report.json').read_text())
        summary = dependency_report['h2h']['summary']
        if summary.get('failed_games',0) or summary.get('score_pct') is None:
            raise RuntimeError('The width screen lacks usable strength evidence.')
        selected = 'candidate' if summary['score_pct']>50 else 'control'
        reference = dependency/selected
        (out/'selection.json').write_text(json.dumps(dict(participant=selected,
            width_screen_sha256=sha(dependency/'report.json'),reference_sha256=sha(reference/'student.bin'),
            reason='Higher aggregate point score; equality favors the smaller control. This is an experimental selection, not production promotion.'),indent=2)+'\n')
        status('preparing_mix',selected=selected)
        manifest = prepare_mix(cfg,reference)
        trainer.load_dataset = masked_loader
        reference_config = json.loads((reference/'requested_config.json').read_text())
        reports = {}; replay = trainer.load_dataset(Path(cfg['dataset']))
        for label,dataset in [('control',cfg['dataset']),('candidate',cfg['mixed_dataset'])]:
            folder=out/label;folder.mkdir(exist_ok=True)
            training = dict(reference_config,data=dataset,out_dir=str(folder),
                init_from=str(reference/'student.bin'),init_hidden=reference_config['hidden'],
                epochs=cfg['epochs'],lr=cfg['lr'],min_lr=cfg['min_lr'],seed=cfg['seed'],
                checkpoint_every=5,resume=True,patience=0,
                policy_mask_runner_sha256=sha(__file__))
            (folder/'requested_config.json').write_text(json.dumps(training,indent=2)+'\n')
            status('training',participant=label,hidden=training['hidden'],epochs=cfg['epochs'])
            reports[label]=trainer.train(training)
            if reports[label]['epochs'] != cfg['epochs']:
                raise RuntimeError('The fine-tune did not complete the requested epochs.')
            trainer.build_candidate(training);parity(folder)
            reports[label]['original_replay_validation'] = replay_validation(folder,training,replay)
        status('h2h',pairs=cfg['pairs'],movetime_ms=cfg['movetime_ms'])
        match_cfg=dict(match_finalists.CONFIG,engine1_name='policy_mix',engine2_name='replay_only_control',
            engine1_executable=str(out/'candidate/zquoridor'),engine1_nnue=str(out/'candidate/student_int8.bin'),
            engine2_executable=str(out/'control/zquoridor'),engine2_nnue=str(out/'control/student_int8.bin'),
            pairs=cfg['pairs'],move_time_ms=cfg['movetime_ms'],workers=cfg['workers'],seed=cfg['seed'],
            openings=dependency_report['config']['openings'],output=str(out/'h2h'))
        categories=run_benchmark._read_opening_categories(Path(match_cfg['openings']))
        matches={}
        for label,opponent in [('control',out/'control'),('initial',reference)]:
            status('h2h',opponent=label,pairs=cfg['pairs'],movetime_ms=cfg['movetime_ms'])
            match_cfg.update(engine2_name='replay_only_control' if label=='control' else 'selected_width_initial',
                engine2_executable=str(opponent/'zquoridor'),engine2_nnue=str(opponent/'student_int8.bin'),
                output=str(out/('h2h_'+label)))
            match=match_finalists.run(match_cfg)
            rows=[json.loads(line) for line in (Path(match_cfg['output'])/'games.jsonl').read_text().splitlines()]
            match['by_category']=run_benchmark.summarize_by_category(rows,categories,bootstrap=20000,seed=cfg['seed'])
            matches[label]=match
        report=dict(config=cfg,selection=selected,dataset=manifest,training=reports,h2h=matches,promoted=False)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        status('complete',summaries={label:match['summary'] for label,match in matches.items()})
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':
    main()
