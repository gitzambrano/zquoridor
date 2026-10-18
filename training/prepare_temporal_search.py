#!/usr/bin/env python3
"""Prepare trajectory-aware search targets from downloaded Drive artifacts."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import subprocess

import numpy as np
import torch
from student_model import Student, encode_features
from read_selfplay import SAMPLE_DTYPE
from prepare_replay import META_DTYPE
from compute_surprise_weights import compute_normalized_multipliers
from target_options import target_settings

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    'downloads': str(ROOT/'data/teaching/drive-temporal-100ep/downloads.jsonl'),
    'output': str(ROOT/'data/teaching/drive-temporal-100ep/dataset'),
    'reference': str(ROOT/'data/nnue/nnue_weights.bin'),
    'architecture': 'multipath_phase_contact_bucketed', 'hidden': 512,
    'target_mode': 'temporal',
    'horizons': [0, 4, 8, 16], 'horizon_weights': [0.5, 0.25, 0.125, 0.0625],
    'outcome_weight': 0.1, 'val_fraction': 0.05, 'seed': 20261013,
    'source_weights': {'selfplay_v301': 0.8, 'selfplay_contact_soup_858': 1.0,
                       'selfplay_858_v1': 1.0, 'claustro_search_games_v1': 1.5,
                       'wall_drops_v1': 0.5, 'gumbel_sims_v1': 1.25},
    'surprise_alpha': 0.5, 'surprise_cap': 4.0, 'threads': 3,
}
STATE_KEYS = ('own_pawn', 'opp_pawn', 'walls_h', 'walls_v', 'walls_left_own',
              'walls_left_opp', 'own_dist', 'opp_dist')


def temporal_target(ply, side, values, length, outcome, horizons, coefficients, terminal_weight):
    """Blend exact available horizons in the original mover perspective."""
    total = 0.0; mass = 0.0; future = False
    for offset, coefficient in zip(horizons, coefficients):
        position = ply + offset
        if position >= length:
            value = outcome
        elif position in values:
            mover, value = values[position]
            value *= 1 if mover == side else -1
        else:
            continue
        total += coefficient * value; mass += coefficient
        future |= offset > 0
    if mass <= 0:
        raise ValueError('The current search value is missing.')
    return (1-terminal_weight)*total/mass + terminal_weight*outcome, future


def split_group(group, cfg):
    number = int(hashlib.sha256((str(cfg['seed'])+group).encode()).hexdigest()[:16],16)/2**64
    return 2 if number < cfg['val_fraction'] else 1


def canonical_policy(root, side):
    """Return current-position visits in the native canonical action frame."""
    visits = np.asarray(root.get('visit_counts_global', []), dtype=np.float64)
    if visits.shape != (209,) or not np.isfinite(visits).all() or (visits < 0).any() or visits.sum() <= 0:
        return None
    if side:
        pawn = np.arange(81).reshape(9,9)[::-1].ravel()
        wall = np.arange(64).reshape(8,8)[::-1].ravel()
        visits = visits[np.r_[pawn,81+wall,145+wall]]
    return (visits/visits.sum()).astype(np.float32)


def encode(commands):
    executable = ROOT/'bin/teacher_encode_state'
    if not executable.exists():
        executable = executable.with_suffix('.exe')
    if len(commands) > 2048:
        return [state for start in range(0, len(commands), 2048) for state in encode(commands[start:start+2048])]
    result = subprocess.run([str(executable)],input='\n'.join(commands)+'\n',
                            text=True,capture_output=True,check=True)
    lines = result.stdout.splitlines()
    if len(lines) != len(commands):
        raise ValueError('The native encoder returned an incorrect row count.')
    output = []
    for line in lines:
        parts = line.split('\t')
        output.append(None if parts[1] != 'ok' else tuple(int(v) for v in parts[2:10]))
    return output


def raw_games(file, cfg, stats):
    commands, rows = [], []
    with gzip.open(file['local'],'rt') as stream:
        for text in stream:
            game = json.loads(text)
            if game.get('status') != 'ok' or game.get('termination') not in ('goal','repetition'):
                stats['excluded_incomplete_games'] += 1; continue
            winner = game.get('winner'); length = len(game['moves'])
            grouped = collections.defaultdict(list)
            for search in game.get('searches',[]):
                root = search.get('root')
                if not root or root.get('root_value_perspective') != 'side-to-move':
                    continue
                policy = canonical_policy(root,search['side_to_move'])
                value = root.get('root_value')
                if policy is None or value is None or not math.isfinite(value) or abs(value)>1:
                    continue
                if root.get('raw',{}).get('stop_reason') == 'solved':
                    stats['excluded_solver_roots'] += 1; continue
                quality = min(2.0,max(0.5,math.sqrt(search.get('budget_ms',50)/200)))
                quality *= min(1.0,max(0.1,sum(root['visit_counts_global'])/64))
                teacher = search['player']=='claustrophobia'
                grouped[search['ply']].append((search,policy,float(value),quality*(3 if teacher else 1),quality))
            merged = {}
            for ply, items in grouped.items():
                main_mass = sum(item[3] for item in items); aux_mass = sum(item[4] for item in items)
                policy = sum(item[1]*item[3] for item in items)/main_mass
                auxiliary = sum(item[1]*item[4] for item in items)/aux_mass
                value = sum(item[2]*item[3] for item in items)/main_mass
                side = items[0][0]['side_to_move']
                merged[ply] = (side,value,policy,auxiliary,items[0][0],main_mass/len(items))
            values = {ply:(item[0],item[1]) for ply,item in merged.items()}
            group = 'opening:'+game.get('opening_group', game['run_id']+':'+str(game['pair_index']))
            for ply,(side,value,policy,auxiliary,search,quality) in merged.items():
                outcome = 0.0 if winner not in (0,1) else 1.0 if winner==side else -1.0
                value,future = temporal_target(ply,side,values,length,outcome,cfg['horizons'],cfg['horizon_weights'],cfg['outcome_weight'])
                stats['temporal_rows'] += 1; stats['rows_with_future'] += future
                commands.append(' '.join(search['history']))
                confidence = cfg['source_weights'][file['kind']]*min(2.5,quality)
                rows.append((value,policy,auxiliary,confidence,split_group(group,cfg),file['kind']))
    states = encode(commands) if commands else []
    for state,row in zip(states,rows):
        if state is None:
            stats['excluded_invalid_states'] += 1
        else:
            yield state,row


def stored_shard(file, meta_file, cfg, stats):
    samples = np.fromfile(file['local'],dtype=SAMPLE_DTYPE)
    metadata = np.fromfile(meta_file['local'],dtype=META_DTYPE)
    if len(samples) != len(metadata):
        raise ValueError('The shard and metadata lengths differ: '+file['path'])
    groups = collections.defaultdict(list)
    for i, meta in enumerate(metadata):
        if int(meta['flags']) & 2 and math.isfinite(float(meta['root'])) and 0<=meta['root']<=1:
            groups[int(meta['game'])].append(i)
        else:
            stats['excluded_missing_search_roots'] += 1
    commands, rows = [], []
    for game, indices in groups.items():
        values = {int(metadata[i]['length'])-int(metadata[i]['plies']):(int(samples[i]['mover']),2*float(metadata[i]['root'])-1) for i in indices}
        group = file['path']+':'+str(game)
        for i in indices:
            row,meta = samples[i],metadata[i]
            policy = np.zeros(209,dtype=np.float32)
            if np.any(row['policy_top_idx']>=209):
                stats['excluded_bad_policy'] += 1; continue
            np.add.at(policy,row['policy_top_idx'].astype(np.int64),row['policy_top_prob'].astype(np.float32))
            if policy.sum()<=0:
                stats['excluded_zero_visits'] += 1; continue
            policy /= policy.sum()
            state = tuple(int(row[k]) for k in STATE_KEYS)
            ply = int(meta['length'])-int(meta['plies']); side = int(row['mover']); outcome = float(row['game_result'])
            value,future = temporal_target(ply,side,values,int(meta['length']),outcome,cfg['horizons'],cfg['horizon_weights'],cfg['outcome_weight'])
            stats['temporal_rows'] += 1; stats['rows_with_future'] += future
            commands.append('@state '+' '.join(str(v) for v in state[:6]))
            rows.append((value,policy,policy,cfg['source_weights'][file['kind']],split_group(group,cfg),file['kind']))
    states = encode(commands) if commands else []
    for state,row in zip(states,rows):
        if state is None:
            stats['excluded_invalid_states'] += 1
        else:
            yield state,row


def wall_positions(file, targets_file, cfg, stats):
    positions = [json.loads(line) for line in Path(file['local']).read_text().splitlines() if line.strip()]
    with np.load(targets_file['local'],allow_pickle=False) as targets:
        policies,values = targets['policy'].copy(),targets['value'].copy()
        identifiers = targets['sample_ids'].tolist()
    lookup = {str(identifier.decode() if isinstance(identifier,bytes) else identifier):i for i,identifier in enumerate(identifiers)}
    commands = [' '.join(pos['history']) for pos in positions]
    for pos,state in zip(positions,encode(commands)):
        if state is None:
            stats['excluded_invalid_states'] += 1; continue
        i = lookup[pos['id']];policy = policies[i];value = float(values[i])
        if not np.isfinite(policy).all() or policy.sum()<=0 or not math.isfinite(value) or abs(value)>1:
            stats['excluded_bad_wall_target'] += 1;continue
        policy = policy/policy.sum()
        yield state,(value,policy,policy,cfg['source_weights'][file['kind']],split_group('wall:'+pos['id'],cfg),file['kind'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads',default=CONFIG['downloads']);parser.add_argument('--output',default=CONFIG['output'])
    parser.add_argument('--reference',default=CONFIG['reference']);parser.add_argument('--hidden',type=int,default=CONFIG['hidden'])
    parser.add_argument('--architecture',default=CONFIG['architecture'])
    parser.add_argument('--target-mode',choices=('current','temporal'),default=CONFIG['target_mode'])
    parser.add_argument('--horizons',type=json.loads,default=None)
    parser.add_argument('--horizon-weights',type=json.loads,default=None)
    parser.add_argument('--outcome-weight',type=float,default=CONFIG['outcome_weight'])
    cfg = dict(CONFIG,**vars(parser.parse_args()));out = Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    cfg.update(target_settings(cfg['target_mode'],cfg['horizons'],cfg['horizon_weights'],cfg['outcome_weight']))
    text = Path(cfg['downloads']).read_text()
    files = [json.loads(line) for line in text.splitlines() if line.strip()] if cfg['downloads'].endswith('.jsonl') else json.loads(text)
    lookup = {f['path']:f for f in files}
    stats = collections.Counter(); sources = collections.Counter()
    db = sqlite3.connect(out/'states.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=NORMAL')
    db.create_function('add_vectors',2,lambda a,b:(np.frombuffer(a,dtype='<f4')+np.frombuffer(b,dtype='<f4')).astype('<f4').tobytes())
    db.execute('CREATE TABLE IF NOT EXISTS states (key BLOB PRIMARY KEY,state BLOB,value REAL,policy BLOB,aux BLOB,mass REAL,n INTEGER,split INTEGER)')
    db.execute('CREATE TABLE IF NOT EXISTS completed (path TEXT PRIMARY KEY)')
    db.execute('CREATE TABLE IF NOT EXISTS provenance (path TEXT PRIMARY KEY,source TEXT,count INTEGER,stats TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS identity (fingerprint TEXT PRIMARY KEY)')
    identity_config = {k:v for k,v in cfg.items() if k not in ('downloads','output')}
    identity_config['preparer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = hashlib.sha256(json.dumps(identity_config,sort_keys=True).encode()).hexdigest()
    previous = db.execute('SELECT fingerprint FROM identity').fetchone()
    if previous is not None and previous[0] != identity:
        raise ValueError('The preparation settings changed. Use a new output directory.')
    if previous is None:
        if db.execute('SELECT COUNT(*) FROM states').fetchone()[0]:
            raise ValueError('The existing preparation cache has no identity certificate.')
        db.execute('INSERT INTO identity VALUES (?)',(identity,));db.commit()
    processed = {row[0] for row in db.execute('SELECT path FROM completed')}
    for f in files:
        if f['path'] in processed:
            continue
        iterator = None
        before_stats = stats.copy()
        if f['title']=='games.jsonl.gz':
            iterator = raw_games(f,cfg,stats)
        elif f['title'].endswith('.bin'):
            metadata = lookup.get(f['path'][:-4]+'.meta')
            if metadata is None:
                stats['excluded_unpaired_shards'] += 1;continue
            iterator = stored_shard(f,metadata,cfg,stats)
        elif f['title']=='positions.jsonl' and f['kind'] in ('wall_drops_v1','gumbel_sims_v1'):
            target = lookup.get(str(Path(f['path']).with_name('targets.npz')))
            if target is not None:
                iterator = wall_positions(f,target,cfg,stats)
        if iterator is None:
            continue
        count = 0
        for state,(value,policy,aux,mass,split,source) in iterator:
            key = struct.pack('<BBQQbb',*state[:6]);blob = struct.pack('<8Q',*state)
            db.execute('INSERT INTO states VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET value=value+excluded.value,policy=add_vectors(policy,excluded.policy),aux=add_vectors(aux,excluded.aux),mass=mass+excluded.mass,n=n+1,split=split|excluded.split',
                       (key,blob,value*mass,(policy*mass).astype('<f4').tobytes(),(aux*mass).astype('<f4').tobytes(),mass,1,split))
            count += 1;sources[source] += 1
        db.execute('INSERT INTO completed VALUES (?)',(f['path'],))
        delta = dict(stats-before_stats)
        db.execute('INSERT OR REPLACE INTO provenance VALUES (?,?,?,?)',(f['path'],f['kind'],count,json.dumps(delta)))
        db.commit()
        print(json.dumps({'file':f['path'],'accepted_rows':count}),flush=True)
    counts = dict(db.execute('SELECT split,COUNT(*) FROM states GROUP BY split'))
    sources = collections.Counter();stats = collections.Counter()
    for source,count,record in db.execute('SELECT source,count,stats FROM provenance'):
        sources[source] += count;stats.update(json.loads(record))
    total = counts.get(1,0)+counts.get(2,0)
    if not counts.get(1) or not counts.get(2):
        raise ValueError('The dataset must contain training and validation states.')
    fields = out/'fields';fields.mkdir(exist_ok=True)
    definitions = {k:(np.uint64 if k in ('walls_h','walls_v') else np.int16,(total,)) for k in STATE_KEYS}
    definitions.update(policy=(np.float32,(total,209)),aux_policy_target=(np.float32,(total,209)),value=(np.float32,(total,)),
                       weight=(np.float32,(total,)),value_sample_weight=(np.float32,(total,)),
                       policy_sample_weight=(np.float32,(total,)),aux_sample_weight=(np.float32,(total,)),is_val=(np.bool_,(total,)))
    arrays = {k:np.lib.format.open_memmap(fields/(k+'.npy'),mode='w+',dtype=dt,shape=shape) for k,(dt,shape) in definitions.items()}
    for i,(state,value,policy,aux,mass,n,split) in enumerate(db.execute('SELECT state,value,policy,aux,mass,n,split FROM states WHERE split IN (1,2) ORDER BY key')):
        for k,v in zip(STATE_KEYS,struct.unpack('<8Q',state)):
            arrays[k][i]=v
        arrays['value'][i]=value/mass;arrays['policy'][i]=np.frombuffer(policy,dtype='<f4')/mass
        arrays['aux_policy_target'][i]=np.frombuffer(aux,dtype='<f4')/mass
        arrays['weight'][i]=mass/n;arrays['is_val'][i]=split==2
    torch.set_num_threads(cfg['threads']);model = Student(cfg['architecture'],cfg['hidden'],qat=True);model.load_float(cfg['reference'])
    surprise = np.empty(total,dtype=np.float32);value_error=np.empty(total,dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,total,4096):
            idx = np.arange(start,min(total,start+4096));x=torch.from_numpy(encode_features(arrays,idx,cfg['architecture']))
            value,logits=model(x);policy=torch.from_numpy(np.asarray(arrays['policy'][idx]))
            surprise[idx]=(policy*(policy.clamp_min(1e-12).log()-torch.log_softmax(logits,dim=1))).sum(1).clamp_min(0).numpy()
            value_error[idx]=(value.sigmoid()-torch.from_numpy((arrays['value'][idx]+1)/2)).abs().numpy()
    train=~arrays['is_val'];base=arrays['weight']
    policy_multiplier=1+cfg['surprise_alpha']*np.minimum(surprise,cfg['surprise_cap'])
    value_multiplier=1+np.minimum(value_error,0.5)
    policy_multiplier/=np.average(policy_multiplier[train],weights=base[train]);value_multiplier/=np.average(value_multiplier[train],weights=base[train])
    arrays['policy_sample_weight'][:]=policy_multiplier;arrays['value_sample_weight'][:]=value_multiplier
    arrays['aux_sample_weight'][:]=policy_multiplier
    base[:]/=base[train].mean()
    for array in arrays.values():array.flush()
    manifest={'schema':'zquoridor.temporal_search_dataset.v1','complete':True,'group_separation_verified':True,
              'samples':total,'train_samples':counts[1],'validation_samples':counts[2],'excluded_cross_split_states':counts.get(3,0),
              'config':cfg,'stats':dict(stats),'sources':dict(sources),'reference_sha256':hashlib.sha256(Path(cfg['reference']).read_bytes()).hexdigest(),
              'downloads_sha256':hashlib.sha256(Path(cfg['downloads']).read_bytes()).hexdigest(),
              'surprise_mean':float(surprise.mean()),'arrays':{}}
    for k in arrays:
        p=fields/(k+'.npy');h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        manifest['arrays'][k]={'filename':str(p.relative_to(out)),'sha256':h.hexdigest()}
    (out/'dataset.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='arrays'},indent=2),flush=True)


if __name__=='__main__':
    main()
