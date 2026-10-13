#!/usr/bin/env python3
"""Run a short value-only experiment from an external JSONL teacher archive."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from student_model import Student, encode_features, export, get_phase_bucket

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "archive": str(ROOT / "data/teaching/external-value/titanium_mix_v2.zip"),
    "output": str(ROOT / "experimental/experiments/external-value-short-20261013"),
    "init": str(ROOT / "data/nnue/nnue_weights.bin"),
    "architecture": "multipath_phase_contact_bucketed",
    "hidden": 512, "epochs": 3, "lr": 1e-5, "batch_size": 2048,
    "threads": 3, "seed": 20261013, "validation_fraction": 0.05,
    "disagreement_scale": 3.0, "min_lr": None, "weight_reference": None,
}


def epoch_lr(cfg, epoch):
    """Return a linear schedule that includes both endpoint rates."""
    if cfg['min_lr'] is None or cfg['epochs'] == 1:
        return cfg['lr']
    return cfg['lr'] + (cfg['min_lr'] - cfg['lr']) * (epoch - 1) / (cfg['epochs'] - 1)


def prepare(cfg, out):
    records, parents = {}, {}

    def find(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    counts = collections.Counter()
    with zipfile.ZipFile(cfg["archive"]) as archive:
        names = [n for n in archive.namelist() if n.endswith('.jsonl.gz')]
        if len(names) != 1:
            raise ValueError("The archive must contain one compressed JSONL file.")
        with gzip.GzipFile(fileobj=archive.open(names[0])) as stream:
            for line in stream:
                row = json.loads(line)
                state = row['state']
                fields = dict(x.split('=', 1) for x in state.split(';'))
                stm = int(fields['t'])
                value = float(row['v_stm'])
                if stm != row['stm'] or not -1 <= value <= 1:
                    raise ValueError("Invalid teacher value or perspective.")
                if abs((1 + value * (1 if stm == 0 else -1))/2 - row['p0_win']) > 0.000051:
                    raise ValueError("Inconsistent teacher perspectives.")
                counts[row['source']] += 1
                group = ('game:' + str(row['game_id'])) if 'game_id' in row else ('state:' + state)
                find(group)
                if state in records:
                    old = records[state]
                    union(old['group'], group)
                    old['sum'] += value
                    old['count'] += 1
                else:
                    records[state] = dict(fields=fields, sum=value, count=1,
                                          group=group, source=row['source'])
    commands, ordered = [], []
    for row in records.values():
        f = row['fields']; stm = int(f['t'])
        def cell(s):
            r, c = int(s[1])-1, ord(s[0])-97
            return (8-r if stm else r)*9+c
        def walls(s):
            if s in ('', '-'):
                return 0
            bits = 0
            for a in s.split(','):
                r, c = int(a[1])-1, ord(a[0])-97
                bits |= 1 << ((7-r if stm else r)*8+c)
            return bits
        own, opp = cell(f['p'+str(stm)]), cell(f['p'+str(1-stm)])
        wh, wv = walls(f['h']), walls(f['v'])
        ow, pw = int(f['w'+str(stm)]), int(f['w'+str(1-stm)])
        commands.append(f'@state {own} {opp} {wh} {wv} {ow} {pw}')
        ordered.append(row)
    encoder = ROOT / 'bin/teacher_encode_state'
    proc = subprocess.run([str(encoder)], input='\n'.join(commands)+'\n', text=True,
                          capture_output=True, check=True)
    lines = proc.stdout.splitlines()
    if len(lines) != len(ordered):
        raise ValueError("The encoder returned an incorrect row count.")
    keys = ['own_pawn', 'opp_pawn', 'walls_h', 'walls_v', 'walls_left_own',
            'walls_left_opp', 'own_dist', 'opp_dist', 'mover']
    columns = {k: [] for k in keys}; values, splits, sources = [], [], []
    excluded = collections.Counter()
    for row, line in zip(ordered, lines):
        parts = line.split('\t')
        if parts[1] != 'ok':
            excluded[parts[-1].split(': ', 1)[-1]] += 1
            continue
        for k, v in zip(keys, parts[2:]):
            columns[k].append(int(v))
        group = find(row['group'])
        fraction = int(hashlib.sha256((str(cfg['seed'])+group).encode()).hexdigest()[:16], 16) / 2**64
        splits.append(fraction < cfg['validation_fraction'])
        values.append(row['sum']/row['count'])
        sources.append(row['source'])
    data = {k: np.array(v, dtype=np.uint64 if k.startswith('walls_') and k in ('walls_h','walls_v') else np.int64) for k,v in columns.items()}
    data['value'] = np.array(values, dtype=np.float32)
    data['is_val'] = np.array(splits, dtype=bool)
    data['source'] = np.array(sources)
    np.savez_compressed(out/'dataset.npz', **data)
    info = dict(raw_rows=sum(counts.values()), source_counts=dict(counts), unique_states=len(records),
                usable_states=len(values), excluded=dict(excluded), validation_states=sum(splits),
                split='Connected game groups and repeated states. Rows without game IDs use state groups.',
                duplicate_target='Mean value for identical state text.',
                provenance='External teacher data. No claim that games involved Zquoridor.')
    print(json.dumps(info), flush=True)
    return data, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('archive', 'output', 'init'):
        parser.add_argument('--'+k, default=CONFIG[k])
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--lr', type=float, default=CONFIG['lr'])
    parser.add_argument('--min-lr', type=float, default=CONFIG['min_lr'])
    parser.add_argument('--weight-reference', default=CONFIG['weight_reference'])
    cfg = dict(CONFIG, **vars(parser.parse_args()))
    if cfg['epochs'] < 1 or cfg['lr'] <= 0 or (cfg['min_lr'] is not None and not 0 < cfg['min_lr'] <= cfg['lr']):
        raise ValueError('Epochs and learning rates must be positive. The final rate cannot exceed the initial rate.')
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg['threads']); torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    started = time.time()
    data, info = prepare(cfg, out)
    model = Student(cfg['architecture'], cfg['hidden'], qat=True)
    model.load_float(cfg['init'])
    reference = None
    if cfg['weight_reference']:
        reference = Student(cfg['architecture'], cfg['hidden'], qat=True)
        reference.load_float(cfg['weight_reference'])
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith('value'))
    n = len(data['value'])
    activations = torch.empty((n, cfg['hidden']), dtype=torch.uint8)
    buckets = torch.from_numpy(get_phase_bucket(data['walls_left_own'] + data['walls_left_opp']))
    baseline = torch.empty(n)
    reference_predictions = torch.empty(n) if reference is not None else baseline
    with torch.no_grad():
        quant_w = torch.round(model.fc1.weight * 255)
        quant_b = torch.round(model.fc1.bias * 255)
        for start in range(0, n, 4096):
            idx = np.arange(start, min(n, start+4096))
            x = torch.from_numpy(encode_features(data, idx, cfg['architecture']))
            if reference is not None:
                reference_predictions[idx] = reference(x)[0].sigmoid()
            a = torch.floor(F.linear(x, quant_w, quant_b).clamp(0,255).square()/255)
            activations[idx] = a.to(torch.uint8)
            for b in range(model.value_buckets):
                mask = buckets[idx] == b
                if mask.any():
                    baseline[torch.as_tensor(idx)[mask]] = model._value_head(b, a[mask]/255, a[mask]).sigmoid()
            if start % 65536 == 0:
                print(f'Cached {min(n,start+4096)}/{n} accumulators.', flush=True)
    target = torch.from_numpy((data['value']+1)/2)
    weights = 1 + cfg['disagreement_scale'] * (reference_predictions-target).abs()
    train = np.flatnonzero(~data['is_val']); val = np.flatnonzero(data['is_val'])
    weights /= weights[train].mean()
    np.savez_compressed(out/'initial_predictions.npz', baseline_probability=baseline.numpy(),
                        weight=weights.numpy(), is_val=data['is_val'])
    def logits(idx):
        a = activations[idx].float()
        result = torch.zeros(len(idx))
        for b in range(model.value_buckets):
            mask = buckets[idx] == b
            if mask.any():
                result[mask] = model._value_head(b, a[mask]/255, a[mask])
        return result
    def evaluate():
        sums = collections.Counter(); count = 0
        with torch.no_grad():
            for start in range(0,len(val),cfg['batch_size']):
                idx = val[start:start+cfg['batch_size']]
                z = logits(idx); p = z.sigmoid(); t = target[idx]
                sums['bce'] += F.binary_cross_entropy_with_logits(z,t,reduction='sum').item()
                sums['brier'] += (p-t).square().sum().item()
                sums['mae_value'] += (2*(p-t).abs()).sum().item()
                count += len(idx)
        return {k:v/count for k,v in sums.items()}
    original = {k:v.detach().clone() for k,v in model.state_dict().items()}
    before = evaluate(); print('Baseline '+json.dumps(before),flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=1e-5)
    history = []; best = float('inf'); best_epoch = None
    for epoch in range(1,cfg['epochs']+1):
        rate = epoch_lr(cfg, epoch)
        for group in optimizer.param_groups:
            group['lr'] = rate
        order = rng.permutation(train); total = 0; mass = 0
        for start in range(0,len(order),cfg['batch_size']):
            idx = order[start:start+cfg['batch_size']]
            z = logits(idx); w = weights[idx]
            loss = (F.binary_cross_entropy_with_logits(z,target[idx],reduction='none')*w).sum()/w.sum()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1)
            optimizer.step(); model.clip_weights()
            total += loss.item()*w.sum().item(); mass += w.sum().item()
        metrics = dict(epoch=epoch, learning_rate=rate, train_weighted_bce=total/mass, validation=evaluate())
        history.append(metrics); print(json.dumps(metrics),flush=True)
        export(model,out/f'epoch_{epoch}.bin')
        if metrics['validation']['bce'] < best:
            best = metrics['validation']['bce']; best_epoch = epoch
            torch.save(model.state_dict(),out/'best.pt')
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True))
    unchanged = all(torch.equal(v,original[k]) for k,v in model.state_dict().items() if not k.startswith('value'))
    if not unchanged:
        raise AssertionError('A frozen parameter changed.')
    manifest = export(model,out/'candidate.bin')
    report = dict(config=cfg, dataset=info, initial_weights_sha256=hashlib.sha256(Path(cfg['init']).read_bytes()).hexdigest(),
                  archive_sha256=hashlib.sha256(Path(cfg['archive']).read_bytes()).hexdigest(),
                  baseline_validation=before, epochs=history, best_epoch=best_epoch, final_validation=evaluate(),
                  frozen_parameters_unchanged=unchanged, weight_formula='1 + 3 * abs(initial_student_probability - teacher_probability), normalized on training rows',
                  weight_reference_sha256=hashlib.sha256(Path(cfg['weight_reference'] or cfg['init']).read_bytes()).hexdigest(),
                  learning_rate_schedule='Linear between endpoint rates.' if cfg['min_lr'] is not None else 'Constant.',
                  weight_min=weights.min().item(),weight_max=weights.max().item(),
                  training_scope='Value heads only. Quantization-aware training. Fixed cached accumulator.',
                  elapsed_seconds=time.time()-started, export=manifest,
                  strength_status='No Elo claim. Candidate only. Production files remain unchanged.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
