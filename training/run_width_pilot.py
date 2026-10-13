#!/usr/bin/env python3
"""Train matched width controls, verify native parity, and run paired games."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import run_experiment as trainer
from training.run_temporal_campaign import parity
from tools import match_finalists, run_benchmark

CONFIG = {
    'output': str(ROOT/'experimental/experiments/width-100ep-20261013'),
    'reference': str(ROOT/'experimental/experiments/temporal-search-100ep-20261013'),
    'dataset': str(ROOT/'data/teaching/drive-temporal-100ep/dataset'),
    'openings': str(ROOT/'results/benchmarks/temporal-search-100ep-strength-20261013/openings.jsonl'),
    'hidden': 768, 'epochs': 100, 'lr': 1e-5, 'min_lr': 1e-7,
    'pairs': 100, 'movetime_ms': 200, 'workers': 3, 'seed': 20261017,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        parser.add_argument('--'+key.replace('_','-'), type=type(value), default=value)
    cfg = vars(parser.parse_args())
    if cfg['hidden'] not in (768, 1024):
        raise ValueError('The candidate width must be 768 or 1024.')
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    reference = Path(cfg['reference'])
    reference_config = json.loads((reference/'requested_config.json').read_text())
    baseline_width = reference_config['hidden']
    def status(phase, **extra):
        payload = dict(phase=phase, pid=os.getpid(), updated_unix=time.time(), promoted=False, **extra)
        temporary = out/'status.tmp'
        temporary.write_text(json.dumps(payload, indent=2)+'\n')
        temporary.replace(out/'status.json')
        print(json.dumps(payload), flush=True)
    (out/'campaign_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    try:
        reports = {}
        for label, width in [('control', baseline_width), ('candidate', cfg['hidden'])]:
            folder = out/label
            folder.mkdir(parents=True, exist_ok=True)
            config = dict(reference_config, data=cfg['dataset'], out_dir=str(folder),
                          init_from=str(reference/'student.bin'), init_hidden=baseline_width,
                          init_architecture=reference_config['architecture'], hidden=width,
                          epochs=cfg['epochs'], lr=cfg['lr'], min_lr=cfg['min_lr'],
                          seed=cfg['seed'], checkpoint_every=5, resume=True,
                          cpu_threads=cfg['workers'], build=True, benchmark=False)
            (folder/'requested_config.json').write_text(json.dumps(config, indent=2)+'\n')
            status('training', participant=label, hidden=width, epochs=cfg['epochs'])
            reports[label] = trainer.train(config)
            if reports[label]['epochs'] != cfg['epochs']:
                raise RuntimeError('The participant did not complete the requested epochs.')
            status('verifying', participant=label, hidden=width)
            trainer.build_candidate(config)
            parity(folder)
        status('h2h', pairs=cfg['pairs'], movetime_ms=cfg['movetime_ms'])
        match_cfg = dict(match_finalists.CONFIG,
                         engine1_name=f"width_{cfg['hidden']}",
                         engine1_executable=str(out/'candidate/zquoridor'),
                         engine1_nnue=str(out/'candidate/student_int8.bin'),
                         engine2_name=f'width_{baseline_width}_matched_control',
                         engine2_executable=str(out/'control/zquoridor'),
                         engine2_nnue=str(out/'control/student_int8.bin'),
                         pairs=cfg['pairs'], move_time_ms=cfg['movetime_ms'],
                         workers=cfg['workers'], seed=cfg['seed'],
                         openings=cfg['openings'], output=str(out/'h2h'))
        match = match_finalists.run(match_cfg)
        rows = [json.loads(line) for line in (out/'h2h/games.jsonl').read_text().splitlines()]
        categories = run_benchmark._read_opening_categories(Path(cfg['openings']))
        match['by_category'] = run_benchmark.summarize_by_category(
            rows, categories, bootstrap=match_cfg['bootstrap'], seed=cfg['seed'])
        report = dict(config=cfg, training=reports, h2h=match, promoted=False)
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        status('complete', summary=match['summary'])
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
