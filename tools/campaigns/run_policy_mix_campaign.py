#!/usr/bin/env python3
"""Wait for a width screen, then compare mixed policy teaching with replay alone."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training import run_experiment as trainer
from training.native_parity import parity
from training.policy_mix import masked_loader, prepare_mix, sha
from training.student_model import Student
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
