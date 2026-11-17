#!/usr/bin/env python3
"""Prepare downloaded trajectories, train 100 epochs, and verify native exports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training import run_experiment as trainer
from training.native_parity import parity

CONFIG = {
    'inventory': str(ROOT/'data/teaching/drive-temporal-100ep/inventory.json'),
    'downloads': str(ROOT/'data/teaching/drive-temporal-100ep/downloads.jsonl'),
    'dataset': str(ROOT/'data/teaching/drive-temporal-100ep/dataset'),
    'output': str(ROOT/'experimental/experiments/temporal-search-100ep-20261013'),
    'epochs': 100, 'lr': 1e-5, 'min_lr': 1e-7, 'cpu_threads': 3,
}


def selected(file):
    return (file['title'].endswith(('.bin','.meta','games.jsonl.gz'))
            or file['kind']=='wall_drops_v1' and file['title'] in ('positions.jsonl','targets.npz','manifest.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=CONFIG['output']);parser.add_argument('--dataset',default=CONFIG['dataset'])
    cfg = dict(CONFIG,**vars(parser.parse_args()));out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    def status(phase, **extra):
        record = dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**extra)
        temporary=out/'status.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(out/'status.json')
        print(json.dumps(record),flush=True)
    try:
        inventory=json.loads(Path(cfg['inventory']).read_text());expected={file['id'] for file in inventory['files'] if selected(file)}
        while True:
            rows=[json.loads(line) for line in Path(cfg['downloads']).read_text().splitlines() if line.strip()]
            downloaded={file['id'] for file in rows}
            if expected <= downloaded:
                break
            status('waiting_downloads',downloaded=len(downloaded),expected=len(expected))
            time.sleep(10)
        for row in rows:
            if not Path(row['local']).is_file() or Path(row['local']).stat().st_size != int(row['size']):
                raise ValueError('Missing or incomplete local artifact: '+row['path'])
        status('preparing',downloaded=len(rows),bytes=sum(int(row['size']) for row in rows))
        subprocess.run([sys.executable,'-u',str(ROOT/'training/prepare_temporal_search.py'),
                        '--downloads',cfg['downloads'],'--output',cfg['dataset']],check=True,cwd=ROOT)
        training=dict(trainer.CONFIG,data=cfg['dataset'],out_dir=str(out),
                      architecture='multipath_phase_contact_bucketed',init_architecture='multipath_phase_contact_bucketed',
                      hidden=512,init_hidden=512,init_from=str(ROOT/'data/nnue/nnue_weights.bin'),
                      from_scratch=False,epochs=cfg['epochs'],lr=cfg['lr'],min_lr=cfg['min_lr'],
                      schedule='cosine',warmup_epochs=0,patience=0,batch_size=2048,trunk_lr_scale=.1,
                      aux_policy_enabled=True,aux_policy_temperature=2.,aux_policy_weight=.15,aux_policy_lr_scale=1.,
                      train_scope='full',qat=True,mirror_h=True,device='cpu',cpu_threads=cfg['cpu_threads'],
                      checkpoint_every=10,resume=True,build=True,benchmark=False,seed=20261013)
        (out/'requested_config.json').write_text(json.dumps(training,indent=2)+'\n')
        status('training',epochs=cfg['epochs'],learning_rate_start=cfg['lr'],learning_rate_end=cfg['min_lr'])
        report=trainer.train(training)
        if report['epochs'] != cfg['epochs']:
            raise RuntimeError('The training run did not complete the requested epoch count.')
        status('verifying',epochs=report['epochs'],best_epoch=report['best_epoch'])
        trainer.build_candidate(training);verification=parity(out)
        package=ROOT.parent/'zquoridor_temporal_search_100ep.zip'
        with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ('student.bin','student_int8.bin','student.architecture.json','train_report.json','requested_config.json','parity.json','best.pt'):
                archive.write(out/name,name)
            archive.write(Path(cfg['dataset'])/'dataset.manifest.json','dataset.manifest.json')
        status('complete',epochs=report['epochs'],best_epoch=report['best_epoch'],parity=verification,
               package=str(package),promoted=False)
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':
    main()
