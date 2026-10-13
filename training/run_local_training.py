#!/usr/bin/env python3
"""Select network capacity, teaching sources, and value targets independently."""
import argparse
import ast
import shutil
import subprocess
import os
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'training'))
from training.target_options import target_settings, matches_targets

CONFIG = {
    'target_mode':'temporal', 'future_horizons':None, 'future_weights':None, 'result_weight':.1,
    'apex_value_fraction':0., 'apex_archive':str(ROOT/'data/teaching/external-value/titanium_mix_v2.zip'),
    'downloads':str(ROOT/'data/teaching/drive-temporal-100ep/downloads.jsonl'),
    'data_cache':str(ROOT/'data/teaching/local-target-options'),
    'hidden':512, 'data':None, 'out_dir':None,
    'init_from':str(ROOT/'data/nnue/nnue_weights.bin'), 'init_hidden':None,
    'epochs':None, 'lr':None, 'min_lr':None, 'device':'auto',
    'batch_size':2048, 'cpu_threads':3, 'build':True, 'dry_run':False,
}


def resolve(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key,value in CONFIG.items():
        options=dict(default=value)
        if key in ('build','dry_run'):
            options['action']=argparse.BooleanOptionalAction
        elif key in ('hidden','init_hidden','epochs','batch_size','cpu_threads'):
            options['type']=int
        elif key in ('lr','min_lr','result_weight','apex_value_fraction'):
            options['type']=float
        if key=='hidden':options['choices']=(512,768,1024)
        if key=='target_mode':options['choices']=('current','temporal')
        if key in ('future_horizons','future_weights'):options['type']=json.loads
        if key=='device':options['choices']=('auto','cpu','cuda')
        parser.add_argument('--'+key.replace('_','-'),**options)
    local=vars(parser.parse_args(argv))
    settings=target_settings(local['target_mode'],local['future_horizons'],local['future_weights'],local['result_weight'])
    if not 0<=local['apex_value_fraction']<1:raise ValueError('The apex value fraction must be between zero and one, excluding one.')
    defaults=(100,1e-5,1e-7)
    for key,value in zip(('epochs','lr','min_lr'),defaults):
        if local[key] is None:local[key]=value
    default_data='drive-temporal-100ep/dataset'
    local['data']=str(Path(local['data']).resolve()) if local['data'] else str(ROOT/'data/teaching'/default_data)
    local['out_dir']=str(Path(local['out_dir']).resolve()) if local['out_dir'] else str(ROOT/'experimental/experiments'/f"local-{local['target_mode']}-{local['hidden']}-apex{local['apex_value_fraction']:g}")
    initial=Path(local['init_from']).resolve()
    architecture_path=initial.with_suffix('.architecture.json')
    architecture=json.loads(architecture_path.read_text()) if architecture_path.exists() else {}
    init_hidden=local['init_hidden'] if local['init_hidden'] is not None else architecture.get('hidden',512)
    if local['hidden'] < init_hidden:
        raise ValueError('Width transfer cannot shrink the initial network.')
    tree=ast.parse((ROOT/'training/run_experiment.py').read_text())
    defaults=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in node.targets))
    config=dict(defaults,data=local['data'],out_dir=local['out_dir'],hidden=local['hidden'],
        architecture='multipath_phase_contact_bucketed',init_from=str(initial),init_hidden=init_hidden,
        init_architecture=architecture.get('architecture','multipath_phase_contact_bucketed'),
        epochs=local['epochs'],lr=local['lr'],min_lr=local['min_lr'],warmup_epochs=0,patience=0,
        device=local['device'],batch_size=local['batch_size'],cpu_threads=local['cpu_threads'],
        qat=True,mirror_h=True,from_scratch=False,train_scope='full',trunk_lr_scale=.1,
        aux_policy_enabled=True,aux_policy_temperature=2.,aux_policy_weight=.15,aux_policy_lr_scale=1.,
        checkpoint_every=5,resume=True,build=local['build'],benchmark=False,
        value_target_settings=settings,apex_value_fraction=local['apex_value_fraction'],local_launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return local,config


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def ensure_encoder():
    executable=ROOT/'bin'/('teacher_encode_state.exe' if os.name=='nt' else 'teacher_encode_state')
    if not executable.exists():
        compiler=shutil.which('g++')
        if compiler is None:raise RuntimeError('g++ is required to prepare trajectory targets.')
        executable.parent.mkdir(exist_ok=True)
        subprocess.run([compiler,'-O2','-std=c++17','-I'+str(ROOT/'src'),str(ROOT/'tools/teacher/encode_state.cpp'),'-o',str(executable)],check=True)


def prepare_data(local,config):
    base=Path(local['data']);path=base/'dataset.manifest.json'
    settings=config['value_target_settings']
    manifest=json.loads(path.read_text()) if path.exists() else {}
    if not matches_targets(manifest,settings):
        ledger=Path(local['downloads'])
        if not ledger.is_file():raise ValueError('Changing targets requires raw trajectories or aligned shard metadata.')
        signature=hashlib.sha256(json.dumps(dict(settings=settings,reference=file_sha(config['init_from']),ledger=file_sha(ledger),preparer=file_sha(ROOT/'training/prepare_temporal_search.py')),sort_keys=True).encode()).hexdigest()[:16]
        base=Path(local['data_cache'])/('replay-'+signature)
        cache_manifest=base/'dataset.manifest.json'
        cached=json.loads(cache_manifest.read_text()) if cache_manifest.exists() else {}
        if not cached.get('complete') or not matches_targets(cached,settings):
            ensure_encoder()
            subprocess.run([sys.executable,str(ROOT/'training/prepare_temporal_search.py'),'--downloads',str(ledger),'--output',str(base),'--reference',config['init_from'],'--hidden',str(config['init_hidden']),'--architecture',config['init_architecture'],'--target-mode',settings['target_mode'],'--horizons',json.dumps(settings['horizons']),'--horizon-weights',json.dumps(settings['horizon_weights']),'--outcome-weight',str(settings['outcome_weight'])],check=True,cwd=ROOT)
    fraction=local['apex_value_fraction']
    if fraction==0:return base
    ready=ROOT/'data/teaching/apex-mix-after-width-20261013';ready_manifest=ready/'dataset.manifest.json'
    if ready_manifest.exists():
        mixed=json.loads(ready_manifest.read_text())
        if mixed.get('apex_training_value_fraction')==fraction and mixed.get('identity',{}).get('base_manifest_sha256')==file_sha(base/'dataset.manifest.json') and mixed.get('identity',{}).get('archive_sha256')==file_sha(local['apex_archive']):return ready
    from training.run_apex_mix_campaign import prepare_mix
    signature=hashlib.sha256(json.dumps(dict(base=file_sha(base/'dataset.manifest.json'),archive=file_sha(local['apex_archive']),fraction=fraction,reference=file_sha(config['init_from'])),sort_keys=True).encode()).hexdigest()[:16]
    output=Path(local['data_cache'])/('mixed-'+signature)
    reference=output/'reference';reference.mkdir(parents=True,exist_ok=True)
    shutil.copy2(config['init_from'],reference/'student.bin')
    (reference/'student.architecture.json').write_text(json.dumps(dict(architecture=config['init_architecture'],hidden=config['init_hidden'])))
    ensure_encoder()
    prepare_mix(dict(dataset=str(base),archive=local['apex_archive'],mixed_dataset=str(output),apex_value_fraction=fraction,workers=local['cpu_threads'],seed=20261019),reference)
    return output


def main():
    local,config=resolve()
    if local['dry_run']:
        print(json.dumps(config,indent=2));return
    from training import run_experiment as trainer
    from training.run_apex_mix_campaign import masked_loader
    from training.run_temporal_campaign import parity
    config['data']=str(prepare_data(local,config))
    if local['apex_value_fraction']>0:trainer.load_dataset=masked_loader
    out=Path(config['out_dir']);out.mkdir(parents=True,exist_ok=True)
    (out/'requested_config.json').write_text(json.dumps(config,indent=2)+'\n')
    report=trainer.train(config)
    if config['build']:
        trainer.build_candidate(config);parity(out)
    print(json.dumps({key:value for key,value in report.items() if key!='history'},indent=2))


if __name__=='__main__':
    main()
