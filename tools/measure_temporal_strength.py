#!/usr/bin/env python3
"""Complete a paired H2H and matched external strength screen."""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_benchmark
from tools.external import local_arena

CONFIG={
    'output':str(ROOT/'results/benchmarks/temporal-search-100ep-strength-20261013'),
    'candidate':str(ROOT/'experimental/experiments/temporal-search-100ep-20261013/student_int8.bin'),
    'baseline':str(ROOT/'data/nnue/nnue_weights_int8.bin'),
    'executable':str(ROOT/'bin/zquoridor'),
    'external_pairs':50,'workers':3,'movetime_ms':200,'seed':20261015,'bootstrap':20000,
}


def load_games(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wdl(rows):
    counts=collections.Counter(row['result'] for row in rows if row.get('status')=='ok')
    return {'wins':counts[1.0],'draws':counts[.5],'losses':counts[0.0],
            'failed':sum(row.get('status')!='ok' for row in rows)}


def pair_scores(rows):
    grouped=collections.defaultdict(dict)
    for row in rows:
        if row.get('status')=='ok':
            grouped[row['opening_index']][row['zq_player']]=row['result']
    return {key:(sides[0]+sides[1])/2 for key,sides in grouped.items() if set(sides)=={0,1}}


def matched_delta(candidate, baseline, seed, bootstrap):
    candidate=pair_scores(candidate);baseline=pair_scores(baseline)
    keys=sorted(set(candidate)&set(baseline))
    if not keys:
        return {'matched_pairs':0,'delta_score_pp':None}
    a=np.array([candidate[k] for k in keys]);b=np.array([baseline[k] for k in keys]);difference=a-b
    rng=np.random.default_rng(seed)
    samples=difference[rng.integers(0,len(keys),size=(bootstrap,len(keys)))].mean(1)
    low,high=np.quantile(samples,[.025,.975])
    return {'matched_pairs':len(keys),'candidate_score_pct':float(a.mean()*100),
            'baseline_score_pct':float(b.mean()*100),'delta_score_pp':float(difference.mean()*100),
            'paired_bootstrap_95_delta_pp':[float(low*100),float(high*100)],
            'candidate_only_pairs':len(set(candidate)-set(baseline)),
            'baseline_only_pairs':len(set(baseline)-set(candidate))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=CONFIG['output'])
    cfg=dict(CONFIG,**vars(parser.parse_args()));out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    def status(phase,**extra):
        payload={'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),**extra}
        p=out/'status.tmp';p.write_text(json.dumps(payload,indent=2)+'\n');p.replace(out/'status.json')
        print(json.dumps(payload),flush=True)
    try:
        while not (out/'h2h/summary.json').is_file():
            status('waiting_h2h');time.sleep(15)
        h2h=json.loads((out/'h2h/summary.json').read_text());rows=load_games(out/'h2h/games.jsonl')
        categories=run_benchmark._read_opening_categories(out/'openings.jsonl')
        h2h['by_category']=run_benchmark.summarize_by_category(rows,categories,bootstrap=cfg['bootstrap'],seed=cfg['seed'])
        h2h['win_draw_loss']=wdl(rows);(out/'h2h/summary.json').write_text(json.dumps(h2h,indent=2)+'\n')
        status('building_external_bots',h2h_score_pct=h2h['summary']['score_pct'])
        subprocess.run([sys.executable,'-u',str(ROOT/'tools/setup_bots.py')],check=True,cwd=ROOT)
        # Build the corrected Titanium bridge before any timed game starts.
        from tools.build_titanium_clock_bridge import ensure_bridge
        ensure_bridge(ROOT,build=True)
        for label,weights in [('baseline',cfg['baseline']),('candidate',cfg['candidate'])]:
            status('external_games',participant=label)
            command=[sys.executable,'-u',str(ROOT/'tools/run_benchmark.py'),'--opponents','claustrophobia,titanium',
                     '--pairs',str(cfg['external_pairs']),'--workers',str(cfg['workers']),'--seed',str(cfg['seed']),
                     '--openings',str(out/'external_openings.jsonl'),'--output',str(out/label),
                     '--zq-executable',cfg['executable'],'--nnue',weights,'--zq-move-time-ms',str(cfg['movetime_ms']),
                     '--claustrophobia-move-time-ms',str(cfg['movetime_ms']),'--titanium-move-time-ms',str(cfg['movetime_ms']),
                     '--claustrophobia-device','cpu','--max-plies','240']
            subprocess.run(command,check=True,cwd=ROOT)
        candidate=load_games(out/'candidate/games.jsonl');baseline=load_games(out/'baseline/games.jsonl')
        report={'h2h':h2h,'external':{},'config':cfg,'promoted':False}
        for offset,opponent in enumerate(('claustrophobia','titanium')):
            a=[row for row in candidate if row['opponent']==opponent];b=[row for row in baseline if row['opponent']==opponent]
            report['external'][opponent]={'candidate':local_arena.summarize_pairs(a,bootstrap=cfg['bootstrap'],seed=cfg['seed']+offset),
                                         'baseline':local_arena.summarize_pairs(b,bootstrap=cfg['bootstrap'],seed=cfg['seed']+offset),
                                         'matched_change':matched_delta(a,b,cfg['seed']+offset,cfg['bootstrap']),
                                         'candidate_wdl':wdl(a),'baseline_wdl':wdl(b),'by_category':{}}
            for category in ('normal','center_rush'):
                aa=[row for row in a if row.get('opening_category')==category]
                bb=[row for row in b if row.get('opening_category')==category]
                report['external'][opponent]['by_category'][category]={
                    'candidate':local_arena.summarize_pairs(aa,bootstrap=cfg['bootstrap'],seed=cfg['seed']+offset),
                    'baseline':local_arena.summarize_pairs(bb,bootstrap=cfg['bootstrap'],seed=cfg['seed']+offset),
                    'matched_change':matched_delta(aa,bb,cfg['seed']+offset,cfg['bootstrap'])}
        (out/'strength_report.json').write_text(json.dumps(report,indent=2)+'\n')
        package=ROOT.parent/'zquoridor_temporal_strength_report.zip'
        with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob('*'):
                if path.is_file() and path.suffix in ('.json','.jsonl'):
                    archive.write(path,str(path.relative_to(out)))
        status('complete',h2h_score_pct=h2h['summary']['score_pct'],package=str(package),promoted=False)
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':
    main()
