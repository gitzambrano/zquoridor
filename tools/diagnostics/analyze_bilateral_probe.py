#!/usr/bin/env python3
import argparse, csv, math
from collections import defaultdict


def load(path):
    rows=[]
    with open(path,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            r={**r,
               'own_dist':int(r['own_dist']),'opp_dist':int(r['opp_dist']),
               'own_walls':int(r['own_walls']),'opp_walls':int(r['opp_walls']),
               'logit':float(r['logit']),'pwin':float(r['pwin']),
               'antisym_error':float(r['antisym_error'])}
            rows.append(r)
    return rows


def monotonic_violations(rows, family, key, expect):
    xs=[r for r in rows if r['family']==family]
    xs=sorted(xs,key=lambda r:r[key])
    v=[]
    for a,b in zip(xs,xs[1:]):
        dp=b['pwin']-a['pwin']
        bad=(expect=='inc' and dp < -1e-6) or (expect=='dec' and dp > 1e-6)
        if bad: v.append((a[key],b[key],a['pwin'],b['pwin']))
    return v


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--max-antisym-error',type=float,default=None)
    ap.add_argument('--fail-on-monotonic',action='store_true')
    args=ap.parse_args()
    rows=load(args.csv)
    if not rows: raise SystemExit('empty probe')

    # As own distance increases, win probability should not improve.
    checks=[
        ('own_distance','own_dist','dec'),
        ('opp_distance','opp_dist','inc'),
        ('own_walls','own_walls','inc'),
        ('opp_walls','opp_walls','dec'),
        ('wall_poor_race','own_dist','dec'),
        ('wall_gap','opp_walls','dec'),
    ]
    total=0
    for fam,key,expect in checks:
        v=monotonic_violations(rows,fam,key,expect)
        total += len(v)
        print(f'{fam:16s}: {len(v)} monotonic violation(s)')
        for item in v[:5]: print('  ',item)

    max_asym=max(abs(r['antisym_error']) for r in rows)
    mean_asym=sum(abs(r['antisym_error']) for r in rows)/len(rows)
    print(f'antisymmetry |z(s)+z(swapped)|: mean={mean_asym:.6f} max={max_asym:.6f}')

    wp=[r for r in rows if r['family']=='wall_poor_race']
    if wp:
        far=max(wp,key=lambda r:r['own_dist'])
        near=min(wp,key=lambda r:r['own_dist'])
        print(f"wall-poor progress signal: Pwin d={far['own_dist']} {far['pwin']:.4f} -> d={near['own_dist']} {near['pwin']:.4f} (delta {near['pwin']-far['pwin']:+.4f})")

    failed=False
    if args.fail_on_monotonic and total: failed=True
    if args.max_antisym_error is not None and max_asym > args.max_antisym_error: failed=True
    raise SystemExit(1 if failed else 0)

if __name__=='__main__': main()
