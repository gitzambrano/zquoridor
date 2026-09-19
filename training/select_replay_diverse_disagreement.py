#!/usr/bin/env python3
"""Select a diverse hard-search teaching corpus from direct-teacher disagreement."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from training import select_replay_disagreement as base

CONFIG={
    "replay_dir":str(ROOT/"data/teaching/replay-historical-2m-cuda"),
    "out":str(ROOT/"data/teaching/search-priority-diverse-50k/positions.jsonl"),
    "max_positions":50000,
    "policy_weight":1.0,
    "value_weight":1.0,
    "stratified_fraction":0.70,
}

STATE_FIELDS=base.STATE_FIELDS


def strata(data:dict[str,np.ndarray])->np.ndarray:
    own_dist=data["own_dist"].astype(np.int64)
    opp_dist=data["opp_dist"].astype(np.int64)
    race=(np.sign(own_dist-opp_dist)+1).astype(np.int64)  # 0 ahead, 1 tie, 2 behind
    ow=data["walls_left_own"].astype(np.int64)
    pw=data["walls_left_opp"].astype(np.int64)
    regime=np.zeros(len(ow),dtype=np.int64)
    regime[(ow==0)&(pw>0)]=1
    regime[(ow>0)&(pw==0)]=2
    regime[(ow==0)&(pw==0)]=3
    return race*4+regime


def select_diverse(policy,value,data,config):
    if not 0.0 <= float(config["stratified_fraction"]) <= 1.0:
        raise ValueError("stratified_fraction must be in [0,1]")
    score=float(config["policy_weight"])*policy + float(config["value_weight"])*value
    target=min(int(config["max_positions"]),len(score))
    sid=strata(data)
    selected=[]
    selected_mask=np.zeros(len(score),dtype=bool)

    stratified_target=int(round(target*float(config["stratified_fraction"])))
    quota=max(1,stratified_target//12) if stratified_target else 0
    if quota:
        for s in range(12):
            idx=np.flatnonzero(sid==s)
            if not len(idx):
                continue
            order=idx[np.argsort(score[idx],kind="stable")[::-1]]
            take=order[:quota]
            selected.extend(int(x) for x in take)
            selected_mask[take]=True

    remaining=target-len(selected)
    if remaining>0:
        pool=np.flatnonzero(~selected_mask)
        order=pool[np.argsort(score[pool],kind="stable")[::-1]]
        selected.extend(int(x) for x in order[:remaining])

    chosen=np.asarray(selected,dtype=np.int64)
    # Final deterministic rank by priority, while retaining the stratified membership.
    order=np.argsort(score[chosen],kind="stable")[::-1]
    return chosen[order],score,sid


def run(config):
    replay_dir=Path(config["replay_dir"])
    with np.load(replay_dir/"dataset.npz",allow_pickle=False) as raw:
        required=(*STATE_FIELDS,"id","is_val","own_dist","opp_dist")
        data={name:raw[name] for name in required}
    ids,policy,value=base.load_direct_predictions(replay_dir,data["id"])
    chosen,score,sid=select_diverse(policy,value,data,config)
    out=Path(config["out"])
    out.parent.mkdir(parents=True,exist_ok=True)
    counts=np.bincount(sid[chosen],minlength=12)
    with out.open("w",encoding="utf-8") as stream:
        for rank,index in enumerate(chosen):
            history=["@state",*(str(int(data[field][index])) for field in STATE_FIELDS)]
            row={
                "schema":"zquoridor.position.v1",
                "id":ids[index].decode(),
                "history":history,
                "side_to_move":0,
                "opening_index":int(index),
                "ply":0,
                "split":"val" if bool(data["is_val"][index]) else "train",
                "metadata":{
                    "source":"v3-snapshot-no-repetition-history",
                    "rank":rank,
                    "policy_js":float(policy[index]),
                    "value_abs_diff":float(value[index]),
                    "priority_score":float(score[index]),
                    "diversity_stratum":int(sid[index]),
                },
            }
            stream.write(json.dumps(row,separators=(",",":"))+"\n")
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    manifest={
        "schema":"zquoridor.teacher.diverse_search_selection.v1",
        "replay_dir":str(replay_dir.resolve()),
        "output":str(out.resolve()),
        "samples":int(len(chosen)),
        "policy_weight":float(config["policy_weight"]),
        "value_weight":float(config["value_weight"]),
        "stratified_fraction":float(config["stratified_fraction"]),
        "strata":"race_sign(3) x wall_regime(4)",
        "stratum_counts":counts.tolist(),
        "sha256":digest,
    }
    out.with_suffix(out.suffix+".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key,value in CONFIG.items():
        p.add_argument("--"+key.replace("_","-"),type=type(value),default=argparse.SUPPRESS)
    cfg=dict(CONFIG,**vars(p.parse_args(argv)))
    print(json.dumps(run(cfg),indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
