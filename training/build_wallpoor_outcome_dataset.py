#!/usr/bin/env python3
"""Blend real game outcomes into value targets, emphasizing wall-poor positions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STATE_FIELDS=("own_pawn","opp_pawn","walls_h","walls_v","walls_left_own","walls_left_opp")

CONFIG={
    "mixed":"data/teaching/historical2m-search10-conservative/dataset.npz",
    "replay":"data/teaching/replay-historical-2m-cuda/dataset.npz",
    "out":"data/teaching/historical2m-search10-wallpoor-outcome/dataset.npz",
    "global_outcome_mix":0.10,
    "wallpoor_outcome_mix":0.50,
    "wallpoor_max_own_walls":1,
    "wallpoor_weight_boost":1.0
}

def state_key(data,i):
    return tuple(int(data[k][i]) for k in STATE_FIELDS)

def run(config):
    with np.load(config["mixed"],allow_pickle=False) as z:
        mixed={k:z[k] for k in z.files}
    with np.load(config["replay"],allow_pickle=False) as z:
        replay={k:z[k] for k in z.files}
    if "game_result" not in replay:
        raise ValueError("replay dataset has no game_result")
    lookup={}
    for i in range(len(replay["game_result"])):
        key=state_key(replay,i)
        lookup[key]=float(replay["game_result"][i])

    outcome=np.empty(len(mixed["value"]),dtype=np.float32)
    missing=0
    for i in range(len(outcome)):
        value=lookup.get(state_key(mixed,i))
        if value is None:
            missing+=1
            outcome[i]=mixed["value"][i]
        else:
            outcome[i]=value
    if missing:
        raise ValueError(f"{missing} mixed states lack a replay outcome; abort instead of silently mixing")

    own_walls=mixed["walls_left_own"].astype(np.int64)
    wallpoor=own_walls <= int(config["wallpoor_max_own_walls"])
    alpha=np.full(len(outcome),float(config["global_outcome_mix"]),dtype=np.float32)
    alpha[wallpoor]=float(config["wallpoor_outcome_mix"])
    if np.any(alpha<0) or np.any(alpha>1):
        raise ValueError("outcome mix must be in [0,1]")

    output={k:v.copy() for k,v in mixed.items()}
    output["value"]=((1.0-alpha)*mixed["value"].astype(np.float32)+alpha*outcome).astype(np.float32)
    output["weight"]=mixed["weight"].astype(np.float32).copy()
    output["weight"][wallpoor]*=float(config["wallpoor_weight_boost"])

    out=Path(config["out"])
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(out,**output)
    manifest={
        "schema":"zquoridor.wallpoor_outcome_mix.v1",
        "mixed":str(Path(config["mixed"]).resolve()),
        "replay":str(Path(config["replay"]).resolve()),
        "output":str(out.resolve()),
        "samples":int(len(outcome)),
        "wallpoor_samples":int(wallpoor.sum()),
        "global_outcome_mix":float(config["global_outcome_mix"]),
        "wallpoor_outcome_mix":float(config["wallpoor_outcome_mix"]),
        "wallpoor_max_own_walls":int(config["wallpoor_max_own_walls"]),
        "wallpoor_weight_boost":float(config["wallpoor_weight_boost"]),
        "mean_abs_value_shift":float(np.abs(output["value"]-mixed["value"]).mean()),
        "wallpoor_mean_abs_value_shift":float(np.abs(output["value"][wallpoor]-mixed["value"][wallpoor]).mean()) if wallpoor.any() else 0.0
    }
    out.with_name("dataset.manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2))
    return manifest

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key,val in CONFIG.items():
        p.add_argument("--"+key.replace("_","-"),type=type(val),default=argparse.SUPPRESS)
    cfg=dict(CONFIG,**vars(p.parse_args(argv)))
    run(cfg)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
