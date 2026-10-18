#!/usr/bin/env python3
"""Structured PER-MOVE telemetry from actual native 3+2 game records.

The referee logs milliseconds, clocks, complete info string, result and all
moves. This only parses data, never supplies any search/simulation timing.
"""
from __future__ import annotations
import argparse,collections,json,re,statistics
from pathlib import Path

KV=re.compile(r"(?<!\S)(diag_[a-z]+)=([^\s]+)")
def main():
  p=argparse.ArgumentParser()
  p.add_argument("--directory",required=True)
  p.add_argument("--mode",choices=["adaptive","fixed"],required=True)
  a=p.parse_args()
  root=Path(a.directory)
  report=json.loads((root/"summary.json").read_text())
  assert report["clock"]=={"initial_ms":180000,"increment_ms":2000,"protocol":"native-game-clock-v2"}
  rows=[json.loads(s) for s in (root/"games.jsonl").read_text().splitlines() if s.strip()]
  assert all(r["status"]=="ok" for r in rows),[r.get("error") for r in rows if r["status"]!="ok"]
  telemetry=[]; byStop=collections.Counter(); byClass=collections.Counter(); noDiag=0; timeLosses=0
  for g in rows:
    if g["termination"]=="time" and g["result"]==0:timeLosses+=1
    for i,m in enumerate(g["move_times"]):
      if m["player"]!="zquoridor":continue
      raw=m.get("search_last","")
      diag={k:v for k,v in KV.findall(raw)}
      if not diag:
        noDiag+=1
        continue
      for name in ("diag_sims","diag_nodes","diag_target","diag_class","diag_stop","diag_adaptive","diag_optimum","diag_expandcap","diag_simcap"):
        assert name in diag,(name,raw)
      ad=int(diag["diag_adaptive"])
      if ad!=(a.mode=="adaptive"):raise AssertionError(f"{a.mode}: adaptive={ad} for UCI move {raw}")
      ms=float(m["elapsed_ms"])
      before=int(m["clock_before_ms"])
      after=int(m["clock_after_ms"])
      assert 0<=ms<=before+25 and after>=0,(ms,before,after)
      info={"game":g["opening_index"],"side":g["zq_player"],"ply_order":i,
        "elapsed_ms":ms,"clock_before_ms":before,"clock_after_ms":after,
        "simulations":int(diag["diag_sims"]),"expanded_nodes":int(diag["diag_nodes"]),
        "target_ms":int(diag["diag_target"]),"optimum_ms":int(diag["diag_optimum"]),
        "class":int(diag["diag_class"]),"stop_reason":int(diag["diag_stop"]),
        "adaptive":ad,"visit_ratio":float(diag.get("diag_visits",0)),
        "q_gap":float(diag.get("diag_qgap",0)),"node_cap":int(diag["diag_expandcap"])}
      telemetry.append(info)
      byStop[info["stop_reason"]]+=1
      byClass[info["class"]]+=1
  assert len(telemetry)>0, "no per-move telemetry was collected"
  assert all(x["stop_reason"] in (0,1,2,3,4,5) for x in telemetry)
  data={"schema":"zquoridor.diagnostic.clock.v1","mode":a.mode,
    "games":len(rows),"moves_instrumented":len(telemetry),"other_moves":noDiag,
    "forfeit_time_losses":timeLosses,
    "mean_elapsed_ms":round(statistics.mean(x["elapsed_ms"] for x in telemetry),2),
    "mean_simulations":round(statistics.mean(x["simulations"] for x in telemetry),2),
    "mean_expanded_nodes":round(statistics.mean(x["expanded_nodes"] for x in telemetry),2),
    "stop_reasons":dict(byStop),"root_classes":dict(byClass),
    "per_move":telemetry}
  (root/"telemetry.json").write_text(json.dumps(data,indent=2)+"\n")
  print("VALID NATIVE CLOCK TELEMETRY",json.dumps({k:v for k,v in data.items() if k!="per_move"}),flush=True)

if __name__=="__main__":main()
