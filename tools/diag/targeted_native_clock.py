#!/usr/bin/env python3
"""Eight frozen 3+2 positions from actual 2026-10-18 games; no opponents, no tournament.

Run main/Lazy/combo and adaptive ON/OFF serially on one runner/CPU.
Each state is searched twice with the identical real-clock UCI command to
observe root tree reuse; preserve the original 180s+2s protocol format.
The OFF control must preserve the ON initial effective node cap.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

# Midgame positions before the tenth own move, extracted from original
# 3+2 games.jsonl, with source-side clock_before_ms.
CASES = [
 ("ti_normal_combo_loss_2_1", "d7h f4v f7h g6v e6h e7v e2 d9 e3 c9 d3 c8 d4 c7 c6h b7 a6h b8 d3 c8 e3 c9 f1h d9 h1h", 114317),
 ("ti_normal_combo_loss_5_0", "a7h e7h f5h e8v a4h e8 e2 d8 e3 d3h f3 h5h c7v d2v f4 e4v f3 d7 d6h d8 d8h h6h f2 d7", 120922),
 ("cl_normal_combo_loss_4_1", "f7h a1v g2v c5v e6h e5v e2 e8 e3 e7 d7v e3h d3 c3h e3 f7 f3 g7 g6h h7 h7v f2v f2 h8 f1", 117816),
 ("cl_normal_combo_loss_10_0", "b6v e5h e2v f4v h7h a3h e2 e8 h1h e7 f1h d7 d2 d6 c5h d7 d3 d8 d4 c4v c3h e8 e4 f8", 120019),
 ("cl_center_combo_win_4_1", "e2 e8 e3 e7 e4 e6 d3h d6h e5v d6 f3h c6 b3h e8v a1h c7h a2v f4h d4 a7h c4 e7h c5 g7h h1h c7 c6 c5 g4v c7 d7", 121338),
 ("cl_center_combo_win_18_0", "e2 e8 d3h e7 d2 f3h e6h c2v d3 d7 c6h e7 d2 d1h e2 f7 f7v f8 f2 f9 e1v g2v g2 g1h f2 g9 f1 g8", 124482),
 ("ti_normal_combo_win_7_0", "c6h h8v f1h g4h c4v d2v e2 e8 f2 f8 f3 e3h g3 g3h h3 h3v f7v h1v h2 e8 g2 e6v f2 e7", 123940),
 ("ti_normal_combo_win_26_1", "g3v d7h c5h g5h b3h b4h f1 e8 f2 f8 f7h g8 f3 h8 h7h g8 f8h e5h c1v f8 e7v g8 f4 h8 g4", 125921),
]
KV = re.compile(r"(?<!\S)(diag_[a-z]+|treeHit|reusedNodes)=([^\s]+)")
def once(exe, nnue, mode, case):
    name, sequence, remaining = case
    line = "position startpos moves " + sequence
    go = f"go wtime {remaining} btime {remaining} winc 2000 binc 2000"
    inp = "uci\nisready\n" + line + "\n" + go + "\n" + go + "\nquit\n"
    cmd = [str(exe), "--nnue", str(nnue)]
    if mode == "off":
        cmd.append("--diag-no-adaptive-time")
    result = subprocess.run(cmd, input=inp, text=True, capture_output=True, timeout=110)
    if result.returncode:
        raise RuntimeError(f"{name}/{mode}/{exe}: {result.stderr[-1500:]}")
    infos = [line for line in result.stdout.splitlines()
             if line.startswith("info depth ") and "diag_sims=" in line]
    moves = [line.split()[1] for line in result.stdout.splitlines()
             if line.startswith("bestmove ")]
    if len(infos) != 2 or len(moves) != 2:
        raise RuntimeError(f"{name}/{mode}/{exe}: missing UCI diagnostics: {result.stdout[-1500:]}")
    rows = []
    for info, move in zip(infos, moves):
        pairs = dict(KV.findall(info))
        for field in ("diag_sims", "diag_nodes", "diag_target", "diag_class",
                      "diag_stop", "diag_adaptive", "diag_optimum", "diag_expandcap"):
            if field not in pairs:
                raise RuntimeError(f"missing {field}: {info}")
        if int(pairs["diag_adaptive"]) != (mode == "on"):
            raise RuntimeError(f"incorrect adaptive mode: {info}")
        rows.append({"move": move, "simulations": int(pairs["diag_sims"]),
                     "expanded_nodes": int(pairs["diag_nodes"]),
                     "target_ms": int(pairs["diag_target"]),
                     "optimum_ms": int(pairs["diag_optimum"]),
                     "root_class": int(pairs["diag_class"]),
                     "stop_reason": int(pairs["diag_stop"]),
                     "node_cap": int(pairs["diag_expandcap"]),
                     "tree_hit": int(pairs.get("treeHit", -1)),
                     "reused_nodes": int(pairs.get("reusedNodes", -1)),
                     "visit_ratio": float(pairs.get("diag_visits", "0")),
                     "q_gap": float(pairs.get("diag_qgap", "0")),
                     "uci_info": info})
    return rows

def main():
    p = argparse.ArgumentParser()
    for engine in ("main", "lazy", "combo"):
        p.add_argument("--" + engine, type=Path, required=True)
    p.add_argument("--nnue", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    binaries = {key: getattr(args, key).resolve() for key in ("main", "lazy", "combo")}
    nnue = args.nnue.resolve()
    output = []
    for i, case in enumerate(CASES):
        rows = {}
        for mode in (("on", "off") if i % 2 == 0 else ("off", "on")):
            # Rotate order to reduce confounding from runner load/frequency drift.
            order = ["main", "lazy", "combo"]
            order = order[i % 3:] + order[:i % 3]
            for eng in order:
                rows[(eng, mode)] = once(binaries[eng], nnue, mode, case)
        for eng in ("main", "lazy", "combo"):
            on, off = rows[(eng, "on")], rows[(eng, "off")]
            # The OFF test must not secretly use the smaller optimum-derived
            # node cap; volatile ON escalation is allowed to differ.
            if on[0]["node_cap"] != off[0]["node_cap"] and on[0]["root_class"] != 1:
                raise RuntimeError(f"unmatched cap in case {case[0]} {eng}: {on[0]['node_cap']} vs {off[0]['node_cap']}")
            for mode in ("on", "off"):
                output.append({"case_id": case[0], "clock_ms": case[2],
                               "engine": eng, "mode": mode, "repeated": rows[(eng, mode)]})
        print(f"COMPLETE {i+1}/8 positions, 3 engines, ON/OFF, 2 searches each", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema":"zquoridor.diag.native_clock.small.v1",
              "protocol":"native UCI clock 3+2", "positions":8,
              "engine_variants":3,"adaptive_modes":2,"repetitions":2,
              "same_runner_serial":True,"results":output},indent=2)+"\n")
    print("TARGETED CLOCK DIAGNOSTIC SUCCESS",args.output,flush=True)
if __name__ == "__main__":
    main()
