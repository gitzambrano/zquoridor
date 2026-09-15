#!/usr/bin/env python3
"""Compare NNUE architectures on shared teaching data and independent arenas."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare_replay
import run_experiment
from tools.external import local_arena
from tools import run_benchmark

CONFIG = {
    "out_dir": str(ROOT / "results/campaign"),
    "source": str(ROOT / "data/selfplay/gen7-montecarlo"),
    "positions": 50000,
    "fresh_positions": 2048,
    "fresh_games": 128,
    "fresh_weight_fraction": 0.2,
    "trajectory_time_ms": 10,
    "claustro_sims": 128,
    "zq_nodes": 512,
    "gamma": 0.995,
    "outcome_weight": 1.0,
    "bootstrap_weight": 1.0,
    "architectures": ["base:256", "race:256", "race:384"],
    "seeds": [20260914, 20260915],
    "epochs": 30,
    "batch_size": 1024,
    "device": "auto",
    "screen_pairs": 100,
    "screen_time_ms": 50,
    "confirmation_pairs": 400,
    "confirmation_time_ms": 200,
    "external_pairs": 100,
    "external_sims": 512,
    "external_device": "gpu",
    "bootstrap": 20000,
    "seed": 2026091407,
    "train": True,
    "arena": True,
    "external": True,
    "dry_run": False,
}


def combine_datasets(replay, fresh, output, fraction, blocked=()):
    """Keep fresh labels on duplicate states and remove cross-split overlap."""
    if not 0 <= fraction < 1:
        raise ValueError("fresh_weight_fraction must be in [0,1)")
    datasets = []
    for path in ([fresh] if fresh is not None else []) + [replay]:
        with np.load(path, allow_pickle=False) as data:
            datasets.append({key:data[key] for key in data.files})
    names = (*prepare_replay.STATE_FIELDS, "own_dist", "opp_dist", "policy", "value", "weight", "is_val", "group_id")
    entries = {}
    conflicts = set(blocked)
    for source, data in enumerate(datasets):
        for i in range(len(data["value"])):
            key = tuple(int(data[k][i]) for k in prepare_replay.STATE_FIELDS)
            if key in entries:
                old_source, old_i = entries[key]
                if bool(datasets[old_source]["is_val"][old_i]) != bool(data["is_val"][i]):
                    conflicts.add(key)
            else:
                entries[key] = (source, i)
    selected = [entry for key, entry in entries.items() if key not in conflicts]
    if not selected:
        raise ValueError("no independent states remain after holdout filtering")
    output_data = {name: np.asarray([datasets[s][name][i] for s,i in selected]) for name in names}
    output_data["group_id"] = np.asarray([f"source{s}:" + str(datasets[s]["group_id"][i]) for s,i in selected])
    source_ids = np.asarray([s for s,_ in selected])
    if fresh is not None and fraction:
        n_fresh = int((source_ids == 0).sum())
        n_replay = int((source_ids == 1).sum())
        if n_fresh == 0 or n_replay == 0:
            raise ValueError("both replay and fresh positions are required")
        output_data["weight"] = output_data["weight"].astype(np.float32)
        for validation in (False, True):
            new_mask = (source_ids == 0) & (output_data["is_val"] == validation)
            old_mask = (source_ids == 1) & (output_data["is_val"] == validation)
            new_sum, old_sum = output_data["weight"][new_mask].sum(), output_data["weight"][old_mask].sum()
            if new_sum <= 0 or old_sum <= 0:
                raise ValueError("fresh and replay data require positive weight in each split")
            output_data["weight"][new_mask] *= fraction * old_sum / ((1-fraction)*new_sum)
    run_experiment.split_indices(output_data)
    np.savez(output, **output_data)
    return dict(samples=len(selected), cross_split_states_removed=len(conflicts),
                train_samples=int((~output_data["is_val"]).sum()), val_samples=int(output_data["is_val"].sum()))


def arena_holdouts(folder, config):
    """Reserve distinct opening states before the training data is assembled."""
    from teachers.teaching_pipeline import _tool
    from build_teacher_soft import encode_states
    generator = _tool("teaching_generate_openings", "tools/teacher/generate_openings.cpp")
    encoder = _tool("teacher_encode_state", "tools/teacher/encode_state.cpp")
    blocked, seen = set(), set()
    books = [("screen", config["screen_pairs"], 71), ("confirm", config["confirmation_pairs"], 149)]
    if config["external"]:
        books.append(("external", config["external_pairs"], 227))
    for label, count, offset in books:
        path = folder / (label + "_openings.jsonl")
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            states = encode_states([dict(history=row["moves"]) for row in rows], encoder)
            keys = [tuple(int(states[k][i]) for k in prepare_replay.STATE_FIELDS) for i in range(len(rows))]
            if len(rows) != count or len(set(keys)) != count or seen.intersection(keys):
                raise ValueError("arena opening states are duplicated or inconsistent")
        else:
            raw = subprocess.check_output([str(generator), str(count*8), "6", str(config["seed"]+offset)], text=True)
            candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
            states = encode_states([dict(history=row["moves"]) for row in candidates], encoder)
            rows, keys = [], []
            for i, row in enumerate(candidates):
                key = tuple(int(states[k][i]) for k in prepare_replay.STATE_FIELDS)
                if key in seen or key in keys:
                    continue
                rows.append(row); keys.append(key)
                if len(rows) == count:
                    break
            if len(rows) != count:
                raise ValueError("could not generate enough distinct arena opening states")
            path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        seen.update(keys)
        histories = {tuple(row["moves"][:end]) for row in rows for end in range(len(row["moves"])+1)}
        states = encode_states([dict(history=list(h)) for h in histories], encoder)
        blocked.update(tuple(int(states[k][i]) for k in prepare_replay.STATE_FIELDS) for i in range(len(histories)))
    return blocked


def paired_match(candidate, baseline, openings, folder, pairs, time_ms, config):
    identity = local_arena.make_manifest(
        dict(pairs=pairs, time_ms=time_ms, seed=config["seed"], protocol="local-paired-v1"),
        dict(candidate=candidate[0], candidate_weights=candidate[1], baseline=baseline[0],
             baseline_weights=baseline[1], openings=openings,
             referee=ROOT/"tools/external/local_arena.py"))
    games_path, rows = local_arena.prepare_resume(folder, identity)
    latest = {(int(r["opening_index"]), int(r["zq_player"])):r for r in rows}
    book = [json.loads(line)["moves"] for line in openings.read_text().splitlines() if line.strip()]
    if len(book) < pairs:
        raise ValueError("not enough independent paired openings")
    def player(engine, name):
        return local_arena.UciPlayer([str(engine[0]), "--nnue", str(engine[1])], name)
    with games_path.open("a", encoding="utf-8", buffering=1) as stream:
        for index, opening in enumerate(book[:pairs]):
            for side in (0,1):
                previous = latest.get((index,side))
                if previous is not None and previous["status"] == "ok":
                    continue
                row = local_arena.play_game(opponent="baseline", opening_index=index, opening=opening,
                    zq_player=side, zq_factory=lambda:player(candidate,"candidate"),
                    opponent_factory=lambda:player(baseline,"baseline"), zq_budget=time_ms,
                    opponent_budget=time_ms, move_timeout_s=30, max_plies=240, run_id=identity["run_id"])
                stream.write(json.dumps(row)+"\n")
                latest[index,side] = row
                if row["status"] != "ok":
                    raise RuntimeError(f"arena failed: {row.get('error')}")
            if (index+1) % 10 == 0:
                print(f"Arena {folder.name}: {index+1}/{pairs} pairs", flush=True)
    report = local_arena.summarize_pairs(list(latest.values()), bootstrap=config["bootstrap"], seed=config["seed"])
    (folder/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def check_external_report(report, expected_pairs):
    if not report.get("summaries"):
        raise RuntimeError("external benchmark returned no opponents")
    for opponent, summary in report["summaries"].items():
        if summary["failed_games"] or summary["complete_pairs"] != expected_pairs:
            raise RuntimeError(f"external benchmark is incomplete for {opponent}")
    return report


_ACTIVE_FOLDER = None


def _run_campaign(argv=None):
    global _ACTIVE_FOLDER
    parser = argparse.ArgumentParser(description=__doc__)
    for key,value in CONFIG.items():
        kwargs = dict(default=argparse.SUPPRESS)
        if isinstance(value,bool): kwargs["action"] = argparse.BooleanOptionalAction
        elif isinstance(value,list): kwargs["type"] = json.loads
        else: kwargs["type"] = type(value)
        parser.add_argument("--"+key.replace("_","-"),**kwargs)
    config = copy.deepcopy(CONFIG)
    config.update(vars(parser.parse_args(argv)))
    for key in ("positions", "epochs", "batch_size", "screen_pairs", "confirmation_pairs"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not config["architectures"] or not config["seeds"]:
        raise ValueError("at least one architecture and seed are required")
    if config["dry_run"]:
        print(json.dumps(config,indent=2)); return 0
    folder = Path(config["out_dir"]).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    config_path = folder/"campaign_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("campaign configuration changed; choose a new out_dir")
    config_path.write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    _ACTIVE_FOLDER = folder
    def status(stage, **extra):
        row = dict(stage=stage, **extra)
        (folder/"status.json").write_text(json.dumps(row,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(row),flush=True)
    status("reserve_arena_holdouts")
    blocked = arena_holdouts(folder, config) if config["arena"] else set()
    status("replay_teaching")
    replay_config = dict(prepare_replay.CONFIG, source=config["source"], out_dir=str(folder/"replay"),
                         max_positions=config["positions"], device=config["device"], seed=config["seed"])
    prepare_replay.run(replay_config)
    fresh = None
    if config["fresh_positions"]:
        status("fresh_search_teaching")
        fresh = folder/"fresh/dataset.npz"
        subprocess.run([sys.executable,str(ROOT/"training/run_teaching.py"),"--mode","mixed",
            "--out-dir",str(folder/"fresh"),"--games",str(config["fresh_games"]),
            "--max-positions",str(config["fresh_positions"]),"--claustro-sims",str(config["claustro_sims"]),
            "--zq-nodes",str(config["zq_nodes"]),"--gamma",str(config["gamma"]),
            "--outcome-weight",str(config["outcome_weight"]),"--bootstrap-weight",str(config["bootstrap_weight"]),
            "--trajectory-movetime-ms",str(config["trajectory_time_ms"]),"--device",config["device"],
            "--seed",str(config["seed"])],check=True,cwd=ROOT)
    dataset = folder/"dataset.npz"
    dataset_info = combine_datasets(folder/"replay/dataset.npz",fresh,dataset,config["fresh_weight_fraction"],blocked)
    (folder/"dataset_manifest.json").write_text(json.dumps(dataset_info,indent=2)+"\n",encoding="utf-8")
    candidates = []
    for arch in config["architectures"]:
        architecture,hidden = arch.split(":")
        for seed in config["seeds"]:
            name = f"{architecture}{hidden}-s{seed}"
            options = dict(run_experiment.CONFIG, data=str(dataset), out_dir=str(folder/name),
                architecture=architecture,hidden=int(hidden),seed=seed,epochs=config["epochs"],
                batch_size=config["batch_size"],device=config["device"],teaching=False,benchmark=False)
            status("training",candidate=name)
            if config["train"]: run_experiment.train(options)
            executable = run_experiment.build_candidate(options)
            candidates.append((name,(executable,folder/name/"student_int8.bin")))
    if not config["arena"]:
        status("trained_pending_arena",candidates=[n for n,_ in candidates]); return 0
    baseline_exe = run_benchmark._build_zq(ROOT/"bin/local_benchmark"/("zquoridor_uci.exe" if sys.platform == "win32" else "zquoridor_uci"))
    baseline = (baseline_exe,ROOT/"data/nnue/nnue_weights_int8.bin")
    from teachers.teaching_pipeline import _tool
    generator = _tool("teaching_generate_openings","tools/teacher/generate_openings.cpp")
    def book(path,count,seed):
        if not path.exists():
            with path.open("w",encoding="utf-8") as stream:
                subprocess.run([str(generator),str(count),"6",str(seed)],stdout=stream,check=True)
        return path
    screen_book = book(folder/"screen_openings.jsonl",config["screen_pairs"],config["seed"]+71)
    scores = []
    for name,candidate in candidates:
        status("screen_arena",candidate=name)
        result = paired_match(candidate,baseline,screen_book,folder/(name+"-arena"),config["screen_pairs"],config["screen_time_ms"],config)
        scores.append((result["score_pct"],name,candidate))
    scores.sort(reverse=True)
    _,name,candidate = scores[0]
    status("independent_confirmation",candidate=name)
    confirm_book = book(folder/"confirm_openings.jsonl",config["confirmation_pairs"],config["seed"]+149)
    result = paired_match(candidate,baseline,confirm_book,folder/"confirmation",config["confirmation_pairs"],config["confirmation_time_ms"],config)
    decision = dict(screen_ranking=[dict(name=n,score_pct=s) for s,n,_ in scores],
        best_screen_candidate=name,confirmation=result,production_weights_changed=False,
        conclusion="candidate_beats_baseline" if result["strength_claim_ready"] and result["decision_95"]["score_low_pct"]>50 else "no_confirmed_improvement_keep_main")
    if config["external"]:
        status("external_confirmation",candidate=name)
        for label,engine in ((name,candidate),("main",baseline)):
            options = dict(run_benchmark.CONFIG,zq_executable=str(engine[0]),nnue=str(engine[1]),
                output=str(folder/("external-"+label)),pairs=config["external_pairs"],
                openings=str(folder/"external_openings.jsonl"), seed=config["seed"],
                zq_move_time_ms=config["confirmation_time_ms"], titanium_move_time_ms=config["confirmation_time_ms"],
                claustrophobia_sims=config["external_sims"],claustrophobia_device=config["external_device"])
            decision["external_"+label] = check_external_report(run_benchmark.run(options), config["external_pairs"])
        candidate_report = decision["external_"+name]["summaries"]
        main_report = decision["external_main"]["summaries"]
        decision["external_comparison"] = {
            opponent: dict(candidate_score_pct=summary["score_pct"],
                main_score_pct=main_report[opponent]["score_pct"],
                score_difference_pct=summary["score_pct"]-main_report[opponent]["score_pct"],
                confirmed_above_opponent=bool(summary["strength_claim_ready"] and
                    summary["decision_95"]["score_low_pct"] > 50))
            for opponent, summary in candidate_report.items()}
    (folder/"decision.json").write_text(json.dumps(decision,indent=2)+"\n",encoding="utf-8")
    status("complete",decision=str(folder/"decision.json"),conclusion=decision["conclusion"])
    return 0


def main(argv=None):
    global _ACTIVE_FOLDER
    _ACTIVE_FOLDER = None
    try:
        return _run_campaign(argv)
    except BaseException as error:
        if _ACTIVE_FOLDER is not None:
            path = _ACTIVE_FOLDER / "status.json"
            previous = json.loads(path.read_text()) if path.exists() else {}
            path.write_text(json.dumps(dict(stage="failed", previous_stage=previous.get("stage"),
                error_type=type(error).__name__, error=str(error)), indent=2)+"\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
