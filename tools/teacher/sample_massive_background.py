#!/usr/bin/env python3
"""Sample massive background replay dataset from historical canonical V3 shards.

Applies oversampling to wall-depletion asymmetry and dense midgame positions.
Labels positions using our champion net (race512-search10-ft) on GPU in fast batches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from read_selfplay import _detect_format, SAMPLE_DTYPE, expand_data_paths
from student_model import Student, encode_features
from prepare_replay import legal_wall_topology, STATE_FIELDS

CONFIG = {
    "source": str(ROOT / "data/selfplay_canonical_v3"),
    "out": str(ROOT / "data/teaching/massive-background-sample/dataset.npz"),
    "champion": str(ROOT / "results/experiments/race512-search10-ft-s20260917/student.bin"),
    "target_positions": 2000000,
    "seed": 20260919,
    "val_fraction": 0.15,
    "batch_size": 8192,
    "asymmetry_oversample": 1.5,
    "dense_oversample": 1.5,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sample_shards(config: dict, blocked: set[tuple]) -> tuple[dict[str, np.ndarray], dict]:
    files = sorted(Path(config["source"]).rglob("*.bin"))
    valid = []
    for path in files:
        # Include all canonical generations with V2/V3 formats
        try:
            dtype, size = _detect_format(str(path))
            if size in (32, 64):
                valid.append((path, dtype, size))
        except Exception:
            continue

    if not valid:
        raise ValueError("no valid 32/64-byte canonical shards found")

    rng = np.random.default_rng(config["seed"])
    rng.shuffle(valid)

    target_n = config["target_positions"]
    quota_per_shard = math.ceil(target_n / len(valid))

    own_pawn = np.empty(target_n, dtype=np.uint8)
    opp_pawn = np.empty(target_n, dtype=np.uint8)
    walls_h = np.empty(target_n, dtype=np.uint64)
    walls_v = np.empty(target_n, dtype=np.uint64)
    walls_left_own = np.empty(target_n, dtype=np.int8)
    walls_left_opp = np.empty(target_n, dtype=np.int8)
    own_dist = np.empty(target_n, dtype=np.uint8)
    opp_dist = np.empty(target_n, dtype=np.uint8)
    game_result = np.empty(target_n, dtype=np.int8)
    ids = np.empty(target_n, dtype="S24")
    groups = np.empty(target_n, dtype="S64")
    splits = np.empty(target_n, dtype=bool)

    seen = set(blocked)
    n_val_shards = max(1, int(len(valid) * config["val_fraction"]))
    accepted_total = 0

    print(f"Scanning {len(valid)} valid canonical shards for {target_n:,} target positions...", flush=True)

    for file_idx, (path, dtype, size) in enumerate(valid):
        digest = sha(path)
        digest_bytes = digest[:16].encode("ascii")
        is_val = file_idx < n_val_shards
        data = np.memmap(path, dtype=dtype, mode="r")
        n_data = len(data)
        if n_data == 0:
            continue

        accepted_this_shard = 0
        perm = rng.permutation(n_data)

        for idx in perm:
            row = data[idx]
            own_p = int(row["own_pawn"])
            opp_p = int(row["opp_pawn"])
            if own_p >= 72 or opp_p <= 8:
                continue

            ow = int(row["walls_left_own"])
            pw = int(row["walls_left_opp"])
            if not (0 <= ow <= 10 and 0 <= pw <= 10):
                continue

            wh = int(row["walls_h"])
            wv = int(row["walls_v"])
            key = (own_p, opp_p, wh, wv, ow, pw)
            if key in seen:
                continue

            # Oversampling filter for weakness positions
            is_asymmetric = abs(ow - pw) >= 2 or ow <= 3 or pw <= 3
            is_dense = (20 - ow - pw) >= 8
            # If neither, downsample plain positions to enrich weakness states
            if not is_asymmetric and not is_dense:
                if rng.random() > 0.40:
                    continue

            if not legal_wall_topology(wh, wv, own_p, opp_p):
                continue

            seen.add(key)
            own_pawn[accepted_total] = own_p
            opp_pawn[accepted_total] = opp_p
            walls_h[accepted_total] = wh
            walls_v[accepted_total] = wv
            walls_left_own[accepted_total] = ow
            walls_left_opp[accepted_total] = pw
            own_dist[accepted_total] = int(row["own_dist"])
            opp_dist[accepted_total] = int(row["opp_dist"])
            game_result[accepted_total] = int(row["game_result"])
            ids[accepted_total] = f"{digest[:8]}_{idx}".encode("ascii")
            groups[accepted_total] = digest_bytes
            splits[accepted_total] = is_val

            accepted_total += 1
            accepted_this_shard += 1
            if accepted_this_shard >= quota_per_shard or accepted_total >= target_n:
                break

        del data
        if accepted_total >= target_n:
            break

    # If shortfall from initial quota pass, do a second sweep
    if accepted_total < target_n:
        print(f"Shortfall pass: accepted {accepted_total:,} of {target_n:,}, sweeping remaining data...", flush=True)
        for file_idx, (path, dtype, size) in enumerate(valid):
            digest = sha(path)
            digest_bytes = digest[:16].encode("ascii")
            is_val = file_idx < n_val_shards
            data = np.memmap(path, dtype=dtype, mode="r")
            n_data = len(data)
            if n_data == 0:
                continue
            for idx in rng.permutation(n_data):
                row = data[idx]
                own_p = int(row["own_pawn"])
                opp_p = int(row["opp_pawn"])
                if own_p >= 72 or opp_p <= 8:
                    continue
                ow = int(row["walls_left_own"])
                pw = int(row["walls_left_opp"])
                if not (0 <= ow <= 10 and 0 <= pw <= 10):
                    continue
                wh = int(row["walls_h"])
                wv = int(row["walls_v"])
                key = (own_p, opp_p, wh, wv, ow, pw)
                if key in seen or not legal_wall_topology(wh, wv, own_p, opp_p):
                    continue
                seen.add(key)
                own_pawn[accepted_total] = own_p
                opp_pawn[accepted_total] = opp_p
                walls_h[accepted_total] = wh
                walls_v[accepted_total] = wv
                walls_left_own[accepted_total] = ow
                walls_left_opp[accepted_total] = pw
                own_dist[accepted_total] = int(row["own_dist"])
                opp_dist[accepted_total] = int(row["opp_dist"])
                game_result[accepted_total] = int(row["game_result"])
                ids[accepted_total] = f"{digest[:8]}_{idx}".encode("ascii")
                groups[accepted_total] = digest_bytes
                splits[accepted_total] = is_val
                accepted_total += 1
                if accepted_total >= target_n:
                    break
            del data
            if accepted_total >= target_n:
                break

    print(f"Sampled {accepted_total:,} distinct positions across {len(valid)} shards.", flush=True)

    arrays = {
        "own_pawn": own_pawn[:accepted_total],
        "opp_pawn": opp_pawn[:accepted_total],
        "walls_h": walls_h[:accepted_total],
        "walls_v": walls_v[:accepted_total],
        "walls_left_own": walls_left_own[:accepted_total],
        "walls_left_opp": walls_left_opp[:accepted_total],
        "own_dist": own_dist[:accepted_total],
        "opp_dist": opp_dist[:accepted_total],
        "game_result": game_result[:accepted_total],
        "id": ids[:accepted_total],
        "group_id": groups[:accepted_total],
        "is_val": splits[:accepted_total],
    }
    return arrays, {"samples": accepted_total, "shards_used": len(valid)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(CONFIG["source"]))
    parser.add_argument("--out", type=Path, default=Path(CONFIG["out"]))
    parser.add_argument("--champion", type=Path, default=Path(CONFIG["champion"]))
    parser.add_argument("--target-positions", type=int, default=CONFIG["target_positions"])
    parser.add_argument("--val-fraction", type=float, default=CONFIG["val_fraction"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Block benchmark opening positions
    blocked = set()
    openings_path = ROOT / "tools/external/openings_titanium.jsonl"
    if openings_path.exists():
        from teachers.teaching_pipeline import _tool
        from build_teacher_soft import encode_states
        openings = [json.loads(line)["moves"] for line in openings_path.read_text().splitlines() if line.strip()]
        histories = {tuple(moves[:end]) for moves in openings for end in range(len(moves) + 1)}
        encoder = _tool("teacher_encode_state", "tools/teacher/encode_state.cpp")
        blocked_data = encode_states([{"history": list(h)} for h in histories], encoder)
        blocked = {tuple(int(blocked_data[k][i]) for k in STATE_FIELDS) for i in range(len(histories))}
        print(f"Blocked {len(blocked)} benchmark opening states.", flush=True)

    # 2. Sample candidate positions
    data, meta = sample_shards(vars(args), blocked)
    n = len(data["id"])

    # 3. Predict targets via GPU inference with champion student
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading champion net from {args.champion} onto {device}...", flush=True)
    model = Student("race", 512, qat=True)
    model.load_float(str(args.champion))
    model.to(device).eval()

    all_pol = np.empty((n, 209), dtype=np.float16)
    all_val = np.empty(n, dtype=np.float16)
    print(f"Generating soft targets for {n:,} positions in batches of {args.batch_size}...", flush=True)

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()

    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            stop = min(n, start + args.batch_size)
            idx = np.arange(start, stop)
            feat = encode_features(data, idx, "race")
            x = torch.from_numpy(feat).to(device)
            v_logit, p_logits = model(x)
            all_pol[start:stop] = F.softmax(p_logits, dim=1).cpu().numpy().astype(np.float16)
            all_val[start:stop] = (torch.sigmoid(v_logit) * 2.0 - 1.0).cpu().numpy().astype(np.float16)

    t1.record()
    torch.cuda.synchronize()

    # Blend network value with true game_result (70% net value + 30% empirical outcome)
    blended_val = 0.70 * all_val + 0.30 * data["game_result"].astype(np.float16)

    data["policy"] = all_pol
    data["value"] = blended_val.astype(np.float16)
    data["weight"] = np.ones(n, dtype=np.float32)

    np.savez(args.out, **data)
    manifest = {
        "schema": "zquoridor.massive_background.v1",
        "out": str(args.out),
        "samples": n,
        "train_samples": int((~data["is_val"]).sum()),
        "val_samples": int(data["is_val"].sum()),
        "champion": str(args.champion),
        "device": device,
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
