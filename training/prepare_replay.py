#!/usr/bin/env python3
"""Build a replay dataset from existing canonical self-play shards.

Input:
    ``data/selfplay/<generation>/selfplay_*.bin`` (32/64-byte canonical
    records; legacy ambiguous 27-byte shards are skipped), plus the old
    NNUE weights and the pinned Claustrophobia checkpoint.

Output:
    ``<out-dir>/dataset.npz`` with train/validation splits and blended policy
    and value targets.  Resumable ``direct_*.npz`` caches, SHA-256 files, and
    ``replay_manifest.json`` are written beside it.  This script never creates
    self-play games, runs search, or trains a student network.

The configuration block is the default; every field can be overridden with a
matching CLI option.  Use a new output directory when inputs or settings
change.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from collections import deque
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training/teachers"))
from read_selfplay import _detect_format
from student_model import Student, encode_features
from teachers.claustrophobia_relabel import encode_positions
from teachers.claustrophobia_inference_worker import infer
from tools.external.bot_setup import ensure_bot

CONFIG = {
    "source": str(ROOT / "data/selfplay_canonical_v3"),
    "out_dir": str(ROOT / "data/teaching/replay"),
    "max_positions": 50000,
    "seed": 20260914,
    "val_fraction": 0.2,
    "chunk_size": 8192,
    "batch_size": 4096,
    "device": "auto",
    "old_nnue": str(ROOT / "checkpoints/gen9-teacher-head/nnue_weights.bin"),
    "old_policy_weight": 0.5,
    "claustro_policy_weight": 0.5,
    "old_value_weight": 0.75,
    "claustro_value_weight": 0.25,
    "outcome_weight": 0.0,
    "resume": True,
}
STATE_FIELDS = ("own_pawn", "opp_pawn", "walls_h", "walls_v", "walls_left_own", "walls_left_opp")


def legal_wall_topology(walls_h: int, walls_v: int, own_pawn: int, opp_pawn: int) -> bool:
    """Validate a recorded board before sending it to the Claustrophobia bridge.

    Historical shards may contain a few corrupt wall sets.  The direct bridge
    correctly rejects them, but filtering at sampling time lets a million-row
    relabel run replace the bad row instead of stopping after hours.  A wall
    crossing uses the same anchor in both bitboards; both pawns must retain a
    path to their respective goal rows.
    """
    walls_h, walls_v = int(walls_h), int(walls_v)
    if walls_h & walls_v:
        return False

    def blocked(row, col, next_row, next_col):
        if row == next_row:
            anchor_row, anchor_col = row, min(col, next_col)
            return ((anchor_row > 0 and (walls_v >> ((anchor_row - 1) * 8 + anchor_col)) & 1)
                    or (anchor_row < 8 and (walls_v >> (anchor_row * 8 + anchor_col)) & 1))
        anchor_row, anchor_col = min(row, next_row), col
        return ((anchor_col > 0 and (walls_h >> (anchor_row * 8 + anchor_col - 1)) & 1)
                or (anchor_col < 8 and (walls_h >> (anchor_row * 8 + anchor_col)) & 1))

    def has_path(cell, goal_row):
        queue, seen = deque([cell]), {cell}
        while queue:
            current = queue.popleft()
            row, col = divmod(current, 9)
            if row == goal_row:
                return True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                nxt = nr * 9 + nc
                if 0 <= nr < 9 and 0 <= nc < 9 and nxt not in seen and not blocked(row, col, nr, nc):
                    seen.add(nxt)
                    queue.append(nxt)
        return False

    return has_path(int(own_pawn), 8) and has_path(int(opp_pawn), 0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sample_states(config, blocked=()):
    files = sorted(Path(config["source"]).rglob("*.bin"))
    valid, skipped = [], []
    for path in files:
        dtype, size = _detect_format(path)
        if size == 27:
            skipped.append(str(path))
        else:
            valid.append((path, dtype, size))
    if len(valid) < 2:
        raise ValueError("replay requires at least two canonical 32/64-byte shards; ambiguous 27-byte data is excluded")
    rng = np.random.default_rng(config["seed"])
    rng.shuffle(valid)
    n_val = max(1, min(len(valid)-1, round(len(valid)*config["val_fraction"])))
    quota = math.ceil(config["max_positions"] / len(valid))
    rows, ids, groups, splits, provenance = [], [], [], [], []
    seen = set(blocked)
    for file_index, (path, dtype, size) in enumerate(valid):
        digest = sha(path)
        provenance.append(dict(path=str(path.resolve()), sha256=digest, record_bytes=size))
        data = np.memmap(path, dtype=dtype, mode="r")
        accepted = 0
        # Use a per-shard random permutation rather than a fixed oversample.
        # Small runs still stop as soon as their quota is filled, while a large
        # run can consume the long tail of a shard when duplicate openings or
        # holdouts reject more rows than the former 4x bound allowed.
        candidates = rng.permutation(len(data))
        for index in candidates:
            row = data[index]
            key = tuple(int(row[name]) for name in STATE_FIELDS)
            if (key in seen or row["own_pawn"] >= 72 or row["opp_pawn"] <= 8
                    or not (0 <= row["walls_left_own"] <= 10)
                    or not (0 <= row["walls_left_opp"] <= 10)
                    or not legal_wall_topology(row["walls_h"], row["walls_v"], row["own_pawn"], row["opp_pawn"])):
                continue
            seen.add(key)
            rows.append(tuple(int(row[name]) for name in (*STATE_FIELDS, "own_dist", "opp_dist", "game_result")))
            ids.append(hashlib.sha256(f"{digest}:{index}".encode()).hexdigest()[:24])
            groups.append(digest)
            splits.append(file_index < n_val)
            accepted += 1
            if accepted >= quota or len(rows) >= config["max_positions"]:
                break
        del data
        if len(rows) >= config["max_positions"]:
            break
    # Some shards contain fewer eligible, distinct positions than the uniform
    # quota. Borrow the shortfall from the training shards first, then from
    # validation shards if necessary. `seen` keeps this second pass disjoint
    # from the balanced first pass and preserves whole-shard group identities.
    if len(rows) < config["max_positions"]:
        for validation in (False, True):
            for file_index, (path, dtype, _size) in enumerate(valid):
                if (file_index < n_val) != validation:
                    continue
                digest = provenance[file_index]["sha256"]
                data = np.memmap(path, dtype=dtype, mode="r")
                for index in rng.permutation(len(data)):
                    row = data[index]
                    key = tuple(int(row[name]) for name in STATE_FIELDS)
                    if (key in seen or row["own_pawn"] >= 72 or row["opp_pawn"] <= 8
                            or not (0 <= row["walls_left_own"] <= 10)
                            or not (0 <= row["walls_left_opp"] <= 10)
                            or not legal_wall_topology(row["walls_h"], row["walls_v"], row["own_pawn"], row["opp_pawn"])):
                        continue
                    seen.add(key)
                    rows.append(tuple(int(row[name]) for name in (*STATE_FIELDS, "own_dist", "opp_dist", "game_result")))
                    ids.append(hashlib.sha256(f"{digest}:{index}".encode()).hexdigest()[:24])
                    groups.append(digest)
                    splits.append(validation)
                    if len(rows) >= config["max_positions"]:
                        break
                del data
                if len(rows) >= config["max_positions"]:
                    break
            if len(rows) >= config["max_positions"]:
                break
    if len(rows) < config["max_positions"]:
        raise ValueError(f"requested {config['max_positions']} positions but found only {len(rows)} distinct eligible samples")
    if len(rows) < 2 or not any(splits) or all(splits):
        raise ValueError("insufficient independent train/validation replay positions")
    names = (*STATE_FIELDS, "own_dist", "opp_dist", "game_result")
    arrays = {name: np.asarray([r[i] for r in rows], dtype=np.uint64 if name in ("walls_h", "walls_v") else np.int64)
              for i, name in enumerate(names)}
    arrays.update(id=np.asarray(ids, dtype="S24"), group_id=np.asarray(groups, dtype="S64"),
                  is_val=np.asarray(splits, dtype=bool))
    return arrays, dict(shards=provenance, skipped_legacy_shards=skipped, samples=len(rows))


def raw_positions(data, start, stop):
    return [dict(id=data["id"][i].decode(), side_to_move=0,
                 history=["@state", *(str(int(data[key][i])) for key in STATE_FIELDS)])
            for i in range(start, stop)]


def run(config):
    for key in ("max_positions", "batch_size", "chunk_size"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < config["val_fraction"] < 1:
        raise ValueError("val_fraction must be in (0,1)")
    for key in ("old_policy_weight", "claustro_policy_weight", "old_value_weight", "claustro_value_weight", "outcome_weight"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError("teacher weights must be finite and nonnegative")
    pw = config["old_policy_weight"] + config["claustro_policy_weight"]
    vw = config["old_value_weight"] + config["claustro_value_weight"] + config["outcome_weight"]
    if pw <= 0 or vw <= 0:
        raise ValueError("at least one teacher weight per head must be positive")
    folder = Path(config["out_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    # Exclude all historical benchmark opening states, including prefixes.
    from teachers.teaching_pipeline import _tool
    from build_teacher_soft import encode_states
    openings = [json.loads(line)["moves"] for line in (ROOT / "tools/external/openings_titanium.jsonl").read_text().splitlines() if line.strip()]
    histories = {tuple(moves[:end]) for moves in openings for end in range(len(moves)+1)}
    blocked_data = encode_states([{"history": list(h)} for h in histories],
                                _tool("teacher_encode_state", "tools/teacher/encode_state.cpp"))
    blocked = {tuple(int(blocked_data[k][i]) for k in STATE_FIELDS) for i in range(len(histories))}
    data, provenance = sample_states(config, blocked)
    bots = ensure_bot("claustrophobia")
    identity = dict(config={k:v for k,v in config.items() if k != "resume"}, provenance=provenance,
                    old_sha=sha(config["old_nnue"]), claustro_sha=sha(bots["checkpoint"]),
                    encoder_sha=sha(bots["encode_bridge"]), code_sha=sha(__file__),
                    dependencies={str(p.relative_to(ROOT)): sha(p) for p in (
                        ROOT/"training/student_model.py", ROOT/"training/train_teacher_policy.py",
                        ROOT/"training/teachers/claustrophobia_inference_worker.py",
                        ROOT/"training/teachers/claustrophobia_relabel.py")})
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = folder / "replay_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["fingerprint"] != fingerprint:
            stable = (previous.get("config") == identity["config"]
                      and previous.get("provenance") == identity["provenance"]
                      and previous.get("old_sha") == identity["old_sha"]
                      and previous.get("claustro_sha") == identity["claustro_sha"]
                      and previous.get("encoder_sha") == identity["encoder_sha"])
            if not (config["resume"] and stable):
                raise ValueError("replay inputs changed; use a new out_dir")
            print("Replay code changed; verifying resumable chunks by sample id.", flush=True)
    manifest = dict(fingerprint=fingerprint, **identity, complete=False)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    old = Student("base", 256, qat=True)
    old.load_float(config["old_nnue"])
    old.to(device).eval()
    teacher = torch.jit.load(str(bots["checkpoint"]), map_location=device).eval()
    n = len(data["id"])
    chunks = []
    for start in range(0, n, config["chunk_size"]):
        stop = min(n, start + config["chunk_size"])
        cache = folder / f"direct_{start:08d}.npz"
        checksum = cache.with_suffix(".sha256")
        reuse = False
        if config["resume"] and cache.exists() and checksum.exists() and checksum.read_text().strip() == sha(cache):
            with np.load(cache, allow_pickle=False) as stored:
                chunk = {k:stored[k] for k in stored.files}
            reuse = np.array_equal(chunk["id"], data["id"][start:stop])
        if not reuse:
            planes, legal = encode_positions(raw_positions(data, start, stop), bots["encode_bridge"])
            cp, cv, op, ov = [], [], [], []
            for b in range(0, stop-start, config["batch_size"]):
                end = min(stop-start, b+config["batch_size"])
                p, v = infer(teacher, planes[b:end], legal[b:end], device)
                cp.append(p); cv.append(v)
                x = torch.from_numpy(encode_features(data, np.arange(start+b,start+end), "base")).to(device)
                with torch.inference_mode():
                    v, p = old(x)
                    p = p.masked_fill(~torch.from_numpy(legal[b:end]).to(device), -torch.inf).softmax(1)
                    op.append(p.cpu().numpy()); ov.append((v.sigmoid()*2-1).cpu().numpy())
            chunk = dict(id=data["id"][start:stop], old_policy=np.concatenate(op), old_value=np.concatenate(ov),
                         claustro_policy=np.concatenate(cp), claustro_value=np.concatenate(cv))
            temporary = cache.with_suffix(".tmp")
            with temporary.open("wb") as out:
                np.savez(out, **chunk)
            temporary.replace(cache)
            checksum.write_text(sha(cache)+"\n", encoding="ascii")
        chunks.append(chunk)
        print(f"Replay teaching: {stop}/{n} positions ({device})", flush=True)
    joined = {key:np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    data["policy"] = (joined["old_policy"]*config["old_policy_weight"] + joined["claustro_policy"]*config["claustro_policy_weight"])/pw
    data["value"] = (joined["old_value"]*config["old_value_weight"] + joined["claustro_value"]*config["claustro_value_weight"] + data["game_result"]*config["outcome_weight"])/vw
    data["weight"] = np.ones(n, np.float32)
    np.savez(folder / "dataset.npz", **data)
    manifest.update(complete=True, dataset=str((folder/"dataset.npz").resolve()),
                    train_samples=int((~data["is_val"]).sum()), val_samples=int(data["is_val"].sum()),
                    dataset_sha256=sha(folder/"dataset.npz"))
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in CONFIG.items():
        kwargs = dict(default=argparse.SUPPRESS)
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(value)
        parser.add_argument("--"+key.replace("_","-"), **kwargs)
    config = dict(CONFIG, **vars(parser.parse_args(argv)))
    print(json.dumps(run(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
