#!/usr/bin/env python3
"""Extract completed StepRecord entries from live generate_rollouts.exe memory,
validate integrity, and generate dataset.npz without stopping the process.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

from build_teacher_soft import encode_states

k32 = ctypes.windll.kernel32
k32.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
k32.VirtualQueryEx.restype = ctypes.c_size_t
k32.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
k32.ReadProcessMemory.restype = wt.BOOL


class MBI(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", wt.DWORD),
        ("PartitionId", wt.WORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wt.DWORD),
        ("Protect", wt.DWORD),
        ("Type", wt.DWORD),
    ]


ID_PATTERN = re.compile(rb"^[0-9a-f]{24}_r\d+_p\d+$")


def find_rollout_pid() -> int | None:
    # Use tasklist / PowerShell to find PID of generate_rollouts
    cmd = ["powershell", "-Command", "Get-Process -Name generate_rollouts -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout.strip()
    if out:
        return int(out.split()[0])
    return None


def read_string(hProc, ptr: int, length: int) -> str | None:
    if length > 256 or length < 0:
        return None
    buf = ctypes.create_string_buffer(length)
    if k32.ReadProcessMemory(hProc, ptr, buf, length, None):
        try:
            return buf.raw[:length].decode("ascii")
        except UnicodeDecodeError:
            return None
    return None


def read_history(hProc, h_start: int, h_finish: int) -> list[str] | None:
    if h_finish < h_start:
        return None
    diff = h_finish - h_start
    if diff == 0:
        return []
    if diff % 32 != 0:
        return None
    count = diff // 32
    if count > 200:
        return None

    h_buf = ctypes.create_string_buffer(diff)
    if not k32.ReadProcessMemory(hProc, h_start, h_buf, diff, None):
        return None

    history = []
    for i in range(count):
        elem = h_buf.raw[i * 32 : (i + 1) * 32]
        p_str = struct.unpack("<Q", elem[0:8])[0]
        e_len = struct.unpack("<Q", elem[8:16])[0]
        if e_len < 0 or e_len > 10:
            return None
        if e_len <= 15:
            move_str = elem[16 : 16 + e_len].decode("ascii", errors="ignore")
        else:
            move_str = read_string(hProc, p_str, e_len)
            if move_str is None:
                return None
        # Quoridor move is 2 or 3 chars (e.g. e2, e3, e4v, d5h)
        if not (2 <= len(move_str) <= 3):
            return None
        history.append(move_str)

    return history


def extract_records(pid: int) -> list[dict]:
    hProc = k32.OpenProcess(0x0410, False, pid)
    if not hProc:
        raise RuntimeError(f"Cannot open process {pid}")

    print(f"Scanning memory of process PID {pid}...", flush=True)
    mbi = MBI()
    addr = 0
    records_by_id = {}

    while addr < 0x7FFF00000000:
        if not k32.VirtualQueryEx(hProc, addr, ctypes.byref(mbi), ctypes.sizeof(mbi)):
            break
        base = mbi.BaseAddress or 0
        size = mbi.RegionSize
        if mbi.State == 0x1000 and (mbi.Protect & 0x04) and size <= 16 * 1024 * 1024:
            buf = ctypes.create_string_buffer(size)
            if k32.ReadProcessMemory(hProc, base, buf, size, None):
                raw = buf.raw[:size]
                # Scan for StepRecord array blocks
                # In each StepRecord (88 bytes):
                # offset 0..7: pointer to id string
                # offset 8..15: id length (28..35)
                # offset 16..23: id capacity
                # offset 32..35: mover (0 or 1)
                # offset 40..47: history.start
                # offset 48..55: history.finish
                # offset 56..63: history.end
                # offset 64..67: chosenAction (0..208)
                # offset 72..79: discountedValue (double, abs <= 1.0)
                # offset 80..83: pliesRemaining (0..120)
                # Fast check: scan for candidate StepRecords
                p = 0
                while p <= size - 88:
                    id_len = struct.unpack("<Q", raw[p + 8 : p + 16])[0]
                    if 28 <= id_len <= 35:
                        mover = struct.unpack("<i", raw[p + 32 : p + 36])[0]
                        if mover in (0, 1):
                            action = struct.unpack("<i", raw[p + 64 : p + 68])[0]
                            if 0 <= action <= 208:
                                plies = struct.unpack("<i", raw[p + 80 : p + 84])[0]
                                if 0 <= plies <= 120:
                                    val = struct.unpack("<d", raw[p + 72 : p + 80])[0]
                                    if -1.0001 <= val <= 1.0001 and val != 0.0:
                                        p_id = struct.unpack("<Q", raw[p : p + 8])[0]
                                        id_str = read_string(hProc, p_id, id_len)
                                        if id_str and ID_PATTERN.match(id_str.encode("ascii")):
                                            h_start = struct.unpack("<Q", raw[p + 40 : p + 48])[0]
                                            h_finish = struct.unpack("<Q", raw[p + 48 : p + 56])[0]
                                            history = read_history(hProc, h_start, h_finish)
                                            if history is not None:
                                                rec = {
                                                    "id": id_str,
                                                    "schema": "zquoridor.position.v1",
                                                    "side_to_move": mover,
                                                    "history": history,
                                                    "best_action": action,
                                                    "root_value": val,
                                                    "plies_remaining": plies,
                                                }
                                                records_by_id[id_str] = rec
                                                p += 88
                                                continue
                    p += 8

        addr = base + size

    k32.CloseHandle(hProc)
    print(f"Extracted {len(records_by_id)} unique, fully-validated StepRecords from memory.", flush=True)
    return list(records_by_id.values())


def main():
    pid = find_rollout_pid()
    if not pid:
        print("No running generate_rollouts process found!", file=sys.stderr)
        sys.exit(1)

    records = extract_records(pid)
    if not records:
        print("Error: No records extracted from process memory!", file=sys.stderr)
        sys.exit(1)

    out_dir = ROOT / "data" / "teaching" / "loss-center-rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_rollouts_file = out_dir / "raw_rollouts.jsonl"
    out_dataset = out_dir / "dataset.npz"

    print(f"Writing {len(records)} records to {raw_rollouts_file}...", flush=True)
    with open(raw_rollouts_file, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")

    # Verify JSONL
    print("Verifying saved JSONL integrity...", flush=True)
    with open(raw_rollouts_file, "r", encoding="utf-8") as fh:
        read_back = [json.loads(line) for line in fh if line.strip()]
    assert len(read_back) == len(records), "Mismatch in written vs read records"

    # Encode states via C++ state encoder
    encoder_bin = ROOT / "bin" / "teacher_encode_state.exe"
    print(f"Encoding {len(records)} states via {encoder_bin}...", flush=True)
    encoded = encode_states(records, encoder_bin)

    n = len(records)
    policy = np.zeros((n, 209), dtype=np.float16)
    value = np.zeros(n, dtype=np.float16)
    seed_ids = []

    for i, r in enumerate(records):
        action = int(r["best_action"])
        policy[i, action] = 1.0
        value[i] = float(r["root_value"])
        m = re.match(r"^(.+)_r\d+_p\d+$", r["id"])
        seed_ids.append(m.group(1) if m else r["id"])

    # Grouped split: 85% train / 15% val by seed ID
    unique_seeds = sorted(set(seed_ids))
    rng = np.random.default_rng(20260919)
    rng.shuffle(unique_seeds)
    n_val_seeds = max(1, int(len(unique_seeds) * 0.15))
    val_seed_set = set(unique_seeds[:n_val_seeds])
    is_val = np.array([sid in val_seed_set for sid in seed_ids], dtype=bool)

    dataset_arrays = {
        "own_pawn": encoded["own_pawn"],
        "opp_pawn": encoded["opp_pawn"],
        "walls_h": encoded["walls_h"],
        "walls_v": encoded["walls_v"],
        "walls_left_own": encoded["walls_left_own"],
        "walls_left_opp": encoded["walls_left_opp"],
        "own_dist": encoded["own_dist"],
        "opp_dist": encoded["opp_dist"],
        "mover": encoded["mover"],
        "policy": policy,
        "value": value,
        "weight": np.full(n, 8.0, dtype=np.float32),
        "is_val": is_val,
        "group_id": np.array([f"rollout:{sid}".encode("utf-8") for sid in seed_ids]),
    }

    n_train = int((~is_val).sum())
    n_val = int(is_val.sum())
    print(f"Split: {n_train} train ({len(unique_seeds) - n_val_seeds} seeds), {n_val} val ({n_val_seeds} seeds)", flush=True)

    np.savez_compressed(out_dataset, **dataset_arrays)
    print(f"SUCCESS: Saved canonical rollout dataset to {out_dataset} ({out_dataset.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)

    manifest = {
        "schema": "zquoridor.rollouts.v1",
        "seeds": len(unique_seeds),
        "rollout_steps": n,
        "train_samples": n_train,
        "val_samples": n_val,
        "gamma": 0.98,
        "explore_plies": 4,
        "sample_weight": 8.0,
        "out": str(out_dataset),
    }
    manifest_file = out_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote manifest to {manifest_file}", flush=True)


if __name__ == "__main__":
    main()
