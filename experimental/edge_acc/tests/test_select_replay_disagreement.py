import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import select_replay_disagreement as selector  # noqa: E402


def test_exports_highest_direct_teacher_disagreements(tmp_path):
    replay = tmp_path / "replay"
    replay.mkdir()
    states = {
        "id": np.asarray([b"a", b"b", b"c"]),
        "is_val": np.asarray([False, True, False]),
        "own_pawn": np.asarray([4, 5, 6]), "opp_pawn": np.asarray([76, 75, 74]),
        "walls_h": np.zeros(3, np.uint64), "walls_v": np.zeros(3, np.uint64),
        "walls_left_own": np.full(3, 10), "walls_left_opp": np.full(3, 10),
    }
    np.savez(replay / "dataset.npz", **states)
    old = np.zeros((3, 209), np.float32); old[:, 0] = 1
    claustro = old.copy(); claustro[1] = 0; claustro[1, 1] = 1
    np.savez(replay / "direct_00000000.npz", id=states["id"], old_policy=old,
             claustro_policy=claustro, old_value=np.asarray([0., 0., 0.]),
             claustro_value=np.asarray([0., .8, .2]))
    out = tmp_path / "selected.jsonl"
    manifest = selector.run(dict(replay_dir=str(replay), out=str(out), max_positions=2,
                                 policy_weight=1.0, value_weight=1.0))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert manifest["samples"] == 2
    assert [row["id"] for row in rows] == ["b", "c"]
    assert rows[0]["history"] == ["@state", "5", "75", "0", "0", "10", "10"]
    assert rows[0]["side_to_move"] == 0
