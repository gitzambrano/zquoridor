import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import build_search_priority_dataset as builder  # noqa: E402


def test_blends_search_targets_and_reweights_disagreement(tmp_path):
    source = tmp_path / "source.npz"
    ids = np.asarray([b"a", b"b"])
    common = dict(id=ids, own_pawn=np.asarray([4, 5]), opp_pawn=np.asarray([76, 75]),
                  walls_h=np.zeros(2, np.uint64), walls_v=np.zeros(2, np.uint64),
                  walls_left_own=np.full(2, 10), walls_left_opp=np.full(2, 10),
                  own_dist=np.ones(2), opp_dist=np.ones(2), game_result=np.zeros(2),
                  group_id=np.asarray([b"g0", b"g1"]), is_val=np.asarray([False, True]),
                  policy=np.zeros((2, 209), np.float32), value=np.zeros(2), weight=np.ones(2, np.float32))
    np.savez(source, **common)
    positions = tmp_path / "positions.jsonl"
    positions.write_text("\n".join(json.dumps(dict(schema="zquoridor.position.v1", id=x, history=["@state"], side_to_move=0)) for x in ("b", "a")) + "\n")
    zq_policy = np.zeros((2, 209), np.float16); zq_policy[:, 0] = 1
    cl_policy = zq_policy.copy(); cl_policy[0] = 0; cl_policy[0, 1] = 1
    zq = tmp_path / "zq.npz"; cl = tmp_path / "cl.npz"
    np.savez(zq, id=np.asarray([b"b", b"a"]), policy=zq_policy, value=np.asarray([0.2, 0.0]), budget_agreement=np.asarray([1.0, 1.0]))
    np.savez(cl, id=np.asarray([b"b", b"a"]), policy=cl_policy, value=np.asarray([-0.2, 0.0]), budget_agreement=np.asarray([1.0, 0.5]))
    out = tmp_path / "dataset.npz"
    manifest = builder.run(dict(builder.CONFIG, source_dataset=str(source), positions=str(positions),
                                zq_targets=str(zq), claustro_targets=str(cl), out=str(out)))
    with np.load(out) as result:
        assert result["id"].tolist() == [b"b", b"a"]
        assert np.isclose(result["policy"][0, :2].sum(), 1.0)
        assert result["weight"][0] > result["weight"][1]
        assert np.isclose(result["value"][0], 0.0)
    assert manifest["samples"] == 2
