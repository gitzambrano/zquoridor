"""Check independent evaluation and data isolation in the local campaign."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import run_campaign as campaign
import prepare_replay as replay


def dataset(path, pawns, val, values=None):
    n = len(pawns)
    data = dict(own_pawn=np.asarray(pawns), opp_pawn=np.full(n, 76),
                walls_h=np.zeros(n, dtype=np.uint64), walls_v=np.zeros(n, dtype=np.uint64),
                walls_left_own=np.full(n, 10), walls_left_opp=np.full(n, 10),
                own_dist=np.full(n, 8), opp_dist=np.full(n, 8),
                policy=np.full((n, 209), 1/209, dtype=np.float32),
                value=np.asarray(values if values is not None else [0.0]*n),
                weight=np.ones(n), is_val=np.asarray(val),
                group_id=np.asarray([f"group{i}" for i in range(n)]))
    np.savez(path, **data)
    return data


def test_merge_excludes_holdouts_and_cross_split_states(tmp_path):
    old, fresh, out = [tmp_path / name for name in ("old.npz", "fresh.npz", "out.npz")]
    d = dataset(old, [4, 13, 22, 31, 40, 67], [False, False, True, False, True, False])
    dataset(fresh, [4, 13, 49, 58], [True, False, False, True], [0, .7, 0, 0])
    blocked = {tuple(int(d[k][3]) for k in replay.STATE_FIELDS)}
    report = campaign.combine_datasets(old, fresh, out, .2, blocked)
    with np.load(out) as result:
        assert set(result["own_pawn"]) == {13, 22, 40, 49, 58, 67}
        assert result["value"][result["own_pawn"] == 13].item() == pytest.approx(.7)
        assert not (set(result["group_id"][result["is_val"]]) & set(result["group_id"][~result["is_val"]]))
    assert report["samples"] == 6


def test_replay_rejects_underfilled_corpus(tmp_path, monkeypatch):
    from read_selfplay import SAMPLE_DTYPE_V2
    for index in range(2):
        rows = np.zeros(3, dtype=SAMPLE_DTYPE_V2)
        rows["own_pawn"] = 4 + index
        rows["opp_pawn"] = 76
        rows.tofile(tmp_path / f"{index}.bin")
    monkeypatch.setattr(replay, "_detect_format", lambda p: (SAMPLE_DTYPE_V2, 32))
    config = dict(replay.CONFIG, source=str(tmp_path), max_positions=8)
    with pytest.raises(ValueError, match="requested 8"):
        replay.sample_states(config)


def test_confirmation_guard_rejects_tiny_perfect_result():
    rows = [dict(opponent="baseline", opening_index=0, zq_player=side, status="ok", result=1.0) for side in (0, 1)]
    result = campaign.local_arena.summarize_pairs(rows, bootstrap=1000, seed=1)
    assert not result["strength_claim_ready"]
    assert result["decision_95"]["score_low_pct"] < 50


def test_campaign_rejects_failed_external_games():
    report = dict(summaries=dict(titanium=dict(failed_games=1, complete_pairs=0)))
    with pytest.raises(RuntimeError, match="incomplete"):
        campaign.check_external_report(report, 100)


def test_fresh_fraction_uses_loss_weight_in_each_split(tmp_path):
    old, fresh, out = [tmp_path / name for name in ("old.npz", "fresh.npz", "out.npz")]
    dataset(old, [4, 13], [False, True])
    data = dataset(fresh, [22, 31], [False, True])
    data["weight"] = np.asarray([.1, .8])
    np.savez(fresh, **data)
    campaign.combine_datasets(old, fresh, out, .2)
    with np.load(out) as combined:
        for val in (False, True):
            mask = combined["is_val"] == val
            new = mask & (combined["own_pawn"] >= 22)
            assert combined["weight"][new].sum() / combined["weight"][mask].sum() == pytest.approx(.2)
