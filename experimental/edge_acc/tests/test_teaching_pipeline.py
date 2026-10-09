import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "teachers"))
sys.path.insert(0, str(ROOT / "training"))

import teaching_pipeline as pipeline  # noqa: E402
import value_targets as values  # noqa: E402


def position(sample_id, history, group, split=None, metadata=None):
    row = {
        "schema": "zquoridor.position.v1",
        "id": sample_id,
        "history": history,
        "side_to_move": len(history) & 1,
        "opening_index": group,
        "ply": len(history),
        "metadata": metadata or {},
    }
    if split is not None:
        row["split"] = split
    return row


def source(ids, policies=None, source_values=None, confidence=None):
    result = {"id": np.asarray(ids, dtype="S24")}
    if policies is not None:
        result["policy"] = np.asarray(policies, dtype=np.float32)
    if source_values is not None:
        result["value"] = np.asarray(source_values, dtype=np.float32)
    if confidence is not None:
        result["confidence"] = np.asarray(confidence, dtype=np.float32)
    return result


def one_hot(index):
    out = np.zeros(209, dtype=np.float32)
    out[index] = 1.0
    return out


def test_game_result_is_signed_for_the_mover_before_temporal_discount():
    assert values.signed_game_outcome(1.0, side_to_move=0) == 1.0
    assert values.signed_game_outcome(1.0, side_to_move=1) == -1.0
    assert values.discount_game_outcome(
        1.0, side_to_move=1, remaining_plies=2, gamma=0.5
    ) == -0.25


@pytest.mark.parametrize("gamma", [-0.1, 1.01, math.nan, math.inf])
def test_gamma_must_be_finite_probability(gamma):
    with pytest.raises(ValueError, match="gamma"):
        values.validate_gamma(gamma)


def test_tempering_keeps_policy_normalized_and_value_signed():
    policy = values.temper_policy([0.8, 0.2], temperature=2.0)
    assert policy == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert values.temper_value(0.8, temperature=2.0) == pytest.approx(0.5)
    assert -1.0 <= values.temper_value(-1.0, temperature=0.5) <= 1.0


def test_value_blend_uses_independent_validated_weights():
    actual = values.blend_value_targets(
        teacher_values=[0.8, -0.2],
        teacher_weights=[3.0, 1.0],
        outcome_value=-0.5,
        outcome_weight=2.0,
        bootstrap_value=0.25,
        bootstrap_weight=2.0,
    )
    assert actual == pytest.approx(0.2125)


@pytest.mark.parametrize(
    "teacher_values,teacher_weights",
    [([math.nan], [1.0]), ([0.0], [math.nan]), ([0.0], [-1.0]), ([0.0], [0.0])],
)
def test_value_blend_rejects_invalid_values_and_weights(teacher_values, teacher_weights):
    with pytest.raises(ValueError):
        values.blend_value_targets(teacher_values, teacher_weights)


def test_outcome_request_requires_real_game_metadata():
    rows = [position("a", [], 7), position("b", ["e2"], 7)]
    with pytest.raises(ValueError, match="game_outcome"):
        pipeline.outcome_targets(rows, gamma=0.9)

    rows[0]["metadata"] = {"game_outcome": 1.0, "remaining_plies": 2}
    rows[1]["metadata"] = {"game_outcome": 1.0, "remaining_plies": 1}
    assert pipeline.outcome_targets(rows, gamma=0.5).tolist() == pytest.approx([0.25, -0.5])


def test_position_loader_rejects_legacy_rows_without_exact_history(tmp_path):
    path = tmp_path / "positions.jsonl"
    path.write_text(json.dumps({"schema": "legacy.state.v1", "id": "a"}) + "\n")
    with pytest.raises(ValueError, match="schema"):
        pipeline.load_positions(path)


def test_group_split_never_leaks_and_rejects_conflicting_declared_splits():
    rows = [
        position("a", [], 10),
        position("b", ["e2"], 10),
        position("c", ["e3"], 11),
        position("d", ["e4"], 12),
    ]
    assigned = pipeline.assign_group_splits(rows, val_fraction=0.34, seed=8)
    assert assigned[0]["split"] == assigned[1]["split"]
    assert {row["split"] for row in assigned} == {"train", "val"}

    conflict = [position("a", [], 3, "train"), position("b", ["e2"], 3, "val")]
    with pytest.raises(ValueError, match="group.*split"):
        pipeline.assign_group_splits(conflict, val_fraction=0.2, seed=1)


def test_position_cap_samples_across_game_groups():
    rows = []
    for game in range(4):
        for ply in range(10):
            rows.append(position(f"{game}-{ply}", [f"m{n}" for n in range(ply)], game))
    sampled = pipeline.sample_balanced_groups(rows, limit=8, seed=4)
    counts = {}
    for row in sampled:
        counts[row["opening_index"]] = counts.get(row["opening_index"], 0) + 1
    assert set(counts) == {0, 1, 2, 3}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_source_join_aligns_ids_and_uses_separate_head_weights():
    rows = [position("a", [], 1), position("b", ["e2"], 2)]
    first = source(["a", "b"], [one_hot(0), one_hot(1)], [0.8, 0.4])
    second = source(["b", "a"], [one_hot(2), one_hot(3)], [-0.2, 0.0])
    joined = pipeline.join_sources(
        rows,
        {"old": first, "claustro": second},
        policy_weights={"old": 1.0, "claustro": 3.0},
        value_weights={"old": 3.0, "claustro": 1.0},
    )
    assert joined["policy"][0, [0, 3]].tolist() == pytest.approx([0.25, 0.75])
    assert joined["policy"][1, [1, 2]].tolist() == pytest.approx([0.25, 0.75])
    assert joined["value"].tolist() == pytest.approx([0.6, 0.25])


def test_source_join_rejects_duplicate_ids_and_nonfinite_head_weights():
    rows = [position("a", [], 1), position("b", ["e2"], 2)]
    duplicate = source(["a", "a"], [one_hot(0), one_hot(1)], [0.0, 0.0])
    with pytest.raises(ValueError, match="duplicate"):
        pipeline.join_sources(rows, {"old": duplicate}, {"old": 1.0}, {"old": 1.0})

    valid = source(["a", "b"], [one_hot(0), one_hot(1)], [0.0, 0.0])
    with pytest.raises(ValueError, match="weight"):
        pipeline.join_sources(rows, {"old": valid}, {"old": math.nan}, {"old": 1.0})


def test_cache_requires_matching_identity_and_complete_chunks(tmp_path):
    identity = pipeline.CacheIdentity(
        input_sha256="input",
        source="old-direct",
        checkpoint_sha256="weights",
        params_sha256="params",
        schema=pipeline.CACHE_SCHEMA,
    )
    cache = tmp_path / "old.npz"
    pipeline.write_cache(
        cache,
        {"id": np.asarray([b"a", b"b"], dtype="S24"), "value": np.asarray([0.1, 0.2])},
        identity,
        completed_chunks=[0, 1],
        total_chunks=2,
    )
    assert pipeline.cache_valid(cache, identity, expected_ids=["a", "b"])
    assert not pipeline.cache_valid(cache, identity, expected_ids=["a", "c"])
    changed = pipeline.CacheIdentity(**(identity.__dict__ | {"params_sha256": "changed"}))
    assert not pipeline.cache_valid(cache, changed, expected_ids=["a", "b"])

    manifest_path = Path(str(cache) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["completed_chunks"] = [0]
    manifest_path.write_text(json.dumps(manifest))
    assert not pipeline.cache_valid(cache, identity, expected_ids=["a", "b"])


def test_source_cache_identity_changes_when_relabel_bridge_changes(tmp_path):
    positions = tmp_path / "positions.jsonl"
    checkpoint = tmp_path / "weights.bin"
    bridge = tmp_path / "bridge.exe"
    positions.write_text("position")
    checkpoint.write_bytes(b"weights")
    bridge.write_bytes(b"bridge-v1")
    first = pipeline.source_identity(
        positions, "teacher", checkpoint, {"temperature": 1.0}, tools=[bridge]
    )
    bridge.write_bytes(b"bridge-v2")
    second = pipeline.source_identity(
        positions, "teacher", checkpoint, {"temperature": 1.0}, tools=[bridge]
    )
    assert first.tool_sha256 != second.tool_sha256
    assert pipeline.cache_key(first) != pipeline.cache_key(second)


def test_cli_defaults_are_finite_and_each_core_setting_can_be_overridden(tmp_path):
    defaults = pipeline.parse_args([])
    assert defaults.mode == "direct"
    assert defaults.games == 24
    assert defaults.max_positions == 512
    overridden = pipeline.parse_args([
        "--mode", "mixed", "--positions", str(tmp_path / "p.jsonl"),
        "--games", "40", "--max-positions", "900", "--gamma", "0.95",
        "--policy-temperature", "1.2", "--value-temperature", "0.8",
        "--chunk-size", "17",
    ])
    assert overridden.mode == "mixed"
    assert overridden.positions == tmp_path / "p.jsonl"
    assert (overridden.games, overridden.max_positions, overridden.chunk_size) == (40, 900, 17)
    assert (overridden.gamma, overridden.policy_temperature, overridden.value_temperature) == (
        0.95, 1.2, 0.8
    )
