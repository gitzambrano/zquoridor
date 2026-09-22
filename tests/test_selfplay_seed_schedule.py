import copy

import pytest

from tools.teacher.build_selfplay_seed_schedule import REQUIRED_FAMILIES, build_schedule, canonical_key


def _row(index, family=None, color=0, side=0):
    row = {
        "id": f"row-{index}",
        "canonical_key": f"state-{family or 'broad'}-{index}",
        "history": [f"e{index + 1}"],
        "physical_color": color,
        "side_to_move": side,
    }
    if family is not None:
        row["opening_category"] = family
    return row


def test_central_schedule_enforces_every_family_floor():
    rows = [
        _row(index, family, color=index % 2, side=(index // 2) % 2)
        for family in REQUIRED_FAMILIES
        for index in range(12)
    ]

    schedule = build_schedule(rows, central_target=50, family_floor=8, broad_target=0)

    assert all(schedule.central_counts[name] >= 8 for name in REQUIRED_FAMILIES)
    assert len(schedule.central_rows) == 50


def test_broad_schedule_excludes_central_keys():
    rows = [_row(index, "front_wall", color=index % 2, side=index % 2) for index in range(4)]
    rows.extend(_row(index + 10, None) for index in range(4))

    schedule = build_schedule(rows, central_target=2, family_floor=0, broad_target=20)

    assert set(schedule.central_keys).isdisjoint(schedule.broad_keys)


def test_schedule_is_deterministic_and_duplicates_do_not_add_rows():
    rows = [_row(index, "pawn_jump", color=index % 2, side=index % 2) for index in range(8)]
    duplicated = rows + [copy.deepcopy(rows[0]), copy.deepcopy(rows[3])]

    first = build_schedule(duplicated, central_target=6, family_floor=0, broad_target=2)
    second = build_schedule(rows, central_target=6, family_floor=0, broad_target=2)

    assert first.central_rows == second.central_rows
    assert first.broad_rows == second.broad_rows
    assert first.central_counts == second.central_counts
    assert len(first.central_keys) == len(set(first.central_keys))
    assert len(first.broad_keys) == len(set(first.broad_keys))


def test_canonical_key_replays_a_legal_jump():
    row = {
        "history": ["e2", "e8", "e3", "e7", "e4", "f7", "a1h", "f6", "e5", "f5", "g5"],
        "side_to_move": 1,
    }

    assert canonical_key(row) == "state:41,42,1,0,10,9,turn=1"


def test_canonical_key_replays_a_wall_blocked_diagonal():
    row = {
        "history": ["e2", "e8", "f5h", "e7", "e3", "f7", "e4", "f6", "e5", "f5", "f4"],
        "side_to_move": 1,
    }

    assert canonical_key(row) == "state:41,32,137438953472,0,10,9,turn=1"


def test_central_schedule_balances_assigned_color_and_side():
    rows = [
        {
            "canonical_key": f"missing-fields-{family}-{index}",
            "opening_category": family,
            "side_to_move": index % 2,
        }
        for family in REQUIRED_FAMILIES
        for index in range(8)
    ]

    schedule = build_schedule(rows, central_target=40, family_floor=8, broad_target=0)

    for family in REQUIRED_FAMILIES:
        selected = [row for row in schedule.central_rows if row["opening_category"] == family]
        assert {row["physical_color"] for row in selected} == {0, 1}
        assert {row["side_to_move"] for row in selected} == {0, 1}
        color_counts = {color: sum(row["physical_color"] == color for row in selected) for color in (0, 1)}
        side_counts = {side: sum(row["side_to_move"] == side for row in selected) for side in (0, 1)}
        assert max(color_counts.values()) - min(color_counts.values()) <= 1
        assert max(side_counts.values()) - min(side_counts.values()) <= 1


def test_central_schedule_rejects_a_family_with_one_physical_color():
    rows = [
        {
            "canonical_key": f"one-color-{family}-{index}",
            "opening_category": family,
            "physical_color": 0,
            "side_to_move": index % 2,
        }
        for family in REQUIRED_FAMILIES
        for index in range(8)
    ]

    with pytest.raises(ValueError, match="physical color"):
        build_schedule(rows, central_target=40, family_floor=8, broad_target=0)


def test_central_schedule_rejects_a_family_with_one_side_to_move():
    rows = [
        {
            "canonical_key": f"one-side-{family}-{index}",
            "opening_category": family,
            "physical_color": index % 2,
            "side_to_move": 0,
        }
        for family in REQUIRED_FAMILIES
        for index in range(8)
    ]

    with pytest.raises(ValueError, match="side"):
        build_schedule(rows, central_target=40, family_floor=8, broad_target=0)
