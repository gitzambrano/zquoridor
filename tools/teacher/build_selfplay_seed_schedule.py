#!/usr/bin/env python3
"""Build deterministic central and broad self-play seed schedules."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FAMILIES = (
    "front_wall",
    "pawn_jump",
    "vertical_channel",
    "reed_rear_wall",
    "sidestep_flank",
)
_FAMILY_SET = set(REQUIRED_FAMILIES)
_MOVE_RE = re.compile(r"^([a-i])([1-9])([hv]?)$", re.IGNORECASE)


@dataclass(frozen=True)
class Schedule:
    """Hold the selected rows and their canonical keys."""

    central_rows: tuple[dict, ...]
    broad_rows: tuple[dict, ...]
    central_counts: dict[str, int]
    central_keys: tuple[str, ...]
    broad_keys: tuple[str, ...]
    manifest: dict


@dataclass
class _ReplayState:
    pawns: list[int]
    walls_h: set[int]
    walls_v: set[int]
    walls_left: list[int]
    turn: int = 0


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _as_int(value: object, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_state_text(text: str) -> tuple[int, int, int, int, int, int] | None:
    fields = text.split()
    if len(fields) != 7 or fields[0] != "@state":
        return None
    values = [_as_int(item) for item in fields[1:]]
    if any(value is None for value in values):
        return None
    return tuple(values)  # type: ignore[return-value]


def _cell(row: int, col: int) -> int:
    return row * 9 + col


def _coords(cell: int) -> tuple[int, int]:
    return divmod(cell, 9)


def _in_bounds(row: int, col: int) -> bool:
    return 0 <= row < 9 and 0 <= col < 9


def _edge_blocked(state: _ReplayState, first: int, second: int) -> bool:
    row_a, col_a = _coords(first)
    row_b, col_b = _coords(second)
    if row_a == row_b:
        slot_col = min(col_a, col_b)
        return slot_col < 0 or slot_col >= 8 or (row_a * 8 + slot_col) in state.walls_h
    if col_a == col_b:
        slot_row = min(row_a, row_b)
        return slot_row < 0 or slot_row >= 8 or (slot_row * 8 + col_a) in state.walls_v
    return True


def _neighbors(state: _ReplayState, cell: int) -> list[int]:
    row, col = _coords(cell)
    result = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if _in_bounds(nr, nc):
            target = _cell(nr, nc)
            if not _edge_blocked(state, cell, target):
                result.append(target)
    return result


def _has_path(state: _ReplayState, pawn: int, player: int) -> bool:
    goal = 8 if player == 0 else 0
    queue = [pawn]
    seen = {pawn}
    while queue:
        cell = queue.pop(0)
        if _coords(cell)[0] == goal:
            return True
        for target in _neighbors(state, cell):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return False


def _can_place_wall(state: _ReplayState, orientation: str, row: int, col: int) -> bool:
    if row >= 8 or col >= 8 or state.walls_left[state.turn] <= 0:
        return False
    slot = row * 8 + col
    if orientation == "h":
        if slot in state.walls_h or slot in state.walls_v:
            return False
        if col > 0 and slot - 1 in state.walls_h:
            return False
        if col < 7 and slot + 1 in state.walls_h:
            return False
    else:
        if slot in state.walls_v or slot in state.walls_h:
            return False
        if row > 0 and slot - 8 in state.walls_v:
            return False
        if row < 7 and slot + 8 in state.walls_v:
            return False
    if orientation == "h":
        state.walls_h.add(slot)
    else:
        state.walls_v.add(slot)
    valid = _has_path(state, state.pawns[0], 0) and _has_path(state, state.pawns[1], 1)
    if orientation == "h":
        state.walls_h.remove(slot)
    else:
        state.walls_v.remove(slot)
    return valid


def _apply_move(state: _ReplayState, text: str) -> bool:
    match = _MOVE_RE.fullmatch(text.strip())
    if match is None:
        return False
    col = ord(match.group(1).lower()) - ord("a")
    row = int(match.group(2)) - 1
    orientation = match.group(3).lower()
    if orientation:
        if not _can_place_wall(state, orientation, row, col):
            return False
        slot = row * 8 + col
        (state.walls_h if orientation == "h" else state.walls_v).add(slot)
        state.walls_left[state.turn] -= 1
        state.turn = 1 - state.turn
        return True

    target = _cell(row, col)
    mover = state.turn
    opponent = 1 - mover
    current = state.pawns[mover]
    if target not in state.pawns and target in _neighbors(state, current):
        state.pawns[mover] = target
        state.turn = opponent
        return True

    opponent_cell = state.pawns[opponent]
    row0, col0 = _coords(current)
    row1, col1 = _coords(opponent_cell)
    if abs(row0 - row1) + abs(col0 - col1) != 1:
        return False
    if _edge_blocked(state, current, opponent_cell):
        return False
    dr, dc = row1 - row0, col1 - col0
    jump_row, jump_col = row1 + dr, col1 + dc
    jump_open = False
    if _in_bounds(jump_row, jump_col):
        jump = _cell(jump_row, jump_col)
        jump_open = not _edge_blocked(state, opponent_cell, jump)
        if jump_open:
            if target != jump:
                return False
            state.pawns[mover] = jump
            state.turn = opponent
            return True

    # A diagonal move is legal only when a direct jump is blocked or out of bounds.
    if jump_open:
        return False
    for side_dr, side_dc in ((dc, -dr), (-dc, dr)):
        diag_row, diag_col = row1 + side_dr, col1 + side_dc
        if _in_bounds(diag_row, diag_col):
            diagonal = _cell(diag_row, diag_col)
            if target == diagonal and not _edge_blocked(state, opponent_cell, diagonal):
                state.pawns[mover] = diagonal
                state.turn = opponent
                return True
    return False


def _replay_history(history: Sequence[object]) -> _ReplayState | None:
    state = _ReplayState([_cell(0, 4), _cell(8, 4)], set(), set(), [10, 10])
    for item in history:
        if not isinstance(item, str) or not _apply_move(state, item):
            return None
    return state


def _state_key(row: Mapping[str, object]) -> str:
    explicit = row.get("canonical_key", row.get("state_key"))
    if explicit is not None:
        if isinstance(explicit, bytes):
            return "bytes:" + explicit.hex()
        return explicit if isinstance(explicit, str) else _stable_json(explicit)
    history = row.get("history", ())
    if isinstance(history, str):
        history = history.split()
    if isinstance(history, Sequence) and history:
        first = history[0]
        if isinstance(first, str) and first.startswith("@state"):
            state_text = first
            if first == "@state" and len(history) >= 7:
                state_text = " ".join(str(item) for item in history[:7])
            values = _parse_state_text(state_text)
            if values is not None:
                own, opp, walls_h, walls_v, own_walls, opp_walls = values
                side = _as_int(row.get("side_to_move"), 0) or 0
                return f"state:{own},{opp},{walls_h},{walls_v},{own_walls},{opp_walls},turn={side}"
        replay = _replay_history(history)
        if replay is not None:
            side = _as_int(row.get("side_to_move"), replay.turn)
            side = replay.turn if side is None else side
            other = 1 - side
            return (
                f"state:{replay.pawns[side]},{replay.pawns[other]},"
                f"{sum(1 << slot for slot in replay.walls_h)},"
                f"{sum(1 << slot for slot in replay.walls_v)},"
                f"{replay.walls_left[side]},{replay.walls_left[other]},turn={side}"
            )
    state = row.get("state")
    if isinstance(state, Mapping):
        row = {**row, **state}
    aliases = {
        "own_pawn": ("own_pawn", "pawn_own"),
        "opp_pawn": ("opp_pawn", "pawn_opp"),
        "walls_h": ("walls_h", "wallsH"),
        "walls_v": ("walls_v", "wallsV"),
        "own_walls": ("walls_left_own", "own_walls", "ownWalls"),
        "opp_walls": ("walls_left_opp", "opp_walls", "oppWalls"),
    }
    values = []
    for name, names in aliases.items():
        value = next((row.get(alias) for alias in names if row.get(alias) is not None), None)
        if value is None:
            break
        values.append(_as_int(value, 0))
    if len(values) == len(aliases):
        side = _as_int(row.get("side_to_move"), 0) or 0
        return "state:" + ",".join(str(value) for value in values) + f",turn={side}"
    payload = {"history": list(history) if isinstance(history, Sequence) else history,
               "side_to_move": row.get("side_to_move")}
    return "history:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def canonical_key(row: Mapping[str, object]) -> str:
    """Return the exact deduplication key for one position row."""
    return _state_key(row)


def _family(row: Mapping[str, object]) -> str | None:
    values = [row.get("opening_category"), row.get("category")]
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        values.extend((metadata.get("opening_category"), metadata.get("category")))
    tags = row.get("tags")
    if isinstance(tags, Sequence) and not isinstance(tags, str):
        values.extend(tags)
    for value in values:
        if isinstance(value, str) and value in _FAMILY_SET:
            return value
    return None


def _balance_value(row: Mapping[str, object], name: str) -> int:
    aliases = {
        "color": ("physical_color", "physical_player", "color", "mover", "zq_player"),
        "side": ("side_to_move", "turn", "side"),
    }
    for alias in aliases[name]:
        value = _as_int(row.get(alias))
        if value is not None:
            return value & 1
    return 0


def _has_balance_field(row: Mapping[str, object], name: str) -> bool:
    aliases = {
        "color": ("physical_color", "physical_player", "color", "mover", "zq_player"),
        "side": ("side_to_move", "turn", "side"),
    }
    return any(row.get(alias) is not None for alias in aliases[name])


def _infer_side(row: Mapping[str, object], key: str) -> int | None:
    for alias in ("side_to_move", "turn", "side"):
        value = _as_int(row.get(alias))
        if value is not None:
            return value & 1
    ply = _as_int(row.get("ply"))
    if ply is not None:
        return ply & 1
    match = re.search(r"turn=([01])", key)
    if match:
        return int(match.group(1))
    history = row.get("history", ())
    if isinstance(history, str):
        history = history.split()
    if isinstance(history, Sequence):
        replay = _replay_history(history)
        if replay is not None:
            return replay.turn
    return None


def _assign_missing_balance_fields(unique: Mapping[str, dict]) -> None:
    """Assign deterministic values when a source omits balance metadata."""
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, row in unique.items():
        groups[_family(row) or "__broad__"].append((key, row))
    for group in groups.values():
        group.sort(key=lambda item: item[0])
        for index, (key, row) in enumerate(group):
            if not _has_balance_field(row, "color"):
                row["physical_color"] = index & 1
            if not _has_balance_field(row, "side"):
                inferred = _infer_side(row, key)
                row["side_to_move"] = index & 1 if inferred is None else inferred


def _require_family_balance(rows: Sequence[dict]) -> None:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        family = _family(row)
        if family is not None:
            by_family[family].append(row)
    for family in REQUIRED_FAMILIES:
        selected = by_family[family]
        if not selected:
            continue
        for field in ("color", "side"):
            counts = Counter(_balance_value(row, field) for row in selected)
            if counts[0] == 0 or counts[1] == 0:
                label = "physical color" if field == "color" else "side to move"
                raise ValueError(f"central family {family!r} lacks a {label} bucket")
            if max(counts.values()) - min(counts.values()) > 1:
                raise ValueError(f"central family {family!r} is not balanced for {field}")


def _deduplicate(rows: Iterable[Mapping[str, object]]) -> dict[str, dict]:
    unique: dict[str, dict] = {}
    for original in rows:
        row = copy.deepcopy(dict(original))
        key = canonical_key(row)
        row["canonical_key"] = key
        existing = unique.get(key)
        if existing is None:
            unique[key] = row
            continue
        old_score = (_family(existing) is not None, str(existing.get("source", "")), _stable_json(existing))
        new_score = (_family(row) is not None, str(row.get("source", "")), _stable_json(row))
        if new_score > old_score:
            unique[key] = row
    return unique


def _ordered_candidates(candidates: Sequence[dict], color_counts: Counter, side_counts: Counter) -> list[dict]:
    return sorted(
        candidates,
        key=lambda row: (
            color_counts[_balance_value(row, "color")],
            side_counts[_balance_value(row, "side")],
            _balance_value(row, "color"),
            _balance_value(row, "side"),
            str(row["canonical_key"]),
        ),
    )


def build_schedule(
    rows: Iterable[Mapping[str, object]],
    central_target: int = 2_500_000,
    family_floor: int = 400_000,
    broad_target: int | None = 1_500_000,
    *,
    source_manifest: Sequence[str] = (),
) -> Schedule:
    """Deduplicate rows, then select balanced central and broad schedules."""
    if central_target < 0 or family_floor < 0 or (broad_target is not None and broad_target < 0):
        raise ValueError("schedule targets must be non-negative")
    if family_floor * len(REQUIRED_FAMILIES) > central_target:
        raise ValueError("family floors exceed the central target")
    source_rows = list(rows)
    unique = _deduplicate(source_rows)
    _assign_missing_balance_fields(unique)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in unique.values():
        family = _family(row)
        if family is not None:
            by_family[family].append(row)
    if any(len(by_family[name]) < family_floor for name in REQUIRED_FAMILIES):
        missing = {name: len(by_family[name]) for name in REQUIRED_FAMILIES if len(by_family[name]) < family_floor}
        raise ValueError(f"central family floor cannot be met: {missing}")

    selected: list[dict] = []
    selected_keys: set[str] = set()
    central_counts = {name: 0 for name in REQUIRED_FAMILIES}
    family_color_counts: dict[str, Counter] = {name: Counter() for name in REQUIRED_FAMILIES}
    family_side_counts: dict[str, Counter] = {name: Counter() for name in REQUIRED_FAMILIES}

    def take_from_family(name: str) -> bool:
        candidates = [row for row in by_family[name] if row["canonical_key"] not in selected_keys]
        if not candidates:
            return False
        choice = _ordered_candidates(
            candidates,
            family_color_counts[name],
            family_side_counts[name],
        )[0]
        choice = copy.deepcopy(choice)
        choice["opening_category"] = name
        selected.append(choice)
        selected_keys.add(choice["canonical_key"])
        central_counts[name] += 1
        family_color_counts[name][_balance_value(choice, "color")] += 1
        family_side_counts[name][_balance_value(choice, "side")] += 1
        return True

    for name in REQUIRED_FAMILIES:
        for _ in range(family_floor):
            if not take_from_family(name):
                raise ValueError(f"central family {name!r} has too few unique rows")
    while len(selected) < central_target:
        available = [name for name in REQUIRED_FAMILIES if any(row["canonical_key"] not in selected_keys for row in by_family[name])]
        if not available:
            break
        name = min(
            available,
            key=lambda item: (
                central_counts[item] / max(1, len(by_family[item])),
                central_counts[item],
                item,
            ),
        )
        take_from_family(name)
    if len(selected) < central_target:
        raise ValueError(f"only {len(selected)} unique central rows are available, target is {central_target}")
    _require_family_balance(selected)

    remaining = [row for key, row in unique.items() if key not in selected_keys]
    remaining.sort(key=lambda row: str(row["canonical_key"]))
    if broad_target is not None:
        remaining = remaining[:broad_target]
    broad = [copy.deepcopy(row) for row in remaining]
    manifest = {
        "schema": "zquoridor.selfplay_seed_schedule.v1",
        "central_target": central_target,
        "central_count": len(selected),
        "central_counts": central_counts,
        "family_floor": family_floor,
        "broad_target": broad_target,
        "broad_count": len(broad),
        "unique_input_rows": len(unique),
        "duplicate_input_rows": max(0, len(source_rows) - len(unique)),
        "source_manifest": list(source_manifest),
        "required_families": list(REQUIRED_FAMILIES),
    }
    return Schedule(tuple(selected), tuple(broad), central_counts, tuple(row["canonical_key"] for row in selected), tuple(row["canonical_key"] for row in broad), manifest)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _source_name(path: Path) -> str:
    """Return a stable source name for a repository or external path."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _opening_rows(path: Path, broad: bool = False) -> list[dict]:
    rows = []
    for opening_number, opening in enumerate(_read_jsonl(path)):
        moves = opening.get("moves", [])
        if not isinstance(moves, list) or not moves:
            continue
        category = opening.get("category")
        color = next(
            (opening.get(name) for name in ("physical_color", "physical_player", "color", "zq_player")
             if opening.get(name) is not None),
            None,
        )
        if color is None:
            color = (_as_int(opening.get("opening_index"), opening_number) or opening_number) & 1
        start = 1 if broad else 4
        for ply in range(start, len(moves) + 1):
            rows.append({
                "schema": "zquoridor.position.v1",
                "id": f"{path.name}:{opening.get('opening_index', 0)}:{ply}",
                "history": moves[:ply],
                "side_to_move": ply & 1,
                "physical_color": _as_int(color, opening_number & 1) & 1,
                "opening_index": opening.get("opening_index", -1),
                "opening_category": category,
                "source": _source_name(path),
            })
    return rows


def _benchmark_rows(path: Path, categories: Mapping[int, str]) -> list[dict]:
    rows = []
    for game in _read_jsonl(path):
        moves = game.get("moves", [])
        if not isinstance(moves, list) or len(moves) < 2:
            continue
        result = _as_int(game.get("result"), 1)
        zq_player = _as_int(game.get("zq_player"), 0)
        is_loss = result is not None and result <= 0 and zq_player is not None
        for ply in range(1, len(moves)):
            if not is_loss and ply > 12:
                continue
            rows.append({
                "schema": "zquoridor.position.v1",
                "id": f"{path}:{game.get('opening_index', -1)}:{ply}",
                "history": moves[:ply],
                "side_to_move": ply & 1,
                "physical_color": zq_player,
                "opening_index": game.get("opening_index", -1),
                "opening_category": categories.get(_as_int(game.get("opening_index"), -1)),
                "source": f"benchmark-loss:{_source_name(path)}" if is_loss else f"benchmark:{_source_name(path)}",
            })
    return rows


def discover_source_rows(source_paths: Sequence[Path] | None = None) -> tuple[list[dict], list[str]]:
    """Read benchmark, opening, and generic position sources."""
    center_paths = [ROOT / "tools/external/openings_center_rush_v1.jsonl", ROOT / "tools/external/openings_center_rush_50pairs.jsonl"]
    rows: list[dict] = []
    used: list[str] = []
    if source_paths:
        for path in source_paths:
            if path.exists():
                rows.extend(_read_jsonl(path))
                used.append(str(path.resolve()))
        return rows, used
    categories: dict[int, str] = {}
    for path in center_paths:
        if not path.exists():
            continue
        for opening in _read_jsonl(path):
            index = _as_int(opening.get("opening_index"), -1)
            if index is not None and isinstance(opening.get("category"), str):
                categories[index] = opening["category"]
        rows.extend(_opening_rows(path))
        used.append(str(path.resolve()))
    broad_opening_paths = [path for path in sorted((ROOT / "tools/external").glob("openings_*.jsonl")) if path not in center_paths]
    for path in broad_opening_paths:
        if path.exists():
            rows.extend(_opening_rows(path, broad=True))
            used.append(str(path.resolve()))
    benchmark_root = ROOT / "benchmark_results"
    for path in sorted(benchmark_root.glob("**/games.jsonl")):
        rows.extend(_benchmark_rows(path, categories))
        used.append(str(path.resolve()))
    for path in sorted((ROOT / "data/teaching").glob("**/*positions*.jsonl")):
        if "trajectories" in path.parts:
            continue
        rows.extend(_read_jsonl(path))
        used.append(str(path.resolve()))
    return rows, used


def write_schedule(schedule: Schedule, out_dir: Path) -> dict:
    """Write both schedules and the manifest to an output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("central_positions.jsonl", schedule.central_rows), ("broad_positions.jsonl", schedule.broad_rows)):
        with (out_dir / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(_stable_json(row) + "\n")
    manifest = dict(schedule.manifest)
    manifest.update({
        "central_path": str((out_dir / "central_positions.jsonl").resolve()),
        "broad_path": str((out_dir / "broad_positions.jsonl").resolve()),
        "central_keys_sha256": hashlib.sha256("\n".join(schedule.central_keys).encode()).hexdigest(),
        "broad_keys_sha256": hashlib.sha256("\n".join(schedule.broad_keys).encode()).hexdigest(),
    })
    (out_dir / "manifest.json").write_text(_stable_json(manifest) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output directory for both schedules.")
    parser.add_argument("--source", type=Path, action="append", help="JSONL source. Repeat this option.")
    parser.add_argument("--central-target", type=int, default=2_500_000)
    parser.add_argument("--family-floor", type=int, default=400_000)
    parser.add_argument("--broad-target", type=int, default=1_500_000)
    args = parser.parse_args(argv)
    rows, sources = discover_source_rows(args.source)
    schedule = build_schedule(rows, args.central_target, args.family_floor, args.broad_target, source_manifest=sources)
    manifest = write_schedule(schedule, args.out.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
