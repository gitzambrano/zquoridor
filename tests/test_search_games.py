from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from tools import run_search_games as search_games
from tools.external.local_arena import Referee
from training.teachers.targets import (
    POLICY_DIM,
    mirror_action_lr,
    sample_id,
    zq_move_to_policy_index,
)


def _root_response(referee: Referee, move: str, *, stop_reason: str = "deadline",
                   root_visits: int = 80, visit_sum: int = 80,
                   extra_visit: tuple[str, int] | None = None) -> dict:
    side = referee.side_to_move
    best = zq_move_to_policy_index(move, side)
    if side == 1:
        best = mirror_action_lr(best)
    visits = [0] * POLICY_DIM
    visits[best] = visit_sum
    if extra_visit is not None:
        illegal, count = extra_visit
        index = zq_move_to_policy_index(illegal, side)
        if side == 1:
            index = mirror_action_lr(index)
        visits[index] = count
    return {
        "protocol": "deadline-and-engine-clock-v2",
        "mode": "movetime",
        "policy_frame": "claustrophobia-canonical-209",
        "side_to_move": side,
        "root_value_perspective": "side-to-move",
        "bestmove": move,
        "best_action": best,
        "move_time_ms": 400,
        "sims": root_visits,
        "root_visits": root_visits,
        "root_value": 0.25,
        "stop_reason": stop_reason,
        "visit_counts": visits,
    }


def _info_for(referee: Referee, move: str, **kwargs) -> list[str]:
    return [json.dumps(_root_response(referee, move, **kwargs))]


def test_default_opening_relative_schedule_tapers_from_400_to_50_ms():
    config = search_games.resolve_config(search_games.build_parser().parse_args([]))

    assert config["schedule_origin"] == "opening"
    assert config["start_move_time_ms"] == 400
    assert config["end_move_time_ms"] == 50
    assert config["decay_start_ply"] == 14
    assert config["decay_end_ply"] == 80
    assert search_games.move_time_ms(0, config) == 400
    assert search_games.move_time_ms(14, config) == 400
    assert search_games.move_time_ms(47, config) == 225
    assert search_games.move_time_ms(80, config) == 50
    assert search_games.move_time_ms(100, config) == 50


@pytest.mark.parametrize("history,move,side", [([], "e2", 0), (["e2"], "e8", 1), (["e2"], "f9", 1)])
def test_parse_root_maps_policy_and_bestmove_for_both_sides(history, move, side):
    referee = Referee()
    for token in history:
        referee.apply(token)
    assert referee.side_to_move == side

    parsed = search_games.parse_root(_info_for(referee, move), referee, move, 400)
    canonical = zq_move_to_policy_index(move, side)

    assert parsed["target_status"] == "ok"
    assert parsed["root_value"] == 0.25
    assert sum(parsed["visit_counts_global"]) == 80
    assert parsed["visit_counts_global"][search_games.global_action(move)] == 80
    assert parsed["policy"][canonical] == 1.0


def test_parse_root_rejects_visits_on_an_illegal_move():
    referee = Referee()
    info = _info_for(referee, "e2", extra_visit=("a1", 1))

    with pytest.raises(search_games.local_arena.EngineError, match="illegal action"):
        search_games.parse_root(info, referee, "e2", 400)


def test_solved_search_accepts_partial_root_policy_but_excludes_it_as_target():
    referee = Referee()
    info = _info_for(referee, "e2", stop_reason="solved", root_visits=80, visit_sum=79)

    parsed = search_games.parse_root(info, referee, "e2", 400)

    assert parsed["root_value"] == 0.25
    assert sum(parsed["visit_counts_global"]) == 79
    assert parsed["target_status"] == "excluded_solver_policy"


def test_parse_root_rejects_inconsistent_deadline_visit_total():
    referee = Referee()
    info = _info_for(referee, "e2", stop_reason="deadline", root_visits=80, visit_sum=79)

    with pytest.raises(search_games.local_arena.EngineError, match="inconsistent visit counts"):
        search_games.parse_root(info, referee, "e2", 400)


def test_opening_selection_is_deterministic_balanced_and_keeps_split_by_history(tmp_path):
    rows = [
        {"moves": [], "category": "empty"},
        {"moves": ["e2"], "category": "one_move"},
    ]
    books = {}
    for name in ("normal", "center_rush"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        books[name] = str(path)
    config = {"opening_books": books, "seed": 91, "pairs": 12, "max_plies": 180,
              "validation_fraction": 0.5}

    selected = search_games.select_openings(config)
    repeated = search_games.select_openings(config)

    assert selected == repeated
    counts = Counter(row["book"] for row in selected)
    assert counts == {"normal": 6, "center_rush": 6}
    split_by_history = defaultdict(set)
    for row in selected:
        split_by_history[tuple(row["opening"])].add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_history.values())
    paired_tasks = [{**row, "zq_player": color} for row in selected for color in (0, 1)]
    assert all(paired_tasks[index]["split"] == paired_tasks[index + 1]["split"]
               for index in range(0, len(paired_tasks), 2))


class _ScriptedPlayer:
    def __init__(self, player_name: str):
        self.player_name = player_name

    def bestmove(self, history, *, budget, timeout_s):
        if self.player_name == "zquoridor":
            move = ["e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"][len(history) // 2]
            return move, 0.001, []

        move = ["e8", "d8", "d7", "d6", "d5", "d4", "d3"][(len(history) - 1) // 2]
        referee = Referee()
        for token in history:
            referee.apply(token)
        return move, 0.001, _info_for(referee, move)

    def close(self):
        pass


def _complete_fake_game() -> dict:
    task = {"pair_index": 0, "zq_player": 0, "opening": [], "book": "normal",
            "category": "normal", "opening_group": sample_id([]), "split": "train"}
    config = {"mode": "match", "schedule_origin": "opening", "start_move_time_ms": 400,
              "end_move_time_ms": 50, "decay_start_ply": 14, "decay_end_ply": 80,
              "max_plies": 180, "move_timeout_s": 5.0}
    factories = {name: (lambda name=name: _ScriptedPlayer(name))
                 for name in ("zquoridor", "claustrophobia")}
    return search_games.play_game(task, config, "fake-run", factories)


def test_complete_fake_game_exports_canonical_targets_and_excludes_truncated_games(tmp_path):
    finished = _complete_fake_game()
    assert finished["status"] == "ok"
    assert finished["termination"] == "goal"
    assert finished["winner"] == 0
    assert len(finished["searches"]) == 15

    games_dir = tmp_path / "games"
    games_dir.mkdir()
    search_games.atomic_json(games_dir / "0000000_0.json", finished)
    for pair_index, status in ((1, "truncated"), (2, "interrupted")):
        excluded = {**finished, "pair_index": pair_index, "status": status}
        search_games.atomic_json(games_dir / f"{pair_index:07d}_0.json", excluded)

    summary = search_games.export_run(tmp_path, {"run_id": "fake-run"}, {"export_targets": True})

    assert summary["statuses"] == {"interrupted": 1, "ok": 1, "truncated": 1}
    assert summary["training_positions"] == 7
    rows = [json.loads(line) for line in (tmp_path / "positions.jsonl").read_text().splitlines()]
    assert len(rows) == 7
    assert all(row["side_to_move"] == 1 for row in rows)
    assert all(row["value"] == 0.25 and row["outcome"] == -1.0 for row in rows)
    assert all(row["policy_frame"] == "zquoridor-canonical-209" for row in rows)

    with np.load(tmp_path / "teacher_targets.npz") as targets:
        assert targets["id"].dtype == np.dtype("S24")
        assert targets["id"].shape == (7,)
        assert targets["policy"].shape == (7, POLICY_DIM)
        assert targets["value"].shape == (7,)
        assert targets["game_result"].shape == (7,)
        assert np.allclose(targets["value"], 0.25)
        assert np.allclose(targets["game_result"], -1.0)
        assert np.allclose(targets["policy"].sum(axis=1), 1.0)
        assert targets["id"][0].decode("ascii") == rows[0]["id"]


def test_weighted_quotas_are_exact_and_default_quotas_remain_balanced(tmp_path):
    books = {}
    for name in ("center", "normal", "weakness"):
        path = tmp_path / (name + ".jsonl")
        path.write_text('{"moves":[]}\n', encoding="utf-8")
        books[name] = str(path)
    config = dict(search_games.CONFIG, opening_books=books, pairs=100, seed=17,
                  opening_weights={"center": 7, "normal": 2, "weakness": 1})
    selected = search_games.select_openings(config)
    assert Counter(row["book"] for row in selected) == {"center": 70, "normal": 20, "weakness": 10}
    assert selected == search_games.select_openings(config)
    assert search_games.opening_quotas(dict(config, pairs=3)) == {"center": 2, "normal": 1, "weakness": 0}
    balanced = search_games.opening_quotas(dict(config, pairs=1500, opening_weights={}))
    assert set(balanced.values()) == {500}


@pytest.mark.parametrize("arguments", [
    ["--opening-weight", "missing=1"],
    ["--opening-weight", "normal=-1"],
    ["--opening-weight", "normal=nan"],
    ["--opening-weight", "normal=1", "--opening-weight", "normal=2"],
    ["--opening-weight", "normal=0", "--opening-weight", "center_rush=0", "--opening-weight", "weakness=0"],
    ["--opening-temperature", "nan"], ["--opening-temperature", "-1"],
    ["--temperature-plies", "-1"], ["--mode", "claustrophobia-selfplay", "--record-both-searches"],
])
def test_collector_rejects_invalid_sampling_configuration(arguments):
    with pytest.raises(ValueError):
        search_games.resolve_config(search_games.build_parser().parse_args(arguments))


def _zq_root_response(referee: Referee, move: str, *, alternative: str | None = None) -> dict:
    visits = [0.0] * POLICY_DIM
    visits[search_games.global_action(move)] = 80.0
    if alternative:
        visits[search_games.global_action(move)] = 1.0
        visits[search_games.global_action(alternative)] = 999.0
    return {"policy_frame": "global-board-209", "root_value_perspective": "side-to-move",
            "side_to_move": referee.side_to_move, "bestmove": move,
            "best_action": search_games.global_action(move), "move_time_ms": 400,
            "root_visits": sum(visits), "root_value": -0.2, "visit_counts": visits,
            "stop_reason": "search-complete", "q_values": [None] * POLICY_DIM,
            "priors": [None] * POLICY_DIM}


class _PairedPlayer:
    def __init__(self, name, calls, diverse=False):
        self.name = name
        self.calls = calls
        self.diverse = diverse

    def bestmove(self, history, *, budget, timeout_s):
        self.calls.append((self.name, tuple(history), budget))
        side = len(history) % 2
        referee = Referee()
        for move in history:
            referee.apply(move)
        script_name = "zquoridor" if side == 0 else "claustrophobia"
        move, _, _ = _ScriptedPlayer(script_name).bestmove(history, budget=budget, timeout_s=timeout_s)
        if self.name == "claustrophobia":
            raw = _root_response(referee, move)
            raw["move_time_ms"] = budget
            info = [json.dumps(raw)]
        else:
            raw = _zq_root_response(referee, move, alternative="d1" if self.diverse and not history else None)
            raw["move_time_ms"] = budget
            info = ["info string root_json " + json.dumps(raw)]
        return move, 0.001, info

    def close(self):
        pass


def _paired_game(**overrides):
    task = {"pair_index": 0, "zq_player": 0, "opening": [], "book": "normal",
            "category": "normal", "opening_group": sample_id([]), "split": "train"}
    calls = []
    config = dict(search_games.CONFIG, record_both_searches=True, **overrides)
    factories = {name: (lambda name=name: _PairedPlayer(name, calls, diverse=True))
                 for name in ("zquoridor", "claustrophobia")}
    return search_games.play_game(task, config, "fake-run", factories), calls


def test_paired_searches_use_identical_positions_budgets_and_preserve_both_roots():
    game, calls = _paired_game()
    assert game["status"] == "ok"
    assert len(game["searches"]) == 30
    assert len(calls) == 30
    for index in range(0, len(calls), 2):
        assert calls[index][1:] == calls[index + 1][1:]
        first, second = game["searches"][index:index + 2]
        assert first["history"] == second["history"]
        assert first["role"] == "mover" and second["role"] == "comparison"
        assert {first["player"], second["player"]} == {"zquoridor", "claustrophobia"}
        assert first["playedmove"] == second["playedmove"]
        assert len(first["root"]["visit_counts_global"]) == POLICY_DIM
        assert len(second["root"]["visit_counts_global"]) == POLICY_DIM
        assert first["outcome"] == second["outcome"]
    rows = list(search_games.training_positions(game))
    assert len(rows) == 15
    assert {row["search_role"] for row in rows} == {"mover", "comparison"}
    assert all(row["source"] == "claustrophobia-in-game-search" for row in rows)
    assert all(row["value"] == 0.25 for row in rows)


def test_temperature_samples_visits_deterministically_and_stops_after_relative_window():
    game, _ = _paired_game(opening_temperature=1.0, temperature_plies=1, max_plies=2)
    repeated, _ = _paired_game(opening_temperature=1.0, temperature_plies=1, max_plies=2)
    assert game == repeated
    assert game["status"] == "truncated"
    assert game["moves"] == ["d1", "e8"]
    first = game["searches"][0]
    assert first["bestmove"] == "e2" and first["playedmove"] == "d1"
    assert first["selection"] == "visit_temperature"
    assert game["searches"][2]["temperature"] == 0
    assert game["searches"][2]["selection"] == "bestmove"
    referee = Referee()
    for move in game["moves"]:
        referee.apply(move)


def test_temperature_respects_solver_proofs_and_does_not_create_zero_visit_policy():
    referee = Referee()
    solved = search_games.parse_root(_info_for(referee, "e2", stop_reason="solved", visit_sum=79),
                                    referee, "e2", 400)
    assert search_games.sampled_move(solved, "e2", referee, 1, random.Random(1)) == ("e2", "solver_bestmove")
    zero = search_games.parse_root(_info_for(referee, "e2", root_visits=0, visit_sum=0),
                                  referee, "e2", 400)
    assert zero["policy"] is None
    assert search_games.sampled_move(zero, "e2", referee, 1, random.Random(1)) == ("e2", "zero_visits_bestmove")


def test_zq_root_accepts_actual_float_counts_and_keeps_unavailable_value_null():
    referee = Referee()
    raw = _zq_root_response(referee, "e2")
    info = ["info string root_json " + json.dumps(raw)]
    root = search_games.parse_zq_root(info, referee, "e2", 400)
    assert root["root_value"] == -0.2 and root["visit_counts_global"][13] == 80.0
    raw.update(root_value=None, root_visits=0.0, visit_counts=[0.0] * POLICY_DIM)
    empty = search_games.parse_zq_root(["info string root_json " + json.dumps(raw)], referee, "e2", 400)
    assert empty["root_value"] is None
    assert sum(empty["visit_counts_global"]) == 0
    with pytest.raises(search_games.local_arena.EngineError, match="complete root JSON"):
        search_games.parse_zq_root(["info string root e2:80"], referee, "e2", 400)


def test_resume_excludes_claimed_interrupted_games_and_rejects_identity_change(tmp_path):
    task = {"pair_index": 0, "zq_player": 0, "book": "normal"}
    manifest = {"run_id": "run-one"}
    assert search_games.prepare_ledger(tmp_path, manifest, [task], True) == [task]
    claim = tmp_path / "games" / "0000000_0.pending"
    search_games.atomic_json(claim, {**task, "run_id": "run-one"})
    assert search_games.prepare_ledger(tmp_path, manifest, [task], True) == []
    record = json.loads((tmp_path / "games" / "0000000_0.json").read_text())
    assert record["status"] == "interrupted"
    assert not claim.exists()
    assert list(search_games.training_positions(record)) == []
    with pytest.raises(ValueError, match="different run identity"):
        search_games.prepare_ledger(tmp_path, {"run_id": "run-two"}, [task], True)
