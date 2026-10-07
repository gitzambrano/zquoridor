"""Exercise native game-clock forwarding without launching external engines."""
from __future__ import annotations

import json
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.external import local_arena


class _ClockPlayer:
    def __init__(self, moves: list[str]) -> None:
        self.moves = iter(moves)
        self.calls: list[dict] = []

    def bestmove_clock(self, history, *, white_ms, black_ms, increment_ms, timeout_s):
        self.calls.append({"history": list(history), "white_ms": white_ms,
                           "black_ms": black_ms, "increment_ms": increment_ms,
                           "timeout_s": timeout_s})
        return next(self.moves), 0.001, []

    def close(self):
        pass


class NativeClockTests(unittest.TestCase):
    def test_play_game_dispatches_both_players_with_complete_clocks(self):
        candidate = _ClockPlayer(["e2"])
        opponent = _ClockPlayer(["e8"])
        row = local_arena.play_game(
            opponent="titanium", opening_index=3, opening=[], zq_player=0,
            zq_factory=lambda: candidate, opponent_factory=lambda: opponent,
            zq_budget=200, opponent_budget=200, move_timeout_s=1.0, max_plies=2,
            run_id="native-clock-test", clock_initial_ms=180000, clock_increment_ms=2000)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(len(candidate.calls), 1)
        self.assertEqual(len(opponent.calls), 1)
        self.assertEqual(candidate.calls[0]["white_ms"], 180000)
        self.assertEqual(candidate.calls[0]["black_ms"], 180000)
        self.assertEqual(opponent.calls[0]["white_ms"], 181999)
        self.assertEqual(opponent.calls[0]["black_ms"], 180000)
        for player in (candidate, opponent):
            self.assertEqual(player.calls[0]["increment_ms"], 2000)
        self.assertTrue(all(move["clock_protocol"] == "native-game-clock-v2"
                            for move in row["move_times"]))

    def test_play_game_refuses_a_clock_mode_player_without_native_api(self):
        class LegacyPlayer:
            def bestmove(self, history, *, budget, timeout_s):
                return "e2", 0.001, []

            def close(self):
                pass

        row = local_arena.play_game(
            opponent="legacy", opening_index=0, opening=[], zq_player=0,
            zq_factory=LegacyPlayer, opponent_factory=LegacyPlayer,
            zq_budget=200, opponent_budget=200, move_timeout_s=1.0, max_plies=1,
            run_id="native-clock-reject", clock_initial_ms=180000, clock_increment_ms=2000)
        self.assertEqual(row["status"], "failed")
        self.assertIn("native game-clock protocol is required", row["error"])

    def test_titanium_forwards_remaining_increment_and_opponent_clock(self):
        player = object.__new__(local_arena.TitaniumPlayer)
        commands = []
        player._send = commands.append
        player._wait_token = lambda token, timeout: None
        replies = iter(["bestmove e2"])
        player._read = lambda timeout: next(replies)
        clock = itertools.count(100000, 1)
        with patch.object(local_arena.time, "monotonic", side_effect=lambda: next(clock) / 1000):
            move, elapsed, _ = player.bestmove_clock(
                ["e2"], white_ms=180000, black_ms=177500, increment_ms=2000, timeout_s=1.0)
        self.assertEqual(move, "e2")
        self.assertGreater(elapsed, 0)
        self.assertEqual(commands, ["position e2", "go rem 177.500000 inc 2.000000 opp 180.000000"])

    def test_claustrophobia_forwards_clocks_and_requires_v2_response(self):
        player = object.__new__(local_arena.ClaustrophobiaPlayer)
        commands = []
        player._send = commands.append
        response = {"protocol": "deadline-and-engine-clock-v2", "mode": "game-clock",
                    "bestmove": "e2", "sims": 12, "root_visits": 12,
                    "stop_reason": "deadline", "white_ms": 180000,
                    "black_ms": 177500, "increment_ms": 2000}
        player._read = lambda timeout: json.dumps(response)
        with patch.object(local_arena.time, "monotonic", return_value=2.0):
            move, _, info = player.bestmove_clock(
                ["e2"], white_ms=180000, black_ms=177500, increment_ms=2000, timeout_s=1.0)
        self.assertEqual(move, "e2")
        self.assertEqual(commands, ["clock\t180000\t177500\t2000\te2"])
        self.assertEqual(json.loads(info[0])["protocol"], "deadline-and-engine-clock-v2")
        self.assertEqual(player.last_info, response)

        legacy = object.__new__(local_arena.ClaustrophobiaPlayer)
        legacy._send = lambda command: None
        legacy._read = lambda timeout: json.dumps({"bestmove": "e2", "move_time_ms": 200})
        with patch.object(local_arena.time, "monotonic", return_value=3.0):
            with self.assertRaisesRegex(local_arena.EngineError, "deadline clock protocol is required"):
                legacy.bestmove_clock(
                    ["e2"], white_ms=180000, black_ms=177500,
                    increment_ms=2000, timeout_s=1.0)


if __name__ == "__main__":
    unittest.main()
