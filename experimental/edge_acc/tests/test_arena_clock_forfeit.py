"""Verify clock losses and process failures in paired clock games."""
import unittest

from tools.external import local_arena as arena


class Player:
    def __init__(self, move="e2", elapsed=0.001, error=None):
        self.move = move
        self.elapsed = elapsed
        self.error = error
        self.timeout = None

    def bestmove_clock(self, history, **kwargs):
        self.timeout = kwargs["timeout_s"]
        if self.error:
            raise self.error
        return self.move, self.elapsed, ["info nodes 17"]

    def bestmove(self, history, **kwargs):
        raise arena.EngineTimeout("fixed-movetime timeout")

    def close(self):
        pass


def game(candidate, opponent=None, **kwargs):
    return arena.play_game(
        opponent="baseline", opening_index=0, opening=[], zq_player=0,
        zq_factory=lambda: candidate, opponent_factory=lambda: opponent or Player("e8"),
        zq_budget=200, opponent_budget=200, move_timeout_s=240, max_plies=2,
        run_id="clock-forfeit-test", clock_initial_ms=kwargs.get("initial", 10),
        clock_increment_ms=2)


class ClockForfeitTests(unittest.TestCase):
    def test_late_move_loses_without_applying_move(self):
        player = Player(elapsed=0.011)
        row = game(player)
        self.assertEqual((row["status"], row["termination"], row["result"]), ("ok", "time", 0.0))
        self.assertEqual(row["winner"], 1)
        self.assertEqual(row["moves"], [])
        self.assertEqual(row["final_clocks_ms"][0], 0)
        self.assertEqual(player.timeout, 0.01)
        self.assertEqual(row["move_times"][-1]["search_last"], "info nodes 17")

    def test_opponent_clock_loss_counts_as_candidate_win(self):
        row = game(Player(), Player("e8", elapsed=0.011))
        self.assertEqual((row["termination"], row["result"], row["winner"]), ("time", 1.0, 0))
        self.assertEqual(row["moves"], ["e2"])

    def test_no_response_at_clock_deadline_loses(self):
        row = game(Player(error=arena.EngineTimeout("deadline")))
        self.assertEqual((row["status"], row["termination"], row["result"]), ("ok", "time", 0.0))

    def test_engine_exit_is_an_explicit_forfeit(self):
        row = game(Player(error=arena.EngineError("process exited with code 1")))
        self.assertEqual((row["status"], row["termination"], row["result"]), ("ok", "engine_error", 0.0))
        self.assertIn("code 1", row["forfeit"]["error"])

    def test_fixed_movetime_timeout_stays_an_infrastructure_failure(self):
        row = game(Player(), initial=0)
        self.assertEqual((row["status"], row["termination"]), ("failed", "error"))

    def test_clock_deadline_uses_remaining_clock(self):
        candidate = Player()
        game(candidate, initial=300000)
        self.assertEqual(candidate.timeout, 300.0)


if __name__ == "__main__":
    unittest.main()
