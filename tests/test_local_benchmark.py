from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from tools import run_benchmark
from tools.external import local_arena


class RefereeTests(unittest.TestCase):
    def test_start_has_full_legal_move_set_and_rejects_illegal_pawn_move(self) -> None:
        state = local_arena.Referee()
        legal = state.legal_moves()
        self.assertEqual(len(legal), 131)
        self.assertIn("e2", legal)
        self.assertIn("a1h", legal)
        with self.assertRaisesRegex(local_arena.IllegalMove, "illegal move"):
            state.apply("e3")

    def test_referee_implements_straight_jump_and_blocked_jump_diagonals(self) -> None:
        straight = local_arena.Referee()
        for move in ("e2", "e8", "e3", "e7", "e4", "e6", "e5"):
            straight.apply(move)
        self.assertIn("e7", straight.legal_moves())
        self.assertNotIn("e6", straight.legal_moves())

        diagonal = local_arena.Referee()
        for move in ("e2", "e8", "e3", "e7", "e4", "e6", "e5", "e6h"):
            diagonal.apply(move)
        self.assertIn("d6", diagonal.legal_moves())
        self.assertIn("f6", diagonal.legal_moves())
        self.assertNotIn("e7", diagonal.legal_moves())

    def test_referee_rejects_crossing_and_path_blocking_walls(self) -> None:
        state = local_arena.Referee()
        state.apply("e2")
        state.apply("e4h")
        self.assertNotIn("e4v", state.legal_moves())
        with self.assertRaises(local_arena.IllegalMove):
            state.apply("e4v")

        trapped = local_arena.Referee()
        sequence = (
            "a6v", "h8v", "e7h", "g6v", "d1v", "e4h",
            "b6h", "f2h", "h5h", "f3v", "e5v", "g7h",
        )
        for move in sequence:
            trapped.apply(move)
        # The candidate has no geometric conflict, but it removes the last path.
        self.assertTrue(trapped._wall_geometry_ok(7, 3, "v"))
        self.assertNotIn("d8v", trapped.legal_moves())


class ProcessTests(unittest.TestCase):
    def _fake_engine(self, directory: Path, behavior: str) -> list[str]:
        script = directory / f"fake_{behavior}.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import sys, time
                behavior = {behavior!r}
                for raw in sys.stdin:
                    cmd = raw.strip()
                    if cmd == 'uci': print('uciok', flush=True)
                    elif cmd == 'isready': print('readyok', flush=True)
                    elif cmd.startswith('go '):
                        if behavior == 'timeout': time.sleep(30)
                        elif behavior == 'error': print('info string error fake failure', flush=True)
                        elif behavior == 'exit': sys.exit(7)
                        else: print('bestmove e2', flush=True)
                    elif cmd == 'quit': break
                """
            ),
            encoding="utf-8",
        )
        return [sys.executable, str(script)]

    def test_engine_reports_error_output_and_closes_process(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            engine = local_arena.UciPlayer(
                self._fake_engine(Path(raw), "error"), "fake", startup_timeout_s=1.0
            )
            try:
                with self.assertRaisesRegex(local_arena.EngineError, "fake failure"):
                    engine.bestmove([], budget=10, timeout_s=1.0)
            finally:
                engine.close()
            self.assertIsNotNone(engine.process.poll())

    def test_engine_timeout_kills_process_tree(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            engine = local_arena.UciPlayer(
                self._fake_engine(Path(raw), "timeout"), "fake", startup_timeout_s=1.0
            )
            started = time.monotonic()
            with self.assertRaisesRegex(local_arena.EngineTimeout, "timeout"):
                engine.bestmove([], budget=10, timeout_s=0.15)
            self.assertLess(time.monotonic() - started, 3.0)
            self.assertIsNotNone(engine.process.poll())

    def test_failed_game_is_recorded_and_excluded_from_pair_statistic(self) -> None:
        rows = [
            {"status": "ok", "opponent": "x", "opening_index": 0,
             "zq_player": 0, "result": 1.0},
            {"status": "failed", "opponent": "x", "opening_index": 0,
             "zq_player": 1, "error": "timeout"},
            {"status": "ok", "opponent": "x", "opening_index": 1,
             "zq_player": 0, "result": 0.5},
            {"status": "ok", "opponent": "x", "opening_index": 1,
             "zq_player": 1, "result": 1.0},
        ]
        report = local_arena.summarize_pairs(rows, bootstrap=1000, seed=9)
        self.assertEqual(report["failed_games"], 1)
        self.assertEqual(report["complete_pairs"], 1)
        self.assertEqual(report["excluded_ok_games"], 1)
        self.assertEqual(report["score_pct"], 75.0)


class ResumeAndConfigTests(unittest.TestCase):
    def test_run_id_hashes_complete_config_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            engine = root / "engine.bin"
            engine.write_bytes(b"version one")
            config = {"pairs": 20, "workers": 1, "opponents": ["titanium"]}
            first = local_arena.make_manifest(config, {"engine": engine})
            engine.write_bytes(b"version two")
            second = local_arena.make_manifest(config, {"engine": engine})
            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertNotEqual(first["artifacts"]["engine"]["sha256"],
                                second["artifacts"]["engine"]["sha256"])

    def test_resume_refuses_a_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            first = {"schema": "zquoridor.local_benchmark.manifest.v1", "run_id": "aaa"}
            second = {"schema": "zquoridor.local_benchmark.manifest.v1", "run_id": "bbb"}
            local_arena.prepare_resume(out, first)
            with self.assertRaisesRegex(ValueError, "different benchmark configuration"):
                local_arena.prepare_resume(out, second)

    def test_cli_overrides_top_config_without_changing_other_defaults(self) -> None:
        parser = run_benchmark.build_parser()
        args = parser.parse_args(["--pairs", "2", "--opponents", "titanium",
                                  "--zq-executable", "custom.exe"])
        cfg = run_benchmark.resolve_config(args)
        self.assertEqual(cfg["pairs"], 2)
        self.assertEqual(cfg["opponents"], ["titanium"])
        self.assertEqual(cfg["zq_executable"], "custom.exe")
        self.assertEqual(cfg["workers"], run_benchmark.CONFIG["workers"])
        self.assertEqual(run_benchmark.CONFIG["opponents"], ["titanium", "claustrophobia"])


if __name__ == "__main__":
    unittest.main()
