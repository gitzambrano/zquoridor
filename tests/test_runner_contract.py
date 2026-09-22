from __future__ import annotations

import importlib


RUNNERS = [
    "tools.run_benchmark",
    "tools.run_clock_smoke",
    "tools.run_full_candidate_suite",
    "tools.benchmark_candidate",
    "tools.setup_bots",
    "training.run_experiment",
    "training.run_teaching",
    "training.run_campaign",
    "training.strong_cycle",
    "tools.match_finalists",
    "tools.teacher.run_four_million_selfplay",
    "tools.teacher.run_weakness_selfplay",
]


def test_runner_modules_expose_config_and_main():
    for name in RUNNERS:
        module = importlib.import_module(name)
        assert hasattr(module, "CONFIG"), name
        assert callable(module.main), name


def test_runner_defaults_are_mappings():
    for name in RUNNERS:
        module = importlib.import_module(name)
        assert isinstance(module.CONFIG, dict), name
