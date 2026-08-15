#!/usr/bin/env python3
"""
run_spsa.py -- launcher do tuner evolutivo/genético.

O nome é mantido para compatibilidade com o projeto. O algoritmo implementado
em tools/spsa/tune_spsa.cpp agora é um Genetic Algorithm robusto, adequado à função
objetivo discreta/ruidosa das partidas de Quoridor.

Ajuste CONFIG e rode:
    python tools/spsa/run_spsa.py
"""

import argparse
import os
import subprocess
import sys

# =============================================================================
# CONFIG
# =============================================================================

GENERATIONS = 40
POPULATION = 24
GAMES_PER_MATCH = 14
THREADS = 10  # avalia N individuos da populacao em paralelo (ver tune_spsa.cpp --help)

ELITE_FRACTION = 0.20
MUTATION_RATE = 0.25
IMMIGRANTS = 0.10

SEED = 120809
TIME_BUDGET_SEC = None

CHECKPOINT = "ga_checkpoint.txt"
HISTORY = "ga_history.csv"
RESULT = "ga_result.txt"

SEARCH_DEPTH = 25
TIME_MS = 80
MAX_PLIES = 140
OPENING_RANDOM_PLIES = 4

USE_NNUE = True
NNUE_WEIGHTS_PATH = None
POLICY_ORDERING = True


def find_project_root():
    here = os.path.abspath(os.path.dirname(__file__))
    candidate = here
    for _ in range(6):
        if os.path.isdir(os.path.join(candidate, "src")):
            return candidate
        candidate = os.path.dirname(candidate)
    return here


def find_exe(root):
    for name in ("tune_spsa.exe", "tune_spsa"):
        p = os.path.join(root, "bin", name)
        if os.path.isfile(p):
            return p
    return None


def needs_recompile(root, exe):
    src = os.path.join(root, "tools", "spsa", "tune_spsa.cpp")
    search = os.path.join(root, "src", "search.hpp")
    if exe is None:
        return True
    return max(os.path.getmtime(src), os.path.getmtime(search)) > os.path.getmtime(exe)


def compile_tuner(root):
    import platform
    if platform.system() == "Windows":
        script = os.path.join(root, "build", "build_tune_spsa.bat")
        if os.path.isfile(script):
            return subprocess.call(["cmd", "/c", script], cwd=root) == 0
    else:
        script = os.path.join(root, "build", "build_tune_spsa.sh")
        if os.path.isfile(script):
            return subprocess.call(["bash", script], cwd=root) == 0
    return False


def parse_args():
    p = argparse.ArgumentParser(description="Genetic tuner do zQuoridor")
    p.add_argument("--generations", type=int, default=GENERATIONS)
    p.add_argument("--population", type=int, default=POPULATION)
    p.add_argument("--games-per-match", type=int, default=GAMES_PER_MATCH)
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--elite-fraction", type=float, default=ELITE_FRACTION)
    p.add_argument("--mutation-rate", type=float, default=MUTATION_RATE)
    p.add_argument("--immigrants", type=float, default=IMMIGRANTS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--time-budget-sec", type=float, default=TIME_BUDGET_SEC)

    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--history", default=HISTORY)
    p.add_argument("--result", default=RESULT)

    p.add_argument("--depth", type=int, default=SEARCH_DEPTH)
    p.add_argument("--time-ms", type=int, default=TIME_MS)
    p.add_argument("--max-plies", type=int, default=MAX_PLIES)
    p.add_argument("--opening-plies", type=int, default=OPENING_RANDOM_PLIES)

    p.add_argument("--heuristic", dest="use_nnue", action="store_false", default=USE_NNUE)
    p.add_argument("--nnue", dest="use_nnue", action="store_true")
    p.add_argument("--nnue-weights", default=NNUE_WEIGHTS_PATH)
    p.add_argument("--policy-order", dest="policy_order", action="store_true", default=POLICY_ORDERING)
    p.add_argument("--no-policy-order", dest="policy_order", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    root = find_project_root()
    exe = find_exe(root)

    if needs_recompile(root, exe):
        print("[GA] compilando tuner...")
        if not compile_tuner(root):
            print("[ERRO] falha ao compilar tune_spsa.", file=sys.stderr)
            sys.exit(1)
        exe = find_exe(root)

    if exe is None:
        print("[ERRO] executável tune_spsa não encontrado.", file=sys.stderr)
        sys.exit(1)

    def path(p):
        return os.path.join(root, p)

    cmd = [
        exe,
        "--generations", str(args.generations),
        "--population", str(args.population),
        "--games-per-match", str(args.games_per_match),
        "--threads", str(args.threads),
        "--elite-fraction", str(args.elite_fraction),
        "--mutation-rate", str(args.mutation_rate),
        "--immigrants", str(args.immigrants),
        "--seed", str(args.seed),
        "--checkpoint", path(args.checkpoint),
        "--history", path(args.history),
        "--result", path(args.result),
        "--depth", str(args.depth),
        "--time-ms", str(args.time_ms),
        "--max-plies", str(args.max_plies),
        "--opening-plies", str(args.opening_plies),
    ]

    if args.time_budget_sec is not None:
        cmd += ["--time-budget-sec", str(args.time_budget_sec)]
    if args.use_nnue:
        if args.nnue_weights:
            cmd += ["--nnue-weights", args.nnue_weights]
    else:
        cmd += ["--heuristic"]
    if not args.policy_order:
        cmd += ["--no-policy-order"]

    print("[GA] executando:")
    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
