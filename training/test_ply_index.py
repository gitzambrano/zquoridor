"""test_ply_index.py -- correctness test for the per-game ply index.

The self-play writer stores one game for each fwrite call, so the samples
of one game stay contiguous and keep their ply order. This test confirms
that `ply_index` recovers that order from a real shard.

The test needs at least one shard under data/selfplay/. Run it from the
repository root:

    python training/test_ply_index.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from read_selfplay import (  # noqa: E402
    SAMPLE_DTYPE, game_start_mask, load_selfplay, ply_index,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARD_DIR = os.path.join(REPO, "data", "selfplay", "gen6-montecarlo")

# The montecarlo temperature window: MC_OBVIOUS_PLIES + MC_TEMP_DECAY_PLIES.
OPENING_PLIES = 26

failures = []


def check(cond, label):
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FALHA {label}")
        failures.append(label)


def test_synthetic():
    """A hand-built array with two known games."""
    print("teste sintetico (2 jogos, comprimentos 3 e 4):")
    n = 7
    arr = np.zeros(n, dtype=SAMPLE_DTYPE)
    # Both games start at index 0 and index 3.
    for i in (0, 3):
        arr[i]["walls_h"] = 0
        arr[i]["walls_v"] = 0
        arr[i]["walls_left_own"] = 10
        arr[i]["walls_left_opp"] = 10
        arr[i]["own_dist"] = 8
        arr[i]["opp_dist"] = 8
    # Every other record must not look like a start.
    for i in (1, 2, 4, 5, 6):
        arr[i]["walls_left_own"] = 10
        arr[i]["walls_left_opp"] = 10
        arr[i]["own_dist"] = 8
        arr[i]["opp_dist"] = 7   # the opponent already moved
    starts = game_start_mask(arr)
    check(int(starts.sum()) == 2, f"2 inicios detectados (obtido {int(starts.sum())})")
    ply = ply_index(arr)
    check(list(ply) == [0, 1, 2, 0, 1, 2, 3], f"ply esperado, obtido {list(ply)}")


def test_real_shard(path):
    print(f"\nshard real: {os.path.basename(path)}")
    arr = load_selfplay(path, quiet=True)
    ply = ply_index(arr)
    starts = game_start_mask(arr)
    n_games = int(starts.sum())
    check(len(ply) == len(arr), "um indice de ply para cada registro")
    check(n_games > 0, f"{n_games} jogos detectados")

    # Segment lengths.
    bounds = np.flatnonzero(starts)
    lengths = np.diff(np.append(bounds, len(arr)))
    check(lengths.min() >= 2, f"nenhum jogo degenerado (minimo {lengths.min()})")
    check(lengths.max() <= 300, f"nenhum jogo acima do corte de 300 (maximo {lengths.max()})")

    # The first record of every game must have ply 0.
    check(bool((ply[bounds] == 0).all()), "todo inicio de jogo tem ply 0")

    # The mover must alternate strictly inside one game. This is the
    # strongest independent check that the boundaries are right.
    ok_alt = 0
    for b, ln in zip(bounds, lengths):
        seg = arr[b:b + ln]["mover"]
        if ln > 1 and np.all(seg[1:] != seg[:-1]):
            ok_alt += 1
        elif ln == 1:
            ok_alt += 1
    frac_alt = ok_alt / max(1, len(bounds))
    check(frac_alt == 1.0, f"mover alterna em 100% dos jogos (obtido {frac_alt:.4%})")

    frac_open = float((ply < OPENING_PLIES).mean())
    print(f"  info  comprimento medio {lengths.mean():.1f} plies, "
          f"{frac_open:.2%} dos registros abaixo do ply {OPENING_PLIES}")
    check(0.30 < frac_open < 0.50,
          f"fracao de abertura plausivel ({frac_open:.2%} entre 30% e 50%)")


def main():
    test_synthetic()
    if not os.path.isdir(SHARD_DIR):
        print(f"\nAVISO: {SHARD_DIR} nao existe -- so o teste sintetico rodou.")
    else:
        shards = sorted(f for f in os.listdir(SHARD_DIR) if f.endswith(".bin"))
        if not shards:
            print(f"\nAVISO: nenhum .bin em {SHARD_DIR} -- so o teste sintetico rodou.")
        for name in shards[:3]:
            test_real_shard(os.path.join(SHARD_DIR, name))

    print()
    if failures:
        print(f"FALHOU: {len(failures)} verificacao(oes)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
