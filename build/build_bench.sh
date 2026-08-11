#!/usr/bin/env bash
# build_bench.sh -- benchmarks de performance (não fazem parte da suíte
# de correção, medem nós/s e profundidade). Equivalente Linux de
# build_bench.bat -- mesmos flags (-mavx2 -mfma explícitos além de
# -march=native: a máquina alvo tem AVX2 garantido, mesmo critério do
# Zchezz). Para ARM/Termux use build_termux.sh em vez deste.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
SRC="$ROOT/src"
SELFPLAY="$ROOT/tools/selfplay"
BENCHMARK="$ROOT/benchmarks"
BIN="$ROOT/bin"

mkdir -p "$BIN"

FLAGS=(-O3 -std=c++17 -march=native -mavx2 -mfma)

echo "[1/4] bench  <-  benchmarks/main.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -I"$SELFPLAY" -o "$BIN/bench" "$BENCHMARK/main.cpp"

echo "[2/4] bench_wall_touch_bonus  <-  benchmarks/bench_wall_touch_bonus.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/bench_wall_touch_bonus" "$BENCHMARK/bench_wall_touch_bonus.cpp"

echo "[3/4] bench_quiescence_toggle  <-  benchmarks/bench_quiescence_toggle.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/bench_quiescence_toggle" "$BENCHMARK/bench_quiescence_toggle.cpp"

echo "[4/4] bench_lmr_pvs  <-  benchmarks/bench_lmr_pvs.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/bench_lmr_pvs" "$BENCHMARK/bench_lmr_pvs.cpp"

echo
echo "OK -- binários em $BIN"
