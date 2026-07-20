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
TESTE="$ROOT/teste"
BIN="$ROOT/bin"

mkdir -p "$BIN"

FLAGS=(-O3 -std=c++17 -march=native -mavx2 -mfma)

echo "[1/2] bench  <-  src/main.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/bench" "$SRC/main.cpp"

echo "[2/3] bench_wall_touch_bonus  <-  teste/bench_wall_touch_bonus.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/bench_wall_touch_bonus" "$TESTE/bench_wall_touch_bonus.cpp"

echo "[3/3] bench_quiescence_toggle  <-  teste/bench_quiescence_toggle.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/bench_quiescence_toggle" "$TESTE/bench_quiescence_toggle.cpp"

echo
echo "OK -- binários em $BIN"
