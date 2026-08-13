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

echo "[1/7] bench  <-  benchmarks/main.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -I"$SELFPLAY" -o "$BIN/bench" "$BENCHMARK/main.cpp"

echo "[2/7] bench_wall_touch_bonus  <-  benchmarks/bench_wall_touch_bonus.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/bench_wall_touch_bonus" "$BENCHMARK/bench_wall_touch_bonus.cpp"

echo "[3/7] bench_quiescence_toggle  <-  benchmarks/bench_quiescence_toggle.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/bench_quiescence_toggle" "$BENCHMARK/bench_quiescence_toggle.cpp"

echo "[4/7] bench_lmr_pvs  <-  benchmarks/bench_lmr_pvs.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/bench_lmr_pvs" "$BENCHMARK/bench_lmr_pvs.cpp"

# [5] baseline da Seção 0 de plan-hybrid-mc-ab.md: carga fixa SEM mcab.hpp
# incluído. Não estava registrado aqui antes; passa a estar porque o bloco A
# de bench_mcab só faz sentido comparado com ele.
echo "[5/7] bench_fixed_depth  <-  benchmarks/bench_fixed_depth.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/bench_fixed_depth" "$BENCHMARK/bench_fixed_depth.cpp"

echo "[6/7] bench_mcab_equivalence  <-  benchmarks/bench_mcab_equivalence.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/bench_mcab_equivalence" "$BENCHMARK/bench_mcab_equivalence.cpp"

echo "[7/7] bench_mcab  <-  benchmarks/bench_mcab.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/bench_mcab" "$BENCHMARK/bench_mcab.cpp"

echo
echo "OK -- binários em $BIN"
echo "  bench_fixed_depth / bench_mcab: a CONTAGEM DE NÓS do bloco A de bench_mcab"
echo "    tem que bater exatamente com bench_fixed_depth (Seção 0)."
echo "  bench_mcab_equivalence [leafDepth] [numPosicoes]"
echo "  bench_mcab [nodeBudget] [leafDepth]  -- rodar da RAIZ do repo."
