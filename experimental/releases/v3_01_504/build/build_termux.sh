#!/usr/bin/env bash
# build_termux.sh -- build nativo em Termux (Android, ARM). Diferenças
# em relação a build_bench.sh/build_tests.sh/build_selfplay.sh:
#   - SEM -mavx2/-mfma: são extensões x86, não existem em ARM (o
#     compilador rejeita a flag). ARM usa NEON, habilitado por
#     -march=native como qualquer outra extensão de CPU.
#   - Termux usa clang por padrão (pacote "clang"); cai para g++ se
#     clang++ não estiver instalado.
#   - Um script só para os três tipos nativos (bench/testes/selfplay) --
#     no celular normalmente não vale a pena separar em três chamadas.
#   - WASM fora de escopo aqui (emsdk não é viável em Termux na prática).
#
# Setup, se ainda não tiver os pacotes:
#   pkg update && pkg install clang python
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
SRC="$ROOT/src"
BENCHMARK="$ROOT/benchmarks"
TESTS="$ROOT/tests"
SELFPLAY="$ROOT/tools/selfplay"
SPSA="$ROOT/tools/spsa"
BIN="$ROOT/bin"

mkdir -p "$BIN" "$ROOT/data"

CXX=g++
if command -v clang++ >/dev/null 2>&1; then
    CXX=clang++
fi
echo "Compilador: $CXX"

PERF_FLAGS=(-O3 -std=c++17 -march=native)
TEST_FLAGS=(-O2 -std=c++17)

echo "[1/8] bench  <-  benchmarks/main.cpp"
"$CXX" "${PERF_FLAGS[@]}" -I"$SRC" -I"$SELFPLAY" -o "$BIN/bench" "$BENCHMARK/main.cpp"

echo "[2/8] bench_wall_touch_bonus  <-  benchmarks/bench_wall_touch_bonus.cpp"
"$CXX" "${PERF_FLAGS[@]}" -I"$SRC" -o "$BIN/bench_wall_touch_bonus" "$BENCHMARK/bench_wall_touch_bonus.cpp"

echo "[3/8] test_rules_sanity"
"$CXX" "${TEST_FLAGS[@]}" -I"$SRC" -o "$BIN/test_rules_sanity" "$TESTS/test_rules_sanity.cpp"

echo "[4/8] test_search_staging"
"$CXX" "${TEST_FLAGS[@]}" -I"$SRC" -o "$BIN/test_search_staging" "$TESTS/test_search_staging.cpp"

echo "[5/8] test_move_ordering"
"$CXX" "${TEST_FLAGS[@]}" -I"$SRC" -o "$BIN/test_move_ordering" "$TESTS/test_move_ordering.cpp"

echo "[6/8] nnue_verify"
"$CXX" "${TEST_FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/nnue_verify" "$TESTS/nnue_verify.cpp"

echo "[7/8] selfplay  <-  tools/selfplay/selfplay_main.cpp"
"$CXX" "${PERF_FLAGS[@]}" -pthread -I"$SRC" -I"$SELFPLAY" -o "$BIN/selfplay" "$SELFPLAY/selfplay_main.cpp"

echo "[8/8] tune_spsa  <-  tools/spsa/tune_spsa.cpp"
"$CXX" "${PERF_FLAGS[@]}" -I"$SRC" -o "$BIN/tune_spsa" "$SPSA/tune_spsa.cpp"

echo
echo "OK -- binários em $BIN (sem extensão, mesmo layout do Linux desktop)"
