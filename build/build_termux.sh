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
TESTE="$ROOT/teste"
BIN="$ROOT/bin"

mkdir -p "$BIN" "$ROOT/data"

CXX=g++
if command -v clang++ >/dev/null 2>&1; then
    CXX=clang++
fi
echo "Compilador: $CXX"

PERF_FLAGS=(-O3 -std=c++17 -march=native)
TEST_FLAGS=(-O2 -std=c++17)

echo "[1/8] bench  <-  src/main.cpp"
"$CXX" "${PERF_FLAGS[@]}" -I"$SRC" -o "$BIN/bench" "$SRC/main.cpp"

echo "[2/8] bench_wall_touch_bonus  <-  teste/bench_wall_touch_bonus.cpp"
"$CXX" "${PERF_FLAGS[@]}" -I"$SRC" -o "$BIN/bench_wall_touch_bonus" "$TESTE/bench_wall_touch_bonus.cpp"

echo "[3/8] test_rules_sanity"
"$CXX" "${TEST_FLAGS[@]}" -I"$SRC" -o "$BIN/test_rules_sanity" "$TESTE/test_rules_sanity.cpp"

echo "[4/8] test_search_staging"
"$CXX" "${TEST_FLAGS[@]}" -I"$SRC" -o "$BIN/test_search_staging" "$TESTE/test_search_staging.cpp"

echo "[5/8] test_move_ordering"
"$CXX" "${TEST_FLAGS[@]}" -I"$SRC" -o "$BIN/test_move_ordering" "$TESTE/test_move_ordering.cpp"

echo "[6/8] nnue_verify"
"$CXX" "${TEST_FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/nnue_verify" "$TESTE/nnue_verify.cpp"

echo "[7/8] selfplay  <-  src/selfplay_main.cpp"
"$CXX" "${PERF_FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/selfplay" "$SRC/selfplay_main.cpp"

echo "[8/8] tune_spsa  <-  teste/tune_spsa.cpp"
"$CXX" "${PERF_FLAGS[@]}" -I"$SRC" -o "$BIN/tune_spsa" "$TESTE/tune_spsa.cpp"

echo
echo "OK -- binários em $BIN (sem extensão, mesmo layout do Linux desktop)"
