#!/usr/bin/env bash
# build_tests.sh -- suíte de correção (regras, staging, move ordering,
# solver de final "mãos vazias", paridade NNUE C++ vs Python). Sem
# -march=native/AVX2: são testes de corretude/precisão numérica, não de
# performance. Equivalente Linux de build_tests.bat.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
SRC="$ROOT/src"
TESTE="$ROOT/teste"
BIN="$ROOT/bin"

mkdir -p "$BIN"

FLAGS=(-O2 -std=c++17)

echo "[1/8] test_rules_sanity"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_rules_sanity" "$TESTE/test_rules_sanity.cpp"

echo "[2/8] test_search_staging"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_search_staging" "$TESTE/test_search_staging.cpp"

echo "[3/8] test_move_ordering"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_move_ordering" "$TESTE/test_move_ordering.cpp"

echo "[4/8] test_endgame_race"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_endgame_race" "$TESTE/test_endgame_race.cpp"

echo "[5/8] test_lmr_pvs"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_lmr_pvs" "$TESTE/test_lmr_pvs.cpp"

echo "[6/8] nnue_verify  (paridade C++ vs Python, precisa -pthread)"
g++ "${FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/nnue_verify" "$TESTE/nnue_verify.cpp"

echo "[7/8] nnue_incremental_check  (acumulador incremental vs rebuild do zero)"
g++ "${FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/nnue_incremental_check" "$TESTE/nnue_incremental_check.cpp"

echo "[8/8] nnue_sign_check  (sanidade de sinal/perspectiva do NNUE vs evalSimple)"
g++ "${FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/nnue_sign_check" "$TESTE/nnue_sign_check.cpp"

echo
echo "OK -- binários em $BIN"
echo "  test_rules_sanity / test_search_staging / test_move_ordering / test_endgame_race / test_lmr_pvs: sem argumentos."
echo "  nnue_verify <pesos_float32.bin> [pesos_int8.bin], ex.:"
echo "    bin/nnue_verify data/nnue/nnue_weights.bin data/nnue/nnue_weights_int8.bin"
echo "  nnue_incremental_check <pesos_int8.bin>: compara acumulador incremental vs"
echo "    rebuild do zero em 30 partidas aleatorias -- deve dar 0 divergencias."
echo "  nnue_sign_check <pesos_int8.bin>: imprime eval NNUE vs evalSimple em 4"
echo "    posicoes de referencia (checagem manual de sinal/perspectiva)."
