#!/usr/bin/env bash
# build_tests.sh -- suíte de correção (regras, staging, move ordering,
# solver de final "mãos vazias", paridade NNUE C++ vs Python). Sem
# -march=native/AVX2: são testes de corretude/precisão numérica, não de
# performance. Equivalente Linux de build_tests.bat.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
SRC="$ROOT/src"
TESTS="$ROOT/tests"
BIN="$ROOT/bin"

mkdir -p "$BIN"

FLAGS=(-O2 -std=c++17)

echo "[1/13] test_rules_sanity"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_rules_sanity" "$TESTS/test_rules_sanity.cpp"

echo "[2/13] test_search_staging"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_search_staging" "$TESTS/test_search_staging.cpp"

echo "[3/13] test_move_ordering"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_move_ordering" "$TESTS/test_move_ordering.cpp"

echo "[4/13] test_endgame_race"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_endgame_race" "$TESTS/test_endgame_race.cpp"

echo "[5/13] test_lmr_pvs"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_lmr_pvs" "$TESTS/test_lmr_pvs.cpp"

echo "[6/13] nnue_verify  (paridade C++ vs Python, precisa -pthread)"
g++ "${FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/nnue_verify" "$TESTS/nnue_verify.cpp"

echo "[7/13] nnue_incremental_check  (acumulador incremental vs rebuild do zero)"
g++ "${FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/nnue_incremental_check" "$TESTS/nnue_incremental_check.cpp"

echo "[8/13] nnue_sign_check  (sanidade de sinal/perspectiva do NNUE vs evalSimple)"
g++ "${FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/nnue_sign_check" "$TESTS/nnue_sign_check.cpp"

echo "[9/13] lazy_acc_parity  (Item 3: update preguicoso por perspectiva vs rebuild do zero)"
g++ "${FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/lazy_acc_parity" "$TESTS/lazy_acc_parity.cpp"

# [10..13] Híbrido MCαβ (plan-hybrid-mc-ab.md). Não precisam de -I extra:
# os .cpp incluem "../tools/common/mcab.hpp" relativo a si mesmos.
echo "[10/13] test_search_leaf_smoke  (Fase 0: searchLeaf/resetOrderingState)"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_search_leaf_smoke" "$TESTS/test_search_leaf_smoke.cpp"

echo "[11/13] test_mcab_core  (scoreToQ, budget do pool, sinal do backup, modo equivalência)"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_mcab_core" "$TESTS/test_mcab_core.cpp"

echo "[12/13] test_mcab_dispatch  (SFINAE: refs antigas sem searchLeaf caem no AB puro)"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_mcab_dispatch" "$TESTS/test_mcab_dispatch.cpp"

echo "[13/13] test_mcab_phase9  (reuso de árvore, ruído Dirichlet, leaf depth adaptativa, teto de tempo)"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_mcab_phase9" "$TESTS/test_mcab_phase9.cpp"

echo
echo "OK -- binários em $BIN"
echo "  test_rules_sanity / test_search_staging / test_move_ordering / test_endgame_race / test_lmr_pvs: sem argumentos."
echo "  nnue_verify <pesos_float32.bin> [pesos_int8.bin], ex.:"
echo "    bin/nnue_verify data/nnue/nnue_weights.bin data/nnue/nnue_weights_int8.bin"
echo "  nnue_incremental_check <pesos_int8.bin>: compara acumulador incremental vs"
echo "    rebuild do zero em 30 partidas aleatorias -- deve dar 0 divergencias."
echo "  nnue_sign_check <pesos_int8.bin>: imprime eval NNUE vs evalSimple em 4"
echo "    posicoes de referencia (checagem manual de sinal/perspectiva)."
echo "  lazy_acc_parity <pesos_int8.bin>: compara AccPair (makeChildAccPair,"
echo "    Item 3) vs rebuild do zero em 80 partidas aleatorias -- deve dar 0 mismatches."
echo "  test_search_leaf_smoke / test_mcab_core / test_mcab_dispatch / test_mcab_phase9:"
echo "    sem argumentos. Rodar a partir da RAIZ do repo (carregam"
echo "    data/nnue/nnue_weights_int8.bin)."
