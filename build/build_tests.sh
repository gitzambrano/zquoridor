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

echo "[1/16] test_rules_sanity"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_rules_sanity" "$TESTS/test_rules_sanity.cpp"

echo "[2/16] test_search_staging"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_search_staging" "$TESTS/test_search_staging.cpp"

echo "[3/16] test_move_ordering"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_move_ordering" "$TESTS/test_move_ordering.cpp"

echo "[4/16] test_endgame_race"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_endgame_race" "$TESTS/test_endgame_race.cpp"

echo "[5/16] test_lmr_pvs"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_lmr_pvs" "$TESTS/test_lmr_pvs.cpp"

echo "[6/16] nnue_verify  (paridade C++ vs Python, precisa -pthread)"
g++ "${FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/nnue_verify" "$TESTS/nnue_verify.cpp"

echo "[7/16] nnue_incremental_check  (acumulador incremental vs rebuild do zero)"
g++ "${FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/nnue_incremental_check" "$TESTS/nnue_incremental_check.cpp"

echo "[8/16] nnue_sign_check  (sanidade de sinal/perspectiva do NNUE vs evalSimple)"
g++ "${FLAGS[@]}" -pthread -I"$SRC" -o "$BIN/nnue_sign_check" "$TESTS/nnue_sign_check.cpp"

# [9..12] Híbrido MCαβ (plan-hybrid-mc-ab.md). Não precisam de -I extra:
# os .cpp incluem "../src/mcab.hpp" relativo a si mesmos.
echo "[9/16] test_search_leaf_smoke  (Fase 0: searchLeaf/resetOrderingState)"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_search_leaf_smoke" "$TESTS/test_search_leaf_smoke.cpp"

echo "[10/16] test_mcab_core  (scoreToQ, budget do pool, sinal do backup, modo equivalência)"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_mcab_core" "$TESTS/test_mcab_core.cpp"

echo "[11/16] test_mcab_dispatch  (SFINAE: refs antigas sem searchLeaf caem no AB puro)"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_mcab_dispatch" "$TESTS/test_mcab_dispatch.cpp"

echo "[12/16] test_mcab_phase9  (reuso de árvore, ruído Dirichlet, leaf depth adaptativa, teto de tempo)"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_mcab_phase9" "$TESTS/test_mcab_phase9.cpp"

echo "[13/16] test_wall_qextension  (inv/qsendgame-ext: caps de quiescência variáveis)"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_wall_qextension" "$TESTS/test_wall_qextension.cpp"

echo "[14/16] test_policy_ab  (inv/ab-policy: defaults bit-exatos, acordo B/C/D)"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_policy_ab" "$TESTS/test_policy_ab.cpp"

echo "[15/16] test_contempt_repetition  (inv/contempt-wandering: sinais de empate, tie-break, semântica de repetição)"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_contempt_repetition" "$TESTS/test_contempt_repetition.cpp"

echo "[16/16] test_endgame_race_fuzz  (inv/race-fuzz: oráculo independente + otimalidade de raiz + budget/cache/degenerados)"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/test_endgame_race_fuzz" "$TESTS/test_endgame_race_fuzz.cpp"

echo
echo "OK -- binários em $BIN"
echo "  test_rules_sanity / test_search_staging / test_move_ordering / test_endgame_race / test_lmr_pvs: sem argumentos."
echo "  nnue_verify <pesos_float32.bin> [pesos_int8.bin], ex.:"
echo "    bin/nnue_verify data/nnue/nnue_weights.bin data/nnue/nnue_weights_int8.bin"
echo "  nnue_incremental_check <pesos_int8.bin>: compara acumulador incremental vs"
echo "    rebuild do zero em 30 partidas aleatorias -- deve dar 0 divergencias."
echo "  nnue_sign_check <pesos_int8.bin>: imprime eval NNUE vs evalSimple em 4"
echo "    posicoes de referencia (checagem manual de sinal/perspectiva)."
echo "  test_search_leaf_smoke / test_mcab_core / test_mcab_dispatch / test_mcab_phase9:"
echo "    sem argumentos. Rodar a partir da RAIZ do repo (carregam"
echo "    data/nnue/nnue_weights_int8.bin)."
echo "  test_wall_qextension / test_policy_ab / test_contempt_repetition /"
echo "  test_endgame_race_fuzz: sem argumentos; o fuzz também carrega"
echo "    data/nnue/nnue_weights_int8.bin quando presente."
