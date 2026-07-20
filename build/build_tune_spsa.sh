#!/usr/bin/env bash
# build_tune_spsa.sh -- tuner SPSA dos 6 pesos de evalSimple (teste/tune_spsa.cpp).
# Ainda não rodado até o fim nesta entrega: robustnessWeight em rules.hpp
# continua no valor placeholder (0.80) -- ver Seção 5 (Fase A) do readme.
# Mesmos flags de performance de build_bench.sh (joga partidas reais em loop).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
SRC="$ROOT/src"
TESTE="$ROOT/teste"
BIN="$ROOT/bin"

mkdir -p "$BIN"

echo "tune_spsa  <-  teste/tune_spsa.cpp"
g++ -O3 -std=c++17 -march=native -I"$SRC" -o "$BIN/tune_spsa" "$TESTE/tune_spsa.cpp"

echo
echo "OK -- $BIN/tune_spsa"
echo "Uso: bin/tune_spsa <iteracoes> [seed] [orcamento_segundos]"
echo "  default: 40 iteracoes, seed 20260719, sem limite de tempo."
echo "  Salva checkpoint em spsa_checkpoint.txt (retoma sozinho se existir) e"
echo "  o resultado final em spsa_result.txt. Rodar a partir da raiz do repo"
echo "  (ou mover esses dois arquivos junto se rodar de outro diretório)."
