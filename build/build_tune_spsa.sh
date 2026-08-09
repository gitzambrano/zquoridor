#!/usr/bin/env bash
# build_tune_spsa.sh -- tuner SPSA dos parametros de busca/ordenacao que
# interagem com a NNUE (contempt, policyOrderScale, catScoreScale) mais o
# sweep discreto de policyOrderingMinDepth (teste/tune_spsa.cpp). NÃO tuna
# mais os pesos de evalSimple (EvalWeights) -- ver comentário no topo do
# .cpp. Normalmente não é necessário rodar este script diretamente: use
# teste/run_spsa.py, que compila sozinho quando o .cpp muda.
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
echo "Uso recomendado: python3 teste/run_spsa.py (config no topo do arquivo)."
echo "Uso direto: bin/tune_spsa --help"
echo "  default: modo spsa, 40 iteracoes, seed 20260719, sem limite de tempo,"
echo "  NNUE + policy ordering ligados (data/nnue/nnue_weights_int8.bin)."
echo "  Salva checkpoint em spsa_checkpoint.txt (retoma sozinho se existir) e"
echo "  o resultado final em spsa_result.txt. Rodar a partir da raiz do repo"
echo "  (ou mover esses dois arquivos junto se rodar de outro diretório)."
