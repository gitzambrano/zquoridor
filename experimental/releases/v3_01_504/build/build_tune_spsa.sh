#!/usr/bin/env bash
# build_tune_spsa.sh -- tuner genetico (GA) dos parametros de busca/ordenacao que
# interagem com a NNUE (contempt, policyOrderScale, catScoreScale) mais o
# parametros discretos, incluindo policyOrderingMinDepth (teste/tune_spsa.cpp). NÃO tuna
# mais os pesos de evalSimple (EvalWeights) -- ver comentário no topo do
# .cpp. Normalmente não é necessário rodar este script diretamente: use
# teste/run_spsa.py, que compila sozinho quando o .cpp muda.
# Mesmos flags de performance de build_bench.sh (joga partidas reais em loop).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
SRC="$ROOT/src"
SPSA="$ROOT/tools/spsa"
BIN="$ROOT/bin"

mkdir -p "$BIN"

echo "tune_spsa  <-  tools/spsa/tune_spsa.cpp"
g++ -O3 -std=c++17 -march=native -pthread -I"$SRC" -o "$BIN/tune_spsa" "$SPSA/tune_spsa.cpp"

echo
echo "OK -- $BIN/tune_spsa"
echo "Uso recomendado: python3 teste/run_spsa.py (config no topo do arquivo)."
echo "Uso direto: bin/tune_spsa --help"
echo "  default: GA, 30 geracoes, populacao 24, 3 partidas antiteticas/confronto,"
echo "  seed 20260809, sem limite de tempo, NNUE + policy ordering ligados"
echo "  (data/nnue/nnue_weights_int8.bin). --threads N paraleliza as partidas"
echo "  da avaliacao de confrontos. O algoritmo mantem populacao, elitismo, crossover, mutacao e imigrantes."
echo "  Salva checkpoint em ga_checkpoint.txt,"
echo "  historico em ga_history.csv"
echo "  e o resultado final em ga_result.txt. Rodar a partir da raiz do repo"
echo "  (ou mover esses arquivos junto se rodar de outro diretório)."
