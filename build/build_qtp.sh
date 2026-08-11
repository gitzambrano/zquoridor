#!/usr/bin/env bash
# build_qtp.sh -- interface texto tipo UCI (stdin/stdout, um comando por
# linha) usando o conjunto de comandos do protocolo QTP referenciado em
# https://github.com/pavlosdais/Quoridor, com A NOSSA notacao de lance
# (ver cabecalho de src/qtp_main.cpp). Equivalente Linux de
# build_qtp.bat -- mesmos flags (-mavx2 -mfma explicitos alem de
# -march=native, mesmo criterio dos outros builds do projeto).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
SRC="$ROOT/src"
BIN="$ROOT/bin"

mkdir -p "$BIN"

FLAGS=(-O3 -std=c++17 -march=native -mavx2 -mfma)

echo "qtp_engine  <-  tools/qtp/qtp_main.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -o "$BIN/qtp_engine" "$ROOT/tools/qtp/qtp_main.cpp"

echo
echo "OK -- $BIN/qtp_engine"
echo "Uso (todos os 3 argumentos sao opcionais):"
echo "  $BIN/qtp_engine [pesos_nnue.bin] [maxdepth] [time_ms]"
echo "  $BIN/qtp_engine data/nnue/nnue_weights_int8.bin 40 200"
echo
echo "Depois, digite comandos linha a linha (ou aponte um referee/GUI pro"
echo "stdin/stdout do processo). Comece com:"
echo "  list_commands"
echo "  playmove black e8"
echo "  playwall white a3h"
echo "  genmove black"
echo "  showboard"
