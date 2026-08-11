#!/usr/bin/env bash
# build_selfplay.sh -- gerador de dados de self-play (multi-thread).
# Equivalente Linux de build_selfplay.bat -- mesmos flags de
# performance (AVX2 explícito + -march=native), é o alvo mais sensível
# a nós/s do projeto.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
SRC="$ROOT/src"
BIN="$ROOT/bin"

mkdir -p "$BIN" "$ROOT/data"

FLAGS=(-O3 -std=c++17 -pthread -march=native -mavx2 -mfma)

echo "selfplay  <-  tools/selfplay/selfplay_main.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -I"$ROOT/tools/selfplay" -o "$BIN/selfplay" "$ROOT/tools/selfplay/selfplay_main.cpp"

echo
echo "OK -- $BIN/selfplay"
echo "Exemplo de uso:"
echo "  $BIN/selfplay --games 2000 --depth 40 --time-ms 100 --threads 8 \\"
echo "      --opening-plies 6 --epsilon 0.25 --out data/selfplay_001.bin"
