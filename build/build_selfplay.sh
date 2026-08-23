#!/usr/bin/env bash
# build_selfplay.sh -- self-play data generator (multi-thread).
# Linux equivalent of build_selfplay.bat -- on x86-64, the same
# performance flags as the .bat (explicit AVX2 plus -march=native). It is
# the target most sensitive to nodes/s in the project. On any other
# architecture the script drops the x86-only flags automatically.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
SRC="$ROOT/src"
BIN="$ROOT/bin"

mkdir -p "$BIN" "$ROOT/data"

FLAGS=(-O3 -std=c++17 -pthread -march=native)

# AVX2 and FMA are x86 extensions. The compiler rejects these two flags on
# other architectures such as AArch64. There, -march=native alone enables
# the local SIMD. Add the two flags only on x86-64 hosts.
case "$(uname -m)" in
    x86_64|amd64) FLAGS+=(-mavx2 -mfma) ;;
esac

echo "selfplay  <-  tools/selfplay/selfplay_main.cpp"
g++ "${FLAGS[@]}" -I"$SRC" -I"$ROOT/tools/selfplay" -o "$BIN/selfplay" "$ROOT/tools/selfplay/selfplay_main.cpp"

echo
echo "OK -- $BIN/selfplay"
echo "Exemplo de uso:"
echo "  $BIN/selfplay --games 2000 --depth 40 --time-ms 100 --threads 8 \\"
echo "      --opening-plies 6 --epsilon 0.25 --out data/selfplay_001.bin"
