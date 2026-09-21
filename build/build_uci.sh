#!/usr/bin/env bash
# Build the UCI-style text protocol engine.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
BIN="$ROOT/bin"
CXX="${CXX:-g++}"

mkdir -p "$BIN"
"$CXX" -O3 -DNDEBUG -std=c++17 -pthread -Isrc \
  "$ROOT/tools/external/zquoridor_uci.cpp" \
  -o "$BIN/zquoridor"

echo "OK: $BIN/zquoridor"
