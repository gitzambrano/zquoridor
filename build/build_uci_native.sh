#!/usr/bin/env bash
# Build the experimental UCI engine for the local CPU. Do not redistribute.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CXX="${CXX:-g++}"
mkdir -p "$ROOT/bin"
"$CXX" -O3 -DNDEBUG -std=c++17 -pthread -march=native -I"$ROOT/src" \
  "$ROOT/tools/external/zquoridor_uci.cpp" \
  -o "$ROOT/bin/zquoridor-native"
echo "OK: $ROOT/bin/zquoridor-native"
