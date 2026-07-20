#!/usr/bin/env bash
# build_wasm.sh -- compila gui_web/engine_wasm.cpp para quoridor.js +
# quoridor.wasm. Requer emsdk ativado no shell (mesmo setup do Zchezz):
#   source /caminho/pro/emsdk/emsdk_env.sh
# engine_wasm.cpp inclui ../src/rules.hpp e ../src/search.hpp diretamente
# (caminho relativo a partir de gui_web/, sem -I necessário aqui).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
GUIWEB="$HERE/../gui_web"

if ! command -v emcc >/dev/null 2>&1; then
    echo "[ERRO] emcc não encontrado no PATH." >&2
    echo "Rode primeiro: source /caminho/pro/emsdk/emsdk_env.sh" >&2
    exit 1
fi

EXPORTED_FUNCS='[
  "_qr_new_game","_qr_turn","_qr_winner","_qr_pawn","_qr_walls_left",
  "_qr_wall_h_bit","_qr_wall_v_bit","_qr_dist_to_goal",
  "_qr_legal_moves_count","_qr_legal_move_is_wall","_qr_legal_move_a",
  "_qr_legal_move_b","_qr_legal_move_c","_qr_apply_pawn_move",
  "_qr_apply_wall_move","_qr_engine_move","_qr_last_move_is_wall",
  "_qr_last_move_a","_qr_last_move_b","_qr_last_move_c",
  "_qr_last_move_eval","_qr_is_draw"
]'

cd "$GUIWEB"

emcc -O3 -std=c++17 \
  engine_wasm.cpp \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=ZquoridorModule \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCS}" \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s ENVIRONMENT=web \
  -o zquoridor.js

python3 build_standalone.py

echo "OK: WASM e bundles atualizados (gui_web/zquoridor.html e root index.html para GitHub Pages)"
echo "Commite e suba: git add index.html gui_web/zquoridor.html && git push"
