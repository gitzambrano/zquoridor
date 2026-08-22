#!/usr/bin/env bash
# build_wasm.sh -- compila gui_web/engine_wasm.cpp para quoridor.js +
# quoridor.wasm. Requer emsdk ativado no shell (mesmo setup do Zchezz):
#   source /caminho/pro/emsdk/emsdk_env.sh
# engine_wasm.cpp inclui ../src/rules.hpp e ../src/search.hpp diretamente
# (caminho relativo a partir de gui_web/, sem -I necessário aqui).
#
# CORREÇÃO: este script estava dessincronizado de build_wasm.bat -- faltavam
# os 3 exports de controle de NNUE (_qr_load_nnue_weights,
# _qr_set_eval_heuristic, _qr_eval_mode_is_nnue), então o app web não tinha
# como ligar NNUE mesmo manualmente via JS, apesar de engine_wasm.cpp já
# exportar essas funções com EMSCRIPTEN_KEEPALIVE. Lista sincronizada com o
# .bat abaixo.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
GUIWEB="$HERE/../gui_web"
ROOT="$HERE/.."
DEFAULT_WEIGHTS="$ROOT/data/nnue/nnue_weights_int8.bin"

if ! command -v emcc >/dev/null 2>&1; then
    echo "[ERRO] emcc não encontrado no PATH." >&2
    echo "Rode primeiro: source /caminho/pro/emsdk/emsdk_env.sh" >&2
    exit 1
fi

EXPORTED_FUNCS='[
  "_qr_an_depth","_qr_an_is_mcab","_qr_an_line_count","_qr_an_line_len",
  "_qr_an_line_move_a","_qr_an_line_move_b","_qr_an_line_move_c",
  "_qr_an_line_move_is_wall","_qr_an_line_score","_qr_an_line_visits",
  "_qr_an_nodes","_qr_analyze","_qr_apply_pawn_move","_qr_apply_wall_move",
  "_qr_dist_to_goal","_qr_edit_begin","_qr_edit_clear","_qr_edit_commit",
  "_qr_edit_get_qfen","_qr_edit_pawn","_qr_edit_set_pawn","_qr_edit_set_qfen",
  "_qr_edit_set_turn","_qr_edit_set_walls_left","_qr_edit_toggle_wall",
  "_qr_edit_turn","_qr_edit_validate","_qr_edit_wall_h_bit",
  "_qr_edit_wall_v_bit","_qr_edit_walls_left","_qr_engine_move",
  "_qr_eval_mode_is_nnue","_qr_get_game_text","_qr_get_qfen","_qr_goto_ply",
  "_qr_hist_move_a","_qr_hist_move_b","_qr_hist_move_c",
  "_qr_hist_move_is_wall","_qr_hist_mover","_qr_history_cursor",
  "_qr_history_len","_qr_is_draw","_qr_is_wall_legal","_qr_last_move_a",
  "_qr_last_move_b","_qr_last_move_c","_qr_last_move_eval",
  "_qr_last_move_is_wall","_qr_legal_move_a","_qr_legal_move_b",
  "_qr_legal_move_c","_qr_legal_move_is_wall","_qr_legal_moves_count",
  "_qr_load_nnue_weights","_qr_mcab_active","_qr_move_notation",
  "_qr_new_game","_qr_path_cell","_qr_path_len","_qr_pawn","_qr_redo",
  "_qr_set_eval_heuristic","_qr_set_game_text","_qr_set_mcab_enabled",
  "_qr_set_qfen","_qr_static_eval","_qr_truncate_here","_qr_turn",
  "_qr_undo","_qr_wall_h_bit","_qr_wall_owner","_qr_wall_v_bit",
  "_qr_walls_left","_qr_winner"
]'

cd "$GUIWEB"

PRELOAD_ARGS=()
if [[ -f "$DEFAULT_WEIGHTS" ]]; then
    echo "[*] Pesos NNUE default encontrados ($DEFAULT_WEIGHTS) -- embutindo no bundle via --preload-file."
    echo "    No FS virtual do WASM eles ficam disponíveis em /data/nnue/nnue_weights_int8.bin;"
    echo "    chame qr_load_nnue_weights(\"/data/nnue/nnue_weights_int8.bin\") do JS para ativar NNUE"
    echo "    (o módulo nasce em modo heurístico -- ver comentário em engine_wasm.cpp)."
    PRELOAD_ARGS=(--preload-file "$DEFAULT_WEIGHTS@/data/nnue/nnue_weights_int8.bin")
else
    echo "[!] Aviso: pesos NNUE default não encontrados em $DEFAULT_WEIGHTS -- bundle sairá sem eles"
    echo "    (rode training/quantize_nnue.py primeiro, ou o app web fica só no modo heurístico)."
fi

emcc -O3 -std=c++17 -msimd128 \
  engine_wasm.cpp \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=ZquoridorModule \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCS}" \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString","stringToNewUTF8"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s ENVIRONMENT=web,worker \
  "${PRELOAD_ARGS[@]}" \
  -o zquoridor.js

python3 build_standalone.py

echo "OK: WASM e bundles atualizados (gui_web/zquoridor.html e root index.html para GitHub Pages)"
echo "Commite e suba: git add index.html gui_web/zquoridor.html && git push"
