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
  "_malloc","_free","_qr_new_game","_qr_turn","_qr_winner","_qr_pawn","_qr_walls_left",
  "_qr_wall_h_bit","_qr_wall_v_bit","_qr_dist_to_goal",
  "_qr_legal_moves_count","_qr_legal_move_is_wall","_qr_legal_move_a",
  "_qr_legal_move_b","_qr_legal_move_c","_qr_apply_pawn_move",
  "_qr_apply_wall_move","_qr_engine_move","_qr_last_move_is_wall",
  "_qr_last_move_a","_qr_last_move_b","_qr_last_move_c",
  "_qr_last_move_eval","_qr_is_draw",
  "_qr_load_nnue_weights","_qr_set_eval_heuristic","_qr_eval_mode_is_nnue",
  "_qr_set_mcab_enabled","_qr_mcab_active",
  "_qr_ply_count","_qr_cursor","_qr_ply_is_wall","_qr_ply_a","_qr_ply_b",
  "_qr_ply_c","_qr_goto_ply","_qr_truncate_history",
  "_qr_scratch_reset","_qr_scratch_from_live","_qr_scratch_from_ply",
  "_qr_scr_apply_pawn","_qr_scr_apply_wall","_qr_scr_turn","_qr_scr_pawn",
  "_qr_scr_walls_left","_qr_scr_wall_h_bit","_qr_scr_wall_v_bit","_qr_scr_dist",
  "_qr_analyze","_qr_an_line_count","_qr_an_line_score","_qr_an_line_len",
  "_qr_an_line_move","_qr_an_nodes","_qr_an_depth",
  "_qr_edit_set_pawn","_qr_edit_set_wall","_qr_edit_set_walls_left",
  "_qr_edit_set_turn","_qr_edit_validity","_qr_edit_apply",
  "_qr_qfen_export","_qr_qfen_export_scratch","_qr_qfen_import_scratch",
  "_qr_last_error"
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
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s ENVIRONMENT=web \
  "${PRELOAD_ARGS[@]}" \
  -o zquoridor.js

python3 build_standalone.py

echo "OK: WASM e bundles atualizados (gui_web/zquoridor.html e root index.html para GitHub Pages)"
echo "Commite e suba: git add index.html gui_web/zquoridor.html && git push"
