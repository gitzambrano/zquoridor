@echo off
call C:\emsdk\emsdk_env.bat >nul 2>&1
cd /d "%~dp0..\gui_web"

set "PRELOAD="
if exist "..\data\nnue\nnue_weights_int8.bin" (
    echo Pesos NNUE default encontrados -- embutindo no bundle via --preload-file.
    set "PRELOAD=--preload-file ..\data\nnue\nnue_weights_int8.bin@/data/nnue/nnue_weights_int8.bin"
) else (
    echo AVISO: pesos NNUE default nao encontrados em data\nnue\nnue_weights_int8.bin
    echo   ^(bundle sai sem eles -- rode training\quantize_nnue.py primeiro^)
)

echo Compiling Zquoridor...
emcc -O3 -std=c++17 -msimd128 ^
  -s MODULARIZE=1 -s EXPORT_NAME=ZquoridorModule ^
  -s "EXPORTED_FUNCTIONS=[""_qr_an_depth"",""_qr_an_is_mcab"",""_qr_an_line_count"",""_qr_an_line_len"",""_qr_an_line_move_a"",""_qr_an_line_move_b"",""_qr_an_line_move_c"",""_qr_an_line_move_is_wall"",""_qr_an_line_score"",""_qr_an_line_visits"",""_qr_an_nodes"",""_qr_analyze"",""_qr_apply_pawn_move"",""_qr_apply_wall_move"",""_qr_dist_to_goal"",""_qr_edit_begin"",""_qr_edit_clear"",""_qr_edit_commit"",""_qr_edit_get_qfen"",""_qr_edit_pawn"",""_qr_edit_set_pawn"",""_qr_edit_set_qfen"",""_qr_edit_set_turn"",""_qr_edit_set_walls_left"",""_qr_edit_toggle_wall"",""_qr_edit_turn"",""_qr_edit_validate"",""_qr_edit_wall_h_bit"",""_qr_edit_wall_v_bit"",""_qr_edit_walls_left"",""_qr_engine_move"",""_qr_eval_mode_is_nnue"",""_qr_get_game_text"",""_qr_get_qfen"",""_qr_goto_ply"",""_qr_hist_move_a"",""_qr_hist_move_b"",""_qr_hist_move_c"",""_qr_hist_move_is_wall"",""_qr_hist_mover"",""_qr_history_cursor"",""_qr_history_len"",""_qr_is_draw"",""_qr_is_wall_legal"",""_qr_last_move_a"",""_qr_last_move_b"",""_qr_last_move_c"",""_qr_last_move_eval"",""_qr_last_move_is_wall"",""_qr_legal_move_a"",""_qr_legal_move_b"",""_qr_legal_move_c"",""_qr_legal_move_is_wall"",""_qr_legal_moves_count"",""_qr_load_nnue_weights"",""_qr_mcab_active"",""_qr_move_notation"",""_qr_new_game"",""_qr_path_cell"",""_qr_path_len"",""_qr_pawn"",""_qr_redo"",""_qr_set_eval_heuristic"",""_qr_set_game_text"",""_qr_set_mcab_enabled"",""_qr_set_qfen"",""_qr_static_eval"",""_qr_truncate_here"",""_qr_turn"",""_qr_undo"",""_qr_wall_h_bit"",""_qr_wall_owner"",""_qr_wall_v_bit"",""_qr_walls_left"",""_qr_winner""]" ^
  -s "EXPORTED_RUNTIME_METHODS=[""ccall"",""cwrap"",""UTF8ToString"",""stringToNewUTF8""]" ^
  -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=268435456 ^
  -s MAXIMUM_MEMORY=536870912 -s STACK_SIZE=4194304 ^
  -s "ENVIRONMENT=web,worker" -s NO_EXIT_RUNTIME=1 ^
  -Wno-unused-variable -Wno-unused-but-set-variable -Wno-uninitialized ^
  -Wno-misleading-indentation -Wno-sign-compare -Wno-unused-function -Wno-parentheses ^
  %PRELOAD% ^
  -o zquoridor.js engine_wasm.cpp
if %ERRORLEVEL% equ 0 (
    echo.
    echo SUCCESS: WASM build complete!
    echo.
) else (
    echo.
    echo ERROR: WASM build failed.
    echo.
    pause
    exit /b 1
)

echo Rebuilding bundle...
set PYTHONIOENCODING=utf-8
python build_standalone.py
if %ERRORLEVEL% equ 0 (
    echo.
    echo SUCCESS: gui_web\zquoridor.html atualizado!
    echo Commite e suba o arquivo para servir via GitHub Pages.
    echo.
) else (
    echo.
    echo ERROR: Bundle build failed.
    echo.
    pause
    exit /b 1
)
pause
