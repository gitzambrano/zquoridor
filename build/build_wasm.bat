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
  -s "EXPORTED_FUNCTIONS=[""_malloc"",""_free"",""_qr_new_game"",""_qr_turn"",""_qr_winner"",""_qr_pawn"",""_qr_walls_left"",""_qr_wall_h_bit"",""_qr_wall_v_bit"",""_qr_dist_to_goal"",""_qr_legal_moves_count"",""_qr_legal_move_is_wall"",""_qr_legal_move_a"",""_qr_legal_move_b"",""_qr_legal_move_c"",""_qr_apply_pawn_move"",""_qr_apply_wall_move"",""_qr_engine_move"",""_qr_last_move_is_wall"",""_qr_last_move_a"",""_qr_last_move_b"",""_qr_last_move_c"",""_qr_last_move_eval"",""_qr_is_draw"",""_qr_load_nnue_weights"",""_qr_set_eval_heuristic"",""_qr_eval_mode_is_nnue"",""_qr_set_mcab_enabled"",""_qr_mcab_active"",""_qr_ply_count"",""_qr_cursor"",""_qr_ply_is_wall"",""_qr_ply_a"",""_qr_ply_b"",""_qr_ply_c"",""_qr_goto_ply"",""_qr_truncate_history"",""_qr_scratch_reset"",""_qr_scratch_from_live"",""_qr_scratch_from_ply"",""_qr_scr_apply_pawn"",""_qr_scr_apply_wall"",""_qr_scr_turn"",""_qr_scr_pawn"",""_qr_scr_walls_left"",""_qr_scr_wall_h_bit"",""_qr_scr_wall_v_bit"",""_qr_scr_dist"",""_qr_analyze"",""_qr_an_line_count"",""_qr_an_line_score"",""_qr_an_line_len"",""_qr_an_line_move"",""_qr_an_nodes"",""_qr_an_depth"",""_qr_edit_set_pawn"",""_qr_edit_set_wall"",""_qr_edit_set_walls_left"",""_qr_edit_set_turn"",""_qr_edit_validity"",""_qr_edit_apply"",""_qr_qfen_export"",""_qr_qfen_export_scratch"",""_qr_qfen_import_scratch"",""_qr_last_error""]" ^
  -s "EXPORTED_RUNTIME_METHODS=[""ccall"",""cwrap"",""HEAPU8""]" ^
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
