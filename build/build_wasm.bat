@echo off
call C:\emsdk\emsdk_env.bat >nul 2>&1
cd /d "%~dp0..\gui_web"

echo Compiling Zquoridor...
emcc -O3 -std=c++17 -msimd128 ^
  -s MODULARIZE=1 -s EXPORT_NAME=ZquoridorModule ^
  -s "EXPORTED_FUNCTIONS=[""_qr_new_game"",""_qr_turn"",""_qr_winner"",""_qr_pawn"",""_qr_walls_left"",""_qr_wall_h_bit"",""_qr_wall_v_bit"",""_qr_dist_to_goal"",""_qr_legal_moves_count"",""_qr_legal_move_is_wall"",""_qr_legal_move_a"",""_qr_legal_move_b"",""_qr_legal_move_c"",""_qr_apply_pawn_move"",""_qr_apply_wall_move"",""_qr_engine_move"",""_qr_last_move_is_wall"",""_qr_last_move_a"",""_qr_last_move_b"",""_qr_last_move_c"",""_qr_last_move_eval""]" ^
  -s "EXPORTED_RUNTIME_METHODS=[""ccall"",""cwrap""]" ^
  -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=268435456 ^
  -s MAXIMUM_MEMORY=536870912 -s STACK_SIZE=4194304 ^
  -s "ENVIRONMENT=web,worker" -s NO_EXIT_RUNTIME=1 ^
  -Wno-unused-variable -Wno-unused-but-set-variable -Wno-uninitialized ^
  -Wno-misleading-indentation -Wno-sign-compare -Wno-unused-function -Wno-parentheses ^
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
