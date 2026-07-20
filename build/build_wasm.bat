@echo off
REM build_wasm.bat -- espelho Windows de gui_web/build_wasm.sh.
REM Compila engine_wasm.cpp -> quoridor.js + quoridor.wasm.
REM Requer emsdk ativado no CMD antes de rodar este script:
REM   caminho\pro\emsdk\emsdk_env.bat
setlocal

where emcc >nul 2>nul
if errorlevel 1 (
    echo [ERRO] emcc nao encontrado no PATH.
    echo Rode primeiro:  caminho\pro\emsdk\emsdk_env.bat
    exit /b 1
)

set ROOT=%~dp0..
set GUIWEB=%ROOT%\gui_web

set EXPORTED_FUNCS=['_qr_new_game','_qr_turn','_qr_winner','_qr_pawn','_qr_walls_left','_qr_wall_h_bit','_qr_wall_v_bit','_qr_dist_to_goal','_qr_legal_moves_count','_qr_legal_move_is_wall','_qr_legal_move_a','_qr_legal_move_b','_qr_legal_move_c','_qr_apply_pawn_move','_qr_apply_wall_move','_qr_engine_move','_qr_last_move_is_wall','_qr_last_move_a','_qr_last_move_b','_qr_last_move_c','_qr_last_move_eval']

pushd "%GUIWEB%"

emcc -O3 -std=c++17 ^
  engine_wasm.cpp ^
  -s MODULARIZE=1 ^
  -s EXPORT_NAME=QuoridorModule ^
  -s EXPORTED_FUNCTIONS="%EXPORTED_FUNCS%" ^
  -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap']" ^
  -s ALLOW_MEMORY_GROWTH=1 ^
  -s ENVIRONMENT=web ^
  -o quoridor.js

if errorlevel 1 (
    popd
    echo [ERRO] Falha na compilacao WASM.
    exit /b 1
)

popd

echo.
echo OK -- gerado gui_web\quoridor.js + gui_web\quoridor.wasm
echo WASM nao carrega via file:// -- sirva a pasta com um servidor HTTP:
echo   cd gui_web ^&^& python -m http.server 8000
echo e abra http://localhost:8000/index.html
echo (opcional: python gui_web\build_standalone.py empacota tudo num quoridor.html unico)
exit /b 0
