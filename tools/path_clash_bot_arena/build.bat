@echo off
setlocal
rem build.bat -- equivalente Windows de build.sh: gera dist\bot.js, um
rem UNICO arquivo JavaScript autocontido (WASM embutido via
rem SINGLE_FILE=1, pesos NNUE embutidos via nnue_weights_data.h) que
rem define chooseAction(state) -- pronto pra submissao na Classe A do
rem quoridor-arena. Mesma logica de build/build_wasm.bat (emsdk em
rem C:\emsdk); nao mexe em nada de gui_web\.

call C:\emsdk\emsdk_env.bat >nul 2>&1

cd /d "%~dp0"
if not exist dist mkdir dist

echo Regenerando pesos NNUE embutidos (src\nnue_weights_data.h)...
set PYTHONIOENCODING=utf-8
python tools\gen_weights_header.py ..\data\nnue\nnue_weights_int8.bin src\nnue_weights_data.h
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERRO: falha ao gerar nnue_weights_data.h -- confira se
    echo   ..\data\nnue\nnue_weights_int8.bin existe.
    echo.
    pause
    exit /b 1
)

echo.
echo Compilando engine_bridge.cpp para WASM sincrono...
rem flags identicas as de build.sh de proposito (mesmo binario nos dois
rem SOs) -- SEM -msimd128: se o Node/V8 que a arena roda nao suportar
rem WASM SIMD, o modulo falharia ao instanciar. Sem -Wno-*: e' codigo
rem nosso, avisos do compilador aqui sao sinal, nao ruido.
emcc -O3 -std=c++17 -Isrc src\engine_bridge.cpp ^
  -s MODULARIZE=1 -s EXPORT_NAME=PathClashBotModule ^
  -s "EXPORTED_FUNCTIONS=[""_qr_pca_init"",""_qr_pca_reset_turn"",""_qr_pca_add_wall"",""_qr_pca_choose"",""_qr_pca_legal_pawn_moves"",""_qr_pca_legal_pawn_dest"",""_qr_pca_nnue_loaded""]" ^
  -s "EXPORTED_RUNTIME_METHODS=[""ccall""]" ^
  -s ENVIRONMENT=node ^
  -s WASM_ASYNC_COMPILATION=0 ^
  -s ALLOW_MEMORY_GROWTH=1 ^
  -s INITIAL_MEMORY=134217728 ^
  -s MAXIMUM_MEMORY=805306368 ^
  -s TOTAL_STACK=8388608 ^
  -s SINGLE_FILE=1 ^
  -o dist\bot_core.js
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERRO: compilacao WASM falhou.
    echo.
    pause
    exit /b 1
)

echo.
echo Montando dist\bot.js (WASM + wrapper, nesta ordem, no topo real do arquivo)...
rem "type" concatena preservando bytes; /b nas duas fontes evita que o
rem copy insira um EOF (0x1A) de modo texto no meio do base64 embutido.
copy /b dist\bot_core.js+src\qr_pca_wrapper.js dist\bot.js >nul
del dist\bot_core.js

if not exist dist\bot.js (
    echo.
    echo ERRO: dist\bot.js nao foi gerado.
    echo.
    pause
    exit /b 1
)

echo.
echo SUCCESS: dist\bot.js gerado.
echo Teste local: node tools\test_local.js
echo.
pause
