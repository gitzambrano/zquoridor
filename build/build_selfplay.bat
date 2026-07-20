@echo off
REM build_selfplay.bat -- gerador de dados de self-play (multi-thread).
REM Mesmos flags de performance do bench (AVX2 explicito + -march=native)
REM porque roda em profundidade/tempo real por milhares de partidas --
REM e o alvo mais sensivel a nos/s do projeto.
setlocal

where g++ >nul 2>nul
if errorlevel 1 (
    echo [ERRO] g++ nao encontrado no PATH. Instale MinGW-w64 e adicione ao PATH.
    exit /b 1
)

set ROOT=%~dp0..
set SRC=%ROOT%\src
set BIN=%ROOT%\bin

if not exist "%BIN%" mkdir "%BIN%"
if not exist "%ROOT%\data" mkdir "%ROOT%\data"

set FLAGS=-O3 -std=c++17 -pthread -march=native -mavx2 -mfma

echo selfplay.exe  ^<-  src\selfplay_main.cpp
g++ %FLAGS% -I"%SRC%" -o "%BIN%\selfplay.exe" "%SRC%\selfplay_main.cpp"
if errorlevel 1 goto :erro

echo.
echo OK -- %BIN%\selfplay.exe
echo Exemplo de uso:
echo   %BIN%\selfplay.exe --games 2000 --depth 40 --time-ms 100 --threads 8 ^^
echo       --opening-plies 6 --epsilon 0.25 --out data\selfplay_001.bin
exit /b 0

:erro
echo.
echo [ERRO] Falha na compilacao.
exit /b 1
