@echo off
REM build_bench.bat -- benchmarks de performance (nao fazem parte da
REM suite de correcao, medem nos/s e profundidade).
REM Origem dos comandos: readme.md, secao "6. Build & Comandos" ->
REM "Core / benchmarks". -mavx2 -mfma adicionados explicitamente alem
REM de -march=native porque a maquina alvo tem AVX2 garantido (mesmo
REM criterio usado no build Windows do Zchezz) -- redundante com
REM -march=native na mesma maquina, mas documenta a intencao e evita
REM depender so da deteccao automatica.
setlocal

where g++ >nul 2>nul
if errorlevel 1 (
    echo [ERRO] g++ nao encontrado no PATH. Instale MinGW-w64 e adicione ao PATH.
    exit /b 1
)

set ROOT=%~dp0..
set SRC=%ROOT%\src
set TESTE=%ROOT%\teste
set BIN=%ROOT%\bin

if not exist "%BIN%" mkdir "%BIN%"

set FLAGS=-O3 -std=c++17 -march=native -mavx2 -mfma

echo [1/2] bench.exe  ^<-  src\main.cpp
g++ %FLAGS% -I"%SRC%" -o "%BIN%\bench.exe" "%SRC%\main.cpp"
if errorlevel 1 goto :erro

echo [2/2] bench_wall_touch_bonus.exe  ^<-  teste\bench_wall_touch_bonus.cpp
g++ %FLAGS% -I"%SRC%" -o "%BIN%\bench_wall_touch_bonus.exe" "%TESTE%\bench_wall_touch_bonus.cpp"
if errorlevel 1 goto :erro

echo.
echo OK -- binarios em %BIN%
exit /b 0

:erro
echo.
echo [ERRO] Falha na compilacao.
exit /b 1
