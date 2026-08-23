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
set SELFPLAY=%ROOT%\tools\selfplay
set BENCHMARK=%ROOT%\benchmarks
set BIN=%ROOT%\bin

if not exist "%BIN%" mkdir "%BIN%"

set FLAGS=-O3 -std=c++17 -march=native -mavx2 -mfma

echo [1/9] bench.exe  ^<-  benchmarks\main.cpp
g++ %FLAGS% -I"%SRC%" -I"%SELFPLAY%" -o "%BIN%\bench.exe" "%BENCHMARK%\main.cpp"
if errorlevel 1 goto :erro

echo [2/9] bench_wall_touch_bonus.exe  ^<-  benchmarks\bench_wall_touch_bonus.cpp
g++ %FLAGS% -I"%SRC%" -o "%BIN%\bench_wall_touch_bonus.exe" "%BENCHMARK%\bench_wall_touch_bonus.cpp"
if errorlevel 1 goto :erro

echo [3/9] bench_quiescence_toggle.exe  ^<-  benchmarks\bench_quiescence_toggle.cpp
g++ %FLAGS% -I"%SRC%" -o "%BIN%\bench_quiescence_toggle.exe" "%BENCHMARK%\bench_quiescence_toggle.cpp"
if errorlevel 1 goto :erro

echo [4/9] bench_lmr_pvs.exe  ^<-  benchmarks\bench_lmr_pvs.cpp
g++ %FLAGS% -I"%SRC%" -o "%BIN%\bench_lmr_pvs.exe" "%BENCHMARK%\bench_lmr_pvs.cpp"
if errorlevel 1 goto :erro

REM [5] baseline da Secao 0 de plan-hybrid-mc-ab.md: carga fixa SEM mcab.hpp
REM incluido. Nao estava registrado aqui antes; passa a estar porque o
REM bloco A de bench_mcab.exe so faz sentido comparado com ele.
echo [5/9] bench_fixed_depth.exe  ^<-  benchmarks\bench_fixed_depth.cpp
g++ %FLAGS% -I"%SRC%" -o "%BIN%\bench_fixed_depth.exe" "%BENCHMARK%\bench_fixed_depth.cpp"
if errorlevel 1 goto :erro

echo [6/9] bench_mcab_equivalence.exe  ^<-  benchmarks\bench_mcab_equivalence.cpp
g++ %FLAGS% -I"%SRC%" -o "%BIN%\bench_mcab_equivalence.exe" "%BENCHMARK%\bench_mcab_equivalence.cpp"
if errorlevel 1 goto :erro

echo [7/9] bench_mcab.exe  ^<-  benchmarks\bench_mcab.cpp
g++ %FLAGS% -I"%SRC%" -o "%BIN%\bench_mcab.exe" "%BENCHMARK%\bench_mcab.cpp"
if errorlevel 1 goto :erro

REM [8] inv/ab-policy: compute map (policy/MCTS vs AB time attribution).
echo [8/9] map_compute.exe  ^<-  benchmarks\map_compute.cpp
g++ %FLAGS% -I"%SRC%" -o "%BIN%\map_compute.exe" "%BENCHMARK%\map_compute.cpp"
if errorlevel 1 goto :erro

REM [9] inv/ab-policy: nodes-to-fixed-depth / time-to-depth por variante.
echo [9/9] bench_policy_ab.exe  ^<-  benchmarks\bench_policy_ab.cpp
g++ %FLAGS% -I"%SRC%" -o "%BIN%\bench_policy_ab.exe" "%BENCHMARK%\bench_policy_ab.cpp"
if errorlevel 1 goto :erro

echo.
echo OK -- binarios em %BIN%
echo   bench_fixed_depth.exe / bench_mcab.exe: a CONTAGEM DE NOS do bloco A de
echo     bench_mcab tem que bater exatamente com bench_fixed_depth (Secao 0).
echo   bench_mcab_equivalence.exe [leafDepth] [numPosicoes]
echo   bench_mcab.exe [nodeBudget] [leafDepth]  -- rodar da RAIZ do repo.
exit /b 0

:erro
echo.
echo [ERRO] Falha na compilacao.
exit /b 1
