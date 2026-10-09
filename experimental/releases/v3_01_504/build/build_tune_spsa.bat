@echo off
REM build_tune_spsa.bat -- tuner genetico (GA) dos parametros de busca/ordenacao
REM que interagem com a NNUE (contempt, policyOrderScale, catScoreScale)
REM mais o parametros discretos, incluindo policyOrderingMinDepth (teste/tune_spsa.cpp).
REM NAO tuna mais os pesos de evalSimple (EvalWeights). Normalmente nao e
REM necessario rodar este script diretamente: use teste/run_spsa.py, que
REM compila sozinho quando o .cpp muda. Mesmos flags de performance de
REM build_bench.bat (joga partidas reais em loop).
setlocal

where g++ >nul 2>nul
if errorlevel 1 (
    echo [ERRO] g++ nao encontrado no PATH. Instale MinGW-w64 e adicione ao PATH.
    exit /b 1
)

set ROOT=%~dp0..
set SRC=%ROOT%\src
set SPSA=%ROOT%\tools\spsa
set BIN=%ROOT%\bin

if not exist "%BIN%" mkdir "%BIN%"

echo tune_spsa.exe  <-  tools\spsa\tune_spsa.cpp
g++ -O3 -std=c++17 -march=native -mavx2 -mfma -pthread -I"%SRC%" -o "%BIN%\tune_spsa.exe" "%SPSA%\tune_spsa.cpp"
if errorlevel 1 goto :erro

echo.
echo OK -- %BIN%\tune_spsa.exe
echo Uso recomendado: python3 teste\run_spsa.py (config no topo do arquivo)
echo Uso direto: bin\tune_spsa.exe --help
echo   default: GA, 30 geracoes, populacao 24, 3 partidas antiteticas/confronto,
echo   seed 20260809, sem limite de tempo, NNUE + policy ordering ligados
echo   (data\nnue\nnue_weights_int8.bin). --threads N paraleliza as partidas
echo   das partidas de cada confronto.
echo   Salva checkpoint em ga_checkpoint.txt,
echo   historico em ga_history.csv (python3 teste\plot_spsa.py pra plotar)
echo   e o resultado final em ga_result.txt. Rodar a partir da raiz do repo.
exit /b 0

:erro
echo.
echo [ERRO] Falha na compilacao.
exit /b 1
