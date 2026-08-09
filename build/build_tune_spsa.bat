@echo off
REM build_tune_spsa.bat -- tuner SPSA dos parametros de busca/ordenacao
REM que interagem com a NNUE (contempt, policyOrderScale, catScoreScale)
REM mais o sweep discreto de policyOrderingMinDepth (teste/tune_spsa.cpp).
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
set TESTE=%ROOT%\teste
set BIN=%ROOT%\bin

if not exist "%BIN%" mkdir "%BIN%"

echo tune_spsa.exe  ^<-  teste\tune_spsa.cpp
g++ -O3 -std=c++17 -march=native -mavx2 -mfma -I"%SRC%" -o "%BIN%\tune_spsa.exe" "%TESTE%\tune_spsa.cpp"
if errorlevel 1 goto :erro

echo.
echo OK -- %BIN%\tune_spsa.exe
echo Uso recomendado: python3 teste\run_spsa.py (config no topo do arquivo)
echo Uso direto: bin\tune_spsa.exe --help
echo   default: modo spsa, 40 iteracoes, seed 20260719, sem limite de tempo,
echo   NNUE + policy ordering ligados (data\nnue\nnue_weights_int8.bin).
echo   Salva checkpoint em spsa_checkpoint.txt (retoma sozinho se existir) e
echo   o resultado final em spsa_result.txt. Rodar a partir da raiz do repo.
exit /b 0

:erro
echo.
echo [ERRO] Falha na compilacao.
exit /b 1
