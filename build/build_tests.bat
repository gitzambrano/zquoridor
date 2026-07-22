@echo off
REM build_tests.bat -- suite de correcao (regras, staging, move ordering,
REM solver de final "maos vazias", paridade NNUE C++ vs Python). Sem
REM -march=native/AVX2: sao testes de corretude/precisao numerica, nao de
REM performance -- mesmo criterio do readme.md.
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

set FLAGS=-O2 -std=c++17

echo [1/5] test_rules_sanity.exe
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_rules_sanity.exe" "%TESTE%\test_rules_sanity.cpp"
if errorlevel 1 goto :erro

echo [2/5] test_search_staging.exe
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_search_staging.exe" "%TESTE%\test_search_staging.cpp"
if errorlevel 1 goto :erro

echo [3/5] test_move_ordering.exe
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_move_ordering.exe" "%TESTE%\test_move_ordering.cpp"
if errorlevel 1 goto :erro

echo [4/5] test_endgame_race.exe
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_endgame_race.exe" "%TESTE%\test_endgame_race.cpp"
if errorlevel 1 goto :erro

echo [5/5] nnue_verify.exe  (paridade C++ vs Python, precisa -pthread)
g++ %FLAGS% -pthread -I"%SRC%" -o "%BIN%\nnue_verify.exe" "%TESTE%\nnue_verify.cpp"
if errorlevel 1 goto :erro

echo.
echo OK -- binarios em %BIN%.
echo   test_rules_sanity / test_search_staging / test_move_ordering / test_endgame_race: sem argumentos.
echo   nnue_verify.exe ^<pesos_float32.bin^> [pesos_int8.bin], ex.:
echo     bin\nnue_verify.exe data\nnue\nnue_weights.bin data\nnue\nnue_weights_int8.bin
exit /b 0

:erro
echo.
echo [ERRO] Falha na compilacao.
exit /b 1
