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
set TESTE=%ROOT%\tests
set BIN=%ROOT%\bin

if not exist "%BIN%" mkdir "%BIN%"

set FLAGS=-O2 -std=c++17

echo [1/17] test_rules_sanity.exe
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_rules_sanity.exe" "%TESTE%\test_rules_sanity.cpp"
if errorlevel 1 goto :erro

echo [2/17] test_search_staging.exe
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_search_staging.exe" "%TESTE%\test_search_staging.cpp"
if errorlevel 1 goto :erro

echo [3/17] test_move_ordering.exe
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_move_ordering.exe" "%TESTE%\test_move_ordering.cpp"
if errorlevel 1 goto :erro

echo [4/17] test_endgame_race.exe
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_endgame_race.exe" "%TESTE%\test_endgame_race.cpp"
if errorlevel 1 goto :erro

echo [5/17] test_lmr_pvs.exe
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_lmr_pvs.exe" "%TESTE%\test_lmr_pvs.cpp"
if errorlevel 1 goto :erro

echo [6/17] nnue_verify.exe  (paridade C++ vs Python, precisa -pthread)
g++ %FLAGS% -pthread -I"%SRC%" -o "%BIN%\nnue_verify.exe" "%TESTE%\nnue_verify.cpp"
if errorlevel 1 goto :erro

echo [7/17] nnue_incremental_check.exe  (acumulador incremental vs rebuild do zero)
g++ %FLAGS% -pthread -I"%SRC%" -o "%BIN%\nnue_incremental_check.exe" "%TESTE%\nnue_incremental_check.cpp"
if errorlevel 1 goto :erro

echo [8/17] nnue_sign_check.exe  (sanidade de sinal/perspectiva do NNUE vs evalSimple)
g++ %FLAGS% -pthread -I"%SRC%" -o "%BIN%\nnue_sign_check.exe" "%TESTE%\nnue_sign_check.cpp"
if errorlevel 1 goto :erro

REM [9..12] Hibrido MCab (plan-hybrid-mc-ab.md). Nao precisam de -I extra:
REM os .cpp incluem "../src/mcab.hpp" relativo a si mesmos.
echo [9/17] test_search_leaf_smoke.exe  (Fase 0: searchLeaf/resetOrderingState)
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_search_leaf_smoke.exe" "%TESTE%\test_search_leaf_smoke.cpp"
if errorlevel 1 goto :erro

echo [10/17] test_mcab_core.exe  (scoreToQ, budget do pool, sinal do backup, modo equivalencia)
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_mcab_core.exe" "%TESTE%\test_mcab_core.cpp"
if errorlevel 1 goto :erro

echo [11/17] test_mcab_dispatch.exe  (SFINAE: refs antigas sem searchLeaf caem no AB puro)
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_mcab_dispatch.exe" "%TESTE%\test_mcab_dispatch.cpp"
if errorlevel 1 goto :erro

echo [12/17] test_mcab_phase9.exe  (reuso de arvore, ruido Dirichlet, leaf depth adaptativa, teto de tempo)
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_mcab_phase9.exe" "%TESTE%\test_mcab_phase9.cpp"
if errorlevel 1 goto :erro

echo [13/17] test_wall_qextension.exe  (inv/qsendgame-ext: caps de quiescencia variaveis)
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_wall_qextension.exe" "%TESTE%\test_wall_qextension.cpp"
if errorlevel 1 goto :erro

REM [14] inv/ab-policy: bit-exactness com toggles off + limiares de acordo
REM com toggles on (direcoes B/C/D + stress).
echo [14/17] test_policy_ab.exe  (inv/ab-policy: defaults bit-exatos, acordo B/C/D)
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_policy_ab.exe" "%TESTE%\test_policy_ab.cpp"
if errorlevel 1 goto :erro

echo [15/17] test_contempt_repetition.exe  (inv/contempt-wandering: sinais de empate, tie-break, semantica de repeticao)
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_contempt_repetition.exe" "%TESTE%\test_contempt_repetition.cpp"
if errorlevel 1 goto :erro

echo [16/17] test_endgame_race_fuzz.exe  (inv/race-fuzz: oracle independente + otimalidade de raiz + budget/cache/degenerados)
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_endgame_race_fuzz.exe" "%TESTE%\test_endgame_race_fuzz.cpp"
if errorlevel 1 goto :erro

echo [17/17] test_mcab_endgame_leaf.exe  (inv/endgame-wander: folha AB de fim de jogo)
g++ %FLAGS% -I"%SRC%" -o "%BIN%\test_mcab_endgame_leaf.exe" "%TESTE%\test_mcab_endgame_leaf.cpp"
if errorlevel 1 goto :erro

echo.
echo OK -- binarios em %BIN%.
echo   test_rules_sanity / test_search_staging / test_move_ordering / test_endgame_race / test_lmr_pvs: sem argumentos.
echo   nnue_verify.exe ^<pesos_float32.bin^> [pesos_int8.bin], ex.:
echo     bin\nnue_verify.exe data\nnue\nnue_weights.bin data\nnue\nnue_weights_int8.bin
echo   nnue_incremental_check.exe ^<pesos_int8.bin^>: acumulador incremental vs rebuild,
echo     deve dar 0 divergencias em 30 partidas aleatorias.
echo   nnue_sign_check.exe ^<pesos_int8.bin^>: eval NNUE vs evalSimple em 4 posicoes
echo     de referencia (checagem manual de sinal/perspectiva).
echo   test_search_leaf_smoke / test_mcab_core / test_mcab_dispatch / test_mcab_phase9:
echo     sem argumentos. Rodar a partir da RAIZ do repo (carregam
echo     data\nnue\nnue_weights_int8.bin).
exit /b 0

:erro
echo.
echo [ERRO] Falha na compilacao.
exit /b 1
