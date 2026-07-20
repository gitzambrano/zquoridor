@echo off
REM build_all.bat -- roda os bats nativos (bench/tests/selfplay/tune_spsa)
REM em sequencia, parando no primeiro erro. O WASM fica de fora do "all"
REM por padrao -- depende do emsdk estar ativado no shell, o que nao é
REM o caso na maioria das maquinas/sessoes; rode build_wasm.bat a parte
REM quando o emsdk estiver pronto, ou passe "wasm" como argumento.
setlocal

set HERE=%~dp0

call "%HERE%build_bench.bat"        || exit /b 1
call "%HERE%build_tests.bat"        || exit /b 1
call "%HERE%build_selfplay.bat"     || exit /b 1
call "%HERE%build_tune_spsa.bat"    || exit /b 1

if /I "%~1"=="wasm" (
    call "%HERE%build_wasm.bat" || exit /b 1
)

echo.
echo ===================================================
echo Nativo compilado. Binarios em %HERE%..\bin
if /I not "%~1"=="wasm" (
    echo ^(WASM nao incluido -- rode build_wasm.bat separadamente com o
    echo  emsdk ativado, ou "build_all.bat wasm" se ja estiver ativado^)
)
echo ===================================================
