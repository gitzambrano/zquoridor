@echo off
REM build_qtp.bat -- interface texto tipo UCI (stdin/stdout, um comando
REM por linha) usando o conjunto de comandos do protocolo QTP referenciado
REM em https://github.com/pavlosdais/Quoridor, com A NOSSA notacao de
REM lance (ver cabecalho de src/qtp_main.cpp). Mesmos flags de performance
REM dos outros builds Windows do projeto (AVX2 + FMA, -O3).
set "PATH=C:\mingw64\bin;%PATH%"
cd /d "%~dp0.."

echo Compilando qtp_engine.exe (AVX2 + FMA, -O3)...
echo.

if not exist "bin" mkdir "bin"

g++ -O3 -std=c++17 -march=native -mavx2 -mfma ^
    -I"src" ^
    -o "bin\qtp_engine.exe" ^
    "tools\qtp\qtp_main.cpp"

if %ERRORLEVEL% equ 0 (
    echo.
    echo SUCCESS: bin\qtp_engine.exe compilado com sucesso!
    echo.
    echo Uso (todos os 3 argumentos sao opcionais):
    echo   bin\qtp_engine.exe [pesos_nnue.bin] [maxdepth] [time_ms]
    echo   bin\qtp_engine.exe data\nnue\nnue_weights_int8.bin 40 200
    echo.
    echo Depois, digite comandos linha a linha (ou aponte um referee/GUI
    echo pro stdin/stdout do processo). Comece com:
    echo   list_commands
    echo   playmove black e8
    echo   playwall white a3h
    echo   genmove black
    echo   showboard
    echo.
) else (
    echo.
    echo ERROR: Compilacao falhou. Verifique se MinGW-w64 esta em C:\mingw64\bin
    echo.
)
pause
