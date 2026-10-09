@echo off
set "PATH=C:\mingw64\bin;%PATH%"
cd /d "%~dp0.."

echo Compilando selfplay.exe (AVX2 + FMA, -O3)...
echo.

if not exist "bin" mkdir "bin"
if not exist "data\selfplay" mkdir "data\selfplay"

g++ -O3 -std=c++17 -pthread -march=native -mavx2 -mfma ^
    -I"src" -I"tools\selfplay" ^
    -o "bin\selfplay.exe" ^
    "tools\selfplay\selfplay_main.cpp"

if %ERRORLEVEL% equ 0 (
    echo.
    echo SUCCESS: bin\selfplay.exe compilado com sucesso!
    echo.
    echo Exemplo de uso com chunks:
    echo   bin\selfplay.exe --games 20000 --chunk-games 2000 --threads 12 --time-ms 200 ^
    echo       --out "data\selfplay\selfplay_{shard:03d}.bin"
    echo.
) else (
    echo.
    echo ERROR: Compilacao falhou. Verifique se MinGW-w64 esta em C:\mingw64\bin
    echo.
)
pause
