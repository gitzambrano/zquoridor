@echo off
REM Build the experimental UCI engine for the local CPU. Do not redistribute.
setlocal
cd /d "%~dp0.."
if not exist bin mkdir bin
g++ -O3 -DNDEBUG -std=c++17 -pthread -march=native -Isrc tools\external\zquoridor_uci.cpp -o bin\zquoridor-native.exe
if errorlevel 1 exit /b 1
echo OK: bin\zquoridor-native.exe
