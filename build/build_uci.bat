@echo off
REM Build the UCI-style text protocol engine.
setlocal
cd /d "%~dp0.."
if not exist bin mkdir bin

g++ -O3 -DNDEBUG -std=c++17 -pthread -Isrc tools\external\zquoridor_uci.cpp -o bin\zquoridor.exe
if errorlevel 1 exit /b 1

echo OK: bin\zquoridor.exe
