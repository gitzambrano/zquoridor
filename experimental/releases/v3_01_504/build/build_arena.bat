@echo off
REM build_arena.bat -- compila teste/arena.cpp como um binario standalone
REM (teste\bin\arena.exe), sem precisar do teste\run_arena.py nem de git.
REM
REM Uso:
REM   build_arena.bat                 rem engine1 = engine2 = checkout local (auto-teste)
REM   build_arena.bat DIR1            rem engine2 = DIR1 tambem
REM   build_arena.bat DIR1 DIR2       rem cada engine usa o src\ de um checkout diferente
REM
REM Toda a logica vive em build_arena_common.py, compartilhada com
REM build_arena.sh -- ver esse arquivo para detalhes.
set "PATH=C:\mingw64\bin;%PATH%"
cd /d "%~dp0"
python build_arena_common.py %*
if %ERRORLEVEL% neq 0 (
    exit /b 1
)
exit /b 0
