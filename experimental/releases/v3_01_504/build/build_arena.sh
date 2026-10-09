#!/usr/bin/env bash
# build_arena.sh -- compila teste/arena.cpp como um binário standalone
# (teste/bin/arena.exe), sem precisar do teste/run_arena.py nem de git.
#
# Uso:
#   build_arena.sh                 # engine1 = engine2 = checkout local (auto-teste)
#   build_arena.sh DIR1            # engine2 = DIR1 também
#   build_arena.sh DIR1 DIR2       # cada engine usa o src/ de um checkout diferente
#
# Toda a lógica (incl. o contorno da peculiaridade do GCC com headers
# byte-idênticos escritos no mesmo segundo) vive em build_arena_common.py,
# compartilhada com build_arena.bat -- ver esse arquivo para detalhes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

PYTHON="python3"
command -v python3 >/dev/null 2>&1 || PYTHON="python"

exec "$PYTHON" "$HERE/build_arena_common.py" "$@"
