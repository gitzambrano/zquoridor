#!/usr/bin/env bash
# build_all.sh -- roda os builds nativos (bench/tests/selfplay/tune_spsa)
# em sequência, parando no primeiro erro (set -e via os scripts
# chamados). WASM fica de fora por padrão -- depende do emsdk ativado no
# shell, o que normalmente não é o caso; rode build_wasm.sh à parte
# quando o emsdk estiver pronto, ou passe "wasm" como argumento.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

"$HERE/build_bench.sh"
"$HERE/build_tests.sh"
"$HERE/build_selfplay.sh"
"$HERE/build_tune_spsa.sh"

if [[ "${1:-}" == "wasm" ]]; then
    "$HERE/build_wasm.sh"
fi

echo
echo "==================================================="
echo "Nativo compilado. Binários em $HERE/../bin"
if [[ "${1:-}" != "wasm" ]]; then
    echo "(WASM não incluído -- rode build_wasm.sh separadamente com o"
    echo " emsdk ativado, ou \"build_all.sh wasm\" se já estiver ativado)"
fi
echo "==================================================="
