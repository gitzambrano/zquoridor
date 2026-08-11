#!/usr/bin/env bash
# build.sh -- gera dist/bot.js: um UNICO arquivo JavaScript autocontido
# (WASM embutido como base64 via SINGLE_FILE=1, pesos NNUE embutidos via
# nnue_weights_data.h) que define uma funcao sincrona chooseAction(state)
# -- pronto para submissao na Classe A do quoridor-arena.
#
# Requer emcc no PATH (Emscripten).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/src"
DIST="$HERE/dist"
ROOT="$HERE/.."

mkdir -p "$DIST"

# regenera os pesos NNUE embutidos a partir do .bin atual -- roda sempre,
# e' barato, e garante que o WASM nunca fique com pesos desatualizados
# em relacao a data/nnue/nnue_weights_int8.bin
python3 "$HERE/tools/gen_weights_header.py" \
    "$ROOT/data/nnue/nnue_weights_int8.bin" \
    "$SRC/nnue_weights_data.h"

# node-acorn (pacote emscripten do apt) fica em /usr/share/nodejs, fora
# de qualquer node_modules -- emcc chama acorn-optimizer.js via `node` e
# precisa achar o pacote por NODE_PATH. Prepende (nao sobrescreve) pra
# nao quebrar um NODE_PATH que o ambiente ja tenha configurado.
if [ -d /usr/share/nodejs ]; then
    export NODE_PATH="/usr/share/nodejs${NODE_PATH:+:$NODE_PATH}"
fi


# NAO usamos --post-js aqui: no template MODULARIZE do emcc 3.1.6, o
# conteudo de --post-js e' inserido DENTRO do corpo da factory function
# (antes do "return Module" dela), nao no topo real do arquivo -- ou
# seja, so passaria a existir depois que alguem chamasse
# PathClashBotModule() manualmente, o que quebra o contrato
# "chooseAction(state) e' uma funcao pronta pra usar" exigido pela
# arena. Em vez disso, compilamos so o modulo WASM e concatenamos nosso
# wrapper DEPOIS, no nivel de topo real do arquivo -- assim
# `chooseAction` fica acessivel tanto como function declaration de topo
# quanto via module.exports, e o `PathClashBotModule` (var de topo do
# arquivo gerado pelo emcc) continua visivel pro wrapper referenciar.
emcc -O3 -std=c++17 -I"$SRC" "$SRC/engine_bridge.cpp" \
  -s MODULARIZE=1 -s EXPORT_NAME=PathClashBotModule \
  -s EXPORTED_FUNCTIONS='["_qr_pca_init","_qr_pca_reset_turn","_qr_pca_add_wall","_qr_pca_choose","_qr_pca_legal_pawn_moves","_qr_pca_legal_pawn_dest","_qr_pca_nnue_loaded"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall"]' \
  -s ENVIRONMENT=node \
  -s WASM_ASYNC_COMPILATION=0 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=134217728 \
  -s MAXIMUM_MEMORY=805306368 \
  -s TOTAL_STACK=8388608 \
  -s SINGLE_FILE=1 \
  -o "$DIST/bot_core.js"

cat "$DIST/bot_core.js" "$SRC/qr_pca_wrapper.js" > "$DIST/bot.js"
rm -f "$DIST/bot_core.js"

echo
echo "OK -- $DIST/bot.js ($(du -h "$DIST/bot.js" | cut -f1))"
echo "Teste local: node tools/test_local.js"
