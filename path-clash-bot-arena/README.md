# path-clash-bot-arena

Adaptador do zquoridor para a **Classe A** (hosted source, JS/Python) do
["path-clash" no quoridor-arena](https://quoridor-arena.sparky.qzz.io/bots/path-clash).

Gera um **único arquivo JavaScript autocontido** (`dist/bot.js`, ~430KB)
que exporta `chooseAction(state)`, com o motor de busca do zquoridor
(NNUE incluída) compilado pra WebAssembly e rodando **sincronamente**
dentro dessa função — sem stdin/stdout, sem rede, sem disco em runtime.

Esta pasta é independente do `gui_web/` (o WASM do jogo em si). Nada
aqui toca nesse build — é um segundo alvo de compilação, com sua própria
ponte C++/WASM voltada especificamente pro protocolo da arena.

## Por quê Classe A e não B

A Classe B (endpoint HTTPS) usaria o motor C++ nativo sem nenhuma
reescrita, mas exige manter um servidor público no ar 24/7. A Classe A
roda no servidor da arena — você só envia o arquivo — mas o runtime é
"chame uma função pura, sem rede, sem disco, 5s por lance incluindo
startup do processo". Reescrever o motor em JS/Python puro perderia toda
a força da busca+NNUE; compilar pra WASM síncrono mantém o motor
original intacto.

## Arquitetura

```
state (JS object, do harness da arena)
   |
   v
qr_pca_wrapper.js  -- chooseAction(state): decodifica o JSON da arena,
   |                  chama o WASM, decodifica a resposta de volta pro
   |                  vocabulario de strings de legal_actions
   v
engine_bridge.cpp  -- reconstroi o State interno a partir de arrays
   |                  simples (sem JSON parsing em C++), roda
   |                  Negamax::chooseMove com o motor real
   v
rules.hpp / search.hpp / nnue.hpp  -- o motor zquoridor, sem modificacoes
```

Os pesos NNUE quantizados (`data/nnue/nnue_weights_int8.bin`) são
embutidos em `src/nnue_weights_data.h` (gerado por
`tools/gen_weights_header.py`) e carregados **da memória** — o container
da arena não tem rede nem disco gravável em runtime, então não dá pra
abrir o `.bin` de um caminho relativo como o motor faz normalmente.

## Mapeamento de coordenadas e jogador (IMPORTANTE)

Deduzido cruzando o exemplo JSON da spec do site com `rules.hpp` — não
há documentação formal disponível publicamente, então isto é uma
inferência de alta confiança, não uma certeza absoluta. Ver seção
"Confiança e riscos conhecidos" abaixo.

- `positions[i]` e `walls[].{row,col}` usam a MESMA convenção
  `(row, col)` 0-indexada que o motor interno (`cellIdx`/`slotIdx`) —
  **sem inversão de eixo**. `row=0` é a borda de partida de
  `player_id 1`; `row=8` a de `player_id 0`.
- **Só o índice do jogador troca**: `player_id` externo `0` corresponde
  ao jogador **interno 1** do motor; `player_id` externo `1` ao jogador
  **interno 0**. Isso foi deduzido comparando `positions`/`goal_rows` do
  exemplo da spec (`positions: [[8,4],[0,4]], goal_rows: [0,8]`) contra
  `GOAL_ROW[]` em `rules.hpp`.
- Muros: `{"dir":"H"/"V", "row", "col"}` mapeiam direto pra
  `slotIdx(row,col)` do motor (orientação H=0/V=1), sem transformação.

## Nomes de lances de peão — auto-detectados, não fixos

A spec não define o vocabulário exato de `MOVE_UP`/`MOVE_DOWN`/
`MOVE_LEFT`/`MOVE_RIGHT` nem das variantes de pulo/diagonal (ex.:
`MOVE_UP_LEFT`, `JUMP_UP`). Em particular, **não há como saber com
certeza se `"UP"` significa linha decrescente ou crescente** sem ver o
protocolo rodando de verdade.

Em vez de fixar um palpite, o wrapper **auto-detecta a convenção a cada
chamada**: o WASM expõe o conjunto geométrico de destinos de peão
legais (`qr_pca_legal_pawn_moves`/`qr_pca_legal_pawn_dest`, sem
ambiguidade nenhuma — vem direto do gerador de lances do motor), e o JS
testa as 4 combinações possíveis de sinal pra UP/DOWN e LEFT/RIGHT,
ficando com a que faz o conjunto decodificado das strings de
`legal_actions` bater exatamente com esse conjunto geométrico. Pulos e
diagonais são decodificados somando os vetores de cada token
reconhecido (`MOVE_UP_LEFT` → UP+LEFT), então cobre qualquer nome
composto por esses quatro rótulos sem precisar adivinhar o vocabulário
inteiro.

Se por algum motivo NENHUMA variante bater perfeitamente (formato
totalmente inesperado), cai no palpite padrão (UP=linha−1) e depois no
fallback por distância; se mesmo assim nada decodificar, devolve
`legal_actions[0]` — nunca um valor fora da lista (que perderia o jogo
na hora).

## Build

Requer [Emscripten](https://emscripten.org/) (`emcc`) no PATH.

**Linux/macOS:**
```bash
cd path-clash-bot-arena
./build.sh
```

**Windows:** (mesma convenção do `build/build_wasm.bat` existente — assume
emsdk instalado em `C:\emsdk`)
```
cd path-clash-bot-arena
build.bat
```

Os dois geram `dist/bot.js`. O script regenera `src/nnue_weights_data.h`
a partir do `.bin` atual toda vez que roda, então nunca fica com pesos
desatualizados.

**Nota sobre o `--post-js` do emcc**: não usamos essa flag de propósito.
No template `MODULARIZE` do emcc 3.1.6, o conteúdo de `--post-js` é
inserido *dentro* do corpo da factory function (antes do `return
Module` dela) — ou seja, só existiria depois que algo chamasse
`PathClashBotModule()` manualmente, quebrando o contrato "chooseAction é
uma função pronta pra usar" que a arena exige. `build.sh`/`build.bat`
compilam só o módulo WASM e depois **concatenam** `qr_pca_wrapper.js`
no fim do arquivo, garantindo que `chooseAction` fique no escopo de
topo real (no Windows via `copy /b`, que concatena bytes sem inserir
EOF de modo texto no meio do base64 embutido).

## Mantendo isso sincronizado com o motor (LEIA se mexeu em nnue.hpp)

**Não há regeneração automática.** `build.sh`/`build.bat` sempre
recompilam do zero quando você os roda — mas nada dispara isso sozinho
quando `rules.hpp`/`search.hpp`/`nnue.hpp` mudam. Não existe git hook,
CI, nem watcher neste repo (nem haveria como um hook rodar
automaticamente sem voce commitar/pushar em algum lugar que o dispare).
Depois de qualquer mudança no motor (principalmente `nnue.hpp` — layout
de pesos, `NUM_FEATURES`/`HIDDEN`/`POLICY_OUT` — ou `rules.hpp`/
`search.hpp`), o fluxo manual é:

1. Rode `./build.sh` (ou `build.bat`) de novo.
2. Rode `node tools/test_local.js`.

O ponto mais frágil é `engine_bridge.cpp::loadNnueFromMemory` — ele
espelha `NNUEWeightsQuant::loadFromFile` (nnue.hpp) **campo a campo, na
mesma ordem**, porque lê os pesos embutidos de um buffer de memória em
vez de abrir o `.bin` do disco (o container da arena não tem
filesystem/rede em runtime). Se alguém adicionar/remover uma cabeça, ou
mudar `NUM_FEATURES`/`HIDDEN`/`POLICY_OUT`, esse loader manual
**não vai acompanhar sozinho** e precisa ser editado à mão pra bater
com a nova ordem de campos de `loadFromFile`.

Isso já aconteceu uma vez de verdade nesta sessão: a remoção da cabeça
auxiliar de imitação da heurística (`wv1_aux`/`bv1_aux`/`wv2_aux`/
`bv2_aux`) mudou o layout do `.bin` (252.820 → 244.464 bytes — a
diferença bate exatamente com o tamanho do bloco removido) e quebraria
silenciosamente o bot se o loader não fosse atualizado junto.

Pra pegar isso automaticamente **na próxima vez que acontecer** (em vez
de descobrir só jogando mal na arena), adicionei duas redes de
segurança:
- `loadNnueFromMemory` agora falha explicitamente (cai pro modo
  heurístico em vez de ler pesos deslocados/lixo) se sobrar ou faltar
  byte depois de ler todos os campos esperados — antes ela não conferia
  isso.
- `tools/test_local.js` tem uma checagem nova, **a primeira da lista**,
  que confirma que a NNUE embutida carregou de verdade (`qr_pca_nnue_
  loaded()`), não que só "não travou". Se essa checagem falhar depois
  de você mexer em `nnue.hpp`, é o sinal de que precisa atualizar
  `loadNnueFromMemory` à mão.

## Teste local

```bash
node tools/test_local.js
```

Roda uma bateria de checagens: o exemplo exato do JSON da spec, os dois
`player_id`, uma parede restringindo o motor, orçamento de tempo (fica
bem abaixo de 5s mesmo incluindo o "cold start" do WASM — no ambiente de
teste, ~4.36s pra um orçamento de 5s), e uma mini-simulação de 8 turnos
alternando os dois lados.

## Submissão no site

1. Rode `./build.sh`.
2. No formulário "Submit bot": **Classe A**, linguagem **JavaScript**,
   cole o conteúdo de `dist/bot.js` (ou faça upload do arquivo).
3. A qualificação é **uma vitória obrigatória** contra `Greedy_bot`
   (lado sorteado). Como o orçamento de busca já fica bem abaixo do
   limite de 5s, isso não deveria ser um problema de tempo — o risco
   real é a decodificação de `legal_actions` (ver seção abaixo).

## Confiança e riscos conhecidos

| Suposição | Confiança | Por quê |
|---|---|---|
| `row`/`col` sem inversão de eixo | Alta | Bate com `GOAL_ROW`/`positions` do exemplo da spec de forma consistente |
| Índice de jogador invertido (`player_id 0` = interno 1) | Alta | Mesma dedução acima, cruzada com `goal_rows` |
| Muros mapeiam direto pra `slotIdx` | Alta | A própria spec descreve `row`/`col` 0–7 "do canto superior esquerdo", igual à convenção interna |
| Vocabulário de `MOVE_*` | Não fixado — auto-detectado em runtime | Sem essa auto-detecção, um palpite errado faria o bot devolver ações erradas sistematicamente |
| Nomes de pulo/diagonal (`MOVE_UP_LEFT` etc.) | Média | Assume composição de tokens direcionais separados por `_`; se o site usar nomes totalmente diferentes (ex. `JUMP_N`), cai no fallback por distância, que ainda devolve algo legal mas não necessariamis o lance ótimo |

**Recomendação**: depois da primeira submissão, seria bom conseguir ver
o replay de pelo menos uma partida (mesmo contra o `Greedy_bot` da
qualificação) pra confirmar visualmente que os lances de pulo/diagonal
(se aparecerem) estão sendo escolhidos como esperado — isso não dá pra
validar sem acesso a uma partida real, já que a página do site é
renderizada via JS e não achei documentação pública indexada sobre o
protocolo desse site especificamente.

## Arquivos

```
path-clash-bot-arena/
  src/
    engine_bridge.cpp       -- ponte WASM sincrona (C++)
    qr_pca_wrapper.js       -- chooseAction(state) (JS)
    nnue_weights_data.h     -- gerado por tools/gen_weights_header.py
  tools/
    gen_weights_header.py   -- .bin -> array C++ embutivel
    test_local.js           -- bateria de testes locais
  build.sh                  -- gera dist/bot.js (Linux/macOS)
  build.bat                 -- gera dist/bot.js (Windows, emsdk em C:\emsdk)
  dist/bot.js               -- ARQUIVO FINAL pra submeter (gerado)
```
