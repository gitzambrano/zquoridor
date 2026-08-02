# Plano Adicional — zquoridor

**Origem:** análise comparativa entre `zquoridor` e `titaniummachine1/titanium-engine`
(Rust, mesmo domínio — Quoridor) mais técnicas clássicas de motores alpha-beta
(Stockfish e afins). Este documento foi podado em 2026-08: itens já
implementados e validados em produção foram condensados num resumo curto
(seção "Já implementado"); só ficou detalhado o que ainda está pendente.
Se precisar do histórico completo de decisão de algum item já implementado
(alternativas descartadas, números de antes/depois, etc.), ele está no
histórico do git deste arquivo.

**Como usar este documento:** cada prioridade pendente tem (1) o que é,
(2) por que vale a pena aqui, (3) como entra no código atual — arquivo e
função de `zquoridor`. Ordem = prioridade real (maior retorno / menor risco
primeiro, e o que você pediu explicitamente primeiro).

---

## Estado atual (baseline — 2026-08)

- **Busca:** negamax + alpha-beta, TT (2M entradas), iterative deepening
  com aspiration windows, LMR log-log + PVS janela nula (mesmo toggle),
  Reverse Futility Pruning, Late Move Pruning, killer moves + history
  heuristic, CAT (calor de corredor) para ordenar/podar muros, cache de BFS
  por nó + cache LRU de distância por topologia, gates de perft como
  regressão de corretude.
- **Final "mãos vazias":** solver exato de corrida (DP retrógrado em 3
  níveis) plugado em `chooseMove` — inclusive no caso de a própria RAIZ já
  estar em `wallsLeft==(0,0)` (bug de escolha de lance corrigido e coberto
  por teste de regressão dedicado; ver nota no resumo de "Já implementado"
  abaixo, item que NÃO pode ser reintroduzido sem ler a lição registrada
  lá).
- **NNUE:** 332 features esparsas (81+81 posição de peão próprio/oponente,
  64+64 slots de muro H/V, 21+21 buckets de distância BFS one-hot),
  acumulador incremental (diff por lance, não rebuild do zero por nó) com
  variantes float32 (treino) e int8/int16 quantizada (busca), heads duplos
  WL + auxiliar (imita `evalSimple`) + política (209 saídas). Integrada de
  fato na busca (`Negamax::EvalMode::NNUE`, `search.hpp`) — não é mais só
  "treinada mas não plugada".
  - **NNUE é o default de avaliação em `selfplay` e `arena`** (tenta
    `data/nnue/nnue_weights_int8.bin` automaticamente; cai para
    `evalSimple` com aviso se o arquivo não existir; `--heuristic` força o
    modo antigo). **Ainda NÃO é o default no build WASM** — ver Prioridade 1
    abaixo, é a lacuna mais visível hoje.
  - Acumulador incremental **validado**: `teste/nnue_incremental_check.cpp`
    compara incremental vs. rebuild-do-zero em milhares de posições reais,
    0 divergências com os pesos treinados atuais.
  - **Dois problemas reais encontrados e ainda não corrigidos** — ver
    Prioridade 1 (são o motivo dela ser #1 agora, à frente de tudo mais):
    quantidade de muros restantes de cada jogador não é feature de entrada
    nenhuma, e a codificação de perspectiva não é geometricamente
    canônica (produz eval diferente pras duas perspectivas até em posições
    perfeitamente simétricas).
- **Build:** `build_bench.sh/.bat`, `build_selfplay.sh/.bat`,
  `build_tests.sh/.bat` (8 binários, incluindo os dois testes de NNUE
  acima, wireados em 2026-08), `build_arena.sh/.bat` (compila
  `teste/bin/arena.exe` standalone, sem precisar de `run_arena.py`/git —
  novo, resolve a lacuna que existia antes de só dar pra compilar a arena
  via o script Python com dois refs git) e `build_wasm.sh/.bat`.
- **Treino NNUE** (`training/train_nnue.py`, PyTorch — roda em CPU tão bem
  quanto uma implementação numpy pura, então não existe mais script
  separado "sem torch"): resume automático por época via
  `data/checkpoints/train_state.pt` (pesos + otimizador + RNGs + histórico
  - early-stopper), captura de Ctrl+C salvando checkpoint de emergência,
    early stopping com restauração do melhor epoch, LR/weight-decay
    schedules com warmup, orçamento de RAM/VRAM calculando batch/chunk size
    automaticamente. **Atualizado em 2026-08** (ver Prioridade 2: bug de
    `--fresh` corrigido + checkpoint em JSON adicionado).

---

## Prioridade 1 — NNUE: feature de muros restantes + correção da assimetria de perspectiva

**Isto é a prioridade #1 agora, à frente de qualquer outra coisa neste
documento.** Objetivo: ter a NNUE **completamente jogável** (default nos
três alvos de produção — selfplay, arena e WASM — sem as duas lacunas
abaixo) antes de seguir pra qualquer item novo.

### 1a. Muros restantes (`wallsLeft`) não é feature de entrada nenhuma

`grep wallsLeft src/nnue.hpp` não retorna nada — confirmado tanto por
leitura do código quanto empiricamente com `nnue_sign_check.cpp`: a
posição inicial e uma posição idêntica exceto por "jogador 0 com 10 muros,
oponente com 0" dão **exatamente o mesmo eval NNUE**. A rede é cega para
quem ainda pode bloquear — informação estratégica real que hoje só entra
no `evalSimple` (heurística), não na rede. Isso é assim desde a versão de
290 features (documentado na versão antiga deste arquivo) até a atual de
332 — não é regressão, é lacuna de design nunca fechada.

**Como implementar:**

1. Adicionar um bloco de features para `wallsLeft[own]`/`wallsLeft[opp]` em
   `nnue.hpp` — mais simples e barato que um valor escalar cru: one-hot por
   contagem (0..10, 11 buckets cada) ou thermometer encoding (11 features
   binárias "tenho ≥N muros", mais fácil da rede aprender monotonicidade).
   `NUM_FEATURES` sobe de 332 para 332+22=354 (thermometer) ou +22 (one-hot,
   mesmo custo). Atualizar `buildAccumulator`/`buildAccumulatorQuant` (as
   duas, float e quantizada) e o `updateAccumulatorForMove`/
   `updateAccumulatorForMoveQuant` incremental — colocar/remover um muro
   decrementa `wallsLeft` de quem jogou, então isso vira mais um par de
   add/remove feature no diff incremental de cada lance de muro (lance de
   peão não mexe nisso).
2. Atualizar `training/quantize_nnue.py` e `training/train_nnue.py`
   (`NUM_FEATURES`, extração de feature a partir do `State` gravado pelo
   selfplay — conferir se `read_selfplay.py`/o formato `.bin` de selfplay
   já registra `wallsLeft` por posição; se não registrar, é preciso
   também mexer no formato de gravação do `selfplay.hpp`/`selfplay_main.cpp`
   antes de re-gerar dados de treino).
3. **Retreinar do zero** (mudança de `NUM_FEATURES` invalida pesos
   antigos — não dá pra fazer warm-start de um checkpoint com shape
   diferente; o próprio `compute_fingerprint`/`try_load_train_state` já
   detecta isso e recusa o checkpoint incompatível, então o pior caso é só
   um aviso, não corrupção).
4. Validar com `nnue_sign_check` (as duas posições de "10 vs 0 muros"
   devem produzir eval visivelmente diferente agora) e
   `nnue_incremental_check` (continua em 0 divergências com o novo bloco
   de features).

### 1b. Eval assimétrica em posições simétricas (bug de perspectiva)

`nnue_sign_check.cpp` na posição inicial (perfeitamente simétrica) dá
`nnue(persp0)=-3` e `nnue(persp1)=70` — deveriam ser iguais (ou os dois
perto de 0), já que a MESMA situação abstrata "quão bem estou" é simétrica
para os dois lados. Causa raiz identificada: `featOwnPawn`/`featWallH`/
`featWallV` usam coordenada **bruta** do tabuleiro — só trocam qual peão é
"meu" vs. "do oponente" (`s.pawn[me]`/`s.pawn[opp]`), sem espelhar
linha/coluna pra canonicalizar a perspectiva do jogador 1. Isso obriga a
rede a aprender duas "geografias" diferentes (uma para ser jogador 0, outra
para ser jogador 1) usando as mesmas colunas de peso, em vez de ganhar
invariância de perspectiva de graça pela própria codificação — motivo
plausível do desvio observado (nunca vai ser exatamente 0 por acaso; sem
espelhar, só convergiria pra simetria com MUITO mais dados/treino).

**Como implementar:** ao computar features pra perspectiva 1, espelhar a
linha (`row -> N-1-row`) na indexação de célula de peão e de slot de muro
(H e V) antes de montar o índice de feature — perspectiva 0 continua sem
transformação (é a canônica). O bucket de distância BFS já está correto
(usa `shortestPathLen(..., me)`, que já é relativo à identidade do
jogador, não à coordenada crua). **Fazer isso JUNTO com 1a** (mesma
mudança de `NUM_FEATURES`/`buildAccumulator*`, mesmo retreino, mesma
rodada de validação) — não vale a pena treinar duas vezes.

**Validação:** `nnue_sign_check` na posição inicial deve dar `nnue(persp0) == nnue(persp1)` (ou muito próximo — pequeno resíduo de treino é aceitável,
73 unidades de diferença como hoje não é).

**Depois de 1a+1b (retreino), fechar a lacuna dos 3 alvos:**

- **WASM ainda não usa NNUE por padrão** — `qr_new_game()` em
  `engine_wasm.cpp` nunca tenta carregar pesos automaticamente, e
  `gui_web/app.js` nunca chama `qr_load_nnue_weights`/
  `qr_set_eval_heuristic` (confirmado por grep, zero ocorrências) — os
  pesos ficam embutidos no bundle (`build_wasm.sh` já faz `--preload-file`
  se `data/nnue/nnue_weights_int8.bin` existir) mas nunca são carregados.
  Implementar um `autoLoadDefaultNnueOnce()` chamado no primeiro
  `qr_new_game()` (mesmo padrão já usado em `selfplay_main.cpp`/
  `arena.cpp`: tenta, cai pra heurístico com silêncio/log se não achar),
  ou no mínimo uma chamada em `app.js` no boot da página.

**Risco:** médio — mudança de arquitetura (NUM_FEATURES) exige retreino
completo, não é hot-fix; mas a mudança em si (mais features + espelho de
coordenada) é mecânica e bem isolada, com os dois testes de NNUE já
existentes cobrindo regressão.

---

## Prioridade 2 — Protocolo UCI + `.exe` standalone

**O que é.** Um front-end de protocolo (UCI — Universal Chess Interface,
adaptado pra Quoridor, ou um protocolo próprio texto simples equivalente)
rodando sobre o `Negamax`/`search.hpp` já existente, empacotado como um
binário standalone (`.exe`/ELF) que qualquer GUI externa (Arena, CuteChess,
ou uma interface própria) ou script consegue conversar via stdin/stdout —
sem precisar do WASM nem do `gui_web/app.js`.

**Por que vale a pena aqui.** Hoje o motor só é acessível via: (a) API C++
direta (`main.cpp`, `selfplay_main.cpp`, `teste/arena.cpp`), todos com
`main()` fechado pra um propósito específico, ou (b) WASM dentro do
`gui_web`. Não existe um jeito de "conversar" com o motor de fora — nem
pra testar interativamente, nem pra plugar em ferramentas de torneio
padrão do xadrez adaptadas (várias delas já falam UCI/protocolos
similares).

**Como implementar em `zquoridor` (esboço, detalhar quando for a vez):**

1. Novo `src/uci_main.cpp` (ou `teste/uci.cpp` inicialmente, promovido pra
   `src/` quando estabilizar): loop de leitura de linha por stdin,
   comandos mínimos primeiro (`position`, `go`, `stop`, `quit`, `isready`,
   `uci`/`uciok`), depois `setoption` pra expor os toggles que já existem
   (NNUE vs. heurístico, profundidade/tempo).
2. Formato de posição/lance precisa de uma notação textual pra Quoridor
   (protocolo UCI original é xadrez-específico — reaproveitar só o
   ESQUELETO do protocolo: `position`/`go`/`bestmove`/`info`, não os
   tokens de lance de xadrez). Definir essa notação é o primeiro passo
   concreto (ex.: algo como `e3h` para peão, `e3wh`/`e3wv` para muro
   horizontal/vertical — decidir formato antes de codar).
3. Novo alvo de build (`build_uci.sh`/`.bat`, mesmo padrão dos outros)
   gerando um `.exe`/binário standalone.
4. NNUE default (mesma convenção dos outros binários de produção — ver
   Prioridade 1) desde o início deste front-end, não como afterthought.

**Risco:** baixo/médio — é um front-end fino sobre a busca já validada,
não toca `search.hpp`/`rules.hpp`; o risco real está em definir bem a
notação de posição/lance antes de comprometer com um formato.

---

## Backlog (pendente, prioridade menor que 1–2 acima)

Itens abaixo continuam válidos mas ficam depois de NNUE completa + treino

- UCI na ordem de trabalho. Descrição condensada — histórico completo
  (alternativas, números) no git.

- **Continuation/countermove/correction history** (extensão de killers +
  history já implementados) — retorno alto, risco baixo/médio, usa
  infraestrutura que já existe.
- **LUT O(1) para movimento de peão** — só vale depois de medir profiling
  mostrando que `pawnStepMoves` pesa de verdade (peão tem no máx. 5
  destinos/turno, ganho absoluto é bem menor que geração de muro).
- **Certificado formal de vitória** (eval floor + refutação do último
  muro, estilo ACE) — depende de CAT + solver de corrida (já prontos);
  risco médio-alto (certificação incorreta é pior que não ter — exige
  bateria adversarial antes de confiar).
- **Livro de abertura** — subproduto quase de graça do pipeline de
  self-play do NNUE; baixa prioridade até haver volume de partidas.
- **Lazy SMP (paralelismo de busca)** — ganho real de nós/s multi-core,
  mas multiplica o que já existe em vez de corrigir ineficiência por nó;
  fazer depois que a fila de algoritmo (LMR/PVS/RFP/LMP, já prontos) e o
  NNUE (Prioridade 1) estiverem estáveis. Lição registrada de outro
  motor do mesmo domínio: não estreitar a busca da thread principal
  achando que as auxiliares compensam — thread principal faz busca normal
  completa, auxiliares são buscas extras com pequenas variações.
- **Gerência de tempo dedicada** (`timeman.hpp` — alocação por lance
  baseada em tempo restante/incremento/estabilidade do bestmove, não só
  profundidade/deadline fixos) — baixo risco, não toca busca nem regras.
- **Revisão de Zobrist/Undo enxuto** — **recomendação é NÃO fazer**: o
  `zquoridor` copia `State` (não faz make/unmake in-place), então isso só
  se aplicaria com evidência de profiling mostrando que a cópia é gargalo
  real. Mantido só como registro de análise.
- **Pondering** (pensar no tempo do adversário) — nice-to-have de produto
  (GUI), não de motor; depende de timeman + UI de turno; menor retorno da
  lista.
- **Internal Iterative Reduction (IIR)** — técnica Stockfish barata e
  isolada (reduzir profundidade em 1 quando não há lance de TT e
  profundidade ≥ limiar), sem dependência de nenhum outro item.

---

## Já implementado (resumo — histórico completo no git)

Busca: CAT, killer moves + history, LMR log-log + PVS (mesmo toggle),
Reverse Futility Pruning, Late Move Pruning, gates de perft, cache de BFS
por nó + cache LRU de distância por topologia, solver exato de final
"mãos vazias" (DP retrógrado, 3 níveis de serviço).

**Lição que precisa sobreviver a qualquer poda deste documento:** um
atalho que devolve só um VALOR (sem lance) é seguro em qualquer nó
INTERNO da árvore (o pai sempre compara os filhos), mas quebra
silenciosamente se o mesmo atalho puder ser atingido diretamente pela
RAIZ de uma busca real — `chooseMove` precisa de um LANCE, não só um
score. Isso já causou um bug real (motor com nós/s saudável mas perdendo
a maioria das partidas, porque jogava lance arbitrário em vez do ótimo
assim que a própria raiz caía em `wallsLeft==(0,0)`), corrigido e coberto
por `test_endgame_race.cpp::testChooseMoveAtEmptyHandedRootPicksOptimalMove`.
Qualquer atalho parecido no futuro (novo nível de serviço, outro solver
exato de subjogo) precisa de um teste que chame `chooseMove` a partir de
uma posição-RAIZ já dentro da zona do atalho — não só o solver isolado.

Também já implementado (fora da busca): acumulador NNUE incremental
(diff por lance, validado por `nnue_incremental_check`), NNUE plugada de
fato na busca com default automático em selfplay/arena, `build_arena.sh`/
`.bat` standalone, checkpoint/resume de treino (ver Prioridade 2).

---

## Resumo de prioridade (visão rápida)

| # | Item | Retorno esperado | Risco | Status |
| -- | --------------------------------------------------------------------------------------------- | --------------------- | ----------------- | ----------------------------------- |
| 1 | NNUE: feature de muros restantes + correção de perspectiva + default WASM | **Alto** | Médio (retreino) | **pendente — prioridade #1** |
| 2 | Protocolo UCI +`.exe` standalone | Alto (acessibilidade) | Baixo/médio | pendente |
| — | Continuation/countermove/correction history | Alto | Baixo/médio | backlog |
| — | LUT O(1) de peão | Baixo-médio | Baixo | backlog (medir antes) |
| — | Certificado de vitória | Médio | Médio-alto | backlog |
| — | Livro de abertura | Médio | Baixo | backlog |
| — | Lazy SMP | Alto (throughput) | Médio-alto | backlog |
| — | Gerência de tempo dedicada | Médio | Baixo | backlog |
| — | Zobrist/Undo enxuto | N/A | N/A | **não fazer sem medir** |
| — | Pondering | Baixo | Baixo | backlog |
| — | Internal Iterative Reduction (IIR) | Médio | Baixo | backlog |
| — | CAT, killers/history, LMR+PVS, RFP, LMP, perft gates, cache BFS/distância, solver de corrida | — | — | ✅ implementado |
