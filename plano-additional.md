# Plano Adicional — Melhorias de Engine (titanium-engine, Stockfish e práticas gerais)

**Origem:** análise comparativa entre `zquoridor` e `titaniummachine1/titanium-engine`
(Rust, mesmo domínio — Quoridor), mais técnicas clássicas de motores alpha-beta
(Stockfish e afins) adaptadas para um jogo sem capturas/material.

**Como usar este documento:** cada item tem (1) o que é, (2) por que vale a pena
aqui, (3) como entra no código atual — arquivo e função de `zquoridor`, sem
scope creep: cada item é uma mudança cirúrgica isolada, testável sozinha antes
da próxima. Itens marcados **[JÁ ALINHADO]** descrevem algo que o `zquoridor`
já faz de forma equivalente — mantidos aqui só para registrar a comparação e
apontar o que falta de cada um.

Ordem = prioridade (maior retorno / menor risco primeiro). Custo e risco de
regressão sobem conforme desce a lista.

---

**Atualização (v17/v18):** a primeira versão deste documento foi escrita em
cima da `main` do titanium-engine tal como o `STATE.md` a descrevia — que já
estava defasado. Uma varredura do histórico completo (`git log`, 178 commits,
mais de 15 branches `v16`/`v17`/`v18`/`experiment/*`/`codex/*`) mostrou bem
mais técnica shippada do que a documentação interna deles registra. As
seções abaixo foram expandidas com isso — itens novos marcados **[NOVO —
v17/v18]**, refinamentos de itens já existentes marcados **[REFINADO]**.

---

## Estado atual (baseline, para referência ao longo do documento)

- Geração de muro: pré-filtro de 2 BFS (`shortestPathTouchSlots`) + DSU
  (`RollbackDSU`/`buildWallDSU`/`wallCandidateAmbiguous`) sobre o grafo dual de
  muros já colocados, caindo em `hasPathToGoal` (BFS exato) só nos candidatos
  ambíguos. — **[JÁ ALINHADO]** com o objetivo do pipeline L1→L2→TOPO→L3 do
  titanium-engine (ver Prioridade 6).
- Busca: negamax + alpha-beta, TT (`TT_BITS=21`, 2M entradas), iterative
  deepening com aspiration windows, lance da TT testado antes de gerar muros
  (`isWallMoveLegal` de 1 candidato), quiescência restrita a muros "críticos"
  (aumentam a distância do oponente ou derrubam `pathRobustness`).
- Eval: `evalSimpleW` (distância + mobilidade + pesos tunáveis via SPSA,
  `tune_spsa.cpp`).
- NNUE: 290 features esparsas (81×2 posições de peão + 128 slots de muro),
  heads duplos valor/política planejados, bootstrap de self-play planejado.
- Benchmark: ~540K nós/s em profundidade 14–15; lição metodológica registrada
  de que self-play não serve para comparar move-ordering (precisa de posições
  fixas).

---

## Prioridade 1 — CAT (Corridor Attention Table): calor de corredor para ordenar e podar muros — **[IMPLEMENTADO]**

> **Status:** implementado em `src/cat.hpp` (`computeCorridorHeat`,
> `wallEdgeHeat`), integrado em `orderWallMoves` (`search.hpp`),
> substituindo o antigo `WALL_TOUCH_BONUS` binário. Rodou como **só
> ordenação**, sem poda (conforme o rollout recomendado abaixo). Testado
> em `test_move_ordering.cpp` (forma do calor + ranking). Benchmark
> ad-hoc (`bench_wall_touch_bonus.cpp`, 40 posições fixas, 200ms/lance):
> **~11,7× menos nós** (2,52M vs 29,54M) até profundidade média
> ligeiramente maior (22,60 vs 22,32) do que a versão sem o sinal. Poda
> de muros frios (o segundo passo mencionado no rollout) **ainda não
> implementada** — próximo item natural de CAT, mas fora do escopo desta
> rodada.

**O que é.** Uma tabela de "calor" por casa e por aresta de muro, calculada por
jogador a cada nó de profundidade suficiente, baseada em quanto cada casa
desvia do caminho mais curto atual:

```
delta(sq)  = dist_do_peão(sq) + dist_até_meta(sq) - dist_mais_curta_total
heat(sq)   = round( 200 / (1 + delta × log2(delta + 2)) )   se delta ≤ 3, senão 0
           + 40 (bônus de gargalo) se delta ≤ 2 e a casa só tem 1 continuação
```

Para um muro em `(r, c, orientação)`, o "calor da aresta" é o máximo do calor
das duas casas que ele toca, mais 1/4 do menor dos dois. Isso captura **rotas
quase-equivalentes** que um único caminho testemunho (o que `shortestPathLen`
já devolve) não vê — o titanium-engine documentou isso como a principal falha
do esquema anterior (v2 → v3): um muro que fecha um desvio de custo igual, mas
fora do caminho testemunho, passava despercebido.

**Por que vale a pena aqui.** O gargalo de Quoridor é sempre o fator de
ramificação de muros (até 128 candidatos por turno). O `zquoridor` já corta
grande parte disso com prefiltro BFS + DSU (legalidade), mas isso decide só
**se** um muro é legal — não decide **quão importante** ele é para ordenação
ou para poda de busca (LMR, prune de muros irrelevantes). CAT ataca
exatamente essa segunda pergunta.

**Como implementar em `zquoridor`.**
1. Novo arquivo `src/cat.hpp`. Função `computeCorridorHeat(wallsH, wallsV, pawnCell, player) -> array<int,81>`
   — reaproveita a mesma BFS multi-fonte que `shortestPathTouchSlots` já roda
   duas vezes (distância a partir do peão + distância até a meta, por casa);
   não é BFS nova, é o mesmo dado já calculado, só que mantendo o vetor de
   distância completo em vez de descartá-lo depois de extrair os slots
   tocados. Ou seja: **primeiro, refatorar `shortestPathTouchSlots` para
   também expor o vetor de distância cru** (ver Prioridade 6, cache por nó —
   os dois itens compartilham a mesma BFS de origem).
2. `wallEdgeHeat(heatP0, heatP1, orientation, r, c) -> int` — máximo das duas
   casas tocadas pelo muro, dos dois jogadores.
3. Em `search.hpp`, no Estágio 3 (geração de muros dentro de `negamax`):
   usar `wallEdgeHeat` como chave de ordenação (maior calor primeiro, depois
   os "sem calor" — candidatos que passaram no prefiltro BFS sem tocar
   ninguém, esses continuam sendo tentados por último ou descartados de LMR
   fundo).
4. Poda: muros com `wallEdgeHeat == 0` e profundidade restante pequena podem
   ser pulados inteiramente em nós não-PV (ver `wall_should_search` do
   titanium — regra: manter se cruza o caminho testemunho do oponente **ou**
   se tem calor ≥ limiar "quente"; descartar prova de zona morta/selada).
5. Constantes sugeridas (mesmos valores do titanium-engine como ponto de
   partida, re-tunáveis depois via SPSA como já é feito para `EvalWeights`):
   `CAT_CORRIDOR_CM=200`, `CAT_HOT_CM=160`, `CAT_COLD_CM=60`,
   `BOTTLENECK_BONUS_CM=40`.

**Risco / ordem de rollout.** Implementar primeiro **só como ordenação**
(sem poda) e medir com o protocolo já estabelecido no projeto (benchmark de
posição fixa, nunca self-play) se reduz nós para mesma profundidade. Só
depois de confirmado, ativar a poda de muros frios como uma segunda mudança
isolada e testável.

---

## Prioridade 2 — Killer moves + history heuristic

**O que é.** Duas tabelas clássicas de ordenação de Stockfish/qualquer
alpha-beta sério, que o titanium-engine **ainda não tem** ("_(not yet)_ — CAT
corridor heat is our positional memory" — eles usam CAT como substituto
parcial, mas killer/history são complementares, não excludentes):

- **Killer moves:** por `ply`, guardar os 1–2 lances que causaram corte
  beta (fail-high) na última vez que essa profundidade foi visitada,
  independente da posição exata. Testados logo depois do lance da TT.
- **History heuristic:** tabela `history[player][from_kind][to] += depth*depth`
  incrementada toda vez que um lance causa corte beta; usada para desempatar
  ordenação entre lances "quietos" (sem calor CAT, sem TT-hit).

**Por que vale a pena aqui.** Quoridor não tem MVV-LVA (não há capturas), então
a ordenação hoje depende de: TT move → calor CAT (Prioridade 1) → resto.
Killer/history são baratos (arrays simples, sem BFS) e cobrem exatamente o
"resto" — lances de peão e muros frios que se repetem como boas respostas em
posições irmãs (transposições da mesma sub-árvore).

**Como implementar em `zquoridor`.**
1. Em `Negamax` (`search.hpp`): adicionar `int killers[MAX_PLY][2]` (guardar
   `moveToPolicyIndex(m)` ou um hash compacto do `Move`) e
   `int history[2][N*N][WS*WS*2]` (ou dimensão mais enxuta — indexar por
   destino do peão e por slot do muro separadamente, não pelo produto
   cartesiano completo).
2. Atualizar ambos só no ramo de corte beta (`localAlpha >= beta`) do laço de
   muros e do laço de peão dentro de `negamax`, nunca em `quiescence`
   (quiescência já tem seu próprio filtro de "crítico").
3. Ordenação: TT move (já existe) → killers do ply atual, se legais → maior
   calor CAT (Prioridade 1, se já implementada) → maior `history[...]` →
   resto em ordem arbitrária estável.

**Risco.** Baixíssimo — é aditivo e não muda legalidade nem eval. Reversível
com uma flag de compilação enquanto mede.

### 2b — **[NOVO — v17/v18]** Histórico completo à la Stockfish: continuation history, countermove, correction history

O código de busca do titanium (`titanium/search/search_impl.rs`, ramo `v17`)
não para no history simples do item 2 — ele porta praticamente **todo** o
conjunto de tabelas de histórico do Stockfish moderno:

- **Fórmula de gravidade** (a mesma do Stockfish) para atualizar qualquer
  tabela de histórico: `h += bonus − h·|bonus|/HIST_MAX` — auto-saturante,
  nunca estoura o range, e decai sozinho com o tempo sem precisar de reset
  manual.
- **Malus, não só bônus:** todo lance testado **antes** do corte beta num nó
  recebe o bônus negativo (é rebaixado), não só o lance que causou o corte é
  premiado. Isso é o que faz o histórico convergir rápido — sem malus, um
  lance que "parecia bom" por acaso nunca é corrigido.
- **Countermove heuristic** (`cm`, indexado pelo "código de histórico denso"
  do lance anterior): guarda qual resposta causou corte beta contra cada
  lance do oponente, testada logo depois dos killers.
- **Continuation history** (`cont_hist`, tabela `[lance_anterior][lance_atual]`
  completa, não só o melhor par como o countermove): pontua "esse muro é uma
  boa resposta a esse lance anterior", cobrindo o par de lances inteiro em
  vez de só o par vencedor. Custo: `HIST_SPAN² × 4 bytes` (eles documentam
  ~256 KiB — barato).
- **Correction history** (`corr_hist[lado][hash_da_estrutura_de_muros]`):
  EMA (média móvel exponencial, clamped ±256) do erro entre o score de busca
  e o eval estático, por lado e por "assinatura" da estrutura de muros no
  tabuleiro. Serve pra **corrigir o viés sistemático do eval estático**
  online, sem retreinar nada — se o `evalSimpleW`/NNUE consistentemente
  superestima um certo tipo de topologia de muros, essa tabela aprende isso
  durante a própria partida e ajusta.
- **Flag `improving`** (Stockfish): compara o eval estático do nó atual com o
  de 2 plies atrás (mesmo lado a mover); se está melhorando, reduções/margens
  ficam mais conservadoras (menos agressivas), porque a posição pode estar
  em transição boa.

**Como implementar em `zquoridor`.**
1. Trocar a atualização simples de `history[...] += depth*depth` (item 2) pela
   fórmula de gravidade acima — é a mesma tabela, só muda a fórmula de
   update, mudança de uma linha.
2. Adicionar malus: no laço de muros/peão de `negamax`, todo lance tentado
   **antes** do que causa o corte beta recebe `bonus = -depth*depth` na mesma
   tabela.
3. `cm[lance_anterior] = lance_que_causou_corte` — array simples do tamanho
   do espaço de lances, atualizado só no corte beta.
4. `cont_hist` como `vector<int>` plano `[lance_anterior * N_MOVES + lance_atual]`
   — mesmo tamanho pequeno citado pelo titanium, cabe tranquilo.
5. `corr_hist[side][hash]`: reaproveitar/gerar um hash barato da configuração
   de muros (pode ser um subconjunto do próprio Zobrist já calculado em
   `State.hash`, reduzido pra um índice pequeno tipo 12–14 bits) — atualizado
   com `search_score - evalSimpleW(...)` depois de cada busca completa de nó,
   lido como correção aditiva ao eval estático em `evalSimpleW`/na chamada do
   NNUE.
6. `improving`: guardar o eval estático por ply (`eval_stack[MAX_PLY]` igual
   ao titanium) e comparar com `ply - 2`.

**Risco.** Baixo a médio — cada peça é isolada e testável separadamente
(ativar uma de cada vez, medir nós/qualidade antes de acumular a próxima).
`corr_hist` é a mais delicada (mexe no eval, não só na ordenação) — testar
por último dessa sublista.

---

## Prioridade 3 — LMR (Late Move Reduction) explícito com fórmula log-log

**O que é.** Reduzir a profundidade de busca de lances tardios na ordenação
(que já foram ordenados por importância, então lances tardios são
provavelmente ruins) e re-buscar em profundidade cheia só se o resultado
reduzido superar alpha. Fórmula usada pelo titanium-engine (mesma família da
usada em Stockfish, adaptada):

```
reduction = clamp( round( ln(depth) * ln(move_index) / 2.25 ), 0, depth / 2 )
```

Com reduções extras (+1) para muros "frios" (calor CAT abaixo de
`CAT_COLD_CM`) e reduções puladas (0) para lances "quentes" (`≥ CAT_HOT_CM`,
"tático — pula LMR").

**Por que vale a pena aqui.** O `zquoridor` já tem TT + aspiration + iterative
deepening, mas a lista atual de negamax não menciona LMR — é o item clássico
que falta na progressão natural TT → aspiration → **LMR** → PVS (Prioridade
8). Sem calor CAT (Prioridade 1) ainda dá pra fazer uma versão simples usando
só o índice de ordenação; fica mais forte depois de 1 estar pronto.

**Como implementar em `zquoridor`.**
1. Em `negamax`, depois do lance da TT e antes/durante o laço principal:
   contar `moveIndex` (1-based) conforme os lances são tentados.
2. Para `moveIndex > LMR_MIN_MOVE_INDEX` (ex.: 3) e `depth >= LMR_MIN_DEPTH`
   (ex.: 3) e o lance não for o da TT nem um killer (Prioridade 2, se pronta):
   calcular `reduction` pela fórmula acima, buscar com
   `negamax(ns, depth - 1 - reduction, -alpha-1, -alpha, stats)` (janela nula,
   ver Prioridade 8), e se `score > alpha`, re-buscar completo
   `negamax(ns, depth - 1, -beta, -alpha, stats)`.
3. **Cuidado específico de Quoridor:** diferente de xadrez, aqui não existe
   "lance de xeque" que precise ficar isento de redução por segurança tática
   simples — mas um muro que muda a topologia de zona morta pode. Usar a
   mesma regra do titanium: nunca reduzir lances que a checagem de
   "quiescência crítica" (já existente em `quiescence`) marcaria como
   crítico.

**Risco.** Médio — LMR mal calibrado pode perder linhas táticas finas
(gargalos de 1 muro). Testar com o mesmo protocolo de posições fixas já
estabelecido, comparando nós-para-mesma-profundidade e, principalmente,
**qualidade do lance escolhido** em posições de gargalo conhecidas antes de
aceitar.

### 3b — **[NOVO — v17/v18]** Reverse Futility Pruning (RFP / static null move)

**O que é.** Em profundidade rasa (`depth ≤ 4` no titanium), se o eval
estático **já** está tão acima de `beta` que nem uma queda razoável o
derrubaria, corta o nó sem gerar lance nenhum. Margem usada por eles:
`(improving ? 70 : 90) × depth` centi-pontos. É o pai do RFP clássico de
Stockfish — mais barato ainda que LMR porque não gera nenhum lance, decide
só olhando o eval estático do nó.

**Como implementar em `zquoridor`.** No topo de `negamax`, depois do teste de
`winner(s)` e da consulta à TT, antes de gerar qualquer lance: se
`depth <= 4` e `evalSimpleW(s, s.turn, weights) - margin(depth, improving) >= beta`,
retornar o eval direto (fail-high). Margem inicial: reaproveitar as mesmas
constantes do titanium como ponto de partida (70/90 por depth), re-tunar
depois com o `tune_spsa.cpp` já existente — é só mais um par de
`EvalWeights`.

**Risco.** Médio — corta demais se a margem for agressiva; testar com
posições de gargalo conhecidas (mesmo protocolo do resto do documento) antes
de aceitar valores diferentes dos do titanium.

### 3c — **[NOVO — v17/v18]** Late Move Pruning (poda por contagem de lances, não por redução)

**O que é.** Diferente de LMR (que **reduz** a profundidade de lances
tardios), Late Move Pruning **descarta inteiramente** lances quietos depois
de já ter tentado um número suficiente deles num nó raso, sem sequer buscar
em profundidade reduzida. No titanium aparece como o corte
`depth <= 2 && i >= (improving ? 14 : 8)` — ou seja: em profundidade ≤2, depois
do 8º (ou 14º, se `improving`) lance quieto testado, para de gerar mais.

**Por que vale a pena junto com LMR.** É o complemento natural: LMR reduz os
lances "médios", LMP corta de vez a cauda longa de lances "quase certamente
irrelevantes" que nem vale a pena reduzir — combina bem com a ordenação por
CAT/killers/history (itens 1–2), porque só é seguro cortar a cauda se a
ordenação na frente for confiável.

**Como implementar em `zquoridor`.** No mesmo laço de `negamax` de LMR: contar
`moveIndex` por nó; se `depth <= LMP_MAX_DEPTH` (começar com 2) e
`moveIndex >= (improving ? LMP_COUNT_IMPROVING : LMP_COUNT_NOT_IMPROVING)`
(começar com 14/8, iguais ao titanium), parar o laço sem testar o resto —
nunca aplicar ao lance da TT, killers, nem a lances "críticos" (mesma
checagem que `quiescence` já usa pra distinguir muro crítico de muro
irrelevante).

**Risco.** Médio — mesma cautela do LMR: validar contra posições de gargalo
conhecidas antes de aceitar, não só nós/segundo.

---

## Prioridade 4 — Solver exato de final "mãos vazias" (race/DP retrógrado)

**O que é.** Quando **ambos** os jogadores ficam sem muros, a topologia de
paredes fica congelada para sempre e o jogo vira uma corrida de peão pura
(com pulos). O titanium-engine trata isso em duas camadas:

- **Serviço A (barato, quase-instantâneo):** se os **conjuntos** de casas do
  caminho mais curto dos dois jogadores são disjuntos, não há interseção
  possível → nenhum pulo pode ocorrer → o resultado é só "quem tem o caminho
  mais curto ajustado pelo turno" — decide sem busca nenhuma.
- **Serviço B (exato, sob demanda):** quando os caminhos se sobrepõem, resolve
  por DP retrógrada **todos os 81×81×2 = 13.122 estados** `(pos0, pos1, turno)`
  daquela topologia fixa de muros — é pouco: cada estado tem no máximo ~5
  sucessores (passos + pulos), a tabela inteira cabe resolvida em memória
  pequena e é rápida de construir do zero a cada vez que a posição entra
  nesse regime (não precisa persistir entre partidas).

**Por que vale a pena aqui.** É o único ganho da lista que troca busca
heurística por **certeza matemática** em parte não-trivial do jogo — finais
"mãos vazias" são comuns (jogadores tendem a gastar todos os muros antes do
fim) e são exatamente onde a busca heurística tradicional (mesmo com NNUE)
ainda erra por horizonte. Resolver exato aqui é estritamente melhor que
qualquer profundidade de busca alcançável ali.

**Como implementar em `zquoridor`.**
1. Novo arquivo `src/endgame_race.hpp`.
2. `raceOutcomeCheap(wallsH, wallsV, pawn0, pawn1, turn) -> optional<Score>`
   — Serviço A: computar os dois conjuntos de casas do caminho mais curto
   (reaproveitar a BFS multi-fonte já usada por `shortestPathTouchSlots`,
   mas coletando **células**, não slots de muro) e testar interseção vazia.
   Se disjuntos, devolver o vencedor por comparação de distância ajustada
   pelo turno (mais barato que qualquer busca).
3. `raceExactDTM(wallsH, wallsV, pawn0, pawn1, turn) -> int` — Serviço B: DP
   retrógrada sobre os 13.122 estados `(p0, p1, turn)`, gerando sucessores
   com a mesma lógica de `pawnStepMoves` já existente em `rules.hpp` (sem
   muros, já que estão congelados). Guardar como `+k`/`-k` plies-até-mate,
   igual ao esquema de score que a TT já usa (`SCORE_INF - k`).
4. Gancho em `negamax`: logo depois do teste de `winner(s)`, se
   `s.wallsLeft[0] == 0 && s.wallsLeft[1] == 0`, tentar Serviço A primeiro; se
   `Unknown`, tentar Serviço B (13k estados resolve em microssegundos, vale a
   pena mesmo custando mais que um nó de busca comum) e retornar o valor
   exato em vez de continuar a busca heurística.
5. **Nunca** confundir isso com "captura obrigatória" — como o próprio
   titanium-engine registra como um bug já corrigido (episódio 11): não é
   sobre esgotar muros, é sobre a interseção dos caminhos.

**Risco.** Baixo, é aditivo e só ativa numa condição de borda claramente
identificável (`wallsLeft[0]==0 && wallsLeft[1]==0`), fácil de testar
isoladamente com posições construídas à mão.

### 4b — **[REFINADO — v17/v18]** o Serviço A na verdade tem 3 níveis, não 1

A primeira versão deste plano descreveu o Serviço A como um único teste
(caminhos disjuntos). O histórico completo mostra que ele evoluiu pra 3
níveis em ordem de custo crescente, cada um só ativado se o anterior não
decidiu:

- **Nível 1 — portão de ETA:** se `delta_eta > 1` (a diferença de distância
  ajustada por turno entre os dois jogadores é grande o bastante), a
  interceptação é fisicamente impossível — decide sem nem checar sobreposição
  de caminho. É o teste mais barato de todos (só subtração de 2 inteiros já
  calculados).
- **Nível 2 — sobreposição de caminho:** o que já estava descrito (conjuntos
  de casas disjuntos → corrida de tempo pura).
- **Nível 3 — certificado de dominância por desvio:** quando os caminhos se
  cruzam mas um lado ainda domina de forma comprovável, um "winner table"
  assimétrico (construído e cacheado por topologia sob demanda, via grafo de
  predecessores) resolve sem precisar da DP completa de 13.122 estados.

Só se os 3 falharem é que o Serviço B (DP retrógrada exata) entra.

**Como isso muda a Prioridade 4 em `zquoridor`.** Implementar nessa mesma
ordem, um nível de cada vez, cada um testável isoladamente antes do próximo:
nível 1 é uma comparação de inteiros (praticamente grátis, fazer primeiro);
nível 2 é o que já foi descrito acima; nível 3 (winner table assimétrico) só
vale a pena implementar depois que 1 e 2 estiverem rodando e sobrar volume
suficiente de posições "sobrepostas mas ainda decidíveis" pra justificar o
cache extra — é o item mais custoso de engenharia dos 3, deixar por último.

### 4c — **[NOVO — v17/v18]** precisão de ±1 tempo com distância "jump-aware"

O titanium ajustou a distância usada nos níveis acima para contar **pulos**
como parte do cálculo de tempo, não só passos simples — sem isso, o
Serviço A subestima/superestima corridas por até 1 tempo em posições onde os
dois peões ficam adjacentes. Ajuste fino, mas fácil de errar: ao portar
qualquer um dos níveis acima, testar explicitamente posições onde os peões
ficam vizinhos (regra de pulo reto/diagonal) antes de confiar no resultado.

**Risco de 4b/4c.** Mesmo do item 4 original — aditivo, mas correção de tempo
por ±1 casa é sutil o bastante pra merecer testes de posição construídos à
mão (pulos retos e diagonais nos dois lados) antes de aceitar qualquer
nível.

### 4d — **[IMPLEMENTADO, com 2 desvios do plano original]** status real após a integração

A Prioridade 4 foi implementada (`src/endgame_race.hpp`, gancho em
`search.hpp::negamax`, `src/test_endgame_race.cpp`) e está em produção,
mas com duas divergências importantes em relação ao que 4b/4c descreviam
— ambas achadas DEPOIS da integração inicial, uma por teste dirigido e
outra por medição de força em arena externa. Registradas aqui pra quem
for portar algo parecido não repetir os mesmos dois erros.

**1. Nível 1 (portão de ETA) foi removido do pipeline de decisão, não só
"refinado".** A margem do item 4c (delta de tempo grande o bastante pra
descartar interceptação) assumia que bloqueio físico custa no máximo 1
tempo a mais que um pulo — um teste aleatório amplo
(`testRandomWallTopologiesGatesAgreeWithExact`) achou um contraexemplo
real onde bloqueio custa mais que isso. Em vez de tentar consertar a
margem (arriscado sem uma prova nova), `raceETAGate` foi mantida como
utilitário isolado, testado, mas **não é mais chamada** por
`resolveEmptyHandedEndgame` — só o Nível 2 (corrigido, ver abaixo) e o
Serviço B decidem em produção.

**2. Nível 2 (sobreposição de caminho) tinha a base geométrica errada —
achado e corrigido depois de já estar em produção.** A versão original
testava disjunção dos **conjuntos de casas em algum caminho mais curto**
(`onShortestPathMask`) e tratava isso como certeza de que nenhuma
interação era possível. Isso é falso: um jogador que está perdendo a
corrida de tempo pura não é obrigado a seguir um caminho mínimo — pode
desviar pra dentro do território do outro só pra bloquear fisicamente,
mesmo sem nunca pisar numa casa do caminho mínimo do oponente. Um
contraexemplo real foi encontrado por busca em topologias sintéticas:
caminhos mínimos disjuntos (colunas 2-3 vs. 4-5) no mesmo tabuleiro
totalmente conectado (as 81 casas continuam mutuamente alcançáveis
ignorando o outro peão) — o gate antigo decidia "vitória do jogador 0 em
10 lances"; o resultado verdadeiro (confirmado pelo Serviço B e por duas
reimplementações independentes de checagem cruzada) era **empate por
perseguição infinita**. Corrigido trocando a base para disjunção da
**região inteira alcançável** (`reachableRegionMask`, BFS única sem meta)
— essa sim é condição necessária e suficiente (regiões disjuntas ⇒
nenhuma aresta atravessável as liga ⇒ os dois jogadores nunca ficam
sequer adjacentes, em qualquer rota, ótima ou não).

**3. Consequência de engenharia da correção acima: o Serviço B precisou
de cache por topologia de muro, e isso não era opcional.** O comentário
original do plano ("13k estados resolve em microssegundos, vale a pena
mesmo custando mais que um nó de busca comum") só era verdade *na
prática* porque o Nível 2 antigo (mesmo incorreto) decidia sozinho com
bastante frequência em tabuleiros típicos, evitando a maioria das
chamadas ao Serviço B. A versão corrigida do Nível 2 decide bem menos
(a condição sã é mais rara — a maioria dos tabuleiros reais continua
totalmente conectada mesmo com vários muros), então passou a cair no
Serviço B em quase todo nó da fase "mãos vazias" — e cada chamada
reconstruía o grafo de 13.122 estados e rodava as duas BFS retrógradas
**do zero**, medido em ~790 microssegundos por chamada (não
"microssegundos" desprezíveis). Isso derrubou nós/s em mais de 50× numa
sessão de arena externa (Elo -166 medido contra a versão anterior) antes
de ser identificado e corrigido. A correção: como
`wallsLeft[0]==0 && wallsLeft[1]==0` (única condição de entrada do
gancho) implica que nenhum muro pode mais ser colocado pro resto daquela
subárvore de busca inteira, `wallsH`/`wallsV` ficam **congelados** por
potencialmente milhares de nós consecutivos — um cache de 1 slot
(`raceExactDTM` só recalcula a DP quando a topologia recebida muda em
relação à última chamada) resolveu: de ~1.267 chamadas/s (sem cache) para
~276.000 chamadas/s (com cache) num benchmark isolado de chamadas
repetidas, e de ~1.600 nós/s pra ~84.000 nós/s num benchmark ponta-a-ponta
com `Negamax::chooseMove` real e orçamento de tempo por lance, mesma
posição. Lição pro roadmap: **qualquer** camada nova que se proponha
"barata o bastante pra não precisar de cache" precisa ser medida sob a
frequência de chamada real esperada em produção antes de assumir isso —
principalmente quando uma correção de corretude futura pode
legitimamente mudar essa frequência pra pior, como aconteceu aqui.

Nível 3 (certificado de dominância por desvio, item 4b) segue não
implementado — nenhum problema de volume o justificou até agora, já que
o Serviço B com cache resolve a fase inteira rápido o bastante.

### 4e — **[CORRIGIDO]** cache por topologia não bastava sozinho, e um bug de ESCOLHA DE LANCE (não de valor) sobrevivera a tudo isso

Depois de 4d (cache de 1 slot por topologia), duas rodadas adicionais de
correção aconteceram, uma de performance e outra — mais séria — de
corretude na escolha do lance em si.

**1. Orçamento de tempo real para o Serviço B (performance).** O cache de
1 slot de 4d só ajuda quando chamadas consecutivas compartilham a MESMA
topologia — e medição em busca real mostrou taxa de acerto de cache
**~0,5%**, não os >90% que o benchmark sintético (chamadas repetidas de
propósito) sugeria. O motivo: perto do fim de jogo, o alpha-beta ainda
está decidindo ONDE colocar os ÚLTIMOS muros de cada lado, e cada
candidato de posição de muro testado nessa borda gera uma topologia
FINAL diferente — o cache não tem o que reaproveitar entre candidatos
irmãos. Isso fazia o Serviço B (~0,6–0,9ms/chamada sem acerto de cache)
rodar do zero em quase todo nó dessa fase de decisão. Corrigido com um
orçamento de TEMPO REAL (medido via `chrono` a cada chamada cara, não
uma contagem estimada de chamadas) reservado ao solver exato — uma
fração pequena (3%) do orçamento de tempo total daquela busca
(`g_raceExactBudgetUs`/`g_raceExactUsedUs`, resetado a cada
`chooseMove`). Quando o orçamento acaba, o nó cai de volta pro heurístico
de sempre em vez de continuar pagando o rebuild caro — no pior caso,
nós/s fica igual ao que seria sem a feature de race, nunca pior. Some-se
a isso: uma posição EXATA repetida (mesmo hash — comum via
transposição/re-busca de PVS/aspiration/iterative deepening) agora é
armazenada e lida da própria TT do motor (`EXACT`, `depth=127` pra nunca
ser sobrescrita por uma entrada de profundidade menor), sem precisar de
estrutura de cache nova pra esse caso.

**2. Bug de escolha de LANCE na raiz — achado só depois de já ter nós/s
saudável (>900k) e AINDA ASSIM perder a maioria das partidas em arena
externa.** `resolveEmptyHandedEndgame` devolve o VALOR exato de uma
posição, mas se comporta como nó-folha: nunca gera nem recursiona sobre
os próprios filhos (igual `winner()`). Isso é correto e barato quando
essa posição é FILHA de outro nó — o pai continua comparando vários
candidatos normalmente, cada um recursando pra um filho que aciona o
atalho só pra fornecer um valor pra comparação. O problema é quando a
própria RAIZ real da partida (a posição que `chooseMove` recebe, vinda
do jogo de verdade — não um nó interno) já satisfaz
`wallsLeft==(0,0)`: não existe "nó pai" nenhum fazendo essa comparação, e
o "melhor lance" que `chooseMove` lia da TT era um PLACEHOLDER
(`legalMoves(s)[0]`, gravado só pra garantir legalidade de retorno, ver
comentário em `negamax`) — não o lance que de fato realiza o DTM ótimo.
O motor "sabia" quem ia ganhar (score correto, TT correta) mas jogava um
lance essencialmente arbitrário pra chegar lá, durante toda a fase de
final — tipicamente a maior parte de uma partida real, já que os muros
costumam acabar bem antes do jogo terminar. Isso explica por que nós/s
podia estar SAUDÁVEL ou até melhor que a linha de base (cada "busca" na
raiz, quando já em mãos-vazias, resolve instantaneamente por não haver
recursão nenhuma) e mesmo assim o motor perder a maioria das partidas —
sintoma que não bate com "está lento", bate com "está decidindo mal".
Corrigido em `chooseMove`: antes do loop de iterative deepening, se a
própria raiz já satisfaz `wallsLeft==(0,0)`, compara os candidatos (só
peão — sem muro nessa fase) por **1 ply usando o valor exato** de cada
filho (mesma fórmula de `resolveEmptyHandedEndgame`) — maximizar sobre
valores já exatos é ótimo por construção, não precisa de busca alguma.

**Validação (não só teste unitário — arena real, mesma ferramenta que
detectou o problema originalmente):** `teste/arena.cpp` compilando o
código REAL de dois refs (não a mesma engine duplicada) — antes da
correção do item 2, ~113 vitórias em 502 jogos contra a v1.1 (Elo
≈−132, apesar de nós/s ALTO); depois da correção, duas amostras
independentes (50 e 50 jogos, seeds diferentes) deram 24–24–2 e
21–28–1 — combinado, 45–52–3 em 100 jogos (score ≈46,5%, Elo ≈−24,
dentro do ruído estatístico dessa quantidade de jogos). Também um teste
de regressão dedicado
(`test_endgame_race.cpp::testChooseMoveAtEmptyHandedRootPicksOptimalMove`)
que constrói uma posição de raiz com um único lance objetivamente ótimo
e confirma que `chooseMove` o escolhe — falha contra a versão sem a
correção (confirmado rodando o teste contra ambas), passa com ela.

**Lição pro roadmap, reforçando a de 4d:** um atalho que devolve só um
VALOR (sem lance) é seguro em qualquer nó INTERNO da árvore (o pai
sempre compara), mas quebra silenciosamente se o mesmo atalho puder ser
atingido diretamente pela RAIZ de uma busca real — `chooseMove` precisa
de um LANCE, não só um score, e nada detecta esse descompasso em tempo
de compilação nem em testes que só chamam o solver isoladamente (como os
de 4d faziam). Qualquer atalho parecido no futuro (Nível 3, ou outro
solver exato de subjogo) precisa de um teste que chame explicitamente o
ponto de entrada de escolha de lance (`chooseMove`) a partir de uma
posição-raiz já dentro da zona do atalho — não só o solver isolado.

---

## Prioridade 5 — Gates de perft como oracle de regressão de corretude

**O que é.** O titanium-engine trava contagens exatas de nós em profundidades
3–6 (`2.062.264`, `247.569.030`, `28.837.934.502`, `3.257.436.276.501`) como
teste de CI — não é benchmark de velocidade, é prova de que a geração de
lances (incluindo regras de pulo, muro, borda) não regrediu, ainda que a
implementação interna mude completamente.

**Por que vale a pena aqui.** O `zquoridor` já tem disciplina forte de
validação (`test_rules_sanity.cpp`, prova formal do DSU documentada em
`dsu.hpp`, prototype Python validado antes do port C++) — isso é o
complemento natural: um número fixo e público que qualquer refatoração de
`legalMoves`/`applyMove` tem que continuar reproduzindo, sem precisar
reler a prova toda vez.

**Como implementar em `zquoridor`.**
1. Adicionar `perft(state, depth) -> uint64_t` em `rules.hpp` (função pura:
   soma recursiva de `legalMoves(applyMove(s, m))` até profundidade 0).
2. Rodar uma vez a partir da posição inicial para profundidades 1–4 (ou até
   onde o tempo permitir), fixar os números encontrados como constantes em
   `teste/test_rules_sanity.cpp` (`PERFT_D3_EXPECTED`, etc. — nomeando ao
   estilo do `PERFT5_STARTPOS`/`PERFT6_STARTPOS` do titanium, já que a
   convenção de nome é boa).
3. Rodar de novo depois de qualquer mudança em `rules.hpp`/`dsu.hpp` como
   parte do `teste/` já existente.

**Risco.** Nenhum — é puramente aditivo, só trava um número já verdadeiro.

---

## Prioridade 6 — Cache de BFS por nó (compartilhar entre ordenação, poda e CAT)

**O que é.** O titanium-engine documentou (episódio 11, "performance pass")
que a busca ficava em ~10K nós/s (vs. ~18M nós/s de perft puro) porque a
**mesma** BFS de distância era recalculada 3× por nó: uma vez em
`collect_search_moves`, outra em `order_moves`, outra dentro do LMR. A
correção foi computar uma vez por nó e compartilhar.

**Por que vale a pena aqui.** É diretamente aplicável e barato: `negamax` já
chama `legalWallMoves` (que já roda a BFS de prefiltro internamente) e depois,
separadamente, chama `shortestPathLen`/`pathRobustness` de novo dentro de
`quiescence` para o mesmo oponente. Se Prioridades 1 e 3 (CAT, LMR) forem
implementadas, cada uma vai querer sua própria BFS de distância — é o momento
certo de consolidar antes que a duplicação se multiplique.

**Como implementar em `zquoridor`.**
1. Criar uma struct `NodeDistCache { array<int,81> distFrom0, distFrom1; }`
   preenchida **uma vez por nó** (não por lance) logo no topo de `negamax`,
   a partir da mesma BFS multi-fonte que `shortestPathTouchSlots` já roda.
2. Refatorar `shortestPathTouchSlots`, `shortestPathLen`, `pathRobustness` e
   a futura `computeCorridorHeat` (Prioridade 1) para aceitarem o vetor de
   distância já calculado como parâmetro opcional, em vez de recalcular BFS
   internamente sempre.
3. Passar `NodeDistCache` por referência para `legalWallMoves`,
   `quiescence` e qualquer coisa de CAT/LMR chamada dentro do mesmo nó.

**Risco.** Médio na refatoração (mexe em várias assinaturas de função core),
baixo na lógica (é só evitar recomputação, sem mudar nenhum resultado) — bom
candidato para fazer com testes de regressão de `test_rules_sanity.cpp`
rodando antes/depois em cada função tocada.

### 6b — **[NOVO — v17/v18]** cache de distância entre nós (LRU por topologia de muros), não só dentro do nó

O item 6 acima resolve a duplicação **dentro** de um nó. O titanium foi além:
um cache **LRU entre nós diferentes**, chaveado pela topologia de muros (não
pela posição inteira — a distância BFS de um jogador só depende de onde estão
os muros, não de onde está o peão do adversário), com tamanho adaptativo "à
la TT" (começa no tamanho de regime permanente, sem rampa fria). Faz sentido
porque a mesma topologia de muros aparece repetidamente em nós irmãos/
transposições — sem esse cache, a mesma BFS é refeita do zero em cada
sub-árvore que chega numa topologia já vista antes.

**Como implementar em `zquoridor`.** Depois do item 6 (cache por nó) estar
funcionando: extrair a chave de cache como `hash(wallsH, wallsV, pawnCell,
player)` (não o `State.hash` completo, que já inclui `turn`/`wallsLeft` —
esses não afetam a distância BFS) e envolver `shortestPathLen`/o cálculo de
distância completo por casa (item 6) numa tabela hash pequena com política
LRU (`std::unordered_map` + lista duplamente ligada, ou um array circular
simples do tamanho da TT). Popular o tamanho inicial já no valor de regime
(evitar a "rampa fria" que o titanium documentou ter sido um desperdício
mensurável).

**Risco.** Médio — cache incorreto (chave que esquece alguma dependência) é
pior que não ter cache; testar explicitamente que a chave realmente não
depende de nada além de `(wallsH, wallsV, pawnCell, player)` antes de
confiar (ex.: mesma topologia + mesmo peão, dois jogos diferentes, tem que
dar o mesmo resultado — teste determinístico fácil de escrever).

---

## Prioridade 7 — Lookup table O(1) para movimento de peão

**O que é.** O titanium-engine pré-computa offline
`PAWN_LEGAL[casa][enemy_key][wall_key] -> bitmask` e extrai o índice de muros
relevantes com uma instrução `PEXT` (BMI2), eliminando o cálculo de
adjacência/pulo por chamada.

**Por que vale a pena aqui — com ressalva.** Diferente da geração de muro
(128 candidatos, gargalo real), o peão só tem no máximo 5 destinos por turno;
o ganho absoluto é bem menor que CAT/LMR/cache-por-nó acima. Vale a pena
**depois** dos itens 1–6, e só se o profiling mostrar que `pawnStepMoves`
aparece de forma mensurável no tempo total (o titanium-engine só valeu a pena
por causa do volume de perft; em busca real com eval cara, o peso relativo é
menor).

**Como implementar em `zquoridor` (se o profiling justificar).**
1. Gerar offline (script Python separado, fora do hot path) uma tabela
   indexada por `(casa do peão, casa do peão adversário relativa, byte
   compacto dos slots de muro ao redor)` → bitmask de destinos legais,
   cobrindo passo simples, pulo reto e pulos diagonais.
2. Como `zquoridor` não depende de BMI2/PEXT hoje (portabilidade WASM/mobile
   é prioridade do projeto — GUI já roda em WebAssembly), usar a versão
   **escalar** do empacotamento de chave (o próprio titanium mantém isso como
   fallback obrigatório para builds sem BMI2) — ainda ganha por eliminar
   ramificação, mesmo sem a instrução PEXT.
3. Tabela cabe em poucos KB (peão tem muito menos combinações relevantes que
   muro); gerar em `teste/` ou script `tools/gen_pawn_lut.py`, versionar o
   `.bin`/header gerado.

**Risco.** Baixo, mas esforço só se justifica depois de medir — não
implementar "porque o outro engine tem".

---

## Prioridade 8 — PVS (Principal Variation Search) com janela nula

**O que é.** Refinamento clássico de alpha-beta: buscar o primeiro lance
(esperado ser o melhor, geralmente o da TT) com janela completa `(alpha,
beta)`; todos os seguintes com **janela nula** `(-alpha-1, -alpha)` — muito
mais barato quando o lance realmente é pior — e só re-buscar com janela
completa se a busca de janela nula "vazar" acima de `alpha`.

**Por que vale a pena aqui.** É o parceiro natural de LMR (Prioridade 3): LMR
já reduz profundidade dos lances tardios, PVS reduz a **largura da janela**
deles também. Junto, é a dupla clássica que separa um alpha-beta simples de
um alpha-beta "sério" (é literalmente o que Stockfish faz em todo nó não-raiz
não-PV).

**Como implementar em `zquoridor`.** Trocar o laço principal de `negamax`
(depois do primeiro lance) de sempre `-negamax(ns, depth-1, -beta, -alpha,
stats)` para: primeiro lance com janela completa; demais com
`-negamax(ns, depth-1, -alpha-1, -alpha, stats)`, promovendo para janela
completa só se `score > alpha && score < beta`. Combina diretamente com a
estrutura de LMR do item 3 (a busca reduzida já é de janela nula; a
re-busca completa acontece só se necessário, exatamente como descrito ali).

**Risco.** Baixo — é uma reestruturação local do laço de busca, sem mudar
geração de lances nem eval; fácil de isolar e comparar nós-por-profundidade
antes/depois.

---

## Prioridade 9 — Certificado formal de vitória (estilo ACE: eval floor + refutação do último muro)

**O que é.** Um verificador que, além do score heurístico, tenta **provar**
formalmente uma vitória combinando: (a) um piso de eval acima do qual só
posições genuinamente ganhas aparecem, com (b) um "portão de refutação do
último muro" — checar se existe algum muro restante do oponente capaz de
desfazer a vantagem antes de aceitar a prova.

**Por que vale a pena aqui.** É mais barato e mais robusto que aumentar
profundidade de busca só para confirmar um final que já parece decidido —
reduz nós desperdiçados "só para ter certeza" em posições de final que a
Prioridade 4 (race exato) não cobre (ainda há muros no jogo, então não é
"mãos vazias" ainda, mas a vantagem já é grande).

**Como implementar em `zquoridor`.** Depende de 1 (CAT) e 4 (race exato)
estarem prontos primeiro — usa calor de corredor pra saber quais muros do
oponente são candidatos plausíveis de refutação, e usa o solver de corrida
como sub-rotina quando a checagem reduz a posição a "mãos vazias". Prioridade
relativamente baixa na ordem geral — é um refinamento avançado, não uma
correção de gargalo.

**Risco.** Médio-alto — lógica de certificação incorreta é pior que não ter
certificação (pode declarar vitória cedo demais). Exige bateria de testes
adversariais antes de confiar, exatamente como o titanium documentou (2
deciders anteriores descobertos **incorretos** em topologias com muro e
descartados).

---

## Prioridade 10 — Livro de abertura

**O que é.** Banco de posições iniciais (até ~15 lances) com estatística de
taxa de vitória, usado para viés de ordenação na raiz (não forçar lance, só
bonificar), minerado de partidas de self-play/engine-vs-engine.

**Por que vale a pena aqui.** Zquoridor já planeja bootstrap de self-play
para o NNUE — o mesmo pipeline de partidas gera dados suficientes para um
livro de abertura como subproduto, quase de graça.

**Como implementar em `zquoridor`.** Depois que o pipeline de self-play do
NNUE estiver rodando (já planejado): agregar `(posição inicial normalizada,
lance, resultado)` das primeiras ~15 jogadas de cada partida de treino num
arquivo simples (JSON ou binário pequeno — Quoridor não precisa de algo tão
elaborado quanto SQLite/DAG do titanium; a árvore de abertura de Quoridor é
muito mais rasa que xadrez). Usar só como bônus de ordenação na raiz, nunca
forçado, e só até a profundidade coberta pelos dados.

**Risco.** Baixo, mas trabalho real só compensa depois que houver volume de
partidas de self-play — não adiantar esse item.

---

## Prioridade 11 — NNUE incremental (accumulator diff) e input de calor CAT — **[REESCRITO — v17/v18]**

A primeira versão deste item falava só da rede de atenção "Ka" (137 saídas)
como algo caro e de retorno incerto — mantenho essa parte, mas o histórico
completo mostrou que o titanium ganhou muito mais performance em algo bem
mais barato e mais próximo do que o `zquoridor` já faz:

- **Diff incremental do accumulator (o "NNUE" de verdade):** em vez de rodar
  a rede inteira do zero em cada nó, atualizar só as features que mudaram
  entre um `State` e o filho (`applyMove`), a partir de um diff de bitboard —
  exatamente a técnica que você **já usou no Zchezz** ("incremental
  accumulator, diff-based undo frame"). O commit deles descreve isso junto
  com "eval cache" e "route by-bit tables" como o pacote que resolveu a
  maior parte do custo de NPS na busca real (não no perft).
- **Input de calor CAT na própria rede** (não só na busca): a rede NNUE deles
  (v16, "ws20 eval inputs") ganhou um plano de entrada extra com o calor de
  corredor (item 1 deste documento) normalizado, alargando a camada oculta de
  32 para 48 — ou seja, o CAT não é usado só pra ordenar/podar, é também
  **feature de eval**. O formato do arquivo de pesos é auto-descritivo
  (cabeçalho com a largura `NET_H`), então pesos antigos continuam
  carregando com o plano novo zerado — mudança de arquitetura sem quebrar
  compatibilidade.

**Como implementar em `zquoridor` — prioridade real, maior que a rede "Ka".**
1. **Primeiro:** confirmar que o NNUE de 290 features do `zquoridor`
   (`nnue.hpp`) já faz diff incremental do accumulator entre `applyMove`s
   consecutivos dentro da árvore de busca, do mesmo jeito que o Zchezz já
   faz — se ainda não faz (por ser mais novo/em desenvolvimento), portar essa
   técnica é o item de maior retorno de toda esta seção, porque evita
   recomputar a rede inteira em cada um dos milhões de nós da busca.
2. **Depois:** se o item 1 (CAT) estiver implementado, considerar adicionar o
   calor de corredor como plano de entrada extra do NNUE (não só como sinal
   de busca) — decisão de arquitetura pra depois que o head duplo
   valor/política planejado estiver estável, seguindo o mesmo princípio de
   cabeçalho auto-descritivo pra não quebrar pesos já treinados.
3. **Rede "Ka" (attention, 137 saídas):** mantém a nota original — só
   considerar depois dos itens 1–2 acima, e só se a head de política do NNUE
   dual-head não bastar.

**Risco.** Item 1 (accumulator incremental) é baixo risco — é uma otimização
de implementação que não muda o resultado da rede, só a velocidade;
testável comparando saída da rede incremental vs. forward completo em cada
posição (devem bater exatamente). Item 2 (input de CAT na rede) é maior
risco por exigir retreino.

---

## Prioridade 12 — Paralelismo de busca (Lazy SMP)

**O que é.** Rodar N threads fazendo a mesma busca iterative-deepening a
partir da raiz, compartilhando a TT (com pequenas variações de ordem/ruído
entre threads para não convergir todas no mesmo caminho), agregando o
resultado da thread que chegou mais fundo.

**Por que vale a pena aqui.** O titanium-engine **rejeita explicitamente**
paralelismo dentro da geração de lances ("Single-thread hot path only — no
movegen multithreading, no GPU") — mas isso é sobre movegen, não sobre busca:
Lazy SMP é ortogonal e paralelizaria a busca do `zquoridor` sem tocar
`rules.hpp`/DSU/BFS. É um ganho real de nós/s em máquinas multi-core, mas
não é prioridade enquanto os itens de algoritmo (1–8) ainda não foram
esgotados — paralelismo multiplica o que já existe, não corrige ineficiência
por nó.

**Como implementar em `zquoridor`.** TT (`std::vector<TTEntry>`) precisa
virar acesso atômico/lock-free por entrada (ou aceitar races benignas,
como Stockfish faz — colisão de TT sob concorrência é tolerada, não é bug
crítico). Cada thread roda `Negamax::search` a partir da mesma raiz com
profundidades levemente escalonadas; thread principal decide o `bestmove`
pela busca mais profunda concluída.

**[REFINADO — v17/v18]** Isso deixou de ser especulativo: o titanium **já
ships** Lazy SMP de verdade (WASM com threads, "v16 threaded WASM"), e eles
mesmos passaram por uma calibração que vale registrar: a thread principal
inicialmente explorava só 10% da largura da raiz (deixando o resto pras
threads auxiliares) e isso piorou — reverteram pra thread principal explorar
quase a largura inteira da raiz (95%), com as auxiliares fazendo variações
mais estreitas/decaladas. Ou seja: **não estreitar demais a busca da thread
principal** achando que "as outras compensam" — comece com a thread
principal fazendo a busca normal completa e só adicione threads auxiliares
como buscas *extras* com pequenas variações de ordenação, sem tirar largura
da principal.

**Risco.** Médio-alto de engenharia (concorrência), mas isolado — não muda
nenhuma regra nem eval, só multiplica throughput.

---

## Prioridade 13 — Gerência de tempo dedicada

**O que é.** Módulo separado (não misturado no laço de `negamax`) que decide
quanto tempo alocar por lance com base em: tempo restante, incremento,
estabilidade do `bestmove` entre iterações do iterative deepening (se o
melhor lance para de mudar, corta cedo; se muda muito, estica).

**Por que vale a pena aqui.** O `zquoridor` já tem `deadline` checado dentro
de `negamax`/`quiescence` — falta só a camada de **decisão** de quanto tempo
dar por jogada em partidas com relógio real (hoje aparenta ser profundidade
fixa ou deadline fixo por chamada).

**Como implementar em `zquoridor`.** Novo `src/timeman.hpp`: função
`allocateTimeMs(remainingMs, incrementMs, movesPlayed, bestMoveStability) ->
int`. Chamada uma vez por jogada (não por nó), define a `deadline` que já é
passada para `Negamax`. Bônus simples: parar o iterative deepening 1 iteração
mais cedo se `bestmove` não mudou nas últimas 3 profundidades e o tempo usado
já passou de ~40% do alocado.

**[REFINADO — v17/v18]** números concretos que eles chegaram, bons como
ponto de partida (em vez de tunar do zero): plano-base de **30 lances
próprios restantes** (derivado de duração média observada de partida, ~60
plies — evitar usar P90/P95 daqui, isso é só telemetria); esse horizonte é
**esticado** quando o Serviço A/B do solver de final (Prioridade 4) já
consegue estimar `min(plies_até_vitória_p0, plies_até_vitória_p1)` — ou seja,
o próprio solver de corrida alimenta a gerência de tempo, não só a busca.
Teto rígido: nunca gastar mais que **1.25× o tempo "ótimo"** calculado
(mesmo `MAX_RATIO` que o Stockfish usa como limite de "roubo" de tempo de
lances futuros). Parada suave dentro da busca em **85%/92%** do tempo
alocado (dois níveis, não um só) além do `deadline` duro que já existe.

**Risco.** Baixo — não toca busca nem regras, só a política de quando parar.

---

## Prioridade 14 — Infra de teste estatístico (estilo SPRT) para variantes do motor

**O que é.** Em vez de medir só "nós por segundo" numa posição fixa, rodar
lotes de partidas motor-vs-motor (versão A vs versão B) e comparar Elo com
significância estatística (SPRT — sequential probability ratio test), a
mesma disciplina que motores de xadrez usam para aceitar/rejeitar mudanças.

**Por que vale a pena aqui.** O projeto já tem a lição metodológica certa
registrada ("self-play não serve para comparar move-ordering; precisa de
posições fixas") — SPRT motor-vs-motor é o complemento que falta: posições
fixas validam *nós por profundidade*; partidas completas com SPRT validam
*força de jogo real*, que é o que efetivamente importa para aceitar CAT, LMR,
etc. como melhorias reais e não só "mais rápido, mesma qualidade".

**Como implementar em `zquoridor`.** Não precisa da infra distribuída
(Cloudflare Worker) do titanium — um script local (`teste/sprt_match.py` ou
C++ simples) que roda N partidas engine-A vs engine-B (dois binários ou dois
conjuntos de pesos/flags do mesmo binário), alternando quem começa, contando
vitórias/derrotas/empates, e aplicando o teste SPRT (fórmula padrão, poucas
linhas) para decidir "aceitar", "rejeitar" ou "precisa de mais jogos".

**Risco.** Nenhum no motor em si — é infraestrutura de validação, roda por
fora do binário de jogo.

---

## Prioridade 15 — Revisão de Zobrist/make-unmake (Undo enxuto)

**O que é.** O titanium-engine lista como próxima prioridade de perf
"profile e enxugar a struct `Undo`" — o custo de desfazer um lance
(recalcular hash Zobrist incremental, restaurar estado) pode dominar em
motores que fazem make/unmake ao invés de copiar estado.

**Por que vale a pena aqui — nota de arquitetura.** O `zquoridor` usa
`applyMove(s, m) -> State` (copia o estado, não faz unmake) — é uma escolha
de design diferente da do titanium (que faz make/unmake in-place com Undo).
**Este item não se aplica diretamente** enquanto `State` for pequeno o
suficiente pra copiar barato (2× `uint64_t` de muros + posições + contagens —
provavelmente cabe em poucas dezenas de bytes, copiável mais rápido que o
overhead de um Undo bem-feito). Incluído aqui só para registrar a análise:
**não migrar para make/unmake** a menos que profiling mostre que a cópia de
`State` é gargalo real — trocar de arquitetura sem essa evidência seria
scope creep sem retorno comprovado.

**Risco.** N/A — recomendação é **não fazer** a menos que medido.

---

## Prioridade 16 (nice-to-have, fora do motor) — Pondering

**O que é.** Pensar no tempo do adversário: assim que o oponente move, o
motor já vinha buscando a posição prevista/child esperado, aproveitando o
tempo ocioso.

**Por que é baixa prioridade aqui.** O próprio titanium-engine marca isso
como "prepared, not implemented" — depende de UI/protocolo (turno do
adversário, WebSocket/worker) mais do que do motor de busca em si. Só faz
sentido depois que os itens 1–8 (algoritmo) e 13 (timeman) estiverem prontos,
e é mais uma feature de produto (GUI) que de engine.

**Como implementar em `zquoridor` (quando chegar a vez).** No `gui_web/app.js`,
depois do lance do adversário via WASM: já ter iniciado uma busca em
background na resposta mais provável do humano assim que a IA joga (nó cap,
sem deadline de relógio); descartar se o lance real do humano não bater com o
previsto, senão reaproveitar a árvore/TT já aquecida.

**Risco.** Baixo, mas é o item de menor retorno da lista — deixar por último.

---

## Prioridade 17 — **[NOVO, Stockfish puro — não encontrado no titanium]** Internal Iterative Reduction (IIR)

**O que é.** Quando um nó não tem lance de TT (não foi visitado antes ou
sofreu colisão de hash) e a profundidade é razoavelmente alta, reduzir a
profundidade de busca desse nó em 1 antes de gerar lances — a lógica é: sem
lance de TT pra guiar a ordenação, o nó provavelmente vai custar caro e
render pouco; melhor gastar menos nele agora e deixar a iteração seguinte do
iterative deepening (ou uma re-busca) aprofundar se realmente valer a pena.
Isso não apareceu em nenhum commit do titanium que encontrei — é uma técnica
puramente do Stockfish moderno, barata e sem dependência de nenhum outro
item deste documento.

**Como implementar em `zquoridor`.** Em `negamax`, logo depois da consulta à
TT: se `!hasTTMove && depth >= IIR_MIN_DEPTH` (começar com 4), decrementar
`depth` em 1 antes de prosseguir para a geração de lances desse nó (não
afeta os filhos, só a profundidade restante deste nó específico).

**Risco.** Baixo — mudança de poucas linhas, isolada, fácil de comparar
nós-por-profundidade antes/depois.

---

## Resumo de prioridade (visão rápida)

| # | Item | Retorno esperado | Risco | Depende de |
|---|------|-------------------|-------|------------|
| 1 | CAT — calor de corredor | Alto | Baixo/médio | — |
| 2 | Killer moves + history | Médio-alto | Baixo | — |
| 2b | Continuation/countermove/correction history (Stockfish real) | Alto | Baixo/médio | 2 |
| 3 | LMR log-log | Alto | Médio | idealmente 1 |
| 3b | Reverse Futility Pruning (RFP) | Médio-alto | Médio | — |
| 3c | Late Move Pruning (poda por contagem) | Médio | Médio | 1, 2 |
| 4 | Solver exato "mãos vazias" (Serviço B) | Alto (certeza) | Baixo | — |
| 4b | Serviço A em 3 níveis (ETA/overlap/winner-table) | Alto | Baixo→médio | 4 |
| 4c | Distância jump-aware (±1 tempo) | Médio | Baixo | 4, 4b |
| 5 | Gates de perft | Médio (correção) | Nenhum | — |
| 6 | Cache de BFS por nó | Alto (eficiência) | Médio | facilita 1 e 3 |
| 6b | Cache LRU de distância entre nós (por topologia) | Alto | Médio | 6 |
| 7 | LUT O(1) de peão | Baixo-médio | Baixo | medir antes |
| 8 | PVS janela nula | Médio-alto | Baixo | combina com 3 |
| 9 | Certificado de vitória | Médio | Médio-alto | 1, 4 |
| 10 | Livro de abertura | Médio | Baixo | self-play do NNUE |
| 11 | NNUE incremental (accumulator diff) + input CAT | **Alto** | Baixo (diff) / médio (input) | 1 p/ input |
| 12 | Lazy SMP (já real no titanium — root 95%/5%) | Alto (throughput) | Médio-alto | esgotar 1–8 antes |
| 13 | Gerência de tempo (30 lances base, MAX_RATIO 1.25, soft-stop 85/92%) | Médio | Baixo | 4/4b alimenta horizonte |
| 14 | Infra SPRT | Metodológico | Nenhum | — |
| 15 | Zobrist/Undo enxuto | N/A | N/A | **não fazer sem medir** |
| 16 | Pondering | Baixo | Baixo | 1–8, 13; é GUI |
| 17 | Internal Iterative Reduction (IIR) | Médio | Baixo | — |

**Nota sobre o item 11:** na primeira versão deste documento ele estava
listado como baixa prioridade ("rede Ka", incerta). A varredura completa do
histórico mostrou que o ganho real e comprovado deles nessa área foi outro
(accumulator incremental + cache de eval), bem mais alinhado ao que o
`zquoridor` já faz — por isso ele subiu de posição na tabela.
