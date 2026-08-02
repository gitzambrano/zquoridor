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
- **NNUE:** 354 features esparsas (81+81 posição de peão próprio/oponente,
  64+64 slots de muro H/V, 21+21 buckets de distância BFS one-hot, 11+11
  buckets de muros restantes — feature nova de 2026-08, ver histórico da
  Prioridade 1 abaixo), acumulador incremental (diff por lance, não
  rebuild do zero por nó) com variantes float32 (treino) e int8/int16
  quantizada (busca), heads duplos WL + auxiliar (imita `evalSimple`) +
  política (209 saídas). Integrada de fato na busca
  (`Negamax::EvalMode::NNUE`, `search.hpp`) — não é mais só "treinada mas
  não plugada". Perspectiva geometricamente canônica (espelho de
  linha/coluna pra jogador 1, ver Prioridade 1) — posições simétricas dão
  eval idêntica pras duas perspectivas.
  - **NNUE é o default de avaliação em `selfplay` e `arena`** (tenta
    `data/nnue/nnue_weights_int8.bin` automaticamente; cai para
    `evalSimple` com aviso se o arquivo não existir; `--heuristic` força o
    modo antigo). **Ainda NÃO é o default no build WASM** — ver Prioridade 1
    abaixo, é a lacuna mais visível hoje.
  - Acumulador incremental **validado**: `teste/nnue_incremental_check.cpp`
    compara incremental vs. rebuild-do-zero, 0 divergências, incluindo o
    bloco novo de muros restantes.
  - **Pesos treinados precisam ser regerados** — mudança de arquitetura
    (332→354 features) invalida qualquer `.bin` anterior; carregar um
    arquivo do formato antigo agora falha alto com mensagem clara (checagem
    de tamanho adicionada nesta sessão) em vez de crashar ou desalinhar
    silenciosamente.
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
  + early-stopper), captura de Ctrl+C salvando checkpoint de emergência,
  early stopping com restauração do melhor epoch, LR/weight-decay
  schedules com warmup, orçamento de RAM/VRAM calculando batch/chunk size
  automaticamente. **Atualizado em 2026-08** (ver Prioridade 2: bug de
  `--fresh` corrigido + checkpoint em JSON adicionado).

---

## Prioridade 1 — NNUE: feature de muros restantes + correção da assimetria de perspectiva

**Status: 1a e 1b implementadas e validadas nesta sessão (2026-08).** O que
falta pra fechar de vez esta prioridade: (i) **retreino de verdade** com
volume real de self-play (o que valida a arquitetura até aqui foi um
smoke-test de 2 epochs em ~1.150 posições sintéticas — suficiente pra
provar que o pipeline inteiro funciona ponta a ponta, não pra jogar bem)
e (ii) o **default no WASM** (item que continua pendente, não fazia parte
do escopo desta rodada).

### 1a. Muros restantes (`wallsLeft`) como feature — **[IMPLEMENTADO]**

`NUM_FEATURES` foi de 332 para **354** (+22 = 2×11 buckets one-hot,
`wallsLeft` de 0 a `WALLS_PER_PLAYER`=10, por lado). `buildAccumulator`/
`buildAccumulatorQuant` (float e quantizada) e o incremental
`updateAccumulatorForMove(Quant)` foram atualizados — colocar um muro
decrementa o `wallsLeft` de quem jogou, e isso agora dispara um par
remove/add de feature no acumulador (só do lado de quem jogou; lance de
peão não mexe nisso). `training/quantize_nnue.py`,
`training/train_nnue.py` e `training/read_selfplay.py` (que tinha sua
própria cópia duplicada da lógica de feature, `to_dense_features` — não
detectada até esta sessão) foram todos atualizados em conjunto.

**Formato de self-play mudou também:** `TrainingSample::wallsLeftOwn/Opp`
já existiam desde antes (não precisou mudar o struct), só não eram usados
como feature — bastou ligar o fio no lado Python.

**Validado:** `nnue_incremental_check` continua em 0 divergências com o
novo bloco de features (4758 posições checadas); `nnue_sign_check` com
pesos treinados (mesmo que só o smoke-test) já não dá mais eval idêntico
entre "10 vs 0 muros" e a posição inicial — a rede agora enxerga a
diferença (ainda não aprendeu o suficiente pra valorizar corretamente,
questão de volume de treino, não de arquitetura).

### 1b. Eval assimétrica em posições simétricas (bug de perspectiva) — **[IMPLEMENTADO]**

Causa raiz era exatamente a identificada: `featOwnPawn`/`featWallH`/
`featWallV` usavam coordenada bruta do tabuleiro, sem espelhar
linha/coluna pra canonicalizar a perspectiva do jogador 1. Corrigido com
`mirroredPawnCell`/`mirroredWallSlot` (`nnue.hpp`) — espelham a linha
(`row -> N-1-row` peão / `WS-1-row` muro) quando `perspective==1`, coluna
nunca muda, perspectiva 0 continua sem transformação (é a canônica).

**Achado durante a implementação, não estava no escopo original:** o
formato de gravação de self-play (`TrainingSample` em `selfplay.hpp`) não
guarda a identidade física (0/1) de quem jogou cada lance — só a célula
crua do peão. Sem isso, seria impossível reconstruir no lado Python se uma
amostra precisa ser espelhada ou não. Resolvido no mesmo espírito já usado
pra `ownDist`/`oppDist` (calculado em C++, onde `mover`/`opp` ainda são
conhecidos com certeza): `selfplay.hpp` agora grava `ownPawn`/`oppPawn`/
`wallsH`/`wallsV`/`policyTarget` **já espelhados** pra perspectiva
canônica do mover, usando os mesmos `mirroredPawnCell`/
`mirrorWallBitboard`/`mirrorMoveForPerspective` de `nnue.hpp`. Efeito
colateral: **datasets `.bin` de self-play gravados ANTES desta mudança
ficam com essas 4 colunas em coordenada crua** (mesmo dtype/tamanho de
arquivo — não dá pra distinguir automaticamente) e precisam ser
regerados; não misturar com dados novos no mesmo treino.

**Validado com pesos treinados reais (não só aleatórios):** posição
inicial simétrica dá `nnue(persp0) == nnue(persp1)` exatamente (testado:
os dois em `-4`) — antes da correção dava `-3`/`70`.

### Bug relacionado encontrado e corrigido durante a implementação: checagem de arquitetura frágil ao carregar pesos

Nem `quantize_nnue.py` nem `NNUEWeightsQuant::loadFromFile` (`nnue.hpp`)
validavam o TAMANHO REAL do arquivo contra a arquitetura esperada —
`quantize_nnue.py` tinha seu próprio `NUM_FEATURES` duplicado e
desatualizado com uma checagem tautológica (comparava a contagem de
floats lida contra a mesma constante usada pra ler — nunca detectava
desalinhamento real); `loadFromFile` só confiava em `fread` falhar por
EOF no meio do caminho, o que por coincidência funcionou pra pegar um
arquivo do layout antigo (332 features) mas não é uma garantia estrutural
(se os tamanhos totais dos dois formatos ficassem parecidos o bastante,
um arquivo errado poderia "carregar" com sucesso e só desalinhar os
campos, silenciosamente). Os dois agora fazem uma checagem explícita de
bytes-no-disco vs. bytes-esperados-pela-arquitetura ANTES de ler, com
mensagem de erro clara em vez de falha silenciosa ou desalinhamento.

**Pendências reais desta prioridade (o que ainda falta):**
1. **Retreino de verdade** — gerar volume real de self-play (milhares de
   partidas, não as ~20 do smoke-test) com o `selfplay.hpp` atualizado, e
   rodar `train_nnue.py` por epochs suficientes (early-stopping já cuida
   de parar na hora certa). Pesos antigos (formato 332-feature) são
   **incompatíveis** — não dá pra fazer warm-start via `--init-from`
   (`try_load_train_state`/o novo checa de tamanho recusa e avisa, não
   corrompe).
2. **WASM ainda não usa NNUE por padrão** — `qr_new_game()` em
   `engine_wasm.cpp` nunca tenta carregar pesos automaticamente, e
   `gui_web/app.js` nunca chama `qr_load_nnue_weights`/
   `qr_set_eval_heuristic` (confirmado por grep, zero ocorrências) — os
   pesos ficam embutidos no bundle (`build_wasm.sh` já faz `--preload-file`
   se `data/nnue/nnue_weights_int8.bin` existir) mas nunca são carregados.
   Implementar um `autoLoadDefaultNnueOnce()` chamado no primeiro
   `qr_new_game()` (mesmo padrão já usado em `selfplay_main.cpp`/
   `arena.cpp`: tenta, cai pra heurístico com silêncio/log se não achar),
   ou no mínimo uma chamada em `app.js` no boot da página.

**Risco:** baixo pro que falta — a parte de arquitetura (a parte
realmente arriscada, por mudar `NUM_FEATURES`) já está feita e validada;
retreino é só tempo de máquina, WASM é mecânico (mesmo padrão já usado
duas vezes em selfplay/arena).

---

## Prioridade 2 — Pipeline de treinamento: checkpoint em JSON + resume + LR — **[IMPLEMENTADO — 2026-08]**

> **Status:** feito e testado ponta a ponta (treino real com dados de
> self-play, incluindo Ctrl+C e resume) nesta rodada.

**Bug real corrigido:** `--fresh` usava `action="store_true"` com
`default=True` em `train_nnue.py` — ou seja, `args.fresh` era **sempre
`True`**, independente de passar a flag ou não (não existia `--no-fresh`
pra desligar). Isso deixava `try_load_train_state`/o resume automático
documentado no topo do arquivo como código morto: nunca disparava na
prática. Corrigido: `FRESH_DEFAULT = False` + par `--fresh`/`--no-fresh`
(mesmo padrão de `--early-stop`/`--no-early-stop` já usado no arquivo).
Testado: rodar de novo com o mesmo `--ckpt-dir` agora realmente imprime
"RETOMANDO treino a partir de .../train_state.pt".

**JSON de pesos + configuração, com opção de reaproveitar ou não:**
- `train_config.json` gravado a cada epoch (e no handler de Ctrl+C) ao
  lado de `train_state.pt`, em `<ckpt-dir>/`: JSON legível (sem
  torch/numpy) com fingerprint da arquitetura, epoch, melhor métrica,
  caminho absoluto dos pesos (`last.bin`) e todos os hiperparâmetros de
  otimização/loss/QAT usados naquela rodada (`lr`, schedules, weight
  decay, early-stop, `qa`/`qb`, etc. — deliberadamente SEM caminhos de
  I/O como `--data`/`--out`, que são específicos da máquina/rodada, não do
  "treino" em si).
- `--resume-config PATH`: aponta pra um `train_config.json` de QUALQUER
  `--ckpt-dir` (não só o atual) e herda os hiperparâmetros de lá como
  default desta rodada — qualquer flag explícita nesta chamada continua
  ganhando prioridade (testado: `--epochs` explícito sobrepõe o herdado).
  Também usa o `weights_path` do JSON como `--init-from` automático se
  `--init-from` não foi passado. Diferente do resume via `--ckpt-dir`
  (que também traz otimizador/RNGs/histórico): isto só traz pesos +
  config, pensado pra começar um CICLO NOVO (ex.: bootstrap em cima de
  self-play novo) com os mesmos hiperparâmetros de uma rodada anterior,
  sem redigitar cada flag.
- **Opção de usar checkpoint anterior ou não:** `--fresh`/`--no-fresh`
  controla os dois (train_state.pt E train_config.json/auto-detect)
  igualmente — `--fresh` ignora ambos e começa do zero.
- **Learning rate inicial:** já existia (`--lr`, default `LR_DEFAULT`) e
  continua funcionando — testado e confirmado que o valor passado é
  aplicado e gravado no JSON corretamente.

---

## Prioridade 3 — Protocolo UCI + `.exe` standalone

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

## Backlog (pendente, prioridade menor que 1–3 acima)

Itens abaixo continuam válidos mas ficam depois de NNUE completa + treino
+ UCI na ordem de trabalho. Descrição condensada — histórico completo
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
- **Infra de teste estatístico (SPRT) motor-vs-motor** — complementa os
  gates de perft (que validam corretude, não força de jogo); script local
  rodando N partidas com SPRT pra aceitar/rejeitar mudanças por Elo.
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
|---|------|-------------------|-------|--------|
| 1 | NNUE: feature de muros restantes + correção de perspectiva | **Alto** | Médio (retreino) | ✅ arquitetura implementada (2026-08) — falta retreino real + default WASM |
| 2 | Treino: checkpoint JSON + resume + LR | Médio (infra) | Baixo | ✅ implementado (2026-08) |
| 3 | Protocolo UCI + `.exe` standalone | Alto (acessibilidade) | Baixo/médio | pendente |
| — | Continuation/countermove/correction history | Alto | Baixo/médio | backlog |
| — | LUT O(1) de peão | Baixo-médio | Baixo | backlog (medir antes) |
| — | Certificado de vitória | Médio | Médio-alto | backlog |
| — | Livro de abertura | Médio | Baixo | backlog |
| — | Lazy SMP | Alto (throughput) | Médio-alto | backlog |
| — | Gerência de tempo dedicada | Médio | Baixo | backlog |
| — | Infra SPRT | Metodológico | Nenhum | backlog |
| — | Zobrist/Undo enxuto | N/A | N/A | **não fazer sem medir** |
| — | Pondering | Baixo | Baixo | backlog |
| — | Internal Iterative Reduction (IIR) | Médio | Baixo | backlog |
| — | CAT, killers/history, LMR+PVS, RFP, LMP, perft gates, cache BFS/distância, solver de corrida | — | — | ✅ implementado |
