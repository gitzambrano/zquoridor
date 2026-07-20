# Zquoridor

Zquoridor é um motor de Quoridor para dois jogadores, jogado num
tabuleiro 9×9 com 10 muros por jogador. A variante de 4 jogadores está
fora do escopo do projeto.

O motor busca lances com negamax e poda alpha-beta, hoje avaliando cada
posição com uma função heurística. A meta de médio prazo é substituir
essa heurística por uma rede neural (NNUE) treinada em cima de partidas
geradas pelo próprio motor, o mesmo caminho já percorrido no **Zchezz**,
motor de xadrez irmão deste projeto.

A meta de força é vencer a esmagadora maioria dos jogadores humanos
(por volta de 99%) e, na sequência, medir o motor contra os
motores públicos de Quoridor mais fortes que existirem.

**Status atual**: a busca (negamax, alpha-beta, tabela de transposição,
quiescência de muro) está implementada e validada por testes de
regressão. A NNUE também está implementada — arquitetura, treino e
quantização — mas ainda não foi plugada na busca, que por enquanto
continua usando só a avaliação heurística (`evalSimple`). O caminho até
lá está descrito no Roadmap (Seção 5).

---

## 1. Estrutura de diretórios

```
zquoridor/
  src/                        # motor, produção
    rules.hpp                 # estado do tabuleiro, geração de lances,
                               #   Zobrist, BFS de caminho, avaliação heurística
    dsu.hpp                   # union-find com rollback (checagem de legalidade de muro)
    search.hpp                # negamax, alpha-beta, tabela de transposição,
                               #   ordenação de lances, quiescência de muro
    nnue.hpp                   # rede neural: acumulador incremental, float32 e int8
    main.cpp                    # benchmark de performance (self-play + custo da NNUE)
    selfplay.hpp                 # geração de dados de treino via self-play multi-thread
    selfplay_main.cpp             # linha de comando do self-play

  teste/                      # testes de corretude e benchmarks
    test_rules_sanity.cpp        # regressão de regras (pré-filtro + DSU)
    test_search_staging.cpp       # geração de lances estagiada vs. referência monolítica
    test_move_ordering.cpp         # ordenação de muros (bônus de toque no caminho)
    nnue_verify.cpp                 # paridade numérica C++ vs. Python (float32 e int8)
    bench_quiescence_toggle.cpp      # nós/s e nós totais com/sem quiescência, posição fixa

  training/                   # pipeline Python de treino da NNUE
    read_selfplay.py             # leitura do .bin de self-play via numpy
    train_nnue.py                  # treino em PyTorch (AdamW, weight decay com annealing,
                                    #   LR schedule com warmup, early stopping)
    train_nnue_numpy.py             # mesmo modelo, fallback sem PyTorch
    quantize_nnue.py                 # quantização int8/int16 pós-treino
    parity_check.py                   # lado Python da checagem de paridade com nnue_verify

  data/
    nnue/
      nnue_weights.bin              # pesos treinados, float32
      nnue_weights_int8.bin          # mesmos pesos quantizados -- é o que a engine carrega

  gui_web/                    # interface humano vs. motor (HTML/JS + WASM)
    engine_wasm.cpp               # casca extern "C" sobre rules.hpp e search.hpp
    build_standalone.py             # empacota tudo num quoridor.html único
    index.html / app.js              # interface, mobile-first
    quoridor.js / quoridor.wasm       # build compilado
    quoridor.html                      # versão standalone (sem servidor)

  build/                      # scripts de compilação
    build_bench.bat / .sh         # main.cpp + bench_quiescence_toggle.cpp
    build_tests.bat / .sh          # os 4 arquivos de teste/
    build_selfplay.bat / .sh        # selfplay_main.cpp
    build_wasm.bat / .sh             # engine_wasm.cpp -> quoridor.js/.wasm
    build_termux.sh                   # build ARM/Android via Termux
    build_all.bat / .sh                # roda bench + tests + selfplay; "wasm" como argumento inclui o wasm

  bin/                        # saída dos builds (não versionado)
```

Tudo em `teste/` inclui os headers de `src/` via `-Isrc` (passado por
todos os scripts de build) — nenhum `.hpp` é duplicado. `gui_web/engine_wasm.cpp`
inclui `../src/rules.hpp` e `../src/search.hpp` diretamente.

---

## 2. Como compilar e rodar

### 2.1 Windows (`build/*.bat`, precisa de MinGW-w64 `g++` no PATH)

| Script | Gera em `bin/` | Flags |
|---|---|---|
| `build_bench.bat` | `bench.exe`, `bench_quiescence_toggle.exe` | `-O3 -march=native -mavx2 -mfma` |
| `build_tests.bat` | `test_rules_sanity.exe`, `test_search_staging.exe`, `test_move_ordering.exe`, `nnue_verify.exe` | `-O2` |
| `build_selfplay.bat` | `selfplay.exe` | `-O3 -march=native -mavx2 -mfma -pthread` |
| `build_wasm.bat` | `gui_web/quoridor.js` + `.wasm` | requer `emsdk_env.bat` ativado antes |
| `build_all.bat` | os alvos nativos acima; `build_all.bat wasm` inclui o WASM | — |

### 2.2 Linux/macOS (`build/*.sh`)

Mesmos alvos e flags, sem extensão `.exe`:

```bash
chmod +x build/*.sh   # uma vez só
build/build_bench.sh
build/build_tests.sh
build/build_selfplay.sh
build/build_wasm.sh      # requer emsdk ativado no shell
build/build_all.sh       # bench+tests+selfplay; "build_all.sh wasm" inclui o wasm
```

### 2.3 Termux (Android/ARM)

ARM não tem AVX2/FMA (são extensões x86) — os scripts normais falham
nessas duas flags. Use `build_termux.sh`: mesmos níveis de otimização,
sem `-mavx2 -mfma`, `-march=native` sozinho já habilita NEON. Usa
`clang++` com fallback para `g++`.

```bash
pkg update && pkg install clang python
chmod +x build/build_termux.sh
build/build_termux.sh
```

WASM não é viável em Termux na prática (emsdk).

### 2.4 GUI web (humano vs. motor)

`gui_web/quoridor.js` + `quoridor.wasm` + `quoridor.html` já vêm
compilados no repositório — não precisa rodar `build_wasm` antes de
testar, só recompilar depois de mexer em `rules.hpp`/`search.hpp`/`engine_wasm.cpp`.

```bash
cd gui_web && python3 -m http.server 8000
# abrir http://localhost:8000/index.html
# (quoridor.html também funciona sozinho, sem servidor)
```

Funcionalidades:
- Colocação de muro por modo de seleção (toca Horizontal/Vertical, os
  slots legais acendem no tabuleiro) ou drag-and-drop.
- Barra de muros restantes por jogador.
- Avaliação do motor exibida no histórico de lances (leitura de posição
  do motor após o próprio lance).
- Modal de nova partida: escolher lado e força do motor (tempo por
  lance) antes de começar.

Para recompilar depois de mexer no motor:

```bash
source /caminho/pro/emsdk/emsdk_env.sh   # ou emsdk_env.bat no Windows
build/build_wasm.sh                        # ou build_wasm.bat
```

Limitação conhecida: o lance do motor roda de forma síncrona no thread
principal e trava a aba durante o tempo configurado (200 ms a 4 s).
Migrar para Web Worker é possível sem mudar `engine_wasm.cpp`, só como o
JS carrega o módulo — ainda não implementado.

### 2.5 Self-play (geração de dados de treino)

```bash
build/build_selfplay.sh   # ou .bat
bin/selfplay --games 2000 --out data/selfplay_001.bin
```

Gera partidas do motor heurístico contra ele mesmo e grava o resultado
direto no formato binário lido pelo treino (Seção 2.6) — não existe
passo de pré-processamento no lado Python. O binário É o dataset: basta
`numpy.fromfile(path, dtype=SAMPLE_DTYPE)` (ver `training/read_selfplay.py`)
para ter um array estruturado pronto para virar tensores.

| Flag | Default | O que faz |
|---|---|---|
| `--games N` | 1000 | número de partidas a jogar |
| `--out PATH` | (obrigatório) | arquivo binário de saída |
| `--depth N` | 40 | profundidade máxima da busca |
| `--time-ms N` | 100 | orçamento de tempo por lance, em ms |
| `--threads N` | núcleos disponíveis | partidas jogadas em paralelo |
| `--opening-plies N` | 6 | quantos lances iniciais de cada partida sofrem ruído aleatório |
| `--epsilon F` | 0.25 | probabilidade de jogar um lance aleatório dentro da janela de abertura acima (evita que toda partida comece igual) |
| `--max-plies N` | 300 | corte de segurança por partida; partidas que passam disso são descartadas, não gravadas |
| `--seed N` | 1 | semente do gerador aleatório |

**Não existe geração automática de vários arquivos por tamanho de shard**
(por exemplo "gere 500 mil posições, mas quebre em arquivos de até 2GB
cada"). Hoje, para ter vários arquivos menores em vez de um único grande,
rode `bin/selfplay` várias vezes com `--seed`/`--out` diferentes — o
treino (Seção 2.6) já aceita múltiplos arquivos via `--data` repetido,
lista separada por vírgula, diretório ou glob, e não carrega tudo em RAM
de uma vez. Ver Fase B do roadmap (Seção 5) para a ideia de automatizar
isso dentro do próprio `bin/selfplay`.

### 2.6 Treino da NNUE (Python)

```bash
pip install torch numpy --break-system-packages   # ou só numpy para o fallback
cd training
python3 train_nnue.py --data ../data/selfplay_*.bin \
    --out ../data/nnue/nnue_weights.bin --plot-dir ../data/plots
# sem PyTorch:
python3 train_nnue_numpy.py --data ../data/selfplay_040k.bin --epochs 60 \
    --out ../data/nnue/nnue_weights.bin
# quantizar pesos já treinados sem re-treinar:
python3 quantize_nnue.py ../data/nnue/nnue_weights.bin ../data/nnue/nnue_weights_int8.bin
```

`train_nnue.py` (PyTorch) e `train_nnue_numpy.py` (fallback sem PyTorch)
aceitam praticamente as mesmas flags — as únicas diferenças são que a
versão numpy não tem `--device` nem `--vram-budget-gb` (não há GPU nesse
caminho) e `--batch-size` é sempre um inteiro fixo, nunca `"auto"`.

**Dados e checkpoint**

| Flag | Default | O que faz |
|---|---|---|
| `--data PATH` | (obrigatório) | arquivo(s) `.bin` de self-play; pode repetir a flag, passar lista separada por vírgula, um diretório ou um glob |
| `--val-data PATH` | nenhum | arquivo(s) usados só para validação (mesmas regras de `--data`); se omitido, a validação sai de uma fração de `--data` |
| `--val-split F` | 0.1 | fração de `--data` reservada para validação quando `--val-data` não é passado |
| `--init-from PATH` | nenhum | pesos `.bin` existentes para continuar um treino em vez de começar do zero |
| `--seed N` | 0 | semente do gerador aleatório |

**Otimização**

| Flag | Default | O que faz |
|---|---|---|
| `--epochs N` | 60 | número de épocas |
| `--batch-size N\|auto` | `auto` | tamanho do batch; `auto` calcula a partir de `--vram-budget-gb` (só em `train_nnue.py`) |
| `--lr F` | 1e-3 | taxa de aprendizado inicial |
| `--lr-min F` | 1e-5 | piso da taxa de aprendizado nos schedules que decaem |
| `--lr-schedule none\|step\|exponential\|cosine` | `cosine` | como a taxa de aprendizado muda ao longo do treino |
| `--warmup-epochs N` | 2 | épocas de aquecimento no início, antes do schedule normal começar |
| `--step-size N` | 10 | épocas por degrau, só usado em `--lr-schedule=step` |
| `--step-gamma F` | 0.5 | fator de redução a cada degrau, só em `--lr-schedule=step` |
| `--exp-gamma F` | 0.97 | fator de decaimento por época, só em `--lr-schedule=exponential` |
| `--device cpu\|cuda` | `cuda` se disponível, senão `cpu` | só em `train_nnue.py` |

**Weight decay com annealing** — regulariza mais forte no início do
treino e afrouxa perto do fim; aplicado só em matrizes de peso, nunca em
bias (AdamW com weight decay desacoplado):

| Flag | Default | O que faz |
|---|---|---|
| `--weight-decay F` | 1e-4 | valor inicial do weight decay |
| `--weight-decay-min F` | 0.0 | valor para o qual o weight decay converge ao final do treino |
| `--wd-schedule none\|constant\|linear\|cosine` | `cosine` | como o weight decay muda entre o valor inicial e o mínimo |

**Early stopping** — para de treinar quando a métrica monitorada para de
melhorar, e por padrão restaura os pesos do melhor epoch (não os do
último) na exportação final:

| Flag | Default | O que faz |
|---|---|---|
| `--early-stop` / `--no-early-stop` | ligado | liga/desliga early stopping |
| `--patience N` | 8 | quantas épocas sem melhora até parar |
| `--min-delta F` | 1e-4 | melhora mínima para contar como "melhorou" |
| `--monitor val_loss\|val_outcome\|val_score\|val_policy\|val_policy_acc` | `val_loss` | qual métrica de validação é monitorada |
| `--no-restore-best` | desligado | exporta os pesos do último epoch em vez dos do melhor |
| `--ckpt-dir PATH` | nenhum | diretório onde salvar `best.bin` (atualizado a cada melhora) e `last.bin` |

**Orçamento de memória (RAM/VRAM)** — evita estourar memória com
datasets grandes sem precisar calcular batch/chunk na mão:

| Flag | Default | O que faz |
|---|---|---|
| `--vram-budget-gb F` | 6.0 | orçamento de VRAM usado para calcular `--batch-size=auto`; só em `train_nnue.py` |
| `--ram-budget-gb F` | 32.0 | orçamento de RAM usado para calcular `--chunk-size=auto` |
| `--ram-chunk-fraction F` | 0.25 | fração do orçamento de RAM reservada ao buffer usado para embaralhar os dados |
| `--chunk-size N\|auto` | `auto` | quantas amostras ficam em memória por vez ao ler os arquivos `.bin`; `auto` calcula a partir de `--ram-budget-gb` |

**Pesos de loss e quantização (QAT)**

| Flag | Default | O que faz |
|---|---|---|
| `--w-score F` | 0.3 | peso da cabeça auxiliar (imita `evalSimple`) na loss total |
| `--w-outcome F` | 1.0 | peso da cabeça WL (resultado real da partida) na loss total |
| `--w-policy F` | 1.0 | peso da cabeça de política na loss total |
| `--qa N` | 255 | fator de quantização QA; precisa bater com `nnue.hpp` e `quantize_nnue.py` |
| `--qb N` | 64 | fator de quantização QB; precisa bater com `nnue.hpp` e `quantize_nnue.py` |

**Saída**

| Flag | Default | O que faz |
|---|---|---|
| `--out PATH` | (obrigatório) | caminho de saída dos pesos treinados, float32 |
| `--no-quantize` | desligado | pula a quantização automática pós-treino |
| `--quant-out PATH` | `<out>` com sufixo `_int8` | caminho de saída dos pesos quantizados |
| `--plot-dir PATH` | nenhum | diretório para salvar plots de convergência/validação em PNG |
| `--log-every N` | 1 | a cada quantas épocas imprimir progresso no terminal |

`quantize_nnue.py` não usa flags nomeadas, só dois argumentos
posicionais: `quantize_nnue.py <entrada.bin> <saida_int8.bin>`.

---

## 3. Status de implementação

Esta seção descreve o que já está pronto no motor, em ordem: primeiro
como uma posição é representada e como os lances legais são encontrados,
depois como a busca decide qual lance jogar, depois o estado da rede
neural, depois os testes que garantem que nada disso quebrou.

### Regras e geração de lances (`rules.hpp`)

- **Representação da posição** (`State`): guarda onde estão os dois
  peões, quais muros já foram colocados (como dois mapas de bits, um
  para muros na horizontal e outro para vertical), de quem é a vez, e um
  hash da posição (Zobrist) usado para reconhecer posições repetidas de
  forma rápida.
- **Geração de lances** (`legalMoves`, `pawnStepMoves`, `legalWallMoves`):
  lista todos os movimentos de peão e colocações de muro válidos na
  posição atual. Colocar um muro só é legal se ainda sobrar pelo menos
  um caminho até a meta para os dois jogadores — essa checagem é a parte
  mais cara da geração de lances, então antes de checar de verdade
  (que exige uma busca no tabuleiro inteiro) existe um filtro geométrico
  rápido que já descarta a maioria dos muros obviamente ilegais, e a
  checagem final usa uma estrutura de dados (union-find com desfazer,
  `dsu.hpp`) que evita refazer esse trabalho do zero a cada muro
  candidato.
- **Busca de caminho no tabuleiro** (`hasPathToGoal`, `shortestPathLen`,
  `shortestPathTouchSlots`, `pathRobustness`): quatro variações da mesma
  busca em largura (BFS) sobre as células do tabuleiro, cada uma
  respondendo uma pergunta diferente — existe algum caminho até a linha
  de meta? qual o caminho mais curto? quais muros, se colocados, cortam
  esse caminho mais curto? e quão frágil é esse caminho, isto é, quantos
  desvios de custo baixo existem se um muro novo bloquear exatamente ele?
  Essa última pergunta (robustez do caminho) é usada tanto na avaliação
  da posição quanto para decidir quando a busca precisa "olhar mais
  fundo" antes de confiar no resultado (quiescência, ver abaixo).
- **Avaliação heurística** (`evalSimple`): a função que dá uma nota
  numérica para uma posição, usada por enquanto no lugar da rede neural.
  Soma, com pesos ajustáveis (`EvalWeights`): a diferença de distância
  até a meta entre os dois jogadores, a mobilidade do peão (quantas
  casas ele pode alcançar), a urgência (o quanto vale estar perto de
  vencer, que pesa mais quando a distância já é pequena) e a robustez do
  caminho mencionada acima.

### Busca (`search.hpp`)

- **Negamax com poda alpha-beta**: o algoritmo de busca em si — explora
  sequências de lances futuros, avalia as posições no final de cada
  sequência com `evalSimple`, e descarta ramos que já provou que não vão
  ser escolhidos, sem precisar terminar de explorá-los.
- **Tabela de transposição**: um cache de posições já analisadas (por
  hash Zobrist), guardando profundidade, resultado e o melhor lance
  encontrado, para não reanalisar do zero uma posição que a busca já
  visitou por outro caminho de lances.
- **Ordenação de lances** (killer moves, history heuristic,
  `orderWallMoves`): a ordem em que os lances candidatos são testados
  importa muito para a eficácia da poda — testar primeiro os lances mais
  promissores corta mais ramos, mais cedo. O motor prioriza lances que já
  causaram poda em posições parecidas antes (killer moves), lances que
  historicamente se saíram bem (history heuristic), e, para muros
  especificamente, muros que atrapalham o caminho mais curto do
  adversário (`WALL_TOUCH_BONUS`).
- **Quiescência de muro**: perto do fim de uma busca, se o melhor lance
  encontrado for um muro que piora bastante o caminho do adversário, o
  motor estende a busca por mais alguns lances antes de aceitar aquele
  resultado, em vez de parar ali e potencialmente errar por não ter
  olhado a consequência imediata da jogada (esse é o problema clássico
  do "efeito horizonte" em motores de jogos). O quanto se estende é
  limitado (`QS_MAX_EXTRA_PLIES`) e o gatilho para considerar um muro
  "crítico" ainda usa valores iniciais, não calibrados por tuning
  (`QS_CRITICAL_BFS_DELTA`, `QS_CRITICAL_ROBUSTNESS_DROP_TO`). Pode ser
  desligada em runtime com `Negamax::setQuiescenceEnabled(false)`, sem
  recompilar — útil para comparar o motor com e sem essa extensão em
  benchmark, ou para descartar essa parte como causa de um bug durante
  debug.

### NNUE (`nnue.hpp`)

A rede neural em si já está implementada — arquitetura, cálculo direto
(forward) e quantização — mas ainda não é usada pela busca, que hoje
continua avaliando posições só com `evalSimple`. Detalhes da arquitetura
na Seção 4; o plano para plugá-la na busca está na Fase B do roadmap
(Seção 5).

### Testes

- `test_rules_sanity.cpp`: regressão de regras e do filtro geométrico +
  union-find usados para checar legalidade de muro.
- `test_search_staging.cpp`: compara a geração de lances estagiada
  (a usada em produção) com uma implementação de referência mais simples
  e direta, para garantir que a versão otimizada não mudou o resultado.
- `test_move_ordering.cpp`: valida o bônus de ordenação de muro (que ele
  favorece os lances certos e não altera a legalidade deles).
- `nnue_verify.cpp`: confirma que a implementação C++ da rede produz os
  mesmos números que a implementação Python, tanto em float32 quanto na
  versão quantizada em int8.
- `bench_quiescence_toggle.cpp`: mede nós por segundo e número total de
  nós com a quiescência ligada e desligada, numa trilha de posições fixa
  e profundidade fixa (não orçamento de tempo, que varia muito de rodada
  para rodada e mascara o efeito medido).

---

## 4. Arquitetura da NNUE

`332 → 256 (acumulador, ativação SCReLU)` seguido de três cabeças
independentes (nada compartilhado além do acumulador):

| Cabeça | Forma | Treinada contra | Papel |
|---|---|---|---|
| WL (resultado) | `256→32→1` | resultado real da partida (+1/-1), BCE | é a que a busca vai consumir (`forwardValueWLQuant`) |
| Auxiliar | `256→32→1` | `evalSimple` no momento do lance, MSE | andaime de treino enquanto o self-play ainda vem da heurística; removida quando o self-play passar a vir da própria rede |
| Política | `256→209` logits | lance jogado, CrossEntropy | ordenação de lances na busca, não é probabilidade de vitória |

A rede não tem conceito fixo de branco/preto: toda entrada e toda saída
são relativas a uma perspectiva (`buildAccumulator(state, perspective)`),
sempre avaliada do ponto de vista de quem vai jogar — o negamax cuida da
troca de sinal entre níveis, a rede nunca precisa saber "quem é branco".

Features de entrada (332): 81 peão próprio + 81 peão oponente + 64 muro
horizontal + 64 muro vertical + 21 bucket de distância BFS própria + 21
bucket de distância BFS oponente (one-hot, não valor cru — mantém a
escala simétrica sob quantização).

Quantização: QAT (durante o treino, não pós-hoc) — `QA=255`/`QB=64`
fixos antes de treinar, os pesos são limitados ao range representável em
int8/int16 a cada passo do otimizador. O acumulador é sempre atualizado
de forma incremental (nenhum lance força recálculo completo); o bucket
de distância fica cacheado no acumulador para não recalcular a BFS a
cada atualização.

---

## 5. Roadmap

Ordem geral: primeiro deixar o motor heurístico (negamax + `evalSimple`,
sem rede) o mais forte possível — é ele quem gera os dados de bootstrap.
Depois treinar a NNUE nesses dados, plugá-la na busca, e então trocar a
fonte do self-play para a própria rede (loop de auto-melhoria, começando
de uma heurística forte em vez de pesos aleatórios).

### Fase A — Fortalecer o motor heurístico (sem rede)

Tudo aqui usa só `evalSimple`, sem NNUE.

1. Teste de força com quiescência ligada vs. desligada (`bench_quiescence_toggle`,
   flag `setQuiescenceEnabled`) — partidas diretas, não só nós/s.
2. Continuation history (1-ply) para combos de muro sequenciais — a
   history hoje é `[lado][lance]`, sem contexto do lance anterior.
3. LMR em muros ordenados tarde, null-move pruning com guarda de
   zugzwang (`wallsLeft[side] > 0` e gap não muito apertado,
   futility/razoring raso em lance de peão quiet com profundidade ≤ 2,
   PVS — nessa ordem, todos dependentes de ordenação já madura (item 2).
4. Calibrar os limiares da quiescência de muro
   (`QS_CRITICAL_BFS_DELTA`, `QS_CRITICAL_ROBUSTNESS_DROP_TO`), hoje
   valores iniciais não calibrados.
5. Ladder interno de ELO para o motor heurístico puro — cada otimização
   acima validada por partidas diretas, não só por nós/s e profundidade.
   Serve também de baseline de força para comparar com a NNUE mais tarde
   (Fase D).

### Fase B — Bootstrap: self-play do motor heurístico → treino da NNUE

Gera dados de treino com o motor da Fase A e usa isso para dar à NNUE seu
primeiro conjunto de pesos utilizável.

6. **Sharding automático no self-play**: hoje `bin/selfplay` só grava um
   arquivo `.bin` por execução (Seção 2.5); rodar milhões de posições de
   uma vez gera um arquivo único enorme, que pode estourar RAM na hora do
   treino mesmo com o carregamento em chunks. Adicionar algo como
   `--games-per-shard N`, fazendo o próprio `bin/selfplay` girar a saída
   em vários arquivos automaticamente em vez de depender de rodar o
   binário várias vezes na mão.
7. Gerar um volume maior de self-play numa máquina com mais
   núcleos/GPU (`bin/selfplay`, Seção 2.5).
8. Rodar um treino completo com o pipeline de regularização
   (`--early-stop --plot-dir`, Seção 2.6) e registrar o resultado.
9. Plugar a NNUE na busca: trocar `evalSimple` por `forwardValueWLQuant`
   na folha do negamax (`search.hpp`).
10. Medir o custo real de nós/s da rede quantizada dentro do laço de
    busca (hoje só medido como microbenchmark isolado, fora do laço).
11. Suíte de força NNUE vs. heurística — a NNUE só substitui `evalSimple`
    se vencer o ladder da Fase A de forma estatisticamente clara.

### Fase C — Self-play da própria NNUE (loop de auto-melhoria)

Começa depois que a Fase B mostrar que a NNUE joga pelo menos tão bem
quanto `evalSimple` (item 11).

12. Gerar self-play usando a NNUE integrada (não mais `evalSimple`) como
    avaliadora da busca.
13. Zerar/remover a cabeça auxiliar (`--w-score 0` ou remoção dos campos
    e do termo de loss correspondente) — ela existe só para imitar
    `evalSimple` enquanto o self-play ainda vem da heurística.
14. Retreinar a NNUE sobre o novo dataset (gerado pela própria rede) e
    repetir o ciclo: joga melhor → gera dados melhores → treina de novo.
15. Migrar `pathRobustness` de termo de `evalSimple` para feature de
    entrada da NNUE — só faz sentido depois que `evalSimple` deixa de
    ser a fonte de verdade.

### Fase D — Infraestrutura e validação externa

16. `engine_cli/`: executável falando um protocolo tipo-UCI por
    stdin/stdout (`uci`, `isready`, `position`, `go movetime`/`go depth`,
    `bestmove`, `stop`, `quit`), reusando `rules.hpp`/`search.hpp`/`nnue.hpp`
    sem modificação. Notação: casas `a1`-`i9`, muro = casa + orientação
    (`e3h`/`e3v`). Só depois do motor estar numa versão mais madura
    (Fase C razoavelmente avançada).
17. Benchmark contra motores externos open-source
    (`github.com/dzionek/quoridorAI`, `github.com/mehrshad-sdtn/AI-Quoridor`)
    e partidas contra humanos reais de níveis variados, para calibrar a
    meta de ~99% empiricamente. Não existe um benchmark público único de
    "melhor motor de Quoridor" — essas duas fontes junto com o ladder
    interno (item 5) são a validação disponível.
