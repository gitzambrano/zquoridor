# Plano: Híbrido MCTS + Alpha-Beta (MCαβ) para zquoridor

Documento de especificação para implementação por outro agente. Todo o texto
abaixo assume acesso ao repositório `zquoridor` (arquivos referenciados:
`src/search.hpp`, `src/nnue.hpp`, `src/rules.hpp`, `tools/arena/arena.cpp`,
`tools/selfplay/selfplay.hpp`, `tools/spsa/tune_spsa.cpp`).

## 0. Requisito não-negociável

**O caminho de busca AB puro (`Negamax::chooseMove`/`Negamax::negamax`) não
pode perder nem 1 nó/s nem 1 ply de profundidade em relação ao estado atual.**
Isso governa toda decisão de design abaixo: o híbrido é aditivo, nunca
modifica lógica existente, e roda inteiramente como um módulo novo que
consome `Negamax` como biblioteca. Qualquer alteração em `search.hpp` listada
na Seção 2 é estritamente uma adição (novo método público) ou uma mudança
de visibilidade (`private` → `public` de um método já existente, sem tocar
no corpo) — zero linhas do hot path de `negamax()` são reescritas.

## 1. Técnica escolhida: MCαβ (alpha-beta rollouts)

Referência: Huang, "Pruning Game Tree by Rollouts" (AAAI 2015); implementado
em produção no Scorpio (Daniel Shawul), onde MCTS com rollouts AB atinge
força equivalente ao AB puro quando a árvore em memória é limitada a zero, e
ganha força conforme a árvore cresce — ou seja, o ganho vem estritamente do
reaproveitamento de esforço de busca por meio de seleção best-first guiada
por política, não de nenhum tipo de mágica do MCTS em si.

Mecânica: constrói-se uma árvore best-first no topo (seleção via PUCT,
usando a cabeça de política já treinada como prior). Cada simulação desce a
árvore, expande um nó folha e o avalia com uma chamada real ao AB existente
(`negamax`) em profundidade rasa — não com playout aleatório nem só o valor
cru da NNUE. O backup na árvore é minimax (propaga o melhor valor), não
média Monte Carlo — é isso que preserva solidez tática (mate, corte de
caminho), ponto fraco conhecido de MCTS com valor de rede pura.

## 2. Escopo do v1

- **Modo NNUE apenas.** O híbrido depende da cabeça de política
  (`forwardPolicyQuant`, `POLICY_OUT=209`) para os priors do PUCT. Em modo
  Heurístico não há política treinada — MCαβ viraria UCT cego, que não é o
  objetivo. `chooseMoveMCAB` deve `assert`/retornar erro se
  `evalMode != EvalMode::NNUE`.
- **Single-thread no v1.** Paralelismo (virtual loss, ABDADA-like busy flag
  para os rollouts AB) fica para v2, depois de validado o ganho de Elo do
  v1 single-thread. Não vale complicar sincronização antes de saber se a
  técnica compensa neste jogo.
- **Sem ruído de Dirichlet na raiz por padrão** (isso é para diversidade de
  self-play/treino, não para força em partida — teria que ser uma flag
  separada, ver Seção 9, item `mcabRootNoiseEpsilon`, default 0 em
  arena/match).
- Ambos os modos — AB puro (`Negamax::chooseMove`) e híbrido
  (`MCABSearch::chooseMoveMCAB`) — devem coexistir na mesma engine/binário
  e ser selecionáveis por flag de linha de comando, exatamente como hoje
  `--e1-heuristic`/`--e2-nnue` selecionam modo de avaliação por engine
  (Seção 10).

## 3. Mudanças mínimas em `src/search.hpp`

Duas alterações, ambas aditivas:

### 3.1 Tornar `resetOrderingState()` público

Hoje está na seção `private:` (linha ~519). Mover a declaração para a
seção `public:` (junto de `clearTT()`/`clearXDistCache()`, que já seguem o
mesmo padrão de "reset de estado exposto para quem orquestra buscas
externas"). Nenhuma linha do corpo do método muda.

### 3.2 Novo método público `searchLeaf`

Adicionar próximo a `clearTT()`/`clearXDistCache()`:

```cpp
// Avaliador de folha para o módulo híbrido MCTS+AB (mcab.hpp). Roda uma
// busca AB completa (mesmo negamax(), mesmas extensões de quiescência de
// muro, LMR/PVS, TT, killers/history desta instância) a partir de uma
// posição arbitrária `s`, até `depth` plies. NÃO reseta TT/killers/history
// entre chamadas -- o chamador (MCABSearch) controla isso 1x por
// chooseMoveMCAB() via resetOrderingState()/clearTT(), do mesmo jeito que
// chooseMove() já faz 1x por busca, não por nó. Isso é o que permite reuso
// de TT entre folhas vizinhas na árvore MCTS sem custo extra de
// bookkeeping.
//
// seedAcc, se não-nulo, é usado como acumulador NNUE já pronto (construído
// incrementalmente pelo chamador via makeChildAccPair ao longo do caminho
// da raiz até `s`) -- evita o rebuild completo (~330 features,
// buildAccPairRoot) a cada folha. Se nulo, cai no comportamento antigo
// (buildAccPairRoot local), usado só por quem chamar isto fora de um
// contexto de árvore incremental.
int searchLeaf(const State& s, int depth, SearchStats& stats,
                RepetitionTable& reptbl, AccPair* seedAcc = nullptr) {
    AccPair local;
    AccPair* accForSearch = nullptr;
    if (evalMode == EvalMode::NNUE) {
        if (seedAcc) {
            local = *seedAcc;
        } else {
            local = buildAccPairRoot(s, &xdistCache);
        }
        nnueAccStack[0] = local;
        accForSearch = &nnueAccStack[0];
    }
    rootDepth = depth;
    return negamax(s, depth, -SCORE_INF, SCORE_INF, stats, reptbl, accForSearch);
}
```

Nada além disso muda em `search.hpp`. `buildAccPairRoot`, `makeChildAccPair`,
`forwardPolicyQuant`, `nnueEvalInt`, `moveToPolicyIndex`, `legalMoves`,
`applyMove`, `winner` já são funções livres/públicas em `nnue.hpp`/`rules.hpp`
— o módulo novo os usa diretamente, sem precisar de mais nenhum método em
`Negamax`.

## 4. Novo arquivo: `tools/common/mcab.hpp` (NÃO em `src/`)

Ponto crítico de arquitetura, ligado direto ao requisito de compatibilidade
retroativa do `arena.cpp` (Seção 4.4): este arquivo **não pode morar em
`src/`**. Tudo em `src/` é versionado por commit e é o que
`run_arena.py`/`compile_arena()` faz checkout via `git worktree add` para
cada ref (`dir1`/`dir2`, ver Seção 4.4) — um ref antigo (anterior a este
plano) não teria `src/mcab.hpp` no worktree, e `arena.cpp` quebraria a
compilação ao tentar incluí-lo condicionado a esse ref.

`tools/common/mcab.hpp` fica FORA da árvore versionada-por-ref: é sempre a
versão do `PROJECT_ROOT` atual (HEAD), igual a `tools/arena/arena.cpp`,
`tools/selfplay/selfplay_main.cpp` e `tools/spsa/tune_spsa.cpp` já são hoje
— nenhum desses arquivos vem do worktree de ref1/ref2, só `search.hpp` (e
o que ele inclui: `nnue.hpp`, `rules.hpp`, `cat.hpp`, `endgame_race.hpp`,
`dsu.hpp`) vem. `mcab.hpp` inclui-se por caminho relativo
(`#include "../common/mcab.hpp"` a partir de `tools/arena/arena.cpp`,
`tools/selfplay/`, `tools/spsa/`) — nenhuma mudança no comando de
compilação de `run_arena.py` é necessária (confirmado: `compile_arena()`
hoje não passa `-I` nenhum, contando com resolução relativa padrão do
compilador para os `#include "..."` de dentro de `search.hpp`; o mesmo
mecanismo resolve `../common/mcab.hpp` a partir de `arena.cpp`).

### 4.1 Duas engines, dois tipos C++ diferentes — `mcab.hpp` como template

Como `arena.cpp` compila `qr_e1::Negamax` e `qr_e2::Negamax` como TIPOS
DIFERENTES (namespaces diferentes via o truque `#define qr qr_eN`),
`MCABSearch` não pode ser uma classe concreta amarrada a `qr::Negamax` — é
um **template**, parametrizado pelo tipo da engine e pelos tipos de estado
associados:

```cpp
// tools/common/mcab.hpp
template <typename Eng, typename StateT, typename MoveT, typename MoveListT,
          typename AccPairT, typename RepTblT, typename SearchStatsT>
class MCABSearch {
    // ... (Seções 4.2-4.3, 5) ...
    // Chamadas a buildAccPairRoot(...), makeChildAccPair(...),
    // forwardPolicyQuant(...), moveToPolicyIndex(...), legalMoves(...),
    // applyMove(...), winner(...) são feitas SEM QUALIFICAÇÃO DE NAMESPACE
    // dentro do template -- resolvidas por ADL (argument-dependent lookup)
    // no ponto de instanciação: como StateT/AccPairT/etc SÃO qr_e1::State/
    // qr_e1::AccPair (ou qr_e2::...) quando instanciado, o compilador acha
    // automaticamente qr_e1::buildAccPairRoot (ou qr_e2::...) -- o mesmo
    // princípio que já faz `applyMove`/`legalMoves` funcionarem sem
    // qualificação dentro do próprio search.hpp de cada ref.
};
```

Instanciação em `arena.cpp` (dentro do escopo onde `qr_e1`/`qr_e2` já
foram `#include`ados):

```cpp
using Mcab1 = MCABSearch<qr_e1::Negamax, qr_e1::State, qr_e1::Move,
                          qr_e1::MoveList, qr_e1::AccPair,
                          qr_e1::RepetitionTable, qr_e1::SearchStats>;
using Mcab2 = MCABSearch<qr_e2::Negamax, qr_e2::State, qr_e2::Move,
                          qr_e2::MoveList, qr_e2::AccPair,
                          qr_e2::RepetitionTable, qr_e2::SearchStats>;
```

Em `selfplay_main.cpp`/`tune_spsa.cpp` (single-namespace `qr`, sem o
truque de rename) a instanciação é análoga, só com `qr::` no lugar de
`qr_e1::`/`qr_e2::`.

### 4.2 Independência de `mcab.hpp` em relação a `search.hpp`

`search.hpp` **não inclui `mcab.hpp`** em nenhuma direção — dependência
unidirecional (`mcab.hpp` → tipos/funções de `search.hpp`/`nnue.hpp`/
`rules.hpp` via os parâmetros de template acima). Nenhum binário que só
usa AB puro (ex. `tests/*.cpp`, `gui_web/engine_wasm.cpp`) paga qualquer
custo de compilação ou risco de regressão por causa deste arquivo, porque
nunca o inclui.

## 4.3 Conversão score → Q, estruturas de dados e pilha incremental

*(conteúdo movido para dentro do template `MCABSearch<...>` descrito acima
— ver Seções 4.3.1-4.3.3 abaixo; a numeração de sub-seções da versão
anterior deste documento — "4.1 Conversão score → Q", "4.2 Estruturas de
dados", "4.3 Pilha de acumuladores incremental" — permanece tecnicamente
idêntica, apenas movida para dentro do arquivo `tools/common/mcab.hpp` em
vez de `src/mcab.hpp`.)*

### 4.3.1 Conversão score → Q

`negamax`/`searchLeaf` devolvem score em unidades inteiras de
`NNUE_EVAL_SCALE` (constante = 200, `nnue.hpp`), definidas por
`nnueEvalInt = round(logit_WL * NNUE_EVAL_SCALE)`. Os priors do PUCT
(`forwardPolicyQuant`) são um softmax em `[0,1]`. Para que o termo de
exploração do PUCT (`c_puct * P * sqrt(N)/(1+n)`) seja comparável ao termo
de exploração (Q), o score de folha precisa ser convertido para a mesma
escala de probabilidade usando a MESMA transformação usada em treino
(`nnueWinProbQuant`, `nnue.hpp`):

```cpp
inline double scoreToQ(int score, double scale /* = NNUE_EVAL_SCALE por padrão */) {
    return 1.0 / (1.0 + std::exp(-(double)score / scale));
}
```

Com `scale = NNUE_EVAL_SCALE` isso é matematicamente idêntico a
`nnueWinProbQuant` aplicado ao logit implícito no score — não é uma
calibração nova, é a mesma curva já usada para gravar `evalNNUE` nos `.bin`
de self-play (`arena.cpp`/`selfplay.hpp`). Scores de mate/corrida
(`SCORE_INF-1`, `RACE_SCORE_BASE - dtm`) saturam para perto de 0/1 sem
overflow (double aguenta `exp(-5000)` tranquilo). `scale` fica exposto como
parâmetro tunável (`mcabScoreScale`, Seção 9) para permitir recalibração
futura sem mexer em `NNUE_EVAL_SCALE` (que é usado por toda a busca AB).

### 4.3.2 Estruturas de dados

Node pool linear (`std::vector<MCABNode>`, sem alocação por nó — índices,
não ponteiros — cache-friendly e sem custo de `new`/`delete` por
simulação):

```cpp
template <typename StateT, typename MoveListT>
struct MCABNode {
    StateT state;
    int side = 0;                // state.turn, cacheado
    bool expanded = false;
    bool terminal = false;
    int terminalScore = 0;       // score (unidades NNUE_EVAL_SCALE); convertido via scoreToQ na hora de usar
    MoveListT moves;              // gerado 1x na expansão (legalMoves(state))
    std::vector<float> P;        // prior por lance, mesmo índice de `moves`
    std::vector<float> N;        // visitas por aresta
    std::vector<float> W;        // soma de valor por aresta (Q = W/N)
    std::vector<int32_t> child;  // índice no pool, -1 = não expandido
    int totalN = 0;
};
```

Nenhum `AccPair` é armazenado por nó (motivo: economia de memória — um
`AccPair` custa ~2 KB, ver comentário em `search.hpp` sobre
`nnueAccStack`; guardar isso em dezenas de milhares de nós custaria dezenas
a centenas de MB à toa).

### 4.3.3 Pilha de acumuladores incremental (não por nó, por caminho de descida)

Em vez de guardar `AccPair` em cada `MCABNode`, `MCABSearch` mantém sua
própria pilha, análoga a `Negamax::nnueAccStack`:

```cpp
std::vector<AccPairT> mcabAccStack; // tamanho = mcabMaxTreeDepth (Seção 9), reservado 1x
```

A cada simulação, a caminhada de seleção raiz→folha recomputa a pilha
incrementalmente: `mcabAccStack[0]` = acumulador da raiz real (construído
1x por `chooseMoveMCAB()` via `buildAccPairRoot`, OU herdado da simulação
anterior se reuso de subárvore está ativo, Seção 8); para cada aresta
raiz→...→folha, `mcabAccStack[i+1] = makeChildAccPair(mcabAccStack[i], ...,
state_i, move_i, &xdistCache)`. Isso é uma atualização incremental (barata,
mesmo mecanismo que `negamax()` já usa internamente na própria busca AB) —
o custo por simulação é O(profundidade da árvore MCTS), não O(features) por
nó. É o mesmo trade-off memória-vs-CPU que qualquer motor de busca faz ao
preferir recomputar via ply-stack a cachear estado por nó.

Quando a expansão de um novo nó folha acontece (Seção 5.2), o
`mcabAccStack[depth]` correspondente já está pronto para: (a) alimentar
`forwardPolicyQuant` (prior), e (b) ser passado como `seedAcc` para
`searchLeaf` — nenhum rebuild completo acontece em nenhum momento fora da
raiz real.

## 4.4 Compatibilidade do `arena.cpp` com refs antigos (git worktree)

**Contexto do mecanismo existente** (não muda): `run_arena.py` →
`prepare_engine_source(ref)` faz `git worktree add --detach TEMP_DIR ref`
para cada ref pedido (`--ref1`/`--ref2`), gerando um checkout COMPLETO do
repositório NAQUELE commit em `TEMP_DIR`. `compile_arena()` então compila
sempre o `arena.cpp` do `PROJECT_ROOT` ATUAL (não do worktree!), passando
só `-DENGINE1_SEARCH_HPP="TEMP_DIR1/src/search.hpp"` e
`-DENGINE2_SEARCH_HPP="TEMP_DIR2/src/search.hpp"`. Ou seja: **o único
arquivo que varia por ref é `search.hpp` (e o que ele inclui de `src/`)**;
tudo em `tools/` — incluindo o novo `tools/common/mcab.hpp` — é sempre a
versão atual do HEAD.

**Consequência direta**: um `--ref1`/`--ref2` apontando para um commit
anterior a este plano vai compilar contra um `qr_e1::Negamax`/
`qr_e2::Negamax` **sem** os métodos `searchLeaf`/`resetOrderingState`
público (Seção 3). Se `MCABSearch<qr_e1::Negamax, ...>` fosse
incondicionalmente instanciado e usado no `arena.cpp` atual, a compilação
quebraria para qualquer ref antigo — inaceitável, já que comparar a versão
nova contra refs antigos em arena é justamente o caso de uso central da
ferramenta.

**Solução: mesmo padrão SFINAE já usado em `arena.cpp` hoje** (ver
`trySetEvalModeNnue`/`trySetPolicyOrdering`/`hasPolicyOrdering`, seção
"Mesmo truque SFINAE..." do arquivo atual). Adicionar em
`tools/common/mcab.hpp` (ou um `tools/common/mcab_dispatch.hpp` separado,
Seção 6):

```cpp
// Trait: Eng tem searchLeaf E resetOrderingState público?
template <typename Eng>
constexpr auto hasMcabSupport(int)
    -> decltype(std::declval<Eng&>().searchLeaf(
                    std::declval<const typename Eng::StateT&>(), 1,
                    std::declval<typename Eng::SearchStatsT&>(),
                    std::declval<typename Eng::RepTblT&>()),
                std::declval<Eng&>().resetOrderingState(),
                bool()) {
    return true;
}
template <typename Eng>
constexpr bool hasMcabSupport(...) { return false; }

// Dispatch: usa MCAB se suportado E habilitado; senão cai para
// eng.chooseMove(...) padrão -- NO-OP silencioso (sem erro de compilação,
// sem crash em runtime) para refs antigos, exatamente como
// trySetPolicyOrdering já faz para setPolicyOrderingEnabled.
template <typename Eng, typename StateT, typename SearchStatsT, typename RepTblT>
auto chooseMoveAuto(Eng& eng, const McabParams& p, const StateT& root,
                     int maxDepthCap, int timeBudgetMs, SearchStatsT& stats,
                     RepTblT& hist, int)
    -> std::enable_if_t<hasMcabSupport<Eng>(0), decltype(eng.chooseMove(root, maxDepthCap, timeBudgetMs, stats, hist))> {
    if (!p.enabled) return eng.chooseMove(root, maxDepthCap, timeBudgetMs, stats, hist);
    MCABSearch<Eng, /* ... */> mcab; // (instância mantida pelo chamador na prática, ver Seção 8)
    return mcab.chooseMoveMCAB(eng, root, timeBudgetMs, p, stats, hist);
}
template <typename Eng, typename StateT, typename SearchStatsT, typename RepTblT>
auto chooseMoveAuto(Eng& eng, const McabParams& p, const StateT& root,
                     int maxDepthCap, int timeBudgetMs, SearchStatsT& stats,
                     RepTblT& hist, ...) {
    // Eng sem suporte a MCAB (ref anterior a este plano) -- ignora `p`
    // completamente e roda AB puro, sem erro de compilação nem aviso
    // de runtime intrusivo (só um log opcional 1x, ver main() do arena).
    return eng.chooseMove(root, maxDepthCap, timeBudgetMs, stats, hist);
}
```

Em `arena.cpp`, o call site (Seção 10) sempre chama `chooseMoveAuto(eng1,
mcabParams1, ...)` incondicionalmente — a decisão "esse ref suporta MCAB
ou não" é 100% resolvida em tempo de compilação por overload resolution,
não por `if` em runtime. Isso significa: **rodar `--e1-mcab` contra um
`--ref1` antigo não é erro** — a flag é aceita, um aviso é impresso uma vez
("[arena] Engine 1: ref não suporta MCAB (compilado antes da feature),
rodando AB puro"), e a partida roda normalmente com AB puro para aquele
lado. Isso é exatamente equivalente ao que já acontece hoje quando
`--e1-policy-order` é passado contra um ref sem `setPolicyOrderingEnabled`.

## 5. Algoritmo, passo a passo

`MCABSearch::chooseMoveMCAB(Negamax& engine, const State& root, int
timeBudgetMs, int leafDepth, MCABStats& stats, const RepetitionTable&
gameHistory)`:

1. Se `root.wallsLeft[0]==0 && root.wallsLeft[1]==0`: delega direto para
   `engine.chooseMove(root, ..., ...)` (o atalho de final "mãos vazias" já
   resolve por solver exato — não há nada para o MCTS ganhar aqui, e
   reimplementar essa lógica no módulo novo seria duplicação sem benefício).
2. `engine.resetOrderingState()`, `engine.clearTT()` opcional (ver Seção 9,
   `mcabClearTTPerMove` — default `false`: preservar TT entre lances dentro
   do mesmo jogo é o comportamento de produção atual do AB puro também).
3. Inicializa/recupera árvore (nova ou reusada, Seção 8). Nó raiz = índice 0
   do pool, `state = root`.
4. Se raiz não expandida: expande (Seção 5.2).
5. Loop de simulações até orçamento (nós OU tempo, o que vier primeiro —
   ambos configuráveis, Seção 9):
   a. **Seleção**: desce da raiz escolhendo em cada nó o filho que maximiza
      PUCT (Seção 5.1), até achar um filho com `child[i] == -1`
      (não expandido) ou um nó terminal. Atualiza `mcabAccStack` a cada
      passo via `makeChildAccPair` (Seção 4.3).
   b. **Expansão + avaliação**: se não-terminal, cria o novo `MCABNode`
      (aloca no pool), gera `moves`, computa `P` via
      `forwardPolicyQuant(mcabAccStack[depth].acc[side], ...)` +
      `moveToPolicyIndex`, e avalia chamando
      `engine.searchLeaf(newState, leafDepth, leafStats, reptbl,
      &mcabAccStack[depth])`. Converte o score retornado em `Q` via
      `scoreToQ`. Se terminal (`winner(newState) != -1`), marca
      `terminal=true` e usa o valor exato (sem chamar `searchLeaf`).
   c. **Backup**: propaga `Q` (do ponto de vista de quem jogou o lance que
      levou ao nó recém-expandido) subindo a árvore, invertendo o sinal a
      cada nível (`Q_pai = 1 - Q_filho`, já que estamos em espaço de
      probabilidade de vitória, análogo a negamax), atualizando `N[i] +=
      1`, `W[i] += Q` em cada aresta do caminho. Backup é **minimax hard**
      (usa o valor do filho diretamente, não faz média de múltiplas
      janelas) — é a escolha do MCαβ original e a única testada pelo
      Scorpio; um "backup médio" fica documentado como variante futura
      (`mcabBackupMode`, Seção 9) mas não é o default.
6. Escolha do lance final na raiz: filho com maior `N[i]` (padrão
   AlphaZero-like — mais robusto a ruído de avaliação single-visita que
   escolher por `Q` cru; configurável via `mcabRootSelectMode`, Seção 9).
7. Se reuso de subárvore está ativo, guarda o subponteiro do filho
   escolhido para a próxima chamada (Seção 8); senão descarta o pool.

### 5.1 Fórmula de seleção (PUCT)

Para cada filho `a` do nó `s`:

```
score(a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

`Q(s,a) = W(s,a)/N(s,a)` se `N(s,a) > 0`; caso contrário usa FPU (first
play urgency, Seção 9, `mcabFpuValue`, default = `Q(s)` do nó pai menos uma
redução pequena — evita que filhos nunca visitados pareçam artificialmente
ótimos só por terem denominador baixo, prática padrão desde AlphaZero/Leela).

### 5.2 Expansão (custo por nó novo)

Por nó expandido: 1 chamada `forwardPolicyQuant` (~53.500 MACs, mesmo custo
já pago hoje pela ordenação de lances assistida por política dentro do AB,
então não é uma técnica nova de custo desconhecido) + 1 chamada
`searchLeaf` em profundidade `leafDepth`. O número de expansões por
`chooseMoveMCAB()` é limitado por `mcabNodeBudget` (Seção 9) — este é o
principal knob de custo total.

## 6. Fallback de equivalência (validação obrigatória antes de qualquer arena real)

Implementar `mcabNodeBudget = 0` (ou `mcabTreeReuse=false` +
`mcabNodeBudget=1`, o nó raiz apenas) como modo especial: neste caso
`chooseMoveMCAB` deve reduzir-se a **uma única chamada** `searchLeaf(root,
leafDepth, ...)` seguida de escolha do lance de maior `Q` entre os filhos
diretos avaliados (sem árvore, sem PUCT) — mesmo espírito do achado do
Scorpio (`treeht=0` ≈ força de AB puro). Este modo serve como teste de
sanidade: rodar e confirmar que o lance escolhido e o `Elo` medido em
arena batem (dentro do ruído estatístico) com `Negamax::chooseMove` puro na
MESMA profundidade `leafDepth`. Se não bater, há bug na integração antes de
sequer chegar a testar se PUCT ajuda.

## 7. Gerenciamento de memória / orçamento de nós

`mcabNodeBudget` (Seção 9) dimensiona `std::vector<MCABNode>` reservado
(`reserve()`, não `resize()`) uma vez por `chooseMoveMCAB()` (ou 1x por
partida, se reuso entre lances estiver ativo, com compactação — ver 8.2).
Custo por nó ≈ `sizeof(State)` (~30 B) + overhead dos `vector`s internos
(`moves`, `P`, `N`, `W`, `child`) proporcional ao número de lances legais
naquele nó (tipicamente poucas dezenas; pode chegar a ~130 perto da
abertura). Estimativa prática: para orçamento de 20.000 nós, esperar algo
entre 5–15 MB de árvore — ordens de grandeza abaixo da TT (~40 MB) e
seguro para não competir por cache/memória com o AB.

## 8. Reuso de subárvore entre lances (economia entre chamadas)

### 8.1 Mecânica

Ao final de `chooseMoveMCAB`, se `mcabTreeReuse=true`: em vez de descartar
o pool, guarda o índice do filho escolhido. Quando o adversário responde e
`chooseMoveMCAB` é chamado de novo com o novo `root`, verifica se o novo
`root` corresponde a um neto do nó salvo (i.e., segue o caminho
`escolhido → resposta_do_oponente` dentro da árvore antiga). Se sim, esse
neto vira a nova raiz — todo o subtree acumulado (estatísticas `N`/`W`/`P`
já calculadas) é reaproveitado, e só falta compactar o pool (mover o
subtree para os índices `[0, k)`, descartando o resto) para não vazar
memória do que ficou fora do caminho jogado.

### 8.2 Quando NÃO reusar

Se o oponente jogou algo fora da árvore que existia (raro, mas possível se
`mcabNodeBudget` cortou a expansão daquele ramo), cai para árvore nova
(`buildAccPairRoot` na nova raiz). Também não reusar quando
`mcabClearTTPerMove=true` estiver ativo (esvaziar TT mas manter árvore MCTS
seria inconsistente — a árvore depende de valores computados com aquela
TT).

## 9. Parâmetros ajustáveis (tabela completa)

Todos como membros de instância de `MCABSearch` (mesmo padrão de
`Negamax::setX/getX` — necessário para permitir 2 configurações
simultâneas em arena/tuner, igual ao motivo documentado em
`Negamax(EvalWeights&)`).

| Parâmetro | Tipo | Range sugerido | Default v1 | Efeito |
|---|---|---|---|---|
| `mcabEnabled` | bool | — | `false` | Liga o caminho híbrido; com `false`, engine se comporta 100% como hoje. |
| `mcabNodeBudget` | int | 0–200000 | `20000` | Nº máx. de nós expandidos por `chooseMoveMCAB`. Principal controle de custo total. `0`/`1` = modo equivalência (Seção 6). |
| `mcabTimeBudgetMs` | int | — | = orçamento de tempo já passado ao chamador | Teto de tempo, independente do nó-orçamento; para de simular ao bater qualquer um dos dois. |
| `mcabLeafDepth` | int | 1–20 | `4` | Profundidade (plies) da busca AB rasa em cada folha (`searchLeaf`). |
| `mcabLeafDepthMax` | int | ≥ `mcabLeafDepth` | `8` | Teto quando `mcabAdaptiveLeafDepth=true`. |
| `mcabAdaptiveLeafDepth` | bool | — | `false` | Se `true`, escala `leafDepth` com `N` do nó pai (nós mais visitados ganham busca mais profunda na próxima expansão do mesmo ramo — reexpansão, ver nota abaixo). V1 pode deixar `false` (profundidade fixa) para simplificar; incluído aqui porque é o próximo ajuste natural. |
| `mcabCPuct` | double | 0.5–5.0 | `1.5` | Constante de exploração do PUCT. |
| `mcabFpuValue` | double | 0.0–1.0 (absoluto) ou delta | `Q(pai) - 0.1` | Valor de first-play-urgency para filhos não visitados. |
| `mcabScoreScale` | double | — | `NNUE_EVAL_SCALE` (200.0) | Escala usada em `scoreToQ`. Separado da constante de busca para permitir recalibração sem afetar o AB. |
| `mcabRootSelectMode` | enum{MaxVisits, MaxQ, MaxVisitsThenQ} | — | `MaxVisits` | Critério de escolha do lance final na raiz. |
| `mcabBackupMode` | enum{MinimaxHard, AvgBlend} | — | `MinimaxHard` | `AvgBlend` fica reservado para experimentos futuros (mistura backup minimax com média das últimas k simulações). |
| `mcabTreeReuse` | bool | — | `true` | Reaproveita subárvore entre lances (Seção 8). |
| `mcabClearTTPerMove` | bool | — | `false` | Consistente com o comportamento atual do AB puro (TT só é limpa entre partidas, não entre lances). |
| `mcabRootNoiseEnabled` | bool | — | `false` | Ruído de Dirichlet nos priors da raiz. Só para geração de dados de self-play, nunca em arena de força. |
| `mcabRootNoiseAlpha` | double | 0.1–1.0 | `0.3` | Parâmetro alfa da Dirichlet (só relevante se `mcabRootNoiseEnabled`). |
| `mcabRootNoiseEpsilon` | double | 0.0–0.5 | `0.25` | Peso do ruído misturado ao prior (só relevante se `mcabRootNoiseEnabled`). |
| `mcabMaxTreeDepth` | int | ≤ `MAX_PLY` | `48` | Dimensiona `mcabAccStack` (Seção 4.3), mesmo espírito do guard já existente em `chooseMove` para `nnueAccStack` vs `maxDepthCap`. |

Todos os parâmetros numéricos (exceto os booleanos de escopo/segurança
como `mcabRootNoiseEnabled` e `mcabClearTTPerMove`) devem entrar depois na
tabela `PARAM_DEFS` de `tools/spsa/tune_spsa.cpp` seguindo exatamente o
padrão já usado para `lmrMinDepth`/`catHotCm`/etc. — candidatos óbvios de
primeira rodada de tuning: `mcabCPuct`, `mcabLeafDepth`, `mcabFpuValue`,
`mcabNodeBudget` (este último provavelmente fixado por orçamento de tempo
de partida real, não tunado por Elo puro).

## 10. Integração com CLI (arena/selfplay/tuner)

Seguir exatamente o padrão já existente em `arena.cpp` para
`--e1-heuristic`/`--e2-nnue`/`--e1-policy-order`/etc. — flags
independentes por engine, permitindo, por ex., E1 = AB puro vs E2 = híbrido
na mesma partida:

```
--e1-mcab / --e2-mcab / --mcab           (liga mcabEnabled para 1/2/ambos)
--e1-mcab-nodes N / --e2-mcab-nodes N    (mcabNodeBudget)
--e1-mcab-leaf-depth N / --e2-...        (mcabLeafDepth)
--e1-mcab-cpuct X / --e2-...             (mcabCPuct)
--e1-mcab-no-tree-reuse / --e2-...       (mcabTreeReuse=false)
--e1-mcab-equiv-mode / --e2-...          (atalho pra mcabNodeBudget=0, Seção 6)
```

Ponto de chamada em `arena.cpp` (hoje `eng1.chooseMove(s1, 40, timeMs, st,
hist1)`): **sempre** trocar por `chooseMoveAuto(eng1, mcabParams1, s1, 40,
timeMs, st, hist1)` (Seção 4.4) — incondicionalmente, mesmo quando
`mcabParams1.enabled == false` (nesse caso a própria função dispatch cai em
`eng.chooseMove(...)` sem overhead extra: é um `if` simples dentro de um
método já resolvido em tempo de compilação por SFINAE, não uma indireção
virtual). Isso elimina a necessidade de dois call sites diferentes
(com/sem MCAB) espalhados pelo arquivo.

Mesmo padrão vale para `tools/selfplay/selfplay_main.cpp` (geração de dados
via híbrido, com `mcabRootNoiseEnabled=true` ali especificamente — não em
arena) e `tools/spsa/tune_spsa.cpp` (novo bloco de `ParamDef` para os
parâmetros MCAB, Seção 9). Como esses dois binários compilam sempre a
`src/` atual (sem o truque de dual-ref), `hasMcabSupport<qr::Negamax>` é
sempre `true` neles — o dispatch nunca cai no fallback ali, mas usar a
mesma função (em vez de chamar `MCABSearch` direto) mantém o código
uniforme entre os três binários.

### 10.1 Regra obrigatória: toda flag de CLI tem uma constante default no topo do arquivo

Para CADA flag nova listada acima (e qualquer outra que vier a ser
adicionada), seguir sem exceção o padrão que `arena.cpp` já usa para
`E1_POLICY_ORDERING_DEFAULT`/`E1_POLICY_ORDER_MIN_DEPTH_DEFAULT`: uma
constante `constexpr` nomeada perto do topo do arquivo (junto do bloco
"CONFIG DEFAULTS" já existente em `arena.cpp`, ou de um bloco equivalente
a criar em `selfplay_main.cpp`/`tune_spsa.cpp`), uma variável `static`
inicializada a partir dela, e o parsing de `argv` só sobrescreve essa
variável quando a flag correspondente é encontrada. Rodar o binário sem
NENHUM argumento tem que produzir exatamente o mesmo comportamento que
rodar com todas as flags passadas explicitamente nos valores default.
Nenhuma flag pode ter seu valor "hardcoded" só no parser de `argv` sem uma
constante nomeada equivalente — isso é o que já permite hoje mudar o
comportamento padrão de `arena.cpp` editando uma linha no topo do arquivo,
sem precisar lembrar/passar flag nenhuma toda vez que rodar `run_arena.py`.

Tabela de constantes novas por arquivo (nomes sugeridos, seguindo a
convenção `E{1,2}_..._DEFAULT` já em uso em `arena.cpp`; em
`selfplay_main.cpp`/`tune_spsa.cpp`, que não têm par E1/E2, usar só
`MCAB_..._DEFAULT`):

**`tools/arena/arena.cpp`** (bloco "CONFIG DEFAULTS", par E1/E2):

| Constante | Tipo | Default sugerido |
|---|---|---|
| `E1_MCAB_ENABLED_DEFAULT` / `E2_MCAB_ENABLED_DEFAULT` | bool | `false` |
| `E1_MCAB_NODE_BUDGET_DEFAULT` / `E2_...` | int | `20000` |
| `E1_MCAB_LEAF_DEPTH_DEFAULT` / `E2_...` | int | `4` |
| `E1_MCAB_LEAF_DEPTH_MAX_DEFAULT` / `E2_...` | int | `8` |
| `E1_MCAB_ADAPTIVE_LEAF_DEPTH_DEFAULT` / `E2_...` | bool | `false` |
| `E1_MCAB_CPUCT_DEFAULT` / `E2_...` | double | `1.5` |
| `E1_MCAB_SCORE_SCALE_DEFAULT` / `E2_...` | double | `200.0` (=`NNUE_EVAL_SCALE`) |
| `E1_MCAB_ROOT_SELECT_MODE_DEFAULT` / `E2_...` | enum | `MaxVisits` |
| `E1_MCAB_BACKUP_MODE_DEFAULT` / `E2_...` | enum | `MinimaxHard` |
| `E1_MCAB_TREE_REUSE_DEFAULT` / `E2_...` | bool | `true` |
| `E1_MCAB_CLEAR_TT_PER_MOVE_DEFAULT` / `E2_...` | bool | `false` |
| `E1_MCAB_ROOT_NOISE_ENABLED_DEFAULT` / `E2_...` | bool | `false` |
| `E1_MCAB_ROOT_NOISE_ALPHA_DEFAULT` / `E2_...` | double | `0.3` |
| `E1_MCAB_ROOT_NOISE_EPSILON_DEFAULT` / `E2_...` | double | `0.25` |
| `E1_MCAB_MAX_TREE_DEPTH_DEFAULT` / `E2_...` | int | `48` |
| `E1_MCAB_EQUIV_MODE_DEFAULT` / `E2_...` | bool | `false` (atalho, força `nodeBudget=0` quando `true`) |

**`tools/selfplay/selfplay_main.cpp`** (novo bloco "CONFIG DEFAULTS",
sem par E1/E2 — só uma engine em self-play):

| Constante | Tipo | Default sugerido |
|---|---|---|
| `MCAB_ENABLED_DEFAULT` | bool | `false` (self-play atual continua AB puro até decidir usar híbrido pra gerar dados) |
| `MCAB_NODE_BUDGET_DEFAULT` | int | `20000` |
| `MCAB_LEAF_DEPTH_DEFAULT` | int | `4` |
| `MCAB_CPUCT_DEFAULT` | double | `1.5` |
| `MCAB_ROOT_NOISE_ENABLED_DEFAULT` | bool | `true` (diferente de arena — aqui é o valor correto por padrão quando `MCAB_ENABLED_DEFAULT` for ligado, já que serve para diversidade de dados) |
| `MCAB_ROOT_NOISE_ALPHA_DEFAULT` | double | `0.3` |
| `MCAB_ROOT_NOISE_EPSILON_DEFAULT` | double | `0.25` |
| (demais parâmetros da Seção 9) | — | mesmos defaults da tabela de `arena.cpp` |

**`tools/spsa/tune_spsa.cpp`**: aqui os "defaults" são as entradas de
`PARAM_DEFS` (Seção 9 — coluna `init`), que já seguem exatamente esse
princípio (`init` = comportamento hoje hardcoded quando o parâmetro nunca
foi tunado). Adicionar uma constante extra, fora de `PARAM_DEFS`
(análoga às outras: liga/desliga o MODO de fitness do tuner, não um valor
tunável em si):

| Constante | Tipo | Default sugerido |
|---|---|---|
| `MCAB_TUNING_MODE_DEFAULT` | bool | `false` (tuner continua avaliando fitness via AB puro até decidirem tunar o híbrido; quando `true`, `applyParams`/fitness passam a chamar `chooseMoveAuto` com `mcabParams.enabled=true`) |

## 11. Fases de implementação e arquivos por fase

Cada fase é um checkpoint de compilação+teste independente — não avançar
para a próxima sem a fase anterior passando nos critérios de saída
listados. Nenhuma fase antes da Fase 4 toca em `arena.cpp`/`run_arena.py`
de propósito — o objetivo é isolar o risco de regressão no AB puro (Seção
0) o máximo possível antes de mexer na ferramenta que faz comparação real
de força.

### Fase 0 — Preparação mínima no engine (único código ref-versionado)

**Arquivos:** `src/search.hpp`

- `resetOrderingState()`: mover de `private:` para `public:` (corpo
  idêntico).
- Novo método público `searchLeaf(...)` (Seção 3.2).

**Critério de saída:** suíte `tests/*.cpp` existente passa sem nenhuma
alteração de código de teste; benchmark `benchmarks/bench_fixed_depth.cpp`
mostra nós/s e profundidade idênticos ao commit anterior (dentro do ruído
de medição normal). Isso valida sozinho o requisito da Seção 0 antes de
qualquer linha de MCAB existir.

### Fase 1 — Núcleo do `MCABSearch` (tool-side, isolado)

**Arquivos novos:**
- `tools/common/mcab.hpp` — `MCABNode`, `MCABSearch<...>` (template),
  `scoreToQ`, PUCT, expansão, backup (Seções 4, 5).
- `tests/test_mcab_core.cpp` — instancia `MCABSearch<qr::Negamax, ...>`
  contra a `src/` local (sem dual-ref, sem `arena.cpp`), testa: pool não
  estoura orçamento, backup propaga sinal corretamente em posições
  triviais construídas à mão, `scoreToQ` bate com `nnueWinProbQuant` para
  os mesmos logits.

**Arquivos não tocados:** `src/*`, `tools/arena/*`, `tools/selfplay/*`,
`tools/spsa/*`.

**Critério de saída:** `test_mcab_core` compila e passa isolado, sem
qualquer binário de produção (arena/selfplay/spsa) ter sido tocado.

### Fase 2 — Camada de compatibilidade / dispatch SFINAE

**Arquivos:**
- `tools/common/mcab.hpp` (ou `tools/common/mcab_dispatch.hpp` separado,
  se o arquivo ficar grande demais) — `hasMcabSupport<Eng>`,
  `chooseMoveAuto(...)` (Seção 4.4).
- `tests/test_mcab_dispatch.cpp` — novo teste que define um "Negamax
  falso" minimalista SEM `searchLeaf` (simula um ref antigo) e confirma
  que `chooseMoveAuto` cai no fallback sem erro de compilação; e um
  "Negamax falso" COM `searchLeaf` confirmando que usa o caminho MCAB.

**Critério de saída:** `test_mcab_dispatch` passa nos dois cenários
(com/sem suporte), sem envolver git worktree nenhum — é um teste de tipos
em C++ puro.

### Fase 3 — Modo equivalência (validação de corretude antes de qualquer ganho de Elo)

**Arquivos:**
- `benchmarks/bench_mcab_equivalence.cpp` (novo, próprio `main()`, mesmo
  padrão dos outros `bench_*.cpp`) — roda `mcabNodeBudget=0`
  (Seção 6) contra `Negamax::chooseMove` puro na mesma `leafDepth`, em um
  conjunto de posições fixas, e compara lance escolhido + score.

**Arquivos não tocados:** ainda nenhum `tools/arena/*`,
`tools/selfplay/*`, `tools/spsa/*`.

**Critério de saída:** lances/scores batem entre os dois caminhos
(dentro do determinismo esperado da busca). Divergência aqui = bug de
sinal/backup/conversão antes de gastar qualquer ciclo de arena real.

### Fase 4 — PUCT completo + benchmark de custo isolado

**Arquivos:**
- `tools/common/mcab.hpp` — completar seleção PUCT/expansão/backup para
  árvore de verdade (além do caso trivial da Fase 3).
- `benchmarks/bench_mcab.cpp` (novo) — mede nós-AB-equivalentes/s do
  híbrido (nós MCTS × `leafDepth` médio) e memória do pool; e, crucial
  para a Seção 0: mede nós/s do MODO AB PURO (`Negamax::chooseMove`) no
  MESMO binário que agora linka `mcab.hpp`, comparado ao binário antigo
  que não linkava — precisa ser idêntico.

**Critério de saída:** zero regressão de nós/s no AB puro (Seção 0
formalmente validada); métricas de custo do híbrido documentadas.

### Fase 5 — Integração no `arena.cpp`

**Arquivos:**
- `tools/arena/arena.cpp`:
  - `#include "../common/mcab.hpp"` (fora do bloco de rename
    `#define qr qr_eN`/`#undef qr`, depois dele).
  - Bloco "CONFIG DEFAULTS": constantes `E1_MCAB_*_DEFAULT`/
    `E2_MCAB_*_DEFAULT` (Seção 10.1).
  - Variáveis `g_e1Mcab*`/`g_e2Mcab*` inicializadas a partir das
    constantes acima (mesmo padrão de `g_e1UsePolicyOrdering`).
  - Parsing de `argv` em `main()`: novas flags da Seção 10.
  - Troca dos call sites `eng1.chooseMove(...)`/`eng2.chooseMove(...)`
    por `chooseMoveAuto(...)` (Seção 10, primeiro parágrafo).
  - Aviso 1x em stderr quando `chooseMoveAuto` cair no fallback por
    incompatibilidade de ref (Seção 4.4) — usar o mesmo padrão de
    `hasPolicyOrdering<qr_eN::Negamax>(0)` já usado para o aviso
    equivalente de policy-ordering.
- **Não tocar** `tools/arena/run_arena.py` — a resolução relativa de
  include já cobre `../common/mcab.hpp` sem mudar o comando de
  compilação (Seção 4).

**Critério de saída:** `run_arena.py --ref1 <commit-antigo-simulado>
--ref2 <local> --e2-mcab` roda sem erro de compilação, com aviso de
fallback para E1 e uso real do híbrido em E2. `run_arena.py` sem
`--ref1`/`--ref2` (versão local vs ela mesma) com `--mcab` (ambos os
lados) roda normalmente.

### Fase 6 — Integração no `selfplay`

**Arquivos:**
- `tools/selfplay/selfplay.hpp` e/ou `tools/selfplay/selfplay_main.cpp`:
  bloco "CONFIG DEFAULTS" próprio (Seção 10.1), flags `--mcab`/
  `--mcab-nodes`/etc. (sem prefixo `e1-`/`e2-`, só uma engine aqui), troca
  do call site interno por `chooseMoveAuto(...)`.

**Critério de saída:** `run_selfplay.py` com `--mcab` gera `.bin` válidos
(mesmo formato `TrainingSample` de sempre); sem `--mcab`, comportamento
idêntico ao anterior.

### Fase 7 — Integração no tuner (`tune_spsa.cpp`)

**Arquivos:**
- `tools/spsa/tune_spsa.cpp`: novos `ParamId`/entradas em `PARAM_DEFS`
  para `mcabCPuct`, `mcabLeafDepth`, `mcabFpuValue`, `mcabScoreScale` (e
  outros candidatos da Seção 9); extensão de `applyParams(...)`;
  constante `MCAB_TUNING_MODE_DEFAULT` (Seção 10.1) controlando se a
  função de fitness roda via `chooseMoveAuto` com MCAB ligado.

**Critério de saída:** `tune_spsa` compila e roda uma geração pequena de
teste (poucos indivíduos/poucos jogos) sem crash, tanto com
`MCAB_TUNING_MODE_DEFAULT=false` (comportamento atual, intocado) quanto
`=true`.

### Fase 8 — Arena real e decisão de mérito

**Arquivos:** nenhum (só execução).

Rodar `run_arena.py` E1 = AB puro (produção atual) vs E2 = híbrido, tempo
por lance igual, conforme Seção 11 (antiga)/critérios da Seção 13. Esta
fase decide se vale a pena prosseguir para a Fase 9.

### Fase 9 — Tuning completo + otimizações incrementais (só se Fase 8 for positiva)

**Arquivos:**
- `tools/spsa/run_spsa.py` (execução, sem mudança de código) com
  `MCAB_TUNING_MODE_DEFAULT=true` no `tune_spsa.cpp` da Fase 7.
- `tools/common/mcab.hpp`: reuso de subárvore entre lances (Seção 8) e
  profundidade adaptativa de folha (`mcabAdaptiveLeafDepth`, Seção 9) —
  otimizações de economia/força incrementais que não bloqueiam a
  validação de mérito da Fase 8.

## 12. Riscos e mitigação

- **Custo de expansão maior que o esperado** (forward de política +
  `searchLeaf` por nó): mitigado por `mcabNodeBudget` como teto duro e por
  medir nós-de-AB-equivalentes/s do híbrido (nós MCTS × `leafDepth` médio)
  contra nós/s do AB puro na mesma janela de tempo, não só "nós MCTS/s"
  isolado, que seria uma métrica enganosa.
- **Conversão score→Q mal calibrada** distorce o PUCT (Q dominado ou
  dominante demais frente ao termo de exploração): mitigado por usar a
  MESMA curva sigmoide já treinada (`nnueWinProbQuant`), não uma nova.
- **Memória do pool de nós crescendo sem controle** com reuso de
  subárvore ativado por muitos lances seguidos: mitigado pela compactação
  obrigatória (Seção 8.1) a cada `chooseMoveMCAB`.
- **Regressão silenciosa no AB puro** por causa da mudança de visibilidade
  de `resetOrderingState`: risco praticamente nulo (mudança é só de
  `private`→`public`, corpo idêntico), mas incluir explicitamente no
  checklist de testes da Fase 0.
- **Quebra de compatibilidade do arena com refs antigos**: mitigado
  estruturalmente pela Seção 4.4 (dispatch SFINAE em tempo de compilação,
  `mcab.hpp` fora da árvore versionada-por-ref). Teste de regressão
  específico: Fase 5 deve incluir pelo menos uma rodada de
  `run_arena.py --ref1 <commit anterior à Fase 0> --e1-mcab` confirmando o
  aviso de fallback e ausência de erro de compilação — não basta testar
  só ref-atual-vs-ref-atual.
- **Flag de CLI sem constante default correspondente** (violação da Seção
  10.1): mitigado por revisão de código dedicada antes de cada merge de
  fase que mexe em `arena.cpp`/`selfplay_main.cpp`/`tune_spsa.cpp` —
  checklist simples: toda entrada nova em `argv` parsing tem uma
  constante `_DEFAULT` correspondente declarada antes dela no arquivo.

## 13. Métricas de sucesso

- Nós/s e profundidade alcançada do modo AB puro: **idênticos** ao
  baseline atual (tolerância zero — é regressão se cair).
- Modo equivalência (Seção 6) vs AB puro na mesma `leafDepth`: Elo dentro
  do erro estatístico do teste (ex. ±10 Elo com N de jogos padrão do
  `run_arena.py`).
- Híbrido completo vs AB puro, mesmo tempo por lance: ganho de Elo
  positivo e estatisticamente significativo é o critério de "vale a pena
  manter"; ganho nulo ou negativo é resultado válido também (Seção 6 do
  argumento original: o próprio Scorpio mostra que isso não é garantido)
  — nesse caso o módulo fica documentado e desligado por padrão
  (`mcabEnabled=false`), sem custo para o resto do engine.
