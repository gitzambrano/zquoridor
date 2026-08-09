// search.hpp -- negamax + alpha-beta + tabela de transposição + iterative
// deepening (Fase 3 do plano), com as melhorias de ordenação/poda da
// Seção 5.5: killer moves, history heuristic, ordenação de muros por
// delta de BFS (restrita aos primeiros plies) e aspiration windows.
// Tudo isso é agnóstico à função de avaliação usada na folha
// (evalSimple hoje, NNUE na Fase 6) -- só reordena/poda lances candidatos.
#pragma once
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cmath>
#include "rules.hpp"
#include "cat.hpp"
#include "endgame_race.hpp"
#include "nnue.hpp"   // AccPair, buildAccPairRoot, makeChildAccPair, nnueEvalInt

namespace qr {

constexpr int TT_BITS = 21;               // 2M entradas
constexpr size_t TT_SIZE = 1ull << TT_BITS;
constexpr int SCORE_INF = 1'000'000;

// RACE_SCORE_BASE (endgame_race.hpp) precisa ficar estritamente abaixo do
// score de vitória imediata (SCORE_INF-1, ver winner() em negamax) com
// folga suficiente pra subtrair o dtm do final resolvido, e acima de
// qualquer eval heurística plausível -- checado aqui em vez de em
// endgame_race.hpp porque SCORE_INF só existe deste ponto em diante.
static_assert(RACE_SCORE_BASE < SCORE_INF - 1, "RACE_SCORE_BASE precisa deixar espaco abaixo do score de vitoria imediata");

// profundidade máxima de iterative deepening suportada (folga generosa
// sobre os ~40 usados hoje em main.cpp/selfplay_main.cpp/gui_web) --
// dimensiona os arrays de killer moves, indexados por ply-a-partir-da-raiz.
constexpr int MAX_PLY = 64;

constexpr int CONTEMPT = -30;  // motor evita empate em posição neutra/melhor

// ordenação de muro por delta de BFS (custo: 1 BFS por muro candidato) só
// vale a pena perto da raiz -- Seção 5.5 do plano. Nos plies mais fundos
// cai de volta pro critério barato "peão antes de muro".
constexpr int WALL_BFS_ORDER_MAX_PLY = 2;

// Quiescência mínima de muro (Fase 4.2.10, item 3 do plano da Seção
// 4.2.10). NÃO é quiescência geral (não existe "captura" em Quoridor) --
// é uma extensão rasa e restrita, gated pelo mesmo critério de "muro
// crítico" já usado na ordenação (delta de BFS) e na eval nova
// (pathRobustness, item 1): na fronteira do horizonte (depth==0), em vez
// de cair direto em evalSimple, testa só o subconjunto de muros do lado a
// mover que fecham significativamente uma rota do OPONENTE (delta de BFS
// grande ou robustez do caminho dele cai a zero -- vira corredor único) e
// estende 1-2 plies só para esses. Extensão máxima pequena de propósito
// (custo por nó de quiescência é 1 legalWallMoves completo, o mesmo custo
// do Estágio 3 do negamax -- não pode virar busca completa disfarçada).
constexpr int QS_MAX_EXTRA_PLIES = 2;
// "drasticamente" = delta de BFS do oponente >= este limiar...
constexpr int QS_CRITICAL_BFS_DELTA = 2;
// ...OU o caminho do oponente, que antes tinha alguma folga
// (pathRobustness > 0), fica sem NENHUM desvio barato depois do muro
// (vira corredor único) -- o caso clássico de horizon effect ("o muro que
// fecha a última rota alternativa bem na borda da busca").
constexpr int QS_CRITICAL_ROBUSTNESS_DROP_TO = 0;

// ---------------------------------------------------------------------
// LMR (Late Move Reduction, plano-additional.md Prioridade 3) + PVS
// (Principal Variation Search, Prioridade 8) -- dupla clássica que separa
// alpha-beta simples de alpha-beta "sério": PVS busca o 1º lance (o
// esperado melhor, normalmente o da TT) em janela completa e todos os
// demais em janela NULA (-alpha-1,-alpha), promovendo pra janela completa
// só se a busca nula "vazar" acima de alpha; LMR, por cima disso, reduz a
// PROFUNDIDADE dos lances tardios na ordenação antes de gastar a busca
// nula, reverificando em profundidade cheia (ainda janela nula) só se o
// resultado reduzido também vazar. Combinados: cada lance depois do 1º
// paga o mínimo possível, e só escala pra busca cara (mais profunda/mais
// larga) quando o resultado barato indica que talvez valha a pena.
//
// Fórmula usada pelo titanium-engine (mesma família da usada em
// Stockfish, adaptada) -- ver plano-additional.md Prioridade 3.
constexpr int LMR_MIN_DEPTH = 3;         // não reduz nós rasos demais
constexpr int LMR_MIN_MOVE_INDEX = 3;    // 1-based; 1º lance (PVS) + os 2
                                          // seguintes nunca são reduzidos
constexpr double LMR_DIVISOR = 2.25;

// Modificadores de calor CAT (cat.hpp) sobre a redução -- só se aplicam a
// lances de MURO (peão não tem calor CAT). Escala de heat: 0..240
// (CAT_CORRIDOR_CM=200 + CAT_BOTTLENECK_BONUS_CM=40 no caso extremo).
// Um muro "quente" (perto do caminho ótimo do oponente) é candidato a
// mudar o resultado tático da posição -- nunca reduzir. Um muro "frio"
// (bem longe de qualquer caminho relevante) quase certamente é ruído de
// ordenação -- reduzir 1 passo a mais que o normal.
constexpr int CAT_HOT_CM = 150;   // heat >= isto -> pula LMR (lance "tático")
constexpr int CAT_COLD_CM = 30;   // heat < isto -> +1 de redução extra ("frio")

// Ordenação por política NNUE (prompt_policy_ordering.md) -- soma o logit
// cru da cabeça de política (forwardPolicyQuant, nnue.hpp) como termo
// extra no score de orderPawnMoves/orderWallMoves, atrás de
// setPolicyOrderingEnabled (default desligada, ver método). O logit cru
// vive numa escala pequena (tipicamente [-5,5], mesma ordem de grandeza
// dos logits da cabeça WL antes de NNUE_EVAL_SCALE) -- POLICY_ORDER_SCALE
// escala isso pra ficar comparável a history/CAT (que somam na casa de
// centenas a milhões por causa de depth*depth e killer bonus). Escolhido
// para ficar na mesma ordem de grandeza dos incrementos de history em
// profundidade típica (depth^2 com depth~8 => ~64 por cutoff único, vários
// cutoffs acumulam facilmente na casa de centenas/milhares) sem dominar
// nem ser irrelevante frente a killer bonus (1'400'000+); valor inicial,
// não tunado -- quem ligar a flag deve validar empiricamente (ver
// benchNegamaxNNUE/run_arena.py) se compensa e se este peso é razoável.
constexpr long long POLICY_ORDER_SCALE = 400;

// reduction = clamp( round( ln(depth) * ln(move_index) / 2.25 ), 0, depth/2 )
// -- ver Prioridade 3 do plano pra derivação/justificativa da fórmula.
inline int lmrReduction(int depth, int moveIndex) {
    double r = std::log((double)depth) * std::log((double)moveIndex) / LMR_DIVISOR;
    int red = (int)std::lround(r);
    if (red < 0) red = 0;
    int maxRed = depth / 2;
    if (red > maxRed) red = maxRed;
    return red;
}

enum TTFlag : uint8_t { EXACT, LOWER, UPPER };

struct TTEntry {
    uint64_t key = 0;
    // Bug corrigido: score pode chegar a SCORE_INF-1 == 999999 (vitória/
    // derrota forçada), mas int16_t só vai até 32767 -- truncava
    // silenciosamente qualquer score de mate armazenado na TT,
    // corrompendo decisões perto do fim de jogo (qualquer score >32767
    // ou <-32768 dava wraparound pra um valor pequeno/errado, tanto na
    // gravação quanto, pior ainda, na leitura por outro nó que confiava
    // nesse valor pra um cutoff). int32_t cobre SCORE_INF com folga.
    int32_t score = 0;
    int8_t depth = -1;
    TTFlag flag = EXACT;
    Move best = Move::pawn(0);
    bool valid = false;
};

struct SearchStats {
    uint64_t nodes = 0;
    int reachedDepth = 0;
    int score = 0;         // avaliação (unidades de evalSimple) da raiz na
                            // profundidade alcançada, do ponto de vista de
                            // quem tinha a vez -- exposta pra GUI mostrar
                            // a avaliação ao lado do lance do motor.
};

class Negamax {
public:
    Negamax() : tt(TT_SIZE), weights(evalWeights()), nnueAccStack(MAX_PLY + QS_MAX_EXTRA_PLIES + 4) {}
    // construtor explícito (Fase 4.2.10, SPSA): cada instância carrega
    // sua própria cópia de EvalWeights, em vez de ler o singleton global
    // evalWeights() a cada chamada de evalSimple(s,side). Necessário pra
    // rodar 2 motores com pesos DIFERENTES no mesmo processo (self-play
    // de tuning) sem risco de uma instância ler os pesos da outra por
    // esquecer de trocar um estado global antes de cada lance -- os
    // pesos ficam fixos por instância, atribuídos uma vez na construção.
    explicit Negamax(const EvalWeights& w) : tt(TT_SIZE), weights(w), nnueAccStack(MAX_PLY + QS_MAX_EXTRA_PLIES + 4) {}

    // Liga/desliga a extensão de quiescência de muro (Fase 4.2.10, item 3)
    // sem precisar recompilar -- pensado pra permitir A/B em benchmark
    // (mesmo binário, mesma posição fixa, só alterna a flag) e pra isolar
    // regressões: se um bug aparecer e não se sabe se é da busca base ou
    // da extensão, desligar aqui confirma/descarta a hipótese na hora.
    // Default true (comportamento de produção inalterado).
    void setQuiescenceEnabled(bool enabled) { quiescenceEnabled = enabled; }
    bool isQuiescenceEnabled() const { return quiescenceEnabled; }

    // Liga/desliga LMR+PVS (plano-additional.md, Prioridades 3 e 8) sem
    // recompilar -- mesmo motivo do toggle de quiescência acima: permite
    // A/B em benchmark e isolar regressões. Default true (produção).
    // IMPORTANTE: test_search_staging.cpp compara a busca "estagiada"
    // contra uma referência monolítica CONGELADA de antes da Fase 4.2.3,
    // que nunca teve LMR/PVS -- LMR muda o VALOR de busca em profundidade
    // fixa por desenho (é instabilidade de busca aceita, não um bug; a
    // garantia de LMR é "nunca poda uma linha sem reverificar em
    // profundidade cheia quando o resultado reduzido supera alpha", não
    // "produz o mesmo valor que busca plena sempre"). testFixedDepthFullWindow
    // (usado por aquele teste) desliga isto antes de rodar, preservando o
    // propósito original do teste (validar SÓ o refactor de staging).
    void setLmrPvsEnabled(bool enabled) { lmrPvsEnabled = enabled; }
    bool isLmrPvsEnabled() const { return lmrPvsEnabled; }
    void setRfpEnabled(bool) {}
    void setLmpEnabled(bool) {}

    // Modo de avaliação de folha:
    //   Heuristic (default) -- usa evalSimpleW, compatível com todos os
    //     testes e benchmarks existentes, não requer pesos carregados.
    //   NNUE -- usa a rede treinada e quantizada (AccumulatorQuant +
    //     forwardValueWLQuant, ver nnue.hpp). Requer que
    //     loadWeightsQuant() tenha sido chamado antes; sem pesos
    //     carregados o resultado é indefinido. Acumuladores são
    //     mantidos incrementalmente na pilha de busca (nnueAccStack)
    //     -- nenhuma recomputação do zero dentro da busca, exceto na
    //     raiz de cada iterative deepening (buildAccumulatorQuant,
    //     O(NUM_FEATURES) = ~330 BFS/features, uma vez por iteração).
    enum class EvalMode { Heuristic, NNUE };
    void setEvalMode(EvalMode m) { evalMode = m; }
    EvalMode getEvalMode() const { return evalMode; }

    // Ordenação de lances assistida pela cabeça de política da NNUE
    // (prompt_policy_ordering.md) -- soma forwardPolicyQuant(curAcc->acc[side])
    // como termo extra em orderPawnMoves/orderWallMoves. Mesmo estilo de
    // toggle que setQuiescenceEnabled/setLmrPvsEnabled acima: permite A/B
    // em benchmark sem recompilar. Default DESLIGADA -- requisito
    // não-negociável do prompt: com a flag off, nenhum forwardPolicyQuant é
    // chamado em lugar nenhum e orderPawnMoves/orderWallMoves executam
    // exatamente o código anterior a esta mudança (custo zero mensurável).
    // Só tem efeito quando evalMode==NNUE E curAcc != nullptr (modo
    // Heurístico não mantém AccPair nenhum pra tirar o forward pass) --
    // ligar isto em modo Heurístico é inofensivo, só não faz nada.
    void setPolicyOrderingEnabled(bool enabled) { policyOrderingEnabled = enabled; }
    bool isPolicyOrderingEnabled() const { return policyOrderingEnabled; }

    // Piso de profundidade (depth restante, não ply): forwardPolicyQuant
    // (~53500 MACs) custa ~5.8x mais que o eval de folha
    // (forwardValueWLQuant, ~9200 MACs) e, sem este piso, era chamado em
    // TODO nó interno -- que numa árvore alfa-beta são dominados por nós
    // rasos perto do horizonte (ordem de grandeza mais numerosos que os
    // nós de topo). Isso derrubava nós/s em ~3x na prática (medido pelo
    // usuário via run_arena.py). O piso restringe o forward pass extra
    // aos POUCOS nós de cima da árvore -- onde um corte alpha-beta corta
    // a subárvore inteira embaixo (maior ROI por chamada), em vez de
    // pagar o custo em nós que já iam podar rápido de qualquer forma.
    // Default 3 é um chute inicial (não tunado) -- ajuste com
    // setPolicyOrderingMinDepth() e valide nós/s + Elo via
    // benchNegamaxNNUE/run_arena.py; 0 volta ao comportamento "todo nó"
    // de antes (não recomendado, é o que causou a queda medida).
    void setPolicyOrderingMinDepth(int d) { policyOrderingMinDepth = d; }
    int getPolicyOrderingMinDepth() const { return policyOrderingMinDepth; }

    // Parâmetros tunáveis por SPSA (teste/tune_spsa.cpp) -- Fase 4.2.10+.
    // Viraram membros de instância (antes eram `constexpr`/`static constexpr`
    // globais) pelo mesmo motivo do construtor explícito Negamax(EvalWeights):
    // o tuner roda 2 engines com valores DIFERENTES no mesmo processo, então
    // cada instância precisa da sua própria cópia em vez de uma constante
    // compartilhada. Default idêntico ao valor antigo hardcoded -- nenhum
    // call-site que não chame o setter muda de comportamento.
    //
    // contempt: default = constante namespace CONTEMPT (-30) logo acima
    // desta classe -- mantida separada (não removida) porque
    // teste/test_search_staging.cpp referencia qr::CONTEMPT diretamente
    // como a referência CONGELADA usada no cálculo de mate/empate daquele
    // teste; alterar o membro da instância via setContempt() não afeta
    // esse valor de referência.
    void setContempt(int c) { contempt = c; }
    int getContempt() const { return contempt; }

    // policyOrderScale: escala do logit cru da cabeça de política somado
    // em orderPawnMoves/orderWallMoves (ver comentário grande de
    // POLICY_ORDER_SCALE acima). Default = essa mesma constante.
    void setPolicyOrderScale(long long s) { policyOrderScale = s; }
    long long getPolicyOrderScale() const { return policyOrderScale; }

    // catScoreScale: peso relativo do calor CAT (cat.hpp) vs. o termo de
    // política no score de ordenação de muro (orderWallMoves) -- os dois
    // somam no mesmo score, então este é o parâmetro que decide o
    // trade-off policy-vs-CAT. Default = valor antigo (2).
    void setCatScoreScale(long long s) { catScoreScale = s; }
    long long getCatScoreScale() const { return catScoreScale; }

    // Limpa toda a tabela de transposição. Deve ser chamado entre partidas
    // no self-play: scores de repetição (path-dependent) ficam gravados na
    // TT e contaminam buscas futuras onde a mesma posição é atingida sem
    // repetição. Não zera killers/history (esses são limpos pelo
    // resetOrderingState() no início de cada chooseMove).
    void clearTT() { std::fill(tt.begin(), tt.end(), TTEntry{}); }

    // Prioridade 6b: ao contrário de clearTT(), NÃO precisa ser chamado
    // entre partidas por corretude (ver PlayerPathCacheTable em
    // rules.hpp) -- exposto só pra medição/benchmark (zerar hits/misses
    // antes de uma rodada isolada).
    void clearXDistCache() { xdistCache.clear(); }
    uint64_t xDistCacheHits() const { return xdistCache.hits(); }
    uint64_t xDistCacheMisses() const { return xdistCache.misses(); }

    int searchShallow(const State& s, int depth, SearchStats& stats) {
        stopped = false;
        rootDepth = depth;
        deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
        g_raceExactBudgetUs = 1e18;  // só chooseMove() aplica o orçamento (ver nota lá)
        AccPair* accForSearch = nullptr;
        if (evalMode == EvalMode::NNUE) {
            nnueAccStack[0] = buildAccPairRoot(s, &xdistCache);
            accForSearch = &nnueAccStack[0];
        }
        RepetitionTable emptyHistory;
        return negamax(s, depth, -SCORE_INF, SCORE_INF, stats, emptyHistory, accForSearch);
    }

    Move chooseMove(const State& root, int maxDepthCap, int timeBudgetMs, SearchStats& stats) {
        RepetitionTable emptyHistory;
        return chooseMove(root, maxDepthCap, timeBudgetMs, stats, emptyHistory);
    }

    Move chooseMove(const State& root, int maxDepthCap, int timeBudgetMs, SearchStats& stats, const RepetitionTable& gameHistory) {
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        deadline = t0 + std::chrono::milliseconds(timeBudgetMs);
        stats = SearchStats{};
        stopped = false;
        resetOrderingState();

        // [CORREÇÃO -- achado por perda de Elo grande numa arena externa
        // apesar de nós/s saudável] Quando a própria RAIZ real (a posição
        // que o jogo de verdade está, não um nó interno da busca) já
        // satisfaz wallsLeft==(0,0), `negamax(root,...)` cai direto no
        // atalho de final "mãos vazias" -- que devolve o VALOR exato mas
        // NUNCA gera nem recursiona sobre os próprios filhos (é tratado
        // como um nó-folha, igual a `winner()`). Isso é correto e barato
        // quando essa posição é FILHA de outro nó (o pai ainda compara
        // vários candidatos normalmente, cada um recursando pra um filho
        // que ai sim aciona o atalho -- só precisa do valor pra comparar).
        // Mas quando essa posição É a raiz, não existe "nó pai" nenhum
        // pra fazer essa comparação -- e o placeholder que o atalho grava
        // na TT (`legalMoves(s)[0]`, só pra garantir legalidade, ver nota
        // em negamax) era o que `chooseMove` lia como "melhor lance",
        // resultando num lance ARBITRÁRIO (não o que realiza o DTM ótimo)
        // durante toda a fase de final -- que é tipicamente a maior parte
        // do fim de uma partida real, já que os muros costumam acabar
        // bem antes do jogo terminar. O motor "sabia" quem ganhava
        // (score correto) mas jogava lances que não necessariamente
        // levavam lá. Resolvido comparando os candidatos (só peão -- sem
        // muro nesta fase) por 1 ply usando o próprio valor EXATO (não
        // heurístico) de cada filho -- maximizar sobre valores já exatos
        // é ótimo por construção, sem precisar de busca nenhuma.
        if (root.wallsLeft[0] == 0 && root.wallsLeft[1] == 0) {
            MoveList rootMoves = legalMoves(root);
            Move best = rootMoves[0];
            int bestScore = -SCORE_INF;
            for (size_t i = 0; i < rootMoves.size(); i++) {
                State ns = applyMove(root, rootMoves[i]);
                int w = winner(ns);
                int childScore;
                if (w != -1) {
                    childScore = (w == ns.turn) ? SCORE_INF - 1 : -(SCORE_INF - 1);
                } else {
                    RaceOutcome ro = resolveEmptyHandedEndgame(ns.wallsH, ns.wallsV, ns.pawn[0], ns.pawn[1], ns.turn);
                    if (ro.winner == -1) {
                        childScore = contempt;
                    } else {
                        int raw = RACE_SCORE_BASE - ro.dtm;
                        childScore = (ro.winner == ns.turn) ? raw : -raw;
                    }
                }
                int scoreForRoot = -childScore;  // negamax: valor do filho é do ponto de vista do OPONENTE
                if (scoreForRoot > bestScore) { bestScore = scoreForRoot; best = rootMoves[i]; }
            }
            stats.reachedDepth = maxDepthCap;  // resolvido com certeza matemática, não com busca heurística parcial
            stats.score = bestScore;
            return best;
        }

        // Orçamento de TEMPO REAL (não contagem de chamadas com custo
        // estimado -- ver endgame_race.hpp) reservado ao solver exato de
        // final "mãos vazias" -- uma fração do orçamento de tempo total
        // desta busca, medida de verdade via chrono a cada chamada cara,
        // então se autocorrige (não depende de acertar hardware/otimização
        // futura). Mantém nós/s perto da linha de base sem race: no pior
        // caso (posições que precisam de muitas topologias diferentes),
        // o excedente cai no heurístico de sempre em vez de continuar
        // pagando o rebuild caro.
        constexpr double RACE_BUDGET_FRACTION = 0.03;
        g_raceExactUsedUs = 0.0;
        g_raceExactBudgetUs = std::max(1000.0, timeBudgetMs * 1000.0 * RACE_BUDGET_FRACTION);

        // Acumuladores NNUE para toda a busca: construídos UMA VEZ antes do
        // iterative deepening (a posição raiz não muda entre iterações). No
        // modo Heuristic, accForSearch fica nullptr e negamax usa evalSimpleW.
        AccPair* accForSearch = nullptr;
        if (evalMode == EvalMode::NNUE) {
            nnueAccStack[0] = buildAccPairRoot(root, &xdistCache);
            accForSearch = &nnueAccStack[0];
        }

        // CORREÇÃO (defensiva, achado em auditoria): nnueAccStack é indexado
        // por aritmética de ponteiro (curAcc+1 = filho) sobre o ply REAL da
        // árvore, sem bounds-check -- ao contrário de killers[]/histórico
        // (indexados por ply com guard explícito `ply>=0 && ply<MAX_PLY`).
        // O tamanho do vetor (MAX_PLY+QS_MAX_EXTRA_PLIES+4) cobre com folga
        // qualquer maxDepthCap<=MAX_PLY, que é o uso atual de todos os
        // call-sites (selfplay/arena usam 40). Mas se algum dia um chamador
        // passar maxDepthCap>MAX_PLY com evalMode==NNUE, a pilha estouraria
        // silenciosamente (escrita fora dos limites do vector). Trava aqui
        // em vez de exigir que todo call-site futuro saiba desse detalhe
        // interno; não afeta o modo heurístico (accForSearch fica nullptr).
        int effectiveMaxDepthCap = maxDepthCap;
        if (evalMode == EvalMode::NNUE) {
            int cap = (int)nnueAccStack.size() - QS_MAX_EXTRA_PLIES - 1;
            if (effectiveMaxDepthCap > cap) effectiveMaxDepthCap = cap;
        }

        Move bestMove = legalMoves(root)[0];
        int prevScore = 0;
        RepetitionTable reptbl = gameHistory;
        reptbl.markRoot();  // tudo antes daqui é histórico real do jogo
        for (int depth = 1; depth <= effectiveMaxDepthCap && !stopped; depth++) {
            rootDepth = depth;
            int score;
            if (depth <= 2) {
                // aspiration window não compensa em profundidades tão rasas
                // (a janela estreita quase sempre falha e obriga rebusca)
                score = negamax(root, depth, -SCORE_INF, SCORE_INF, stats, reptbl, accForSearch);
            } else if (prevScore <= -RACE_SCALE_THRESHOLD || prevScore >= RACE_SCALE_THRESHOLD) {
                // prevScore já está em escala de mate/race (RACE_SCORE_BASE
                // ou SCORE_INF-1), não em escala de evalSimple. Uma janela
                // de +-50 centrada nesse valor sempre falha e força
                // rebusca em janela cheia -- pulando direto evita pagar
                // essa profundidade duas vezes.
                score = negamax(root, depth, -SCORE_INF, SCORE_INF, stats, reptbl, accForSearch);
            } else {
                int alpha = prevScore - ASPIRATION_DELTA;
                int beta = prevScore + ASPIRATION_DELTA;
                score = negamax(root, depth, alpha, beta, stats, reptbl, accForSearch);
                if (!stopped && (score <= alpha || score >= beta)) {
                    // falhou fora da janela estreita -- rebusca com janela cheia
                    score = negamax(root, depth, -SCORE_INF, SCORE_INF, stats, reptbl, accForSearch);
                }
            }
            if (stopped) break;
            prevScore = score;
            TTEntry& e = probe(root.hash);
            if (e.valid && e.key == root.hash) bestMove = e.best;
            stats.reachedDepth = depth;
            stats.score = prevScore;
            if (clock::now() >= deadline) break;
        }
        return bestMove;
    }

private:
    static constexpr int ASPIRATION_DELTA = 50;  // unidades de evalSimple
    static constexpr int RACE_SCALE_THRESHOLD = 100'000;

    // bônus de ordenação de muro via CAT (Corridor Attention Table,
    // cat.hpp, plano-additional.md Prioridade 1) -- substitui o antigo
    // WALL_TOUCH_BONUS binário ("toca o caminho testemunha ou não") por um
    // calor CONTÍNUO por casa, que também enxerga desvios de custo baixo
    // fora do único caminho testemunha devolvido por
    // shortestPathTouchSlots. wallEdgeHeat() (cat.hpp) devolve no máximo
    // ~CAT_CORRIDOR_CM + CAT_CORRIDOR_CM/4 + CAT_BOTTLENECK_BONUS_CM (uma
    // casa de gargalo em delta==0 tocada junto de outra quase tão quente)
    // -- catScoreScale (membro de instância, ver setter público acima)
    // leva isso pra uma faixa comparável à do antigo WALL_TOUCH_BONUS
    // (600), mantendo a mesma ordem de grandeza relativa a killer
    // (1.4-1.5M) e history (depth^2 por corte) já calibrados.

    std::vector<TTEntry> tt;
    EvalWeights weights;
    // Prioridade 6b do plano-additional.md: cache de BFS de distância
    // ENTRE nós (chaveado por wallsH/wallsV/pawnCell/player, não por
    // s.hash inteiro) -- vive pela duração da instância inteira de
    // Negamax (mesmo padrão de `tt` acima), então uma topologia de muro
    // vista em qualquer nó de qualquer busca anterior desta instância
    // continua disponível. Ao contrário de `tt`, nunca precisa ser
    // limpo entre partidas por corretude (é uma função pura da chave,
    // não guarda score dependente de caminho) -- ver PlayerPathCacheTable
    // em rules.hpp.
    PlayerPathCacheTable xdistCache;
    std::chrono::steady_clock::time_point deadline;
    bool stopped = false;
    int rootDepth = 0;
    bool quiescenceEnabled = true;
    bool lmrPvsEnabled = true;
    EvalMode evalMode = EvalMode::Heuristic;
    bool policyOrderingEnabled = false;
    int policyOrderingMinDepth = 3;
    // Membros tunáveis por SPSA -- ver setContempt/setPolicyOrderScale/
    // setCatScoreScale (públicos, acima). Default = valor antigo hardcoded
    // das constantes CONTEMPT/POLICY_ORDER_SCALE (namespace, ainda
    // existem) e 2 (era CAT_SCORE_SCALE, static constexpr da classe).
    int contempt = CONTEMPT;
    long long policyOrderScale = POLICY_ORDER_SCALE;
    long long catScoreScale = 2;
    // Pilha de pares de acumuladores NNUE -- um AccPair por ply da busca,
    // indexado por aritmética de ponteiro a partir da raiz (curAcc+1 = filho).
    // Dimensionado para rootDepth até MAX_PLY + quiescência até QS_MAX_EXTRA_PLIES
    // com 4 slots de folga. Heap-alocado (std::vector) para não onerar a
    // pilha de função quando Negamax é instanciado como variável local.
    // Custo: ~2 KB × 70 = ~140 KB por instância -- desprezível frente à TT (~40 MB).
    std::vector<AccPair> nnueAccStack;

    // killer moves: 2 slots por ply, lances que causaram beta-cutoff em
    // nós irmãos na mesma distância da raiz.
    Move killers[MAX_PLY][2];
    bool killerValid[MAX_PLY][2] = {};

    // history heuristic: indexada por [lado que joga][índice canônico do
    // lance] (moveToPolicyIndex, rules.hpp), incrementada por depth^2 a
    // cada beta-cutoff.
    int history[2][NUM_MOVE_INDICES] = {};

    void resetOrderingState() {
        std::memset(killerValid, 0, sizeof(killerValid));
        std::memset(history, 0, sizeof(history));
    }

    TTEntry& probe(uint64_t hash) { return tt[hash & (TT_SIZE - 1)]; }

    void store(uint64_t hash, int depth, int score, TTFlag flag, const Move& best) {
        TTEntry& e = probe(hash);
        if (!e.valid || depth >= e.depth) {
            e = {hash, (int32_t)score, (int8_t)depth, flag, best, true};
        }
    }

    // registra um lance que causou beta-cutoff: atualiza killer slots
    // (sem duplicar o slot 0 se já for o mesmo lance) e a history table.
    void recordCutoff(const Move& m, int ply, int side, int depth) {
        if (ply >= 0 && ply < MAX_PLY) {
            if (!(killerValid[ply][0] && killers[ply][0] == m)) {
                killers[ply][1] = killers[ply][0];
                killerValid[ply][1] = killerValid[ply][0];
                killers[ply][0] = m;
                killerValid[ply][0] = true;
            }
        }
        history[side][moveToPolicyIndex(m)] += depth * depth;
    }

    // buffer fixo (stack, sem heap alloc) para reordenar lances -- máximo
    // de lances legais em Quoridor é 3 peão + 128 muro = 131; 256 dá folga.
    // static thread_local: evita realocar a cada chamada (chamada em
    // praticamente todo nó da árvore) sem precisar de heap.
    static constexpr size_t ORDER_BUF_CAP = 256;

    // Ordenação dentro do bloco de peão (Fase 4.2.3: peão e muro agora são
    // dois pools separados, gerados em estágios distintos -- não faz mais
    // sentido comparar score de peão contra score de muro num único sort,
    // o próprio staging já garante peão antes de muro). Só killer + history
    // decidem a ordem dentro do bloco de peão (no máximo 5 candidatos, o
    // impacto de ordenação fina aqui é pequeno de qualquer forma).
    // `policy` (opcional, default nullptr, prompt_policy_ordering.md):
    // se não-nulo, aponta pro array de 209 logits crus já computado UMA
    // VEZ por nó (forwardPolicyQuant sobre curAcc->acc[side], ver negamax)
    // -- soma policyLogitForMove(*policy, m, side)*POLICY_ORDER_SCALE como
    // termo extra. nullptr (caso comum, flag desligada ou modo
    // Heurístico) preserva o código/custo exatos de antes desta mudança.
    void orderPawnMoves(MoveList& moves, int ply, int side,
                         const std::array<float, POLICY_OUT>* policy = nullptr) {
        bool ply0 = ply >= 0 && ply < MAX_PLY;
        size_t n = moves.size();
        std::pair<long long, Move> buf[ORDER_BUF_CAP];
        for (size_t i = 0; i < n; i++) {
            const Move& m = moves[i];
            long long sc = history[side][moveToPolicyIndex(m)];
            if (policy) {
                sc += (long long)std::lround(policyLogitForMove(*policy, m, side) * (float)policyOrderScale);
            }
            if (ply0) {
                if (killerValid[ply][0] && m == killers[ply][0]) sc += 1'500'000;
                else if (killerValid[ply][1] && m == killers[ply][1]) sc += 1'400'000;
            }
            buf[i] = {sc, m};
        }
        std::sort(buf, buf + n, [](const auto& x, const auto& y) { return x.first > y.first; });
        for (size_t i = 0; i < n; i++) moves[i] = buf[i].second;
    }

    // Ordenação dentro do bloco de muro -- killer/history de antes, mais
    // duas camadas de sinal de "quão tático" é o muro (Seção 4.2.4/4.2.5
    // do plano, mais Prioridade 1 do plano-additional.md):
    //   1. wallByBFS (já existia): delta EXATO de shortestPathLen do
    //      oponente antes/depois do lance -- preciso, mas paga 2 BFS por
    //      candidato, então continua restrito a ply <= WALL_BFS_ORDER_MAX_PLY
    //      (poucos nós, perto da raiz -- custo agregado pequeno).
    //   2. CAT (cat.hpp, novo): calor de corredor do oponente, calculado
    //      UMA VEZ por nó (2 BFS, não por candidato) e recebido já pronto
    //      em `oppHeat` -- substitui o antigo WALL_TOUCH_BONUS binário.
    //      Ao contrário do touch bitmask (só enxerga o único caminho
    //      testemunha), CAT também dá crédito a muros que fecham desvios
    //      de custo baixo fora desse caminho específico -- ver cat.hpp
    //      para a derivação completa. Roda em TODOS os plies (o custo não
    //      escala com o número de candidatos, diferente de wallByBFS).
    //
    // oppCache (opcional, default nullptr, Prioridade 6 do
    // plano-additional.md): se não-nulo e `valid`, `before` é lido direto
    // do cache (0 BFS) em vez de chamar shortestPathLen -- o chamador
    // (negamax, Estágio 3) já paga essa BFS dentro de legalWallMoves logo
    // antes desta chamada, para o pré-filtro; reaproveitar aqui evita
    // pagá-la de novo para a MESMA topologia/jogador.
    // `policy`: mesmo parâmetro/contrato de orderPawnMoves acima.
    void orderWallMoves(MoveList& moves, int ply, int side, const State& s,
                         const CorridorHeat& oppHeat, const PlayerPathCache* oppCache = nullptr,
                         const std::array<float, POLICY_OUT>* policy = nullptr) {
        bool ply0 = ply >= 0 && ply < MAX_PLY;
        bool wallByBFS = ply <= WALL_BFS_ORDER_MAX_PLY;
        int opp = 1 - side;
        size_t n = moves.size();

        std::pair<long long, Move> buf[ORDER_BUF_CAP];
        // `before` não depende do candidato `m` (só de `s`, fixo pra toda
        // a chamada) -- hoisted pra fora do laço. Achado de revisão:
        // estava sendo recomputado (BFS completa) em CADA iteração, até
        // ~128 vezes por nó perto da raiz (ply<=WALL_BFS_ORDER_MAX_PLY),
        // sempre com o mesmo resultado. Achado desta rodada (Prioridade
        // 6): mesmo hoisted, ainda era uma BFS A MAIS por nó além da que
        // legalWallMoves já tinha acabado de pagar para o mesmo par
        // (topologia atual, opp) -- reaproveita o cache quando disponível.
        int before = 0;
        if (wallByBFS) {
            before = (oppCache && oppCache->valid)
                ? cachedShortestPathLen(*oppCache)
                : shortestPathLen(s.wallsH, s.wallsV, s.pawn[opp], opp);
        }
        for (size_t i = 0; i < n; i++) {
            const Move& m = moves[i];
            long long sc = 0;
            if (wallByBFS) {
                // quanto o muro aumenta o caminho do oponente até a meta
                // dele (mesma shortestPathLen já usada em evalSimple).
                // Prioridade 6b: passa por xdistCache -- restrito a
                // ply<=WALL_BFS_ORDER_MAX_PLY (perto da raiz), onde nós
                // IRMÃOS testam os MESMOS candidatos de muro sobre a
                // MESMA topologia base com frequência real, então esta
                // topologia-depois-do-candidato tende a se repetir entre
                // chamadas (não só dentro de uma única chamada).
                State ns = applyMove(s, m);
                PlayerPathCache afterCache;
                computeDistCached(ns.wallsH, ns.wallsV, ns.pawn[opp], opp, &xdistCache, afterCache);
                int after = cachedShortestPathLen(afterCache);
                sc = (long long)(after - before) * 1000;
            }
            sc += wallEdgeHeat(oppHeat, m.a, m.b, m.c) * catScoreScale;
            sc += history[side][moveToPolicyIndex(m)];
            if (policy) {
                sc += (long long)std::lround(policyLogitForMove(*policy, m, side) * (float)policyOrderScale);
            }
            if (ply0) {
                if (killerValid[ply][0] && m == killers[ply][0]) sc += 1'500'000;
                else if (killerValid[ply][1] && m == killers[ply][1]) sc += 1'400'000;
            }
            buf[i] = {sc, m};
        }
        std::sort(buf, buf + n, [](const auto& x, const auto& y) { return x.first > y.first; });
        for (size_t i = 0; i < n; i++) moves[i] = buf[i].second;
    }

    // Quiescência mínima de muro (Fase 4.2.10, item 3 -- ver constantes
    // QS_* acima). Chamada só na fronteira do horizonte (depth==0 do
    // negamax), no lugar do antigo `return evalSimple(...)` direto.
    // Stand-pat: evalSimple já é um limite inferior razoável (não existe
    // "captura obrigatória" em Quoridor), então funciona como em
    // quiescência de xadrez -- só estende se algum muro crítico do lado a
    // mover melhorar sobre o stand-pat.
    int quiescence(const State& s, int alpha, int beta, int qply, SearchStats& stats, RepetitionTable& reptbl,
                   bool rootParity, AccPair* curAcc = nullptr) {
        stats.nodes++;
        if ((stats.nodes & 0x3FF) == 0 && std::chrono::steady_clock::now() >= deadline) {
            stopped = true;
            return 0;
        }
        int w = winner(s);
        if (w != -1) return (w == s.turn) ? SCORE_INF - 1 : -(SCORE_INF - 1);

        // Ver comentário equivalente em negamax() -- mesmo critério de
        // 2 vs 3 ocorrências e mesma âncora de sinal na paridade do ply
        // em relação à raiz (rootParity), não em s.turn.
        if (reptbl.isRepetitionDraw(s.hash)) return rootParity ? contempt : -contempt;

        // Stand-pat: modo NNUE usa o acumulador já pronto (O(HIDDEN×32) ops);
        // modo heurístico usa evalSimpleW que também preenche sideCache/oppCache
        // para as checagens abaixo. No modo NNUE os caches são preenchidos
        // separadamente via computeDistCached (mesma BFS, sem pagar a fórmula
        // evalSimple). Em ambos os casos sideCache/oppCache são necessários
        // para legalWallMoves e para o critério de muro crítico.
        int side = s.turn, opp = 1 - side;
        PlayerPathCache sideCache, oppCache;
        int standPat;
        if (curAcc) {
            // NNUE: avaliação do acumulador já pronto; BFS dos dois caches
            // são pagas aqui (não somam ao stand-pat -- são usadas só abaixo
            // por legalWallMoves e oppDistBefore/oppRobustBefore). O xtable
            // amortiza o custo quando a MESMA topologia+peão já foi vista.
            computeDistCached(s.wallsH, s.wallsV, s.pawn[side], side, &xdistCache, sideCache);
            computeDistCached(s.wallsH, s.wallsV, s.pawn[opp],  opp,  &xdistCache, oppCache);
            standPat = nnueEvalInt(*curAcc, side);
        } else {
            // Heurístico: evalSimpleW preenche sideCache/oppCache como efeito
            // colateral (Prioridade 6 do plano-additional.md) -- 2 BFS fundidas.
            standPat = evalSimpleW(s, s.turn, weights, &sideCache, &oppCache, &xdistCache);
        }
        if (standPat >= beta) return standPat;
        int localAlpha = alpha > standPat ? alpha : standPat;
        int best = standPat;

        if (qply >= QS_MAX_EXTRA_PLIES) return best;

        if (s.wallsLeft[side] <= 0) return best;  // sem muro pra jogar, nada a estender

        // Os caches (sideCache/oppCache) já estão válidos em ambos os modos
        // acima -- legalWallMoves não roda BFS nenhuma internamente.
        MoveList wallMoves;
        uint64_t touchH0, touchV0, touchH1, touchV1;
        PlayerPathCache* cache0 = (side == 0) ? &sideCache : &oppCache;
        PlayerPathCache* cache1 = (side == 0) ? &oppCache  : &sideCache;
        legalWallMoves(s, side, wallMoves, &touchH0, &touchV0, &touchH1, &touchV1, cache0, cache1, &xdistCache);
        uint64_t touchHOpp = (side == 0) ? touchH1 : touchH0;
        uint64_t touchVOpp = (side == 0) ? touchV1 : touchV0;

        int oppDistBefore   = cachedShortestPathLen(oppCache);
        int oppRobustBefore = cachedPathRobustness(oppCache, s.wallsH, s.wallsV);

        for (size_t i = 0; i < wallMoves.size(); i++) {
            const Move& m = wallMoves[i];
            int slot = slotIdx(m.b, m.c);
            bool touches = (m.a == 0) ? ((touchHOpp >> slot) & 1ull) : ((touchVOpp >> slot) & 1ull);
            if (!touches) continue;  // não toca o caminho atual do oponente -> não pode ser "crítico" aqui

            State ns = applyMove(s, m);
            // 1 BFS só (computeDistFull), não mais até 2 (shortestPathLen
            // sempre + pathRobustness condicional) -- a topologia de `ns`
            // é única por candidato (não há cache de nó pra reaproveitar
            // aqui), mas a robustez sai de graça do mesmo cache quando
            // precisa (só percorre parent[], não roda BFS nova).
            PlayerPathCache oppCacheAfter;
            computeDistCached(ns.wallsH, ns.wallsV, ns.pawn[opp], opp, &xdistCache, oppCacheAfter);
            int oppDistAfter = cachedShortestPathLen(oppCacheAfter);
            bool critical = (oppDistAfter - oppDistBefore) >= QS_CRITICAL_BFS_DELTA;
            if (!critical && oppRobustBefore > QS_CRITICAL_ROBUSTNESS_DROP_TO) {
                int oppRobustAfter = cachedPathRobustness(oppCacheAfter, ns.wallsH, ns.wallsV);
                critical = (oppRobustAfter <= QS_CRITICAL_ROBUSTNESS_DROP_TO);
            }
            if (!critical) continue;

            // Acumulador do filho de quiescência: mesmo padrão do negamax;
            // lances de quiescência são sempre de MURO (nunca peão aqui).
            AccPair* childAcc = nullptr;
            if (curAcc) {
                childAcc = curAcc + 1;
                makeChildAccPair(*curAcc, *childAcc, s, m, &xdistCache);
            }
            reptbl.push(ns.hash);
            int score = -quiescence(ns, -beta, -localAlpha, qply + 1, stats, reptbl, !rootParity, childAcc);
            reptbl.pop();
            if (stopped) return 0;
            if (score > best) best = score;
            if (score > localAlpha) localAlpha = score;
            if (localAlpha >= beta) break;
        }
        return best;
    }

    int negamax(const State& s, int depth, int alpha, int beta, SearchStats& stats, RepetitionTable& reptbl, AccPair* curAcc = nullptr) {
        stats.nodes++;
        if ((stats.nodes & 0x3FF) == 0 && std::chrono::steady_clock::now() >= deadline) {
            stopped = true;
            return 0;
        }
        int w = winner(s);
        if (w != -1) return (w == s.turn) ? SCORE_INF - 1 : -(SCORE_INF - 1);

        int ply = rootDepth - depth;

        // Empate por repetição (ver RepetitionTable::isRepetitionDraw em
        // rules.hpp pro critério de 2 vs 3 ocorrências). O sinal do
        // CONTEMPT é ancorado na paridade do ply em relação à RAIZ desta
        // busca (ply par = mesmo lado que está pensando agora), não em
        // `s.turn` puro -- porque `s.turn` alterna a cada lance e não tem
        // relação fixa com "nós" vs "o oponente": a mesma fórmula
        // aplicada direto a s.turn dá um viés de sinal que troca
        // aleatoriamente conforme a paridade da profundidade em que a
        // repetição aparece na árvore, em vez de refletir uma preferência
        // consistente. É o mesmo motivo pelo qual o Stockfish ancora
        // "Analysis Contempt" ao lado que tem o lance NA RAIZ, não ao
        // lado do nó terminal (ver UCI option "Analysis Contempt": "By
        // default, contempt is set to prefer the side to move [na raiz]").
        if (reptbl.isRepetitionDraw(s.hash)) return (ply % 2 == 0) ? contempt : -contempt;

        // Final "mãos vazias" (endgame_race.hpp, plano-additional.md
        // Prioridade 4): quando os dois ficam sem muros, a topologia
        // congela pra sempre e o jogo vira corrida de peão pura -- resolve
        // por solver exato (Nível 1 -> Nível 2 -> Serviço B, ver
        // endgame_race.hpp) em vez de continuar a busca heurística, que
        // não tem como bater certeza matemática aqui. Colocado depois do
        // check de repetição de propósito: se a mesma posição já se
        // repetiu de fato na partida real, isso tem prioridade sobre o
        // resultado teórico do subjogo (mesmo doravante).
        bool raceResolved = false;
        int raceScore = 0;
        if (s.wallsLeft[0] == 0 && s.wallsLeft[1] == 0) {
            // Achado de regressão de nós/s (arena externa): o cache de
            // 1 slot em endgame_race.hpp (indexado só por topologia de
            // muro) quase nunca acerta numa busca real -- medido em
            // ~0,5% de hit rate, porque o alpha-beta ainda está decidindo
            // ONDE colocar os últimos muros de cada lado quando entra
            // nesta fase, e cada candidato de muro testado nessa borda
            // gera uma topologia final DIFERENTE. Isso faz o solver
            // exato (Serviço B, ~0,6-0,9ms por chamada) rodar do zero em
            // quase todo nó desta fase. Uma posição EXATA (mesmo
            // wallsH/wallsV + mesmo par de peões + mesmo turno -- ou
            // seja, o mesmo s.hash) sim se repete com frequência real via
            // transposição/re-busca de PVS/aspiration/iterative
            // deepening, então armazenar o resultado na TT (chave =
            // s.hash, já calculada, já rege o restante do motor) resolve
            // esse caso sem precisar de estrutura nova: EXACT com
            // depth=127 pra nunca ser descartado por um `e.depth>=depth`
            // de profundidade menor.
            TTEntry& re = probe(s.hash);
            if (re.valid && re.key == s.hash && re.flag == EXACT) {
                return re.score;
            }
            // Orçamento agora cobre a chamada INTEIRA (gate barato +
            // solver exato), não só o rebuild caro -- ver nota grande em
            // endgame_race.hpp. O gate sozinho (~1,85us/chamada, 4 BFS de
            // 81 casas) já dobrava o custo por nó mesmo quando o solver
            // exato nunca era acionado, porque rodava incondicionalmente
            // em todo nó desta fase. Checando o orçamento ANTES de pagar
            // qualquer coisa (nem o gate), o pior caso (orçamento
            // esgotado) fica com custo zero adicional -- nó igual ao que
            // seria sem a feature de race, garantindo que nós/s nunca cai
            // mais que a fração do orçamento reservada (RACE_BUDGET_FRACTION).
            if (g_raceExactUsedUs < g_raceExactBudgetUs) {
                auto __raceT0 = std::chrono::steady_clock::now();
                RaceOutcome ro = resolveEmptyHandedEndgame(s.wallsH, s.wallsV, s.pawn[0], s.pawn[1], s.turn);
                g_raceExactUsedUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - __raceT0).count();
                raceResolved = true;
                if (ro.winner == -1) {
                    raceScore = contempt;  // empate -- perseguição infinita/repetição (bug corrigido: era -CONTEMPT, recompensava o empate)
                } else {
                    int raw = RACE_SCORE_BASE - ro.dtm;
                    raceScore = (ro.winner == s.turn) ? raw : -raw;
                }
            }
            // orçamento esgotado -- raceResolved fica false, cai pro resto
            // da função (TT normal, geração de lances, recursão)
            // EXATAMENTE como se este `if` nem existisse: no pior caso o
            // nó custa o mesmo que custava antes da feature de race
            // existir, nunca mais.
        }
        if (raceResolved) {

            int score = raceScore;
            // Bug pego pelo test_search_staging.cpp (posicao 3) durante a
            // implementacao deste cache: chooseMove le e.best da TT SEM
            // checar legalidade quando root.hash bate (ver loop de
            // iterative deepening acima) -- se a raiz de verdade cair
            // nesta fase (jogo real com os dois sem muro), um placeholder
            // tipo Move::pawn(0) pode nao ser um lance legal ali e virar
            // um lance ilegal devolvido pelo motor. legalMoves(s)[0] e
            // sempre legal (peao, ja que sem muro so ha lance de peao) --
            // nao e o lance DTM-otimo (o solver so devolve o VALOR, nao
            // qual lance realiza esse valor), mas garante seguranca; so
            // paga o custo de legalMoves numa chamada por posicao nova
            // (miss), nao por no.
            store(s.hash, 127, score, EXACT, legalMoves(s)[0]);
            return score;
        }

        // Fix (Fase 4.2.10, pós-implementação): depth==0 tem que ser
        // resolvido ANTES de consultar a TT para cutoff. Motivo: a
        // condição `e.depth >= depth` logo abaixo é TRIVIALMENTE
        // verdadeira quando depth==0 (qualquer entrada válida tem
        // depth>=0), então um nó folha podia ser "curto-circuitado" por
        // uma entrada EXACT deixada por uma busca MAIS PROFUNDA da MESMA
        // posição alcançada em outro ramo/iteração -- devolvendo o valor
        // de um negamax completo de N plies no lugar do valor de
        // quiescência (que só olha um subconjunto restrito de muros
        // críticos). Com evalSimple (função pura, sem esse desenho de
        // busca restrita) isso nunca mudava o resultado observável nos
        // testes feitos; com quiescência, mudava com frequência real --
        // quebrava a garantia de exatidão de que o aspiration window de
        // chooseMove depende (achado desta rodada, ver Seção 4.2.10).
        // Nós de depth==0 nunca são armazenados na TT (retornam antes do
        // `store()` no fim da função), então mover este check pra cá não
        // perde nenhum reuso real -- só remove uma substituição indevida.
        if (depth == 0) {
            // flag desligada -> comportamento de folha pré-4.2.10 (eval
            // direta, sem extensão de muro crítico). winner() já foi
            // resolvido acima, então não há necessidade de checá-lo de
            // novo aqui -- mesma garantia que quiescence() já tinha.
            if (!quiescenceEnabled) {
                if (curAcc) return nnueEvalInt(*curAcc, s.turn);
                return evalSimpleW(s, s.turn, weights, nullptr, nullptr, &xdistCache);
            }
            return quiescence(s, alpha, beta, 0, stats, reptbl, ply % 2 == 0, curAcc);
        }

        int alphaOrig = alpha;
        TTEntry& e = probe(s.hash);
        // lance da TT copiado pra variável local logo aqui -- não guarda
        // ponteiro pra dentro da tabela (e.best), que pode ser sobrescrita
        // por qualquer uma das chamadas recursivas de negamax mais abaixo
        // (mesmo slot pode colidir por hash de outra posição). Estágio 1
        // do staging (abaixo) só usa hasTTMove/ttMoveVal a partir daqui.
        bool hasTTMove = false;
        Move ttMoveVal = Move::pawn(0);
        if (e.valid && e.key == s.hash) {
            hasTTMove = true;
            ttMoveVal = e.best;
            // Corte de valor: `e.depth >= depth` (reuso de qualquer busca
            // igual-ou-mais-profunda), a técnica padrão usada por
            // praticamente todo motor de alpha-beta+TT sério.
            if (e.depth >= depth) {
                if (e.flag == EXACT) return e.score;
                if (e.flag == LOWER) alpha = std::max(alpha, (int)e.score);
                else if (e.flag == UPPER) beta = std::min(beta, (int)e.score);
                if (alpha >= beta) return e.score;
            }
        }

        int side = s.turn;

        // Ordenação assistida por política (prompt_policy_ordering.md):
        // no máximo UM forward pass de política por nó (não por
        // candidato), e só nos nós com depth restante >= policyOrderingMinDepth
        // (ver comentário do setter acima -- forwardPolicyQuant é ~5.8x
        // mais caro que o eval de folha e, sem este piso, dominava o
        // custo total por rodar em todo nó interno). Gated também por
        // policyOrderingEnabled && curAcc, mesmo guarda que nnueEvalInt já
        // usa pra saber se há AccPair mantido pra esta busca (modo
        // Heurístico nunca entra aqui). curAcc->acc[side]
        // é a perspectiva de quem vai jogar agora -- garantidamente já
        // resolvida (eager) pela invariante de AccPair documentada em
        // nnue.hpp, então nenhuma chamada extra a resolvePending é
        // necessária aqui (mesma premissa que o `nnueEvalInt(*curAcc, s.turn)`
        // de depth==0 acima já depende). policyArr fica fora do `if` (RAII
        // de pilha, sem custo quando não inicializado/usado) -- policyPtr
        // só aponta pra ela quando de fato computada.
        std::array<float, POLICY_OUT> policyArr;
        const std::array<float, POLICY_OUT>* policyPtr = nullptr;
        if (policyOrderingEnabled && curAcc && depth >= policyOrderingMinDepth) {
            forwardPolicyQuant(curAcc->acc[side], policyArr);
            policyPtr = &policyArr;
        }

        // --- Geração estagiada de lances (Fase 4.2.3 do plano) ---------
        // Estágio 1: lance da TT, testado antes de gerar qualquer muro --
        // se ele já causar corte, o Estágio 3 (o caro, com pré-filtro +
        // DSU + até BFS completo por candidato ambíguo, Seção 4.2.1) nem
        // chega a rodar. Legalidade do lance da TT verificada de forma
        // barata: se for peão, checando pertencimento no bloco de peão
        // (simpre gerado, Estágio 2, sem BFS/alocação -- gerar esse bloco
        // não paga o custo que queremos evitar); se for muro,
        // isWallMoveLegal (rules.hpp) testa só ESSE candidato, sem gerar
        // os outros 127. Guarda contra colisão de hash na TT: um lance
        // "fantasma" (de outra posição com mesmo hash) nunca passa nessas
        // checagens, então nunca é jogado.
        MoveList pawnMoves;
        pawnStepMoves(s, side, pawnMoves);
        orderPawnMoves(pawnMoves, ply, side, policyPtr);

        bool ttTried = false;
        if (hasTTMove && ttMoveVal.isWall) {
            if (isWallMoveLegal(s, side, ttMoveVal.a, ttMoveVal.b, ttMoveVal.c))
                ttTried = true;
        } else if (hasTTMove) {
            for (size_t i = 0; i < pawnMoves.size(); i++) {
                if (pawnMoves[i] == ttMoveVal) { ttTried = true; break; }
            }
        }

        int best = -SCORE_INF;
        Move bestMove = pawnMoves.empty() ? ttMoveVal : pawnMoves[0];
        bool haveMove = false;
        bool cutoff = false;
        bool ply0 = ply >= 0 && ply < MAX_PLY;
        int moveCount = 0;  // 1-based, TT+peão+muro combinados -- alimenta LMR+PVS abaixo

        // aplica um lance candidato; devolve true se a busca deve parar
        // de avaliar mais lances neste nó (beta-cutoff OU tempo esgotado
        // -- `stopped` é checado pelo chamador logo em seguida).
        //
        // moveIndex (1-based, ordem de tentativa neste nó) e catHeat (só
        // faz sentido pra lance de muro -- ver wallEdgeHeat/cat.hpp; -1 =
        // "sem calor conhecido", pula os modificadores CAT) alimentam
        // LMR+PVS (Prioridades 3 e 8 do plano-additional.md, constantes
        // no topo deste arquivo). Com lmrPvsEnabled desligado, cai de
        // volta no comportamento antigo -- sempre janela completa em
        // profundidade cheia (ver setLmrPvsEnabled acima).
        auto tryMove = [&](const Move& m, int moveIndex, int catHeat) -> bool {
            // Acumulador do filho: se modo NNUE ativo, makeChildAccPair
            // (nnue.hpp, Item 3) atualiza AGORA só a perspectiva de quem vai
            // jogar no filho (a que nnueEvalInt vai ler se ele for folha);
            // a perspectiva de quem jogou `m` fica pending -- só paga o
            // update (O(HIDDEN) + até 2 BFS de muro) se a busca de fato
            // chegar a precisar dela (um ply abaixo, se não houver cutoff
            // antes). No modo Heurístico, curAcc == nullptr e nenhum
            // acumulador é mantido.
            AccPair* childAcc = nullptr;
            if (curAcc) {
                childAcc = curAcc + 1;
                makeChildAccPair(*curAcc, *childAcc, s, m, &xdistCache);
            }
            State ns = applyMove(s, m);
            reptbl.push(ns.hash);
            int score;
            if (!lmrPvsEnabled || moveIndex == 1) {
                // 1º lance (normalmente o da TT, a aposta da ordenação de
                // ser o melhor): sempre janela completa, profundidade
                // cheia -- é o valor de referência que os demais lances
                // (busca de janela nula, abaixo) tentam apenas SUPERAR ou
                // não, sem precisar do valor exato quando não superam.
                score = -negamax(ns, depth - 1, -beta, -alpha, stats, reptbl, childAcc);
            } else {
                // LMR (Prioridade 3): nunca reduz lance killer, nunca
                // reduz muro "quente" (perto do caminho ótimo do
                // oponente -- candidato a mudar o resultado tático,
                // mesmo cuidado que o plano pede para lances que a
                // checagem crítica de quiescência marcaria). Muro "frio"
                // ganha 1 passo extra de redução.
                bool isKillerMove = ply0 && ((killerValid[ply][0] && m == killers[ply][0]) ||
                                             (killerValid[ply][1] && m == killers[ply][1]));
                int reduction = 0;
                if (moveIndex > LMR_MIN_MOVE_INDEX && depth >= LMR_MIN_DEPTH && !isKillerMove) {
                    bool hot = (catHeat >= 0 && catHeat >= CAT_HOT_CM);
                    if (!hot) {
                        reduction = lmrReduction(depth, moveIndex);
                        if (catHeat >= 0 && catHeat < CAT_COLD_CM) reduction += 1;
                        int maxRed = depth / 2;
                        if (reduction > maxRed) reduction = maxRed;
                    }
                }
                // PVS (Prioridade 8): janela nula em profundidade (talvez
                // reduzida por LMR).
                score = -negamax(ns, depth - 1 - reduction, -alpha - 1, -alpha, stats, reptbl, childAcc);
                if (!stopped && reduction > 0 && score > alpha) {
                    // vazou acima de alpha na profundidade reduzida --
                    // reverifica em profundidade CHEIA antes de confiar
                    // (a redução é heurística, não prova nada sozinha),
                    // ainda em janela nula.
                    score = -negamax(ns, depth - 1, -alpha - 1, -alpha, stats, reptbl, childAcc);
                }
                if (!stopped && score > alpha && score < beta) {
                    // janela nula vazou sem provar corte (score<beta) --
                    // só profundidade cheia + janela completa dá o valor
                    // exato aqui (mesmo lance pode genuinamente ser o
                    // novo melhor, não só um pouco melhor que alpha).
                    score = -negamax(ns, depth - 1, -beta, -alpha, stats, reptbl, childAcc);
                }
            }
            reptbl.pop();
            if (stopped) return true;
            if (!haveMove || score > best) { best = score; bestMove = m; haveMove = true; }
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                recordCutoff(m, ply, side, depth);
                return true;
            }
            return false;
        };

        // Estágio 1: lance da TT primeiro, se validado acima.
        if (ttTried) {
            moveCount++;
            cutoff = tryMove(ttMoveVal, moveCount, -1);
            if (stopped) return 0;
        }

        // Estágio 2: resto dos lances de peão (bloco já gerado/ordenado).
        if (!cutoff) {
            for (size_t i = 0; i < pawnMoves.size() && !cutoff; i++) {
                const Move& m = pawnMoves[i];
                if (ttTried && m == ttMoveVal) continue;  // já tentado no Estágio 1
                moveCount++;
                cutoff = tryMove(m, moveCount, -1);
                if (stopped) return 0;
            }
        }

        // Estágio 3: muros, gerados sob demanda -- só se os estágios
        // anteriores não deram corte. Paga aqui o custo de
        // legalWallMoves (pré-filtro + DSU + BFS residual, Seção 4.2.1)
        // só numa fração dos nós internos, não em todos.
        if (!cutoff) {
            MoveList wallMoves;
            // Prioridade 6 do plano-additional.md: legalWallMoves já roda
            // (no máximo) 1 BFS por jogador para o pré-filtro de muro --
            // devolvendo esses caches aqui, orderWallMoves (logo abaixo)
            // reaproveita o do oponente pra "before" em vez de pagar mais
            // uma BFS pra mesma topologia/jogador.
            PlayerPathCache cache0, cache1;
            legalWallMoves(s, side, wallMoves, nullptr, nullptr, nullptr, nullptr, &cache0, &cache1, &xdistCache);
            // Achado de revisão de performance: computeCorridorHeat (2
            // BFS) rodava incondicionalmente aqui, mesmo quando
            // wallMoves está vazio -- o que acontece sempre que o lado a
            // mover já não tem mais muros (wallsLeft[side]<=0), mas o
            // oponente ainda tem (senão o hook de final "mãos vazias",
            // mais acima nesta função, já teria resolvido o nó). Isso é
            // comum durante boa parte do "meio de jogo tardio"/"final" --
            // exatamente a fase que mais sofria com custo extra por nó.
            // Só vale a pena pagar o calor de corredor se há algum muro
            // pra de fato ordenar com ele.
            CorridorHeat oppHeat;
            bool haveOppHeat = false;
            if (!wallMoves.empty()) {
                int opp = 1 - side;
                // computeCorridorHeat continua com sua própria BFS
                // dedicada (não reaproveita cache0/cache1) -- ver nota em
                // rules.hpp sobre por que essa fusão específica (com
                // early-exit parcial) fica de fora por risco de correção.
                oppHeat = computeCorridorHeat(s.wallsH, s.wallsV, s.pawn[opp], opp);
                haveOppHeat = true;
                const PlayerPathCache& oppCache = (opp == 0) ? cache0 : cache1;
                orderWallMoves(wallMoves, ply, side, s, oppHeat, &oppCache, policyPtr);
            }
            for (size_t i = 0; i < wallMoves.size() && !cutoff; i++) {
                const Move& m = wallMoves[i];
                if (ttTried && m == ttMoveVal) continue;  // já tentado no Estágio 1
                moveCount++;
                int heat = haveOppHeat ? wallEdgeHeat(oppHeat, m.a, m.b, m.c) : -1;
                cutoff = tryMove(m, moveCount, heat);
                if (stopped) return 0;
            }
        }

        TTFlag flag = (best <= alphaOrig) ? UPPER : (best >= beta ? LOWER : EXACT);
        store(s.hash, depth, best, flag, bestMove);
        return best;
    }

#ifdef QR_ENABLE_TEST_HOOKS
    // wrappers só para teste (test_move_ordering.cpp) -- orderWallMoves/
    // orderPawnMoves são detalhe de implementação privado; produção nunca
    // define QR_ENABLE_TEST_HOOKS, então isso não existe fora dos testes.
public:
    void testOrderWallMoves(MoveList& moves, int ply, int side, const State& s,
                             const CorridorHeat& oppHeat) {
        orderWallMoves(moves, ply, side, s, oppHeat);
    }
    void testOrderPawnMoves(MoveList& moves, int ply, int side) {
        orderPawnMoves(moves, ply, side);
    }
    int testQuiescence(const State& s, int alpha, int beta, SearchStats& stats) {
        stopped = false;
        deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
        RepetitionTable emptyHistory;
        return quiescence(s, alpha, beta, 0, stats, emptyHistory, true);
    }
    // igual a testFixedDepthFullWindow, mas SEM desligar LMR/PVS -- para
    // testes que precisam comparar explicitamente o comportamento COM
    // LMR/PVS ligado (produção) contra a referência de janela cheia
    // acima (ver test_lmr_pvs.cpp). Usa o valor atual de lmrPvsEnabled
    // da instância (default true, via setLmrPvsEnabled se o chamador
    // quiser testar desligado explicitamente também).
    int testFixedDepthFullWindowLmr(const State& s, int depth, SearchStats& stats) {
        resetOrderingState();
        std::fill(tt.begin(), tt.end(), TTEntry{});
        stopped = false;
        rootDepth = depth;
        deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
        g_raceExactBudgetUs = 1e18;
        stats = SearchStats{};
        AccPair* accForSearch = nullptr;
        if (evalMode == EvalMode::NNUE) {
            nnueAccStack[0] = buildAccPairRoot(s, &xdistCache);
            accForSearch = &nnueAccStack[0];
        }
        RepetitionTable emptyHistory;
        return negamax(s, depth, -SCORE_INF, SCORE_INF, stats, emptyHistory, accForSearch);
    }
    int testFixedDepthFullWindow(const State& s, int depth, SearchStats& stats) {
        resetOrderingState();
        std::fill(tt.begin(), tt.end(), TTEntry{});
        stopped = false;
        rootDepth = depth;
        deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
        g_raceExactBudgetUs = 1e18;  // só chooseMove() aplica o orçamento (ver nota lá) -- senão uma chamada anterior de chooseMove() deixa o orçamento finito "vazando" pra esta busca de referência
        stats = SearchStats{};
        AccPair* accForSearch = nullptr;
        if (evalMode == EvalMode::NNUE) {
            nnueAccStack[0] = buildAccPairRoot(s, &xdistCache);
            accForSearch = &nnueAccStack[0];
        }
        RepetitionTable emptyHistory;
        // LMR/PVS desligado aqui de propósito: esta função existe pra
        // comparar a busca ESTAGIADA contra a referência monolítica
        // congelada (test_search_staging.cpp, Fase 4.2.3) -- a referência
        // nunca teve LMR/PVS, então compará-las com LMR ligado testaria
        // duas coisas diferentes ao mesmo tempo (staging E instabilidade
        // de busca do LMR) e quebraria a garantia de "0 divergências" que
        // aquele teste existe pra validar. Salva/restaura em vez de só
        // desligar, caso um teste futuro precise ligar isto de propósito.
        bool prevLmrPvs = lmrPvsEnabled;
        lmrPvsEnabled = false;
        int result = negamax(s, depth, -SCORE_INF, SCORE_INF, stats, emptyHistory, accForSearch);
        lmrPvsEnabled = prevLmrPvs;
        return result;
    }
    // igual, mas SEM limpar a TT -- pra reproduzir de propósito o reuso
    // entre iterações do chooseMove (não limpa killers/history também).
    int testNegamaxKeepTT(const State& s, int depth, int alpha, int beta, SearchStats& stats) {
        stopped = false;
        rootDepth = depth;
        deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
        g_raceExactBudgetUs = 1e18;  // só chooseMove() aplica o orçamento (ver nota lá)
        stats = SearchStats{};
        AccPair* accForSearch = nullptr;
        if (evalMode == EvalMode::NNUE) {
            nnueAccStack[0] = buildAccPairRoot(s, &xdistCache);
            accForSearch = &nnueAccStack[0];
        }
        RepetitionTable emptyHistory;
        return negamax(s, depth, alpha, beta, stats, emptyHistory, accForSearch);
    }
    void testClearTT() { std::fill(tt.begin(), tt.end(), TTEntry{}); }
#endif
};

} // namespace qr
