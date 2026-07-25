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
#include "rules.hpp"
#include "cat.hpp"
#include "endgame_race.hpp"

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
    Negamax() : tt(TT_SIZE), weights(evalWeights()) {}
    // construtor explícito (Fase 4.2.10, SPSA): cada instância carrega
    // sua própria cópia de EvalWeights, em vez de ler o singleton global
    // evalWeights() a cada chamada de evalSimple(s,side). Necessário pra
    // rodar 2 motores com pesos DIFERENTES no mesmo processo (self-play
    // de tuning) sem risco de uma instância ler os pesos da outra por
    // esquecer de trocar um estado global antes de cada lance -- os
    // pesos ficam fixos por instância, atribuídos uma vez na construção.
    explicit Negamax(const EvalWeights& w) : tt(TT_SIZE), weights(w) {}

    // Liga/desliga a extensão de quiescência de muro (Fase 4.2.10, item 3)
    // sem precisar recompilar -- pensado pra permitir A/B em benchmark
    // (mesmo binário, mesma posição fixa, só alterna a flag) e pra isolar
    // regressões: se um bug aparecer e não se sabe se é da busca base ou
    // da extensão, desligar aqui confirma/descarta a hipótese na hora.
    // Default true (comportamento de produção inalterado).
    void setQuiescenceEnabled(bool enabled) { quiescenceEnabled = enabled; }
    bool isQuiescenceEnabled() const { return quiescenceEnabled; }

    // Limpa toda a tabela de transposição. Deve ser chamado entre partidas
    // no self-play: scores de repetição (path-dependent) ficam gravados na
    // TT e contaminam buscas futuras onde a mesma posição é atingida sem
    // repetição. Não zera killers/history (esses são limpos pelo
    // resetOrderingState() no início de cada chooseMove).
    void clearTT() { std::fill(tt.begin(), tt.end(), TTEntry{}); }

    int searchShallow(const State& s, int depth, SearchStats& stats) {
        stopped = false;
        rootDepth = depth;
        deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
        g_raceExactBudgetUs = 1e18;  // só chooseMove() aplica o orçamento (ver nota lá)
        RepetitionTable emptyHistory;
        return negamax(s, depth, -SCORE_INF, SCORE_INF, stats, emptyHistory);
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
                        childScore = CONTEMPT;
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

        Move bestMove = legalMoves(root)[0];
        int prevScore = 0;
        RepetitionTable reptbl = gameHistory;
        reptbl.markRoot();  // tudo antes daqui é histórico real do jogo
        for (int depth = 1; depth <= maxDepthCap && !stopped; depth++) {
            rootDepth = depth;
            int score;
            if (depth <= 2) {
                // aspiration window não compensa em profundidades tão rasas
                // (a janela estreita quase sempre falha e obriga rebusca)
                score = negamax(root, depth, -SCORE_INF, SCORE_INF, stats, reptbl);
            } else if (prevScore <= -RACE_SCALE_THRESHOLD || prevScore >= RACE_SCALE_THRESHOLD) {
                // prevScore já está em escala de mate/race (RACE_SCORE_BASE
                // ou SCORE_INF-1), não em escala de evalSimple. Uma janela
                // de +-50 centrada nesse valor sempre falha e força
                // rebusca em janela cheia -- pulando direto evita pagar
                // essa profundidade duas vezes.
                score = negamax(root, depth, -SCORE_INF, SCORE_INF, stats, reptbl);
            } else {
                int alpha = prevScore - ASPIRATION_DELTA;
                int beta = prevScore + ASPIRATION_DELTA;
                score = negamax(root, depth, alpha, beta, stats, reptbl);
                if (!stopped && (score <= alpha || score >= beta)) {
                    // falhou fora da janela estreita -- rebusca com janela cheia
                    score = negamax(root, depth, -SCORE_INF, SCORE_INF, stats, reptbl);
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
    // -- CAT_SCORE_SCALE leva isso pra uma faixa comparável à do antigo
    // WALL_TOUCH_BONUS (600), mantendo a mesma ordem de grandeza relativa
    // a killer (1.4-1.5M) e history (depth^2 por corte) já calibrados.
    static constexpr long long CAT_SCORE_SCALE = 2;

    std::vector<TTEntry> tt;
    EvalWeights weights;
    std::chrono::steady_clock::time_point deadline;
    bool stopped = false;
    int rootDepth = 0;
    bool quiescenceEnabled = true;

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
    void orderPawnMoves(MoveList& moves, int ply, int side) {
        bool ply0 = ply >= 0 && ply < MAX_PLY;
        size_t n = moves.size();
        static thread_local std::pair<long long, Move> buf[ORDER_BUF_CAP];
        for (size_t i = 0; i < n; i++) {
            const Move& m = moves[i];
            long long sc = history[side][moveToPolicyIndex(m)];
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
    void orderWallMoves(MoveList& moves, int ply, int side, const State& s,
                         const CorridorHeat& oppHeat, const PlayerPathCache* oppCache = nullptr) {
        bool ply0 = ply >= 0 && ply < MAX_PLY;
        bool wallByBFS = ply <= WALL_BFS_ORDER_MAX_PLY;
        int opp = 1 - side;
        size_t n = moves.size();

        static thread_local std::pair<long long, Move> buf[ORDER_BUF_CAP];
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
                // dele (mesma shortestPathLen já usada em evalSimple)
                State ns = applyMove(s, m);
                int after = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[opp], opp);
                sc = (long long)(after - before) * 1000;
            }
            sc += wallEdgeHeat(oppHeat, m.a, m.b, m.c) * CAT_SCORE_SCALE;
            sc += history[side][moveToPolicyIndex(m)];
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
                   bool rootParity) {
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
        if (reptbl.isRepetitionDraw(s.hash)) return rootParity ? CONTEMPT : -CONTEMPT;

        // Prioridade 6 do plano-additional.md: evalSimpleW por si só já
        // funde 4 BFS em 2 (ver rules.hpp); capturando esses 2 caches
        // (sideCache/oppCache) aqui, o resto desta função (legalWallMoves
        // logo abaixo, e oppDistBefore/oppRobustBefore) reaproveita o
        // MESMO resultado em vez de recalcular -- 0 BFS adicionais no
        // caminho comum (extensão não disparada), e só 1 BFS por
        // candidato de muro testado (em vez de até 2) no caminho que
        // dispara a extensão.
        PlayerPathCache sideCache, oppCache;
        int standPat = evalSimpleW(s, s.turn, weights, &sideCache, &oppCache);
        if (standPat >= beta) return standPat;
        int localAlpha = alpha > standPat ? alpha : standPat;
        int best = standPat;

        if (qply >= QS_MAX_EXTRA_PLIES) return best;

        int side = s.turn, opp = 1 - side;
        if (s.wallsLeft[side] <= 0) return best;  // sem muro pra jogar, nada a estender

        // mesmo BFS já pago acima por evalSimpleW -- legalWallMoves recebe
        // os caches (mapeados por índice de jogador, não por side/opp) já
        // válidos e não roda BFS nenhuma internamente.
        MoveList wallMoves;
        uint64_t touchH0, touchV0, touchH1, touchV1;
        PlayerPathCache* cache0 = (side == 0) ? &sideCache : &oppCache;
        PlayerPathCache* cache1 = (side == 0) ? &oppCache : &sideCache;
        legalWallMoves(s, side, wallMoves, &touchH0, &touchV0, &touchH1, &touchV1, cache0, cache1);
        uint64_t touchHOpp = (side == 0) ? touchH1 : touchH0;
        uint64_t touchVOpp = (side == 0) ? touchV1 : touchV0;

        int oppDistBefore = cachedShortestPathLen(oppCache);
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
            computeDistFull(ns.wallsH, ns.wallsV, ns.pawn[opp], opp, oppCacheAfter);
            int oppDistAfter = cachedShortestPathLen(oppCacheAfter);
            bool critical = (oppDistAfter - oppDistBefore) >= QS_CRITICAL_BFS_DELTA;
            if (!critical && oppRobustBefore > QS_CRITICAL_ROBUSTNESS_DROP_TO) {
                int oppRobustAfter = cachedPathRobustness(oppCacheAfter, ns.wallsH, ns.wallsV);
                critical = (oppRobustAfter <= QS_CRITICAL_ROBUSTNESS_DROP_TO);
            }
            if (!critical) continue;

            reptbl.push(ns.hash);
            int score = -quiescence(ns, -beta, -localAlpha, qply + 1, stats, reptbl, !rootParity);
            reptbl.pop();
            if (stopped) return 0;
            if (score > best) best = score;
            if (score > localAlpha) localAlpha = score;
            if (localAlpha >= beta) break;
        }
        return best;
    }

    int negamax(const State& s, int depth, int alpha, int beta, SearchStats& stats, RepetitionTable& reptbl) {
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
        if (reptbl.isRepetitionDraw(s.hash)) return (ply % 2 == 0) ? CONTEMPT : -CONTEMPT;

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
                    raceScore = CONTEMPT;  // empate -- perseguição infinita/repetição (bug corrigido: era -CONTEMPT, recompensava o empate)
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
            if (!quiescenceEnabled) return evalSimpleW(s, s.turn, weights);
            return quiescence(s, alpha, beta, 0, stats, reptbl, ply % 2 == 0);
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
        orderPawnMoves(pawnMoves, ply, side);

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

        // aplica um lance candidato; devolve true se a busca deve parar
        // de avaliar mais lances neste nó (beta-cutoff OU tempo esgotado
        // -- `stopped` é checado pelo chamador logo em seguida).
        auto tryMove = [&](const Move& m) -> bool {
            State ns = applyMove(s, m);
            reptbl.push(ns.hash);
            int score = -negamax(ns, depth - 1, -beta, -alpha, stats, reptbl);
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
            cutoff = tryMove(ttMoveVal);
            if (stopped) return 0;
        }

        // Estágio 2: resto dos lances de peão (bloco já gerado/ordenado).
        if (!cutoff) {
            for (size_t i = 0; i < pawnMoves.size() && !cutoff; i++) {
                const Move& m = pawnMoves[i];
                if (ttTried && m == ttMoveVal) continue;  // já tentado no Estágio 1
                cutoff = tryMove(m);
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
            legalWallMoves(s, side, wallMoves, nullptr, nullptr, nullptr, nullptr, &cache0, &cache1);
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
            if (!wallMoves.empty()) {
                int opp = 1 - side;
                // computeCorridorHeat continua com sua própria BFS
                // dedicada (não reaproveita cache0/cache1) -- ver nota em
                // rules.hpp sobre por que essa fusão específica (com
                // early-exit parcial) fica de fora por risco de correção.
                CorridorHeat oppHeat = computeCorridorHeat(s.wallsH, s.wallsV, s.pawn[opp], opp);
                const PlayerPathCache& oppCache = (opp == 0) ? cache0 : cache1;
                orderWallMoves(wallMoves, ply, side, s, oppHeat, &oppCache);
            }
            for (size_t i = 0; i < wallMoves.size() && !cutoff; i++) {
                const Move& m = wallMoves[i];
                if (ttTried && m == ttMoveVal) continue;  // já tentado no Estágio 1
                cutoff = tryMove(m);
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
    int testFixedDepthFullWindow(const State& s, int depth, SearchStats& stats) {
        resetOrderingState();
        std::fill(tt.begin(), tt.end(), TTEntry{});
        stopped = false;
        rootDepth = depth;
        deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
        g_raceExactBudgetUs = 1e18;  // só chooseMove() aplica o orçamento (ver nota lá) -- senão uma chamada anterior de chooseMove() deixa o orçamento finito "vazando" pra esta busca de referência
        stats = SearchStats{};
        RepetitionTable emptyHistory;
        return negamax(s, depth, -SCORE_INF, SCORE_INF, stats, emptyHistory);
    }
    // igual, mas SEM limpar a TT -- pra reproduzir de propósito o reuso
    // entre iterações do chooseMove (não limpa killers/history também).
    int testNegamaxKeepTT(const State& s, int depth, int alpha, int beta, SearchStats& stats) {
        stopped = false;
        rootDepth = depth;
        deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
        g_raceExactBudgetUs = 1e18;  // só chooseMove() aplica o orçamento (ver nota lá)
        stats = SearchStats{};
        RepetitionTable emptyHistory;
        return negamax(s, depth, alpha, beta, stats, emptyHistory);
    }
    void testClearTT() { std::fill(tt.begin(), tt.end(), TTEntry{}); }
#endif
};

} // namespace qr
