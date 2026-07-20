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

namespace qr {

constexpr int TT_BITS = 21;               // 2M entradas
constexpr size_t TT_SIZE = 1ull << TT_BITS;
constexpr int SCORE_INF = 1'000'000;

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

        Move bestMove = legalMoves(root)[0];
        int prevScore = 0;
        RepetitionTable reptbl = gameHistory;
        for (int depth = 1; depth <= maxDepthCap && !stopped; depth++) {
            rootDepth = depth;
            int score;
            if (depth <= 2) {
                // aspiration window não compensa em profundidades tão rasas
                // (a janela estreita quase sempre falha e obriga rebusca)
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

    // bônus barato de ordenação de muro (Fase 4.2.5 do plano): muro que
    // toca uma aresta do caminho mais curto ATUAL do oponente, via
    // shortestPathTouchSlots (mesma função do pré-filtro de
    // legalWallMoves, rules.hpp) -- O(1) por candidato depois de UMA BFS
    // por nó (não por candidato, ao contrário do wallByBFS exato abaixo).
    // Escala abaixo do menor efeito prático de um wallByBFS exato
    // (delta>=1 já vale 1000) e acima da history típica em profundidades
    // rasas -- desempate honesto, não domina um delta exato quando os
    // dois estão presentes.
    static constexpr long long WALL_TOUCH_BONUS = 600;

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
    // do plano):
    //   1. wallByBFS (já existia): delta EXATO de shortestPathLen do
    //      oponente antes/depois do lance -- preciso, mas paga 2 BFS por
    //      candidato, então continua restrito a ply <= WALL_BFS_ORDER_MAX_PLY
    //      (poucos nós, perto da raiz -- custo agregado pequeno).
    //   2. WALL_TOUCH_BONUS (novo): sinal barato de "muro toca o caminho
    //      mínimo atual do oponente". touchHOpp/touchVOpp são recebidos
    //      JÁ CALCULADOS pelo chamador (legalWallMoves já roda essa
    //      mesma BFS internamente pro pré-filtro, Seção 4.2.1) -- achado
    //      do benchmark ad-hoc desta sessão: recalcular aqui dentro
    //      duplicava a BFS e custava ~11% de nós/s sem ganho
    //      compensador; reaproveitar o resultado deixa o sinal
    //      efetivamente grátis.
    void orderWallMoves(MoveList& moves, int ply, int side, const State& s,
                         uint64_t touchHOpp, uint64_t touchVOpp) {
        bool ply0 = ply >= 0 && ply < MAX_PLY;
        bool wallByBFS = ply <= WALL_BFS_ORDER_MAX_PLY;
        int opp = 1 - side;
        size_t n = moves.size();

        static thread_local std::pair<long long, Move> buf[ORDER_BUF_CAP];
        for (size_t i = 0; i < n; i++) {
            const Move& m = moves[i];
            long long sc = 0;
            if (wallByBFS) {
                // quanto o muro aumenta o caminho do oponente até a meta
                // dele (mesma shortestPathLen já usada em evalSimple)
                State ns = applyMove(s, m);
                int before = shortestPathLen(s.wallsH, s.wallsV, s.pawn[opp], opp);
                int after = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[opp], opp);
                sc = (long long)(after - before) * 1000;
            }
            int slot = slotIdx(m.b, m.c);
            bool touches = (m.a == 0) ? ((touchHOpp >> slot) & 1ull) : ((touchVOpp >> slot) & 1ull);
            if (touches) sc += WALL_TOUCH_BONUS;
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
    int quiescence(const State& s, int alpha, int beta, int qply, SearchStats& stats, RepetitionTable& reptbl) {
        stats.nodes++;
        if ((stats.nodes & 0x3FF) == 0 && std::chrono::steady_clock::now() >= deadline) {
            stopped = true;
            return 0;
        }
        int w = winner(s);
        if (w != -1) return (w == s.turn) ? SCORE_INF - 1 : -(SCORE_INF - 1);

        if (reptbl.count(s.hash) >= 2) return -CONTEMPT;

        int standPat = evalSimpleW(s, s.turn, weights);
        if (standPat >= beta) return standPat;
        int localAlpha = alpha > standPat ? alpha : standPat;
        int best = standPat;

        if (qply >= QS_MAX_EXTRA_PLIES) return best;

        int side = s.turn, opp = 1 - side;
        if (s.wallsLeft[side] <= 0) return best;  // sem muro pra jogar, nada a estender

        // mesmo BFS já pago pelo pré-filtro de legalWallMoves -- não é
        // custo adicional além do que o Estágio 3 do negamax já pagaria
        // se este nó não fosse folha.
        MoveList wallMoves;
        uint64_t touchH0, touchV0, touchH1, touchV1;
        legalWallMoves(s, side, wallMoves, &touchH0, &touchV0, &touchH1, &touchV1);
        uint64_t touchHOpp = (side == 0) ? touchH1 : touchH0;
        uint64_t touchVOpp = (side == 0) ? touchV1 : touchV0;

        int oppDistBefore = shortestPathLen(s.wallsH, s.wallsV, s.pawn[opp], opp);
        int oppRobustBefore = pathRobustness(s.wallsH, s.wallsV, s.pawn[opp], opp);

        for (size_t i = 0; i < wallMoves.size(); i++) {
            const Move& m = wallMoves[i];
            int slot = slotIdx(m.b, m.c);
            bool touches = (m.a == 0) ? ((touchHOpp >> slot) & 1ull) : ((touchVOpp >> slot) & 1ull);
            if (!touches) continue;  // não toca o caminho atual do oponente -> não pode ser "crítico" aqui

            State ns = applyMove(s, m);
            int oppDistAfter = shortestPathLen(ns.wallsH, ns.wallsV, ns.pawn[opp], opp);
            bool critical = (oppDistAfter - oppDistBefore) >= QS_CRITICAL_BFS_DELTA;
            if (!critical && oppRobustBefore > QS_CRITICAL_ROBUSTNESS_DROP_TO) {
                int oppRobustAfter = pathRobustness(ns.wallsH, ns.wallsV, ns.pawn[opp], opp);
                critical = (oppRobustAfter <= QS_CRITICAL_ROBUSTNESS_DROP_TO);
            }
            if (!critical) continue;

            reptbl.push(ns.hash);
            int score = -quiescence(ns, -beta, -localAlpha, qply + 1, stats, reptbl);
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

        if (reptbl.count(s.hash) >= 2) return -CONTEMPT;

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
            return quiescence(s, alpha, beta, 0, stats, reptbl);
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

        int ply = rootDepth - depth;
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
            uint64_t touchH0, touchV0, touchH1, touchV1;
            legalWallMoves(s, side, wallMoves, &touchH0, &touchV0, &touchH1, &touchV1);
            uint64_t touchHOpp = (side == 0) ? touchH1 : touchH0;
            uint64_t touchVOpp = (side == 0) ? touchV1 : touchV0;
            orderWallMoves(wallMoves, ply, side, s, touchHOpp, touchVOpp);
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
                             uint64_t touchHOpp, uint64_t touchVOpp) {
        orderWallMoves(moves, ply, side, s, touchHOpp, touchVOpp);
    }
    void testOrderPawnMoves(MoveList& moves, int ply, int side) {
        orderPawnMoves(moves, ply, side);
    }
    int testQuiescence(const State& s, int alpha, int beta, SearchStats& stats) {
        stopped = false;
        deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
        RepetitionTable emptyHistory;
        return quiescence(s, alpha, beta, 0, stats, emptyHistory);
    }
    int testFixedDepthFullWindow(const State& s, int depth, SearchStats& stats) {
        resetOrderingState();
        std::fill(tt.begin(), tt.end(), TTEntry{});
        stopped = false;
        rootDepth = depth;
        deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
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
        stats = SearchStats{};
        RepetitionTable emptyHistory;
        return negamax(s, depth, alpha, beta, stats, emptyHistory);
    }
    void testClearTT() { std::fill(tt.begin(), tt.end(), TTEntry{}); }
#endif
};

} // namespace qr
