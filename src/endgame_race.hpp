// endgame_race.hpp -- solver exato de final "mãos vazias" (ambos os
// jogadores sem muros), plano-additional.md Prioridade 4/4b/4c --
// comparação com o "Serviço A" (barato, 2 níveis) + "Serviço B" (exato,
// DP retrógrada) do titaniummachine1/titanium-engine.
//
// --- O que resolve ---
// Quando `wallsLeft[0]==0 && wallsLeft[1]==0`, a topologia de muros fica
// congelada pra sempre: o jogo vira uma corrida de peão pura (com pulos
// retos/diagonais, rules.hpp::pawnStepMoves). Isso é o único ponto do
// motor em que dá pra trocar busca heurística por CERTEZA MATEMÁTICA --
// resolver exato aqui é estritamente melhor que qualquer profundidade de
// busca alcançável, porque o resultado final não depende de heurística
// nenhuma, só das regras.
//
// --- Por que não é só "quem tem o caminho mais curto" ---
// Um jogador pode BLOQUEAR o outro por tempo indefinido (perseguição:
// ficar sempre na frente do caminho do oponente, forçando desvio atrás
// de desvio) -- isso não é só uma corrida de distância BFS, é um jogo de
// 2 jogadores com objetivo de alcançabilidade, que pode terminar em
// EMPATE por repetição perpétua (nenhum dos dois consegue forçar vitória)
// além de vitória de um dos dois. Por isso o núcleo exato (Serviço B,
// `raceExactDTM`) é uma análise retrógrada de jogo (mesmo algoritmo geral
// usado em bancos de final de xadrez/damas -- Allis, "Searching for
// Solutions in Games and Artificial Intelligence"), não um cálculo de
// distância simples.
//
// --- 2 camadas ativas (não 3 -- ver nota sobre o Nível 1 abaixo), em
// ordem de custo crescente (só ativa a próxima se a anterior não decidir) ---
// Nível 2 (raceDisjointGate): se as REGIÕES INTEIRAS alcançáveis por cada
//   jogador (ignorando o outro peão, só pela topologia de muro -- não
//   apenas as casas de ALGUM caminho mais curto, ver nota de correção
//   abaixo) são disjuntas, não há como um pulo OU bloqueio físico nunca
//   ocorrer entre eles, mesmo fora de rota ótima -- decide por tempo
//   puro, com certeza geométrica (não é uma margem numérica torcida, é
//   impossibilidade física de interação: regiões disjuntas = nenhuma
//   aresta atravessável as conecta = nunca ficam sequer adjacentes).
//   Custo: 2 BFS completas (mesmo padrão de computeCorridorHeat, cat.hpp).
// Serviço B (raceExactDTM): DP retrógrada exata sobre os 81×81×2=13.122
//   estados (pos0, pos1, turno) daquela topologia de muros fixa --
//   sempre correto, inclusive detecta empate por perseguição infinita.
//   Só roda quando o Nível 2 não decidiu (regiões se tocam).
//
// [NOTA -- desvio do plano-additional.md, achado durante os testes desta
// implementação] O documento original descrevia um "Nível 1" (portão de
// ETA por diferença bruta de distância, com margem fixa de segurança
// assumindo que um pulo economiza no máximo 1 lance -- item 4c) como
// decisão válida SEM checar sobreposição de caminho. `test_endgame_race.cpp`
// (testRandomWallTopologiesGatesAgreeWithExact) encontrou um contraexemplo
// real: com caminhos se cruzando, um BLOQUEIO FÍSICO (o peão à frente
// ocupando a única continuação, sem espaço pra pulo diagonal por causa da
// topologia de muro) pode custar MAIS que 1 lance extra -- no caso
// encontrado, 2 lances (dtm exato 21 contra a previsão ingênua de 19 com
// gap=3, a margem que o item 4c do plano sugeria como segura). Como o
// empate por perseguição infinita (raceExactDTM) já prova que esse custo
// não tem limite superior fixo em geral (bloqueio pode se estender
// indefinidamente), NENHUMA margem constante de gap é uma condição
// suficiente sem também garantir a ausência física de interação -- só a
// disjunção de caminhos (Nível 2) dá essa garantia. `raceETAGate` foi
// mantida como função utilitária isolada (testável, não usada por
// `resolveEmptyHandedEndgame`) para eventual refinamento futuro -- por
// exemplo, combinada com uma prova geométrica adicional de que não há
// espaço de manobra pro bloqueio, o que reintroduziria a otimização de
// custo praticamente zero que o Nível 1 prometia. Sem essa prova, usá-la
// sozinha na busca real seria uma fonte de erro de avaliação silencioso
// (o motor "acreditaria" ter certeza matemática quando na verdade não
// tem) -- pior que simplesmente não ter a otimização.
//
// [NOTA] Nível 3 do titanium-engine ("winner table" assimétrico,
// cacheado por topologia) também NÃO foi portado aqui -- o próprio
// plano-additional.md recomenda deixá-lo por último ("item mais custoso
// de engenharia dos 3"), só valendo a pena depois que o Nível 2 estiver
// rodando em produção e sobrar volume de posições "sobrepostas mas ainda
// decidíveis" pra justificar o cache extra. Sem o nível 3, essas
// posições caem direto no Serviço B exato (mais caro por chamada, nunca
// incorreto) -- correção não fica comprometida, só uma otimização de
// performance fica pra depois.
#pragma once
#include <cstdint>
#include <array>
#include <vector>
#include "rules.hpp"

namespace qr {

// Score-base reservado para vitórias/derrotas de final resolvido (Prioridade
// 4). Precisa ficar ABAIXO de SCORE_INF-1 (o score de vitória imediata,
// ver search.hpp) com folga suficiente para subtrair o dtm (distância em
// plies até a vitória forçada, tipicamente pequena -- o tabuleiro é 9x9,
// então dtm nunca chega perto de esgotar essa folga) e ACIMA de qualquer
// valor plausível de evalSimpleW (termos limitados a algumas centenas).
// search.hpp tem um static_assert conferindo a relação com SCORE_INF.
constexpr int RACE_SCORE_BASE = 999'000;

struct RaceOutcome {
    int winner;  // 0, 1, ou -1 = empate (perseguição infinita / repetição)
    int dtm;     // plies até a vitória forçada (só significativo se winner != -1)
};

// Nível 1 -- "portão de ETA" (plano-additional.md, item 4b) -- NÃO É
// USADA por `resolveEmptyHandedEndgame` (ver nota grande no topo do
// arquivo: um contraexemplo real de teste mostrou que essa margem NÃO é
// segura em geral -- bloqueio físico, não só pulo, pode custar mais que 1
// lance extra quando os caminhos se cruzam). Mantida isolada e testada
// (`test_endgame_race.cpp`) só como utilitário para eventual refinamento
// futuro combinado com uma prova geométrica adicional -- NÃO usar sozinha
// como fonte de verdade fora desse contexto. `rawDist0`/`rawDist1` são as
// distâncias BFS (shortestPathLen) de cada jogador até sua meta, ignorando
// o peão do outro. off(p) = 0 se p move agora, 1 caso contrário -- ply
// absoluto (0-based, a partir de AGORA) em que p completaria sua corrida
// em linha reta, SE ninguém o atrapalhasse: pl(p) = off(p) + 2*(rawDist(p)-1).
inline bool raceETAGate(int rawDist0, int rawDist1, int turn, int& winnerOut, int& dtmOut) {
    int off0 = (turn == 0) ? 0 : 1;
    int off1 = (turn == 1) ? 0 : 1;
    int pl0 = off0 + 2 * (rawDist0 - 1);
    int pl1 = off1 + 2 * (rawDist1 - 1);
    int gap = pl1 - pl0;  // > 0 favorece o jogador 0
    if (gap >= 3) { winnerOut = 0; dtmOut = pl0 + 1; return true; }
    if (gap <= -3) { winnerOut = 1; dtmOut = pl1 + 1; return true; }
    return false;
}

// Máscara de casas que pertencem a ALGUM caminho mais curto de `player`
// (delta==0 na mesma definição de cat.hpp::computeCorridorHeat -- reaproveita
// a mesma técnica de 2 BFS, aqui devolvendo o conjunto bruto de células em
// vez do calor). Usada só pelo Nível 2 abaixo.
inline std::array<bool, N * N> onShortestPathMask(uint64_t wallsH, uint64_t wallsV, int startCell, int player) {
    std::array<bool, N * N> mask{};
    int goalRow = GOAL_ROW[player];
    static thread_local int distStart[N * N];
    static thread_local int distGoal[N * N];
    static thread_local uint64_t genStart[N * N] = {};
    static thread_local uint64_t genGoal[N * N] = {};
    static thread_local uint64_t gen = 0;
    static thread_local int queue[N * N];
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    ++gen;

    {
        int head = 0, tail = 0;
        queue[tail++] = startCell;
        distStart[startCell] = 0;
        genStart[startCell] = gen;
        while (head < tail) {
            int cell = queue[head++];
            int r = rowOf(cell), c = colOf(cell);
            int d0 = distStart[cell];
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d], nc = c + dc[d];
                if (!inBounds(nr, nc)) continue;
                int ncell = cellIdx(nr, nc);
                if (genStart[ncell] == gen) continue;
                if (edgeBlocked(wallsH, wallsV, r, c, nr, nc)) continue;
                distStart[ncell] = d0 + 1;
                genStart[ncell] = gen;
                queue[tail++] = ncell;
            }
        }
    }
    {
        int head = 0, tail = 0;
        for (int c = 0; c < N; c++) {
            int cell = cellIdx(goalRow, c);
            distGoal[cell] = 0;
            genGoal[cell] = gen;
            queue[tail++] = cell;
        }
        while (head < tail) {
            int cell = queue[head++];
            int r = rowOf(cell), c = colOf(cell);
            int d0 = distGoal[cell];
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d], nc = c + dc[d];
                if (!inBounds(nr, nc)) continue;
                int ncell = cellIdx(nr, nc);
                if (genGoal[ncell] == gen) continue;
                if (edgeBlocked(wallsH, wallsV, r, c, nr, nc)) continue;
                distGoal[ncell] = d0 + 1;
                genGoal[ncell] = gen;
                queue[tail++] = ncell;
            }
        }
    }

    if (genStart[startCell] != gen || genGoal[startCell] != gen) return mask;  // defensivo
    int total = distGoal[startCell];
    for (int cell = 0; cell < N * N; cell++) {
        if (genStart[cell] != gen || genGoal[cell] != gen) continue;
        if (distStart[cell] + distGoal[cell] == total) mask[cell] = true;
    }
    return mask;
}

// Máscara de REGIÃO alcançável (BFS única, sem meta -- todas as casas que
// `startCell` consegue alcançar seguindo só a topologia de muro
// congelada, ignorando o outro peão e sem restrição de destino). Base
// sonora do Nível 2 abaixo -- ver nota de correção logo acima de
// `raceDisjointGate` sobre por que `onShortestPathMask` sozinha NÃO
// bastava.
inline std::array<bool, N * N> reachableRegionMask(uint64_t wallsH, uint64_t wallsV, int startCell) {
    std::array<bool, N * N> mask{};
    static thread_local int queue[N * N];
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    int head = 0, tail = 0;
    queue[tail++] = startCell;
    mask[startCell] = true;
    while (head < tail) {
        int cell = queue[head++];
        int r = rowOf(cell), c = colOf(cell);
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (!inBounds(nr, nc)) continue;
            int ncell = cellIdx(nr, nc);
            if (mask[ncell]) continue;
            if (edgeBlocked(wallsH, wallsV, r, c, nr, nc)) continue;
            mask[ncell] = true;
            queue[tail++] = ncell;
        }
    }
    return mask;
}

// [NOTA -- correção pós-implementação, achado por
// test_endgame_race.cpp::testInfinitePursuitDraw] A primeira versão desta
// função testava só a disjunção dos conjuntos de casas em ALGUM CAMINHO
// MAIS CURTO (`onShortestPathMask`) e tratava isso como "certeza
// geométrica" de que nenhuma interação era possível. Isso é FALSO em
// geral: disjunção dos caminhos MAIS CURTOS não implica disjunção das
// REGIÕES alcançáveis -- um jogador que está perdendo a corrida pura de
// tempo não é obrigado a seguir um caminho mais curto; ele pode desviar
// pra dentro do território do outro (fora do próprio conjunto de caminho
// mínimo) só pra bloquear fisicamente o oponente, mesmo sem nunca pisar
// numa casa que pertença ao caminho mínimo do oponente. Contraexemplo
// real encontrado: dois corredores com conjuntos de caminho mínimo
// disjuntos (colunas 2-3 vs. colunas 4-5) no mesmo tabuleiro TOTALMENTE
// CONECTADO (as 81 casas continuam mutuamente alcançáveis por qualquer
// um dos dois peões, ignorando o outro) -- o gate antigo decidia vitória
// do jogador 0 em 10 lances; o Serviço B exato (confirmado por 2
// reimplementações independentes de checagem cruzada, BFS retrógrada por
// predecessor e fixpoint iterativo "forward") mostra que o resultado
// verdadeiro é EMPATE por perseguição infinita nessa mesma posição
// (turno do jogador 1). A correção usa `reachableRegionMask` (toda a
// REGIÃO alcançável, não só o caminho mínimo): se as duas regiões são
// disjuntas enquanto CONJUNTOS DE CÉLULAS, então, por definição de
// componente conexo do grafo de casas menos as arestas bloqueadas por
// muro, não existe aresta atravessável entre elas -- logo nenhum dos
// dois jogadores jamais consegue ficar sequer ADJACENTE a uma casa do
// outro lado, o que é condição necessária e suficiente pra excluir
// qualquer bloqueio/pulo pra sempre (não só ao longo de rotas ótimas).
// Essa é uma prova estritamente mais forte que a anterior.
//
// Nível 2 -- disjunção de REGIÃO alcançável (plano-additional.md, item 4b,
// corrigido -- ver nota acima). Se as REGIÕES inteiras que os dois
// jogadores conseguem alcançar (ignorando o outro peão, só pela
// topologia de muro) forem disjuntas, nenhuma interação física (pulo OU
// bloqueio, dentro ou fora de rota ótima) pode ocorrer entre eles jamais
// -> a corrida vira comparação pura de tempo (pl0 vs pl1, sem a margem de
// segurança do Nível 1 -- aqui não há incerteza de pulo pra compensar,
// porque a garantia de não-interação já é absoluta). Empate exato de ply
// (pl0==pl1) não pode ocorrer -- paridade de turno sempre desempata (ver
// prova no corpo) -- mas o caso é tratado defensivamente caindo pro
// Serviço B em vez de assumir.
inline bool raceDisjointGate(uint64_t wallsH, uint64_t wallsV, int pawn0, int pawn1, int turn,
                              int rawDist0, int rawDist1, int& winnerOut, int& dtmOut) {
    auto m0 = reachableRegionMask(wallsH, wallsV, pawn0);
    auto m1 = reachableRegionMask(wallsH, wallsV, pawn1);
    for (int i = 0; i < N * N; i++) {
        if (m0[i] && m1[i]) return false;  // regiões alcançáveis se tocam -- não decide aqui
    }
    int off0 = (turn == 0) ? 0 : 1;
    int off1 = (turn == 1) ? 0 : 1;
    int pl0 = off0 + 2 * (rawDist0 - 1);
    int pl1 = off1 + 2 * (rawDist1 - 1);
    // Prova de que pl0 != pl1: pl0 tem a paridade de off0, pl1 a de off1,
    // e off0 != off1 (exatamente um dos dois jogadores move agora) ->
    // pl0 e pl1 têm paridades opostas -> nunca iguais.
    if (pl0 < pl1) { winnerOut = 0; dtmOut = pl0 + 1; return true; }
    if (pl1 < pl0) { winnerOut = 1; dtmOut = pl1 + 1; return true; }
    return false;  // defensivo -- não deveria ser alcançado (ver prova acima)
}

// Serviço B -- solver exato via análise retrógrada (plano-additional.md,
// item 4). Resolve TODOS os 81×81×2 estados (pos0, pos1, turno) daquela
// topologia de muros fixa (`wallsH`/`wallsV` congelados) e devolve o
// resultado exato para o estado de consulta (pawn0, pawn1, turn).
//
// Algoritmo: mesma análise retrógrada usada em bancos de final de
// xadrez/damas, adaptada para um jogo de alcançabilidade pura (sem
// capturas, sem material -- só "quem chega primeiro", incluindo a
// possibilidade de EMPATE por perseguição infinita). Para cada jogador X,
// calcula o "atrator" -- conjunto de estados de onde X força vitória
// independente do que o oponente jogar -- via BFS retrógrada a partir dos
// estados terminais (X já na fileira de meta):
//   - Se quem move no estado é X: existencial -- basta 1 sucessor no
//     atrator de X (X escolhe esse lance).
//   - Se quem move é o oponente: universal -- só entra quando TODOS os
//     sucessores já estiverem no atrator de X (oponente não tem escapatória).
// Estados que nunca entram no atrator de nenhum dos dois são EMPATE (nem
// um nem outro consegue forçar sua própria vitória -- cenário real de
// bloqueio mútuo perpétuo em Quoridor, não só teórico).
//
// Custo: ~13k estados, até ~5 sucessores cada (pawnStepMoves já cobre
// passos + pulos retos/diagonais) -- microssegundos, não persiste entre
// chamadas (plano-additional.md confirma que não vale a pena persistir
// entre partidas).
inline RaceOutcome raceExactDTM(uint64_t wallsH, uint64_t wallsV, int pawn0, int pawn1, int turn) {
    constexpr int NS = N * N;         // 81
    constexpr int NST = NS * NS * 2;  // 13122
    auto sidx = [](int p0, int p1, int t) { return (p0 * NS + p1) * 2 + t; };

    // Grafo do jogo (sucessores + predecessores), reconstruído a cada
    // chamada a partir da topologia de muros fixa recebida -- barato o
    // bastante (ver custo acima) pra não precisar de cache entre chamadas.
    static thread_local std::vector<std::vector<int>> succ, pred;
    if ((int)succ.size() != NST) { succ.assign(NST, {}); pred.assign(NST, {}); }
    for (int i = 0; i < NST; i++) { succ[i].clear(); pred[i].clear(); }

    State tmp;
    tmp.wallsH = wallsH;
    tmp.wallsV = wallsV;
    MoveList mv;
    for (int p0 = 0; p0 < NS; p0++) {
        for (int p1 = 0; p1 < NS; p1++) {
            if (p0 == p1) continue;  // dois peões nunca ocupam a mesma casa
            bool term0 = rowOf(p0) == GOAL_ROW[0];
            bool term1 = rowOf(p1) == GOAL_ROW[1];
            if (term0 || term1) continue;  // terminal -- sem sucessores necessários
            for (int t = 0; t < 2; t++) {
                int from = sidx(p0, p1, t);
                tmp.pawn[0] = p0;
                tmp.pawn[1] = p1;
                mv.n = 0;
                pawnStepMoves(tmp, t, mv);
                for (size_t i = 0; i < mv.size(); i++) {
                    int dest = mv[i].a;
                    int np0 = (t == 0) ? dest : p0;
                    int np1 = (t == 1) ? dest : p1;
                    int to = sidx(np0, np1, 1 - t);
                    succ[from].push_back(to);
                    pred[to].push_back(from);
                }
            }
        }
    }

    static thread_local std::vector<int8_t> win0, win1;
    static thread_local std::vector<int> dtm0, dtm1;

    auto solveFor = [&](int X, std::vector<int8_t>& win, std::vector<int>& dtm) {
        win.assign(NST, 0);
        dtm.assign(NST, -1);
        std::vector<int> cnt(NST);
        for (int i = 0; i < NST; i++) cnt[i] = (int)succ[i].size();
        std::vector<int> queue;
        queue.reserve(NST);
        for (int p0 = 0; p0 < NS; p0++) {
            for (int p1 = 0; p1 < NS; p1++) {
                if (p0 == p1) continue;
                int pX = (X == 0) ? p0 : p1;
                if (rowOf(pX) != GOAL_ROW[X]) continue;
                for (int t = 0; t < 2; t++) {
                    int s = sidx(p0, p1, t);
                    win[s] = 1;
                    dtm[s] = 0;
                    queue.push_back(s);
                }
            }
        }
        size_t head = 0;
        while (head < queue.size()) {
            int s = queue[head++];
            for (int p : pred[s]) {
                if (win[p]) continue;
                int t = p & 1;  // sidx(p0,p1,t) = (...)*2+t -> paridade == turno do estado p
                if (t == X) {
                    // existencial: 1 sucessor vencedor já basta
                    win[p] = 1;
                    dtm[p] = dtm[s] + 1;
                    queue.push_back(p);
                } else if (--cnt[p] == 0) {
                    // universal: só quando TODOS os sucessores entraram --
                    // BFS processa em ordem crescente de dtm, então o
                    // último sucessor a entrar é o de MAIOR dtm (o
                    // oponente escolhe adiar o máximo possível) --
                    // semântica correta de "pior caso pro lado que perde".
                    win[p] = 1;
                    dtm[p] = dtm[s] + 1;
                    queue.push_back(p);
                }
            }
        }
    };

    solveFor(0, win0, dtm0);
    solveFor(1, win1, dtm1);

    int s0 = sidx(pawn0, pawn1, turn);
    if (win0[s0]) return {0, dtm0[s0]};
    if (win1[s0]) return {1, dtm1[s0]};
    return {-1, 0};  // empate -- nenhum dos dois força a própria vitória (perseguição infinita)
}

// Entrada única (Nível 2 -> Serviço B, ver nota no topo do arquivo sobre
// por que o Nível 1 do plano-additional.md não entra aqui) -- só chamada
// quando wallsLeft[0]==0 && wallsLeft[1]==0 (ver gancho em search.hpp).
inline RaceOutcome resolveEmptyHandedEndgame(uint64_t wallsH, uint64_t wallsV, int pawn0, int pawn1, int turn) {
    int rawDist0 = shortestPathLen(wallsH, wallsV, pawn0, 0);
    int rawDist1 = shortestPathLen(wallsH, wallsV, pawn1, 1);
    int w, dtm;
    if (raceDisjointGate(wallsH, wallsV, pawn0, pawn1, turn, rawDist0, rawDist1, w, dtm)) return {w, dtm};
    return raceExactDTM(wallsH, wallsV, pawn0, pawn1, turn);
}

} // namespace qr
