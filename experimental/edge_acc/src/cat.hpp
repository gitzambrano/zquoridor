// cat.hpp -- Corridor Attention Table (CAT): calor de corredor por casa,
// usado para ordenar candidatos de muro na busca (plano-additional.md,
// Prioridade 1 -- comparação com titaniummachine1/titanium-engine).
//
// --- O que resolve que o sinal antigo (WALL_TOUCH_BONUS) não resolvia ---
// orderWallMoves (search.hpp) já tinha um bônus binário: "este muro toca
// alguma aresta do ÚNICO caminho mais curto testemunha (shortestPathTouchSlots)
// do oponente?". Isso enxerga só UMA rota witness -- um muro que fecha uma
// rota alternativa de custo igual, mas que não é a rota que o BFS reportou
// como testemunha, tem bônus 0, mesmo sendo tão relevante quanto um muro
// que toca a rota testemunha. CAT generaliza isso: em vez de "toca a rota
// testemunha ou não" (booleano), calcula um "calor" por CASA que mede o
// quanto essa casa se desvia do caminho ótimo (delta = distância a partir
// do peão + distância até a meta - distância total mínima). delta==0
// siginifca "está em algum caminho ótimo" (não só o witness específico);
// delta pequeno (1..3) significa "desvio de custo baixo" -- ainda
// relevante, o witness binário não capturava isso.
//
// --- Custo ---
// 2 BFS completos por chamada (um a partir do peão, um multi-fonte a
// partir de toda a fileira de meta) -- mesma ordem de custo de UMA
// chamada a shortestPathLen + shortestPathTouchSlots juntas. Chamado UMA
// VEZ POR NÓ (não por candidato de muro, ao contrário do wallByBFS exato
// já existente em orderWallMoves, que roda até 128 BFS por nó e por isso
// fica restrito a WALL_BFS_ORDER_MAX_PLY). CAT pode rodar em TODOS os
// plies porque o custo não escala com o número de candidatos.
//
// --- Rollout (plano-additional.md) ---
// Implementado aqui só como ORDENAÇÃO (substituindo WALL_TOUCH_BONUS).
// Poda de muros "frios" (heat==0) fica para depois, como mudança isolada
// e testável separadamente -- não faz parte desta etapa.
#pragma once
#include <cstdint>
#include <array>
#include <cmath>
#include "rules.hpp"

namespace qr {

// Constantes de partida (mesmos valores usados como referência pelo
// titanium-engine, ver plano-additional.md Prioridade 1) -- ainda não
// tunadas via SPSA; candidatas a entrar em EvalWeights/uma tabela própria
// numa rodada de tuning futura, mesmo tratamento que QS_CRITICAL_BFS_DELTA
// recebeu antes de ser tunado.
constexpr int CAT_CORRIDOR_CM = 200;      // calor máximo (delta==0)
constexpr int CAT_MAX_DELTA = 3;          // delta > isto -> calor 0

// Tabela pré-computada de delta -> calor base (achado de perfilamento:
// std::log2+std::lround custam ~14ns/chamada contra ~0,6ns de um acesso
// de array -- e isso rodava por CADA casa alcançável com delta<=CAT_MAX_DELTA
// TODA chamada de computeCorridorHeat, ou seja, até 81 vezes por nó. Como
// delta só assume os valores 0..CAT_MAX_DELTA, a fórmula
// `CAT_CORRIDOR_CM / (1 + delta*log2(delta+2))` tem resultado fixo pra
// cada delta -- computável uma vez em tempo de compilação, não por casa
// por chamada. Isto sozinho respondia por quase todo o custo medido de
// computeCorridorHeat (~1,1us/chamada).
constexpr int catHeatByDeltaCompute(int delta) {
    // constexpr não aceita std::log2 (não é constexpr até C++26 em alguns
    // compiladores) -- log2(x) = ln(x)/ln(2); usamos double e um cálculo
    // manual só pra gerar a tabela em compilação, não em runtime.
    // Como delta é só 0..3 (poucos valores), calculamos via série/Newton
    // seria overkill -- em vez disso, os 4 valores foram conferidos batendo
    // com computeCorridorHeat original (log2/lround em runtime) antes de
    // fixar aqui: delta=0->200, delta=1->77, delta=2->40, delta=3->25.
    return delta == 0 ? CAT_CORRIDOR_CM
         : delta == 1 ? 77
         : delta == 2 ? 40
         : delta == 3 ? 25
         : 0;
}
constexpr int CAT_HEAT_BY_DELTA[CAT_MAX_DELTA + 1] = {
    catHeatByDeltaCompute(0), catHeatByDeltaCompute(1),
    catHeatByDeltaCompute(2), catHeatByDeltaCompute(3)
};
constexpr int CAT_BOTTLENECK_BONUS_CM = 40; // bônus se a casa (delta<=2) só tem 1 continuação

struct CorridorHeat {
    std::array<int, N * N> heat{};  // heat[cell] -- default 0 (celula fora do alcance ou delta>CAT_MAX_DELTA)
};

// Calcula o calor de corredor de UM jogador na topologia de muro atual.
// `startCell` normalmente é s.pawn[player] -- separado como parâmetro
// (em vez de receber `State` inteiro) para deixar claro que o resultado
// só depende de (wallsH, wallsV, startCell, player), nunca do peão do
// oponente nem de wallsLeft/turn -- mesma independência de chave que o
// plano-additional.md (Prioridade 6b) descreve como pré-requisito para um
// cache entre nós, ainda que esse cache em si não seja implementado aqui.
inline CorridorHeat computeCorridorHeat(uint64_t wallsH, uint64_t wallsV, int startCell, int player) {
    CorridorHeat out;
    int goalRow = GOAL_ROW[player];

    // Duas BFS independentes, mesma técnica de geração (contador `gen`)
    // já usada em hasPathToGoal/shortestPathLen/shortestPathTouchSlots
    // (rules.hpp) -- evita zerar os 81 elementos a cada chamada.
    static thread_local int distStart[N * N];
    static thread_local int distGoal[N * N];
    static thread_local uint64_t genStart[N * N] = {};
    static thread_local uint64_t genGoal[N * N] = {};
    static thread_local uint64_t gen = 0;
    static thread_local int queue[N * N];
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    ++gen;

    // BFS 1: a partir do peão -- distStart[cell] = passos mínimos até `cell`.
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

    // BFS 2: multi-fonte a partir de TODA a fileira de meta -- grafo não
    // direcionado (edgeBlocked é simétrico), então isto dá exatamente
    // "passos mínimos até a fileira de meta" para qualquer casa, com uma
    // única busca (em vez de uma BFS por casa candidata).
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

    // Não deveria acontecer (legalidade de muro já garante caminho para os
    // dois jogadores) -- defensivo: devolve calor 0 em toda parte.
    if (genStart[startCell] != gen || genGoal[startCell] != gen) return out;
    int total = distGoal[startCell];  // == distStart[<melhor célula da fileira de meta>]

    for (int cell = 0; cell < N * N; cell++) {
        if (genStart[cell] != gen || genGoal[cell] != gen) continue;  // não alcançável nesta topologia
        int delta = distStart[cell] + distGoal[cell] - total;
        if (delta < 0) delta = 0;  // não deveria ser negativo; defensivo contra arredondamento
        if (delta > CAT_MAX_DELTA) continue;  // calor 0 (default de CorridorHeat)

        int h = CAT_HEAT_BY_DELTA[delta];

        // Bônus de gargalo: entre as casas "quase ótimas" (delta<=2), uma
        // que só tem 1 continuação possível em direção à meta (sem desvio
        // lateral) é um gargalo genuíno -- qualquer muro ali tende a valer
        // mais do que o calor de delta sozinho sugere. "Continuação" aqui
        // usa o mesmo critério de vizinhança de pathRobustness (rules.hpp):
        // vizinho alcançável com distStart[vizinho] == distStart[cell]+1.
        if (delta <= 2) {
            int r = rowOf(cell), c = colOf(cell);
            int continuations = 0;
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d], nc = c + dc[d];
                if (!inBounds(nr, nc)) continue;
                int ncell = cellIdx(nr, nc);
                if (genStart[ncell] != gen) continue;
                if (edgeBlocked(wallsH, wallsV, r, c, nr, nc)) continue;
                if (distStart[ncell] == distStart[cell] + 1) continuations++;
            }
            if (continuations <= 1) h += CAT_BOTTLENECK_BONUS_CM;
        }
        out.heat[cell] = h;
    }
    return out;
}

// Calor de uma ARESTA de muro candidato, a partir da CorridorHeat de UM
// jogador (tipicamente o oponente do lado que está gerando lances --
// orderWallMoves usa a CorridorHeat do oponente, mesma perspectiva de
// touchHOpp/touchVOpp já usados hoje). Geometria: tanto um muro H(r,c)
// quanto um muro V(r,c) tocam exatamente o mesmo bloco de 4 casas --
// (r,c), (r,c+1), (r+1,c), (r+1,c+1) -- só mudando QUAL par de arestas
// entre essas casas fica bloqueado (ver dsu.hpp para a derivação da
// geometria completa; edgeBlocked, rules.hpp, confirma que H(r,c) e
// V(r,c) leem os mesmos 4 vizinhos ao testar essas arestas). Por isso
// `orientation` não entra na conta abaixo -- o conjunto de casas tocadas
// é o mesmo nos dois casos.
//
// Score = maior calor entre as 4 casas + 1/4 do segundo maior -- captura
// tanto "há uma casa muito quente aqui" (hi) quanto "as casas ao redor
// também importam um pouco, não só a mais quente" (lo/4), sem deixar um
// segundo pico quase tão importante virar ruído puro.
inline int wallEdgeHeat(const CorridorHeat& heat, int /*orientation*/, int r, int c) {
    int cells[4] = {
        cellIdx(r, c), cellIdx(r, c + 1),
        cellIdx(r + 1, c), cellIdx(r + 1, c + 1)
    };
    int hi = 0, lo = 0;
    for (int cell : cells) {
        int h = heat.heat[cell];
        if (h > hi) { lo = hi; hi = h; }
        else if (h > lo) { lo = h; }
    }
    return hi + lo / 4;
}

} // namespace qr
