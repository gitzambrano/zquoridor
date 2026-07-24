// rules.hpp -- núcleo de regras do Quoridor em C++ (port do rules.py),
// otimizado para nodes/sec alto: peões e muros em inteiros pequenos,
// muros em bitboards de 64 bits, BFS com arrays fixos (sem alocação).
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <array>
#include <vector>
#include <random>
#include <cassert>

namespace qr {

constexpr int N = 9;             // tabuleiro 9x9
constexpr int WS = 8;            // 8x8 slots de muro por orientação
constexpr int WALLS_PER_PLAYER = 10;
constexpr int GOAL_ROW[2] = {N - 1, 0};

inline int cellIdx(int r, int c) { return r * N + c; }
inline int rowOf(int cell) { return cell / N; }
inline int colOf(int cell) { return cell % N; }
inline bool inBounds(int r, int c) { return r >= 0 && r < N && c >= 0 && c < N; }

// slot de muro (r,c), r,c em 0..7 -> índice 0..63
inline int slotIdx(int r, int c) { return r * WS + c; }

} // namespace qr

// dsu.hpp usa N/WS/slotIdx acima (reabre namespace qr internamente).
#include "dsu.hpp"

namespace qr {

struct Move {
    bool isWall;
    uint8_t a, b, c;   // peão: a=cell destino (b,c não usados)
                       // muro: a=orientação(0=H,1=V), b=r, c=col
    static Move pawn(int destCell) { return {false, (uint8_t)destCell, 0, 0}; }
    static Move wall(int orientation, int r, int c) { return {true, (uint8_t)orientation, (uint8_t)r, (uint8_t)c}; }
    bool operator==(const Move& o) const { return isWall == o.isWall && a == o.a && b == o.b && c == o.c; }
};

// Lista de lances de capacidade fixa, sem heap allocation (Fase 4.2.2 do
// plano) -- substitui o std::vector<Move> antes retornado/preenchido por
// legalMoves/pawnStepMoves/legalWallMoves, que alocava a cada chamada
// (praticamente todo nó da árvore de busca, ver negamax em search.hpp).
// Mesmo padrão/motivação do buffer `static thread_local` já usado em
// orderMoves (search.hpp, Seção 4/5.6 do plano): capacidade 256 dá folga
// generosa sobre o máximo real de lances legais (3 peão + 128 muro =
// 131). Interface mínima compatível com o uso existente de
// std::vector<Move> nos call sites (size/empty/operator[]/begin/end/
// push_back) -- troca de tipo por `auto`/assinatura, sem mudar a lógica
// de quem consome.
struct MoveList {
    static constexpr size_t CAP = 256;
    std::array<Move, CAP> data;
    size_t n = 0;

    void push_back(const Move& m) {
        assert(n < CAP && "MoveList::push_back overflow -- aumente CAP");
        data[n++] = m;
    }
    size_t size() const { return n; }
    bool empty() const { return n == 0; }
    Move& operator[](size_t i) { return data[i]; }
    const Move& operator[](size_t i) const { return data[i]; }
    Move* begin() { return data.data(); }
    Move* end() { return data.data() + n; }
    const Move* begin() const { return data.data(); }
    const Move* end() const { return data.data() + n; }

    std::vector<Move> toVector() const { return std::vector<Move>(begin(), end()); }
};

// índice canônico 0..208 de um lance (81 destino de peão + 128 slot de
// muro = 209), usado tanto como policy_target do treino (selfplay.hpp)
// quanto como chave da history heuristic (search.hpp) -- mesma
// codificação nos dois lugares, uma única fonte de verdade. Mesma
// codificação de ação usada nas saídas da NNUE (Seção 7.2 do plano):
// 0..80 = célula destino do peão (absoluta), 81..144 = slot de muro
// horizontal, 145..208 = slot de muro vertical.
inline uint16_t moveToPolicyIndex(const Move& m) {
    if (!m.isWall) return (uint16_t)m.a;
    int slot = slotIdx(m.b, m.c);
    return (uint16_t)(N * N + (m.a == 0 ? slot : WS * WS + slot));
}
constexpr int NUM_MOVE_INDICES = N * N + 2 * WS * WS;  // 209

struct State {
    uint8_t pawn[2];
    uint64_t wallsH = 0, wallsV = 0;   // bit slotIdx(r,c)
    int8_t wallsLeft[2] = {WALLS_PER_PLAYER, WALLS_PER_PLAYER};
    int turn = 0;
    uint64_t hash = 0;
};

// --- Zobrist ----------------------------------------------------------
struct Zobrist {
    uint64_t pawnKey[2][N * N];
    uint64_t wallHKey[WS * WS];
    uint64_t wallVKey[WS * WS];
    uint64_t turnKey;

    Zobrist() {
        std::mt19937_64 rng(0xC0FFEEu);
        for (int p = 0; p < 2; p++)
            for (int i = 0; i < N * N; i++) pawnKey[p][i] = rng();
        for (int i = 0; i < WS * WS; i++) wallHKey[i] = rng();
        for (int i = 0; i < WS * WS; i++) wallVKey[i] = rng();
        turnKey = rng();
    }
};
inline Zobrist& zobrist() { static Zobrist z; return z; }

inline State initialState() {
    State s;
    s.pawn[0] = cellIdx(0, N / 2);
    s.pawn[1] = cellIdx(N - 1, N / 2);
    Zobrist& z = zobrist();
    s.hash = z.pawnKey[0][s.pawn[0]] ^ z.pawnKey[1][s.pawn[1]];
    return s;
}

// --- bloqueio de arestas -------------------------------------------------
inline bool edgeBlocked(uint64_t wallsH, uint64_t wallsV, int ra, int ca, int rb, int cb) {
    if (ra == rb) {
        int r = ra, c = ca < cb ? ca : cb;
        bool blocked = false;
        if (r - 1 >= 0) blocked |= (wallsV >> slotIdx(r - 1, c)) & 1ull;
        if (r < WS) blocked |= (wallsV >> slotIdx(r, c)) & 1ull;
        return blocked;
    } else {
        int r = ra < rb ? ra : rb, c = ca;
        bool blocked = false;
        if (c - 1 >= 0) blocked |= (wallsH >> slotIdx(r, c - 1)) & 1ull;
        if (c < WS) blocked |= (wallsH >> slotIdx(r, c)) & 1ull;
        return blocked;
    }
}

inline bool wallSlotAvailable(uint64_t wallsH, uint64_t wallsV, int orientation, int r, int c) {
    if (r < 0 || r >= WS || c < 0 || c >= WS) return false;
    if (orientation == 0) {
        if ((wallsH >> slotIdx(r, c)) & 1ull) return false;
        if (c - 1 >= 0 && ((wallsH >> slotIdx(r, c - 1)) & 1ull)) return false;
        if (c + 1 < WS && ((wallsH >> slotIdx(r, c + 1)) & 1ull)) return false;
        if ((wallsV >> slotIdx(r, c)) & 1ull) return false;
    } else {
        if ((wallsV >> slotIdx(r, c)) & 1ull) return false;
        if (r - 1 >= 0 && ((wallsV >> slotIdx(r - 1, c)) & 1ull)) return false;
        if (r + 1 < WS && ((wallsV >> slotIdx(r + 1, c)) & 1ull)) return false;
        if ((wallsH >> slotIdx(r, c)) & 1ull) return false;
    }
    return true;
}

// BFS sem alocação (arrays fixos de 81 células) -- existe caminho até a meta?
//
// Otimização de contador de geração (perfilado com gprof: shortestPathLen
// sozinha era 36,5% do tempo total de busca, 21M chamadas; pathRobustness
// mais 17,9%, 8,25M chamadas -- juntas mais da metade do tempo). As 4
// funções de BFS abaixo faziam `for(i=0;i<81;i++) reset` em TODA chamada,
// mesmo quando o BFS visita poucas células antes de achar a meta. Troca:
// em vez de "limpo tudo antes, visited[i]==true significa visitado", uso
// um contador `gen` incrementado a cada chamada e um array paralelo
// `visitGen[i]` -- célula i está "visitada nesta chamada" sse
// `visitGen[i] == gen` (comparação O(1), sem zerar nada entre chamadas).
// uint64_t de propósito: nenhuma contagem realista de chamadas (mesmo ao
// longo de todo um self-play) dá a volta, então não há risco de colisão
// de geração revalidando por engano uma célula de uma chamada antiga.
inline bool hasPathToGoal(uint64_t wallsH, uint64_t wallsV, int startCell, int player) {
    int goalRow = GOAL_ROW[player];
    if (rowOf(startCell) == goalRow) return true;
    static thread_local uint64_t visitGen[N * N] = {};
    static thread_local uint64_t gen = 0;
    static thread_local int queue[N * N];
    ++gen;
    int head = 0, tail = 0;
    queue[tail++] = startCell;
    visitGen[startCell] = gen;
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    while (head < tail) {
        int cell = queue[head++];
        int r = rowOf(cell), c = colOf(cell);
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (!inBounds(nr, nc)) continue;
            int ncell = cellIdx(nr, nc);
            if (visitGen[ncell] == gen) continue;
            if (edgeBlocked(wallsH, wallsV, r, c, nr, nc)) continue;
            if (nr == goalRow) return true;
            visitGen[ncell] = gen;
            queue[tail++] = ncell;
        }
    }
    return false;
}

// BFS com distância -- usado pela eval simples (Fase 3). Mesma técnica de
// geração acima: `distGen[i] == gen` substitui o antigo `dist[i] != -1`
// como marcador de "já visitado nesta chamada" -- `dist[i]` só é lido
// depois de checar a geração, então o valor de uma chamada anterior nunca
// vaza pra esta.
inline int shortestPathLen(uint64_t wallsH, uint64_t wallsV, int startCell, int player) {
    int goalRow = GOAL_ROW[player];
    if (rowOf(startCell) == goalRow) return 0;
    static thread_local int dist[N * N];
    static thread_local uint64_t distGen[N * N] = {};
    static thread_local uint64_t gen = 0;
    static thread_local int queue[N * N];
    ++gen;
    int head = 0, tail = 0;
    queue[tail++] = startCell;
    dist[startCell] = 0;
    distGen[startCell] = gen;
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    while (head < tail) {
        int cell = queue[head++];
        int r = rowOf(cell), c = colOf(cell);
        int d0 = dist[cell];
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (!inBounds(nr, nc)) continue;
            int ncell = cellIdx(nr, nc);
            if (distGen[ncell] == gen) continue;
            if (edgeBlocked(wallsH, wallsV, r, c, nr, nc)) continue;
            if (nr == goalRow) return d0 + 1;
            dist[ncell] = d0 + 1;
            distGen[ncell] = gen;
            queue[tail++] = ncell;
        }
    }
    return -1;  // não deveria acontecer (legalidade de muro já garante caminho)
}

// Pré-filtro barato antes do BFS caro (Fase 4.2.1 do plano). Calcula, a
// partir dos muros ATUAIS (antes de qualquer candidato novo), o conjunto
// de slots de muro que -- se colocados -- cortariam pelo menos uma aresta
// do caminho mais curto de `player` até sua meta. Prova de corretude: um
// muro candidato que não toca nenhum slot marcado aqui não pode ter
// desconectado esse jogador, porque o caminho testemunha (a mesma rota
// já usada por essa BFS) continua inteiramente livre -- é o único muro
// novo sendo avaliado, e ele não passa por nenhuma aresta dessa rota. Ou
// seja, para esse jogador, hasPathToGoal(candidato) é garantidamente
// verdadeiro sem precisar rodar o BFS de novo. Só quando o slot toca o
// caminho é que a legalidade fica genuinamente incerta e o BFS completo
// (hasPathToGoal) precisa rodar.
//
// Reconstrói o caminho via ponteiros de pai (mesma BFS de hasPathToGoal,
// mas mantendo `parent[]` em vez de parar no primeiro achado) e, para
// cada aresta do caminho, marca os slots de muro que a bloqueariam --
// mesma lógica inversa de edgeBlocked(): aresta horizontal (mesma linha)
// é bloqueada por muro vertical; aresta vertical (mesma coluna) é
// bloqueada por muro horizontal, cada aresta podendo ser bloqueada por
// até 2 slots adjacentes.
inline void shortestPathTouchSlots(uint64_t wallsH, uint64_t wallsV, int startCell, int player,
                                    uint64_t& touchH, uint64_t& touchV) {
    touchH = 0;
    touchV = 0;
    int goalRow = GOAL_ROW[player];
    if (rowOf(startCell) == goalRow) return;  // já na meta -- nenhum muro desconecta mais
    // Geração em vez de reset O(N) (mesma técnica de hasPathToGoal/
    // shortestPathLen acima). `parent[]` não precisa de reset: só é
    // escrito/lido para células marcadas nesta geração, e `parent[startCell]`
    // é setado explicitamente (O(1)) em vez de depender de um reset global
    // pra virar -1 -- a caminhada de volta em parent[] (mais abaixo, na
    // reconstrução do caminho) só visita células desta mesma geração.
    static thread_local int parent[N * N];
    static thread_local uint64_t visitGen[N * N] = {};
    static thread_local uint64_t gen = 0;
    static thread_local int queue[N * N];
    ++gen;
    int head = 0, tail = 0;
    queue[tail++] = startCell;
    visitGen[startCell] = gen;
    parent[startCell] = -1;
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    int goalCell = -1;
    while (head < tail && goalCell == -1) {
        int cell = queue[head++];
        int r = rowOf(cell), c = colOf(cell);
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (!inBounds(nr, nc)) continue;
            int ncell = cellIdx(nr, nc);
            if (visitGen[ncell] == gen) continue;
            if (edgeBlocked(wallsH, wallsV, r, c, nr, nc)) continue;
            visitGen[ncell] = gen;
            parent[ncell] = cell;
            if (nr == goalRow) { goalCell = ncell; break; }
            queue[tail++] = ncell;
        }
    }
    if (goalCell == -1) return;  // não deveria acontecer (estado já é legal)
    int cur = goalCell;
    while (parent[cur] != -1) {
        int prev = parent[cur];
        int ra = rowOf(prev), ca = colOf(prev);
        int rb = rowOf(cur), cb = colOf(cur);
        if (ra == rb) {
            // aresta horizontal -- bloqueada por muro vertical
            int r = ra, c = ca < cb ? ca : cb;
            if (r - 1 >= 0) touchV |= (1ull << slotIdx(r - 1, c));
            if (r < WS) touchV |= (1ull << slotIdx(r, c));
        } else {
            // aresta vertical -- bloqueada por muro horizontal
            int r = ra < rb ? ra : rb, c = ca;
            if (c - 1 >= 0) touchH |= (1ull << slotIdx(r, c - 1));
            if (c < WS) touchH |= (1ull << slotIdx(r, c));
        }
        cur = prev;
    }
}

// Robustez de caminho (Fase 4.2.10, item 1 do plano da Seção 4.2.10).
// `shortestPathLen`/`shortestPathTouchSlots` só enxergam o COMPRIMENTO do
// caminho mínimo; não distinguem um corredor de 1 célula (qualquer muro
// no lugar certo bloqueia) de um caminho igualmente curto com desvios
// quase tão bons ao lado. Métrica implementada aqui (decisão de projeto:
// termo de `evalSimple`, não feature de entrada da NNUE -- ver nota no
// readme, Seção 4.2.10, sobre por que essa troca fica pra depois): 1 BFS
// completo com `parent[]`/`dist[]` (mesma forma de `shortestPathTouchSlots`,
// função própria pra não arriscar mexer em código já validado por
// `testWallPrefilterRegression`), e depois, para cada célula do caminho
// mínimo reconstruído, conta quantos vizinhos ortogonais alcançáveis
// (aresta não bloqueada) fora do caminho têm `dist[]` <= dist da célula do
// caminho + 1 -- ou seja, oferecem um desvio de custo marginal (0 ou +1)
// que sai e volta perto do caminho principal. Retorna a contagem bruta
// (0 = corredor de 1 célula do início ao fim, sem NENHUM desvio barato;
// quanto maior, mais redundante/robusto). Não normalizado pelo
// comprimento do caminho de propósito -- caminhos mais longos "ganham"
// mais chances de robustez, o que é o comportamento desejado (a mesma
// vantagem de distância vale menos se for um corredor único e vale mais
// se for uma rede larga de rotas).
inline int pathRobustness(uint64_t wallsH, uint64_t wallsV, int startCell, int player) {
    int goalRow = GOAL_ROW[player];
    if (rowOf(startCell) == goalRow) return 0;  // já chegou -- robustez não importa mais
    // Geração em vez de reset O(N) (mesma técnica das 3 funções acima).
    // `dist[]`/`parent[]` só são lidos para células desta geração (checada
    // via `visitGen[]`); `parent[startCell]`/`dist[startCell]` setados
    // explicitamente em vez de via reset global.
    static thread_local int dist[N * N];
    static thread_local int parent[N * N];
    static thread_local uint64_t visitGen[N * N] = {};
    static thread_local uint64_t gen = 0;
    static thread_local int queue[N * N];
    ++gen;
    int head = 0, tail = 0;
    queue[tail++] = startCell;
    visitGen[startCell] = gen;
    parent[startCell] = -1;
    dist[startCell] = 0;
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    int goalCell = -1;
    while (head < tail && goalCell == -1) {
        int cell = queue[head++];
        int r = rowOf(cell), c = colOf(cell);
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (!inBounds(nr, nc)) continue;
            int ncell = cellIdx(nr, nc);
            if (visitGen[ncell] == gen) continue;
            if (edgeBlocked(wallsH, wallsV, r, c, nr, nc)) continue;
            visitGen[ncell] = gen;
            parent[ncell] = cell;
            dist[ncell] = dist[cell] + 1;
            if (nr == goalRow) { goalCell = ncell; break; }
            queue[tail++] = ncell;
        }
    }
    if (goalCell == -1) return 0;  // não deveria acontecer (estado já é legal)

    int robustness = 0;
    int cur = parent[goalCell];  // o passo final até a linha de meta não conta como "célula do caminho"
    while (cur != -1) {
        int r = rowOf(cur), c = colOf(cur);
        int prevOnPath = parent[cur];
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (!inBounds(nr, nc)) continue;
            int ncell = cellIdx(nr, nc);
            if (ncell == prevOnPath) continue;               // não conta o passo de volta no próprio caminho
            if (visitGen[ncell] != gen) continue;              // não alcançável dentro do BFS já feito
            if (edgeBlocked(wallsH, wallsV, r, c, nr, nc)) continue;
            if (dist[ncell] <= dist[cur] + 1) robustness++;    // desvio de custo marginal <=1
        }
        cur = prevOnPath;
    }
    return robustness;
}

// lances de peão (ortogonais + saltos retos/diagonais) -- grava direto
// como Move::pawn(destCell) em out (Fase 4.2.2: sem std::vector<int>
// intermediário, out é o MoveList final de legalMoves).
inline void pawnStepMoves(const State& s, int player, MoveList& out) {
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    int me = s.pawn[player], opp = s.pawn[1 - player];
    int mr = rowOf(me), mc = colOf(me);
    for (int d = 0; d < 4; d++) {
        int r1 = mr + dr[d], c1 = mc + dc[d];
        if (!inBounds(r1, c1) || edgeBlocked(s.wallsH, s.wallsV, mr, mc, r1, c1)) continue;
        int step1 = cellIdx(r1, c1);
        if (step1 != opp) { out.push_back(Move::pawn(step1)); continue; }
        int r2 = r1 + dr[d], c2 = c1 + dc[d];
        if (inBounds(r2, c2) && !edgeBlocked(s.wallsH, s.wallsV, r1, c1, r2, c2)) {
            out.push_back(Move::pawn(cellIdx(r2, c2)));
            continue;
        }
        // diagonais
        int pd0, pd1;
        if (d < 2) { pd0 = 2; pd1 = 3; } else { pd0 = 0; pd1 = 1; }
        for (int pd : {pd0, pd1}) {
            int rdi = r1 + dr[pd], cdi = c1 + dc[pd];
            if (inBounds(rdi, cdi) && !edgeBlocked(s.wallsH, s.wallsV, r1, c1, rdi, cdi))
                out.push_back(Move::pawn(cellIdx(rdi, cdi)));
        }
    }
}

// touchOutH0/touchOutV0/touchOutH1/touchOutV1 (opcionais, default nullptr):
// se não-nulos, recebem os mesmos bitmasks do pré-filtro calculados
// abaixo -- permite ao chamador (orderWallMoves, search.hpp) reaproveitar
// esse resultado pra ordenação sem recalcular a mesma BFS de novo
// (achado do benchmark ad-hoc desta sessão: recomputar
// shortestPathTouchSlots dentro de orderWallMoves duplicava trabalho já
// feito aqui, custando nós/s sem ganho compensador).
inline void legalWallMoves(const State& s, int player, MoveList& out,
                            uint64_t* touchOutH0 = nullptr, uint64_t* touchOutV0 = nullptr,
                            uint64_t* touchOutH1 = nullptr, uint64_t* touchOutV1 = nullptr) {
    if (s.wallsLeft[player] <= 0) {
        if (touchOutH0) *touchOutH0 = 0; if (touchOutV0) *touchOutV0 = 0;
        if (touchOutH1) *touchOutH1 = 0; if (touchOutV1) *touchOutV1 = 0;
        return;
    }

    // Pré-filtro (Fase 4.2.1 do plano): calculado uma única vez por
    // chamada (2 BFS, não 2×128). Um slot candidato que não toca o
    // caminho mais curto atual de nenhum dos dois jogadores é
    // garantidamente legal -- ver prova em shortestPathTouchSlots. Só
    // roda hasPathToGoal (BFS completo) para o(s) jogador(es) cujo
    // caminho o candidato realmente toca.
    uint64_t touchH0, touchV0, touchH1, touchV1;
    shortestPathTouchSlots(s.wallsH, s.wallsV, s.pawn[0], 0, touchH0, touchV0);
    shortestPathTouchSlots(s.wallsH, s.wallsV, s.pawn[1], 1, touchH1, touchV1);
    if (touchOutH0) *touchOutH0 = touchH0; if (touchOutV0) *touchOutV0 = touchV0;
    if (touchOutH1) *touchOutH1 = touchH1; if (touchOutV1) *touchOutV1 = touchV1;

    // DSU (Fase 4.2.1, item 2 do plano) sobre o grafo dual de muros --
    // ver prova de corretude e geometria em dsu.hpp. Construído uma
    // única vez por chamada a partir dos muros JÁ colocados em `s`
    // (barato: no máximo 20 muros no jogo todo). Usado só para os
    // candidatos que tocam o caminho testemunha de algum jogador (os
    // que sobram do pré-filtro acima); para esses, evita o BFS caro
    // sempre que o candidato provadamente não fecha nenhuma barreira
    // esquerda-direita -- se fecha, cai de volta no hasPathToGoal exato
    // (mesmo comportamento de antes, sem risco de regressão).
    RollbackDSU dsu;
    buildWallDSU(dsu, s.wallsH, s.wallsV);

    for (int orientation = 0; orientation < 2; orientation++) {
        uint64_t touch0 = orientation == 0 ? touchH0 : touchV0;
        uint64_t touch1 = orientation == 0 ? touchH1 : touchV1;
        for (int r = 0; r < WS; r++) {
            for (int c = 0; c < WS; c++) {
                if (!wallSlotAvailable(s.wallsH, s.wallsV, orientation, r, c)) continue;
                int slot = slotIdx(r, c);
                bool touches0 = (touch0 >> slot) & 1ull;
                bool touches1 = (touch1 >> slot) & 1ull;
                if (!touches0 && !touches1) {
                    // não toca o caminho testemunha de ninguém -> legal, sem BFS
                    out.push_back(Move::wall(orientation, r, c));
                    continue;
                }
                if (!wallCandidateAmbiguous(dsu, s.wallsH, s.wallsV, orientation, r, c)) {
                    // DSU prova que este muro não fecha nenhum ciclo
                    // (bolso/cercado), nem conecta esquerda-direita
                    // (barreira completa), nem fecha nenhum bolso de
                    // canto (superior/inferior × esquerda/direita) ->
                    // nenhum jogador pode ficar sem caminho até sua
                    // meta -> legal, sem BFS (prova em dsu.hpp).
                    out.push_back(Move::wall(orientation, r, c));
                    continue;
                }
                // Ambíguo (fecha ciclo e/ou conecta esquerda-direita):
                // pode ou não bloquear de fato -- cai no BFS exato como
                // antes.
                uint64_t nh = s.wallsH, nv = s.wallsV;
                if (orientation == 0) nh |= (1ull << slot);
                else nv |= (1ull << slot);
                if (touches0 && !hasPathToGoal(nh, nv, s.pawn[0], 0)) continue;
                if (touches1 && !hasPathToGoal(nh, nv, s.pawn[1], 1)) continue;
                out.push_back(Move::wall(orientation, r, c));
            }
        }
    }
}

// Legalidade de UM único candidato de muro, sem gerar a lista inteira de
// 128 slots (Fase 4.2.3 -- staged move generation, search.hpp). Usada só
// para testar o lance da TT antes de decidir se vale a pena gerar
// legalWallMoves inteiro: se o lance da TT já causar corte, o Estágio 3
// (muro) nem precisa rodar. Mesma checagem de legalidade de
// legalWallMoves/legalWallMovesReference (disponibilidade de slot +
// hasPathToGoal dos dois jogadores), só que para 1 candidato em vez de
// até 128 -- não usa o pré-filtro/DSU (que só compensam quando há muitos
// candidatos a testar de uma vez), 2 BFS bastam aqui.
inline bool isWallMoveLegal(const State& s, int player, int orientation, int r, int c) {
    if (s.wallsLeft[player] <= 0) return false;
    if (!wallSlotAvailable(s.wallsH, s.wallsV, orientation, r, c)) return false;
    uint64_t nh = s.wallsH, nv = s.wallsV;
    int slot = slotIdx(r, c);
    if (orientation == 0) nh |= (1ull << slot);
    else nv |= (1ull << slot);
    if (!hasPathToGoal(nh, nv, s.pawn[0], 0)) return false;
    if (!hasPathToGoal(nh, nv, s.pawn[1], 1)) return false;
    return true;
}

inline MoveList legalMoves(const State& s) {
    MoveList moves;
    pawnStepMoves(s, s.turn, moves);
    legalWallMoves(s, s.turn, moves);
    return moves;
}

inline int winner(const State& s) {
    if (rowOf(s.pawn[0]) == GOAL_ROW[0]) return 0;
    if (rowOf(s.pawn[1]) == GOAL_ROW[1]) return 1;
    return -1;
}

inline State applyMove(const State& s, const Move& m) {
    State ns = s;
    Zobrist& z = zobrist();
    int player = s.turn;
    if (!m.isWall) {
        int destCell = m.a;
        ns.hash ^= z.pawnKey[player][s.pawn[player]];
        ns.pawn[player] = destCell;
        ns.hash ^= z.pawnKey[player][destCell];
    } else {
        int orientation = m.a, r = m.b, c = m.c;
        int idx = slotIdx(r, c);
        if (orientation == 0) { ns.wallsH |= (1ull << idx); ns.hash ^= z.wallHKey[idx]; }
        else { ns.wallsV |= (1ull << idx); ns.hash ^= z.wallVKey[idx]; }
        ns.wallsLeft[player] -= 1;
    }
    ns.turn = 1 - player;
    ns.hash ^= z.turnKey;
    return ns;
}

// pesos da evalSimple, isolados num struct pra permitir tuning automático
// (ver tune_spsa.cpp) sem mexer na lógica da eval em si. Ponto de partida
// original era "chutado" (10/1/6/2/1); os valores abaixo já são o
// resultado do tuning SPSA (ver tune_log.csv e validate_weights.cpp), que
// venceu o baseline chutado em teste de validação com busca real.
// evalWeights() é a instância de PRODUÇÃO, usada por evalSimple(s,player);
// o tuner troca essa instância entre partidas (single-threaded, ver
// tune_spsa.cpp) pra comparar conjuntos de pesos via jogo real.
struct EvalWeights {
    double distWeight      = 11.30; // peso da diferença de distância BFS (termo dominante) [tunado via SPSA, ver tune_log.csv]
    double urgencyScale    = 0.64;  // multiplicador da URGENCY_TABLE (forma da curva fica fixa) [tunado]
    double wallWeightClose = 7.76;  // peso de muro restante quando a corrida está apertada (|gap|<=3) [tunado]
    double wallWeightFar   = 3.26;  // peso de muro restante quando a corrida já está decidida [tunado]
    double mobWeight       = 1.62;  // peso da diferença de mobilidade do peão [tunado]
    // NÃO tunado -- adicionado nesta rodada (Fase 4.2.10, item 1) depois
    // do SPSA que calibrou os 5 pesos acima. Valor abaixo é um chute
    // inicial pequeno (mesma categoria dos "10/1/1" originais antes do
    // tuning), só pra manter o termo funcional/testável por regressão de
    // corretude; precisa de uma rodada de SPSA própria (6 parâmetros)
    // antes de qualquer teste de força valer como evidência.
    double robustnessWeight = 0.80; // peso da diferença de pathRobustness() [NÃO tunado -- placeholder]
};
inline EvalWeights& evalWeights() { static EvalWeights w; return w; }

// variante de pawnStepMoves que só conta os lances de peão em vez de
// gravá-los -- evalSimple só precisa do tamanho, não dos destinos, e
// pawnStepMoves grava num std::vector (alocação de heap) a cada chamada.
// Chamada 2x por folha da busca (uma por jogador), então essa alocação
// evitada é o mesmo tipo de regressão de nós/s já identificado e corrigido
// em orderMoves (Seção 5.6 do plano) -- mesma lógica de geração de lance
// de pawnStepMoves acima, sem o vector.
inline int pawnMobilityCount(const State& s, int player) {
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    int me = s.pawn[player], opp = s.pawn[1 - player];
    int mr = rowOf(me), mc = colOf(me);
    int count = 0;
    for (int d = 0; d < 4; d++) {
        int r1 = mr + dr[d], c1 = mc + dc[d];
        if (!inBounds(r1, c1) || edgeBlocked(s.wallsH, s.wallsV, mr, mc, r1, c1)) continue;
        int step1 = cellIdx(r1, c1);
        if (step1 != opp) { count++; continue; }
        int r2 = r1 + dr[d], c2 = c1 + dc[d];
        if (inBounds(r2, c2) && !edgeBlocked(s.wallsH, s.wallsV, r1, c1, r2, c2)) {
            count++;
            continue;
        }
        int pd0, pd1;
        if (d < 2) { pd0 = 2; pd1 = 3; } else { pd0 = 0; pd1 = 1; }
        for (int pd : {pd0, pd1}) {
            int rdi = r1 + dr[pd], cdi = c1 + dc[pd];
            if (inBounds(rdi, cdi) && !edgeBlocked(s.wallsH, s.wallsV, r1, c1, rdi, cdi))
                count++;
        }
    }
    return count;
}

// tabela de "urgência" por distância restante até a meta: peso extra
// não-linear por estar perto de vencer -- mesma ideia de bônus de peão
// passado crescente por fileira em PSQTs de xadrez (zchezz/Stockfish):
// perto da meta cada tempo vale muito mais do que no meio do tabuleiro,
// porque sobra pouco espaço pro oponente reagir com muro. Indexada pela
// distância BFS (clampada); decrescente e convexa. A FORMA da curva fica
// fixa -- só a magnitude (EvalWeights::urgencyScale) é tunada, pra manter
// a dimensionalidade do SPSA baixa (5 parâmetros, não 17).
constexpr int URGENCY_MAX_DIST = 16;  // acima disso a urgência já é ~0
constexpr int URGENCY_TABLE[URGENCY_MAX_DIST + 1] = {
    80, 65, 52, 41, 32, 25, 19, 14, 10, 7, 5, 3, 2, 1, 1, 0, 0
};
inline int urgency(int dist) {
    if (dist < 0) dist = 0;
    if (dist > URGENCY_MAX_DIST) dist = URGENCY_MAX_DIST;
    return URGENCY_TABLE[dist];
}

// eval heurística (Fase 3 do plano, reforçada): distância BFS (termo
// dominante, linear) + urgência não-linear por proximidade da meta +
// muros restantes (peso maior quando a corrida está apertada, regime em
// que um muro a mais costuma decidir) + mobilidade do peão, sem alocação
// (pawnMobilityCount acima). Serve dois papéis: eval de folha da busca
// negamax E alvo auxiliar (searchScore) do bootstrap supervisionado da
// NNUE (selfplay.hpp) -- quanto mais forte e mais bem escalada aqui, mais
// rápido a rede aprende antes de assumir a busca (Fase 6).
//
// evalSimpleW recebe os pesos explicitamente -- usada pelo tuner
// (tune_spsa.cpp) pra comparar dois conjuntos de pesos no mesmo processo
// sem depender do estado global. evalSimple(s,player) é o atalho de
// produção, sempre com os pesos de evalWeights() (arredondado só no final,
// pra não pagar custo de double no meio do cálculo mais do que uma vez).
inline int evalSimpleW(const State& s, int player, const EvalWeights& w) {
    int me = player, opp = 1 - player;
    int myDist = shortestPathLen(s.wallsH, s.wallsV, s.pawn[me], me);
    int oppDist = shortestPathLen(s.wallsH, s.wallsV, s.pawn[opp], opp);
    int myMob = pawnMobilityCount(s, me);
    int oppMob = pawnMobilityCount(s, opp);

    double score = w.distWeight * (oppDist - myDist);
    score += w.urgencyScale * (urgency(myDist) - urgency(oppDist));

    // Robustez de caminho (Fase 4.2.10, item 1): mesma distância BFS vale
    // mais se o caminho for uma rede larga de rotas (difícil de fechar em
    // 1-2 muros) do que se for um corredor único. Termo simétrico, mesmo
    // padrão dos outros (diferença própria-menos-oponente).
    int myRobust = pathRobustness(s.wallsH, s.wallsV, s.pawn[me], me);
    int oppRobust = pathRobustness(s.wallsH, s.wallsV, s.pawn[opp], opp);
    score += w.robustnessWeight * (myRobust - oppRobust);

    int distGap = std::abs(myDist - oppDist);
    double wallWeight = (distGap <= 3) ? w.wallWeightClose : w.wallWeightFar;
    score += wallWeight * (s.wallsLeft[me] - s.wallsLeft[opp]);

    score += w.mobWeight * (myMob - oppMob);
    return (int)std::lround(score);
}

inline int evalSimple(const State& s, int player) {
    return evalSimpleW(s, player, evalWeights());
}

struct RepetitionTable {
    static constexpr int MAX_HIST = 512;
    uint64_t hist[MAX_HIST];
    int size = 0;

    void push(uint64_t hash) { if (size < MAX_HIST) hist[size++] = hash; }
    void pop()               { if (size > 0) --size; }
    int count(uint64_t hash) const {
        int c = 0;
        for (int i = 0; i < size; i++) {
            if (hist[i] == hash) c++;
        }
        return c;
    }
    void clear() { size = 0; }
};

} // namespace qr

namespace qr {

// perft (Prioridade 5, plano-additional.md): contagem exata de nós-folha
// da árvore de geração de lances até `depth`, a partir de `s` -- não é
// benchmark de velocidade, é prova de que legalMoves/applyMove/pawnStepMoves/
// legalWallMoves (pulo, muro, borda) não regrediram, mesmo que a
// implementação interna mude por completo. Convenção igual à do
// titanium-engine (perft "puro": nunca para cedo em `winner(s)!=-1`, conta
// só profundidade de geração de lances -- não surge diferença prática pras
// profundidades usadas aqui, já que uma vitória não é alcançável tão cedo
// a partir da posição inicial).
inline uint64_t perft(const State& s, int depth) {
    if (depth == 0) return 1;
    MoveList moves = legalMoves(s);
    if (depth == 1) return moves.size();
    uint64_t total = 0;
    for (size_t i = 0; i < moves.size(); i++) {
        total += perft(applyMove(s, moves[i]), depth - 1);
    }
    return total;
}

} // namespace qr
