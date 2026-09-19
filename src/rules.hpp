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

constexpr int cellIdx(int r, int c) { return r * N + c; }
constexpr int rowOf(int cell) { return cell / N; }
constexpr int colOf(int cell) { return cell % N; }
constexpr bool inBounds(int r, int c) { return r >= 0 && r < N && c >= 0 && c < N; }

// slot de muro (r,c), r,c em 0..7 -> índice 0..63
constexpr int slotIdx(int r, int c) { return r * WS + c; }

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
// orderMoves (search.hpp, Seção 4/5.6 do plano). O limite pelas regras é
// 133: no máximo 128 ações de muro e no máximo 5 ações de peão. Um peão
// tem 4 direções ortogonais; se uma está ocupada pelo adversário, essa ação
// vira no máximo 2 diagonais, logo 3+2=5. Mantemos 144 por folga/alinhamento,
// em vez de 256. MCAB guarda MoveList inline por nó, então reduzir o slack
// melhora residência em cache sem alterar a enumeração de lances.
// Interface mínima compatível com o uso existente de
// std::vector<Move> nos call sites (size/empty/operator[]/begin/end/
// push_back) -- troca de tipo por `auto`/assinatura, sem mudar a lógica
// de quem consome.
struct MoveList {
    static constexpr size_t CAP = 144;
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
// Precomputed board-neighbour and wall-slot masks for the four orthogonal
// directions used by every BFS/pawn-move expansion. Direction order matches the
// historical dr/dc arrays: 0=N, 1=S, 2=W, 3=E.
//
// This moves fixed 9x9 geometry out of the hot loop. A blocked-edge test becomes
// two mask ANDs instead of row/column arithmetic plus conditional slot probes.
constexpr std::array<std::array<int8_t, 4>, N * N> makeNeighborTable() {
    std::array<std::array<int8_t, 4>, N * N> a{};
    for (int cell = 0; cell < N * N; ++cell) {
        int r = cell / N, col = cell % N;
        a[(size_t)cell][0] = (int8_t)(r > 0 ? cell - N : -1);