// Differential test: new RepetitionTable::isRepetitionDraw (bounded,
// parity-stepped) vs the frozen pre-optimization full-scan semantics.
// Simulates real games (walls monotonic), random push/pop interleavings,
// and compares verdicts on every queried hash.
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include "rules.hpp"

using namespace qr;

struct OldReptbl {
    uint64_t hist[512];
    int size = 0;
    int rootSize = 0;
    void push(uint64_t h) { if (size < 512) hist[size++] = h; }
    void pop() { if (size > 0) --size; }
    void markRoot() { rootSize = size; }
    bool isRepetitionDraw(uint64_t hash) const {
        int total = 0;
        bool preRoot = false;
        for (int i = 0; i < size; i++) {
            if (hist[i] == hash) { total++; if (i < rootSize) preRoot = true; }
        }
        return preRoot ? (total >= 3) : (total >= 2);
    }
};

int main() {
    std::mt19937 rng(987654321u);
    long long checks = 0;
    for (int trial = 0; trial < 300; trial++) {
        State s = initialState();
        RepetitionTable nt;
        OldReptbl ot;
        std::vector<uint64_t> queries;
        // log de operações pra depurar violação de invariante
        std::vector<std::string> ops;
        auto checkInvariant = [&](const char* where) {
            // qualquer par de índices com hashes iguais deve estar >= lastIrrev
            for (int i = nt.lastIrrev - 1; i >= 0; i--) {
                for (int j = i + 1; j < nt.size; j++) {
                    if (nt.hist[i] == nt.hist[j]) {
                        printf("INVARIANTE VIOLADA (%s): trial %d, i=%d j=%d lastIrrev=%d size=%d\n",
                               where, trial, i, j, nt.lastIrrev, nt.size);
                        printf("  hist[%d]=%llu\n  hist[%d]=%llu\n", i, (unsigned long long)nt.hist[i],
                               j, (unsigned long long)nt.hist[j]);
                        int lo = (int)ops.size() - 70 < 0 ? 0 : (int)ops.size() - 70;
                        for (int k = lo; k < (int)ops.size(); k++)
                            printf("  op[%d]: %s\n", k, ops[k].c_str());
                        return true;
                    }
                }
            }
            return false;
        };
        int plies = 40 + (int)(rng() % 200);
        for (int p = 0; p < plies; p++) {
            // query current position before moving
            queries.push_back(s.hash);
            auto moves = legalMoves(s);
            if (moves.empty()) break;
            std::uniform_int_distribution<size_t> d(0, moves.size() - 1);
            const Move& m = moves[d(rng)];
            // occasionally pop (simulating search backtracking)
            if (p > 0 && (rng() % 7) == 0) {
                nt.pop(); ot.pop();
                char buf[64]; snprintf(buf, 64, "pop -> size=%d lastIrrev=%d", nt.size, nt.lastIrrev);
                ops.push_back(buf);
            }
            nt.push(applyMove(s, m).hash, m.isWall);
            ot.push(applyMove(s, m).hash);
            { char buf[96]; snprintf(buf, 96, "push idx=%d %s hash=%llu -> lastIrrev=%d",
                nt.size - 1, m.isWall ? "WALL" : "pawn", (unsigned long long)applyMove(s, m).hash, nt.lastIrrev);
              ops.push_back(buf); }
            s = applyMove(s, m);
            if (checkInvariant("fase1")) return 1;
            if (winner(s) != -1) break;
        }
        nt.markRoot(); ot.markRoot();
        // more mid-search pops/pushes around the root
        for (int k = 0; k < 50 && nt.size > 2; k++) {
            if ((rng() % 3) == 0) {
                nt.pop(); ot.pop();
                char buf[64]; snprintf(buf, 64, "pop -> size=%d lastIrrev=%d", nt.size, nt.lastIrrev);
                ops.push_back(buf);
            }
            else {
                State ns = s;
                auto moves = legalMoves(ns);
                if (moves.empty()) break;
                const Move& m = moves[rng() % moves.size()];
                ns = applyMove(ns, m);
                nt.push(ns.hash, m.isWall);
                ot.push(ns.hash);
                { char buf[96]; snprintf(buf, 96, "push idx=%d %s -> lastIrrev=%d", nt.size - 1, m.isWall ? "WALL" : "pawn", nt.lastIrrev);
                  ops.push_back(buf); }
                queries.push_back(ns.hash);
                s = ns;
                if (checkInvariant("fase2")) return 1;
            }
            queries.push_back(s.hash);
            if (winner(s) != -1) break;
        }
        for (uint64_t h : queries) {
            bool a = nt.isRepetitionDraw(h);
            bool b = ot.isRepetitionDraw(h);
            checks++;
            if (a != b) {
                printf("DIVERGENCIA: novo=%d velho=%d (trial %d)\n", (int)a, (int)b, trial);
                printf("size=%d rootSize=%d lastIrrev=%d hash=%llu\n", nt.size, nt.rootSize, nt.lastIrrev,
                       (unsigned long long)h);
                for (int i = 0; i < ot.size; i++) {
                    if (ot.hist[i] == h) printf("  ocorrencia em indice %d (%s root)\n", i, i < ot.rootSize ? "PRE-" : "pos-");
                }
                return 1;
            }
        }
        // count() must stay consistent too
    }
    printf("OK: %lld checagens, 0 divergencias entre varredura limitada e completa\n", checks);

    // Real-game smoke: forced repetition cycle must still be detected
    State s = initialState();
    RepetitionTable rt;
    // build a 2x2 shuffle between two cells for both players
    Move forth0 = Move::pawn(cellIdx(1, N / 2));
    Move back0  = Move::pawn(cellIdx(0, N / 2));
    Move forth1 = Move::pawn(cellIdx(N - 2, N / 2));
    Move back1  = Move::pawn(cellIdx(N - 1, N / 2));
    rt.markRoot();
    bool sawDraw = false;
    for (int i = 0; i < 30 && !sawDraw; i++) {
        Move m = (i % 4 == 0) ? forth0 : (i % 4 == 1) ? forth1 : (i % 4 == 2) ? back0 : back1;
        s = applyMove(s, m);
        rt.push(s.hash, false);
        if (rt.isRepetitionDraw(s.hash)) sawDraw = true;
    }
    if (!sawDraw) { printf("FALHOU: ciclo de repeticao nao detectado\n"); return 1; }
    printf("OK: ciclo de repeticao detectado como empate\n");
    return 0;
}
