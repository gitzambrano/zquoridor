// qfen.hpp -- notation and position serialization for the web GUI.
//
// Owned by the W1 (engine) workstream together with engine_wasm.cpp. It is a
// header rather than a block inside engine_wasm.cpp so that a standalone test
// harness can include exactly these functions without dragging in the
// emscripten bindings or the module-global game object.
//
// Everything here is pure: no global state, no dependency on the live game.
//
// Notation (fixed board frame, never flipped -- see gui_web/GUI_PLAN.md):
//   - columns a..i map to engine columns 0..8 left to right;
//   - ranks 1..9 map to engine rows 8..0, i.e. rank = 9 - engine_row;
//   - a pawn move is "<col><rank>", e.g. "e5";
//   - a wall is "<col><rank><h|v>" where rank = 8 - r (the south side of the
//     corridor), e.g. orientation 0, r=3, c=2 -> "c5h".
//
// QFEN is a single line with 6 space separated fields:
//   <pawn0> <pawn1> <wallsLeft0> <wallsLeft1> <turn> <walls>
// `walls` is a comma separated list sorted ascending by (orientation, r, c),
// or "-" when the board is empty. The field may be omitted entirely, which is
// treated as "-".
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../src/rules.hpp"

namespace qgui {

// --- helpers -----------------------------------------------------------

// Recomputes the Zobrist hash of `s` from scratch. Needed whenever a State is
// built field by field (QFEN parsing, the position editor) instead of by
// applyMove(), which maintains the hash incrementally.
inline void recomputeHash(qr::State& s) {
    qr::Zobrist& z = qr::zobrist();
    uint64_t h = z.pawnKey[0][s.pawn[0]] ^ z.pawnKey[1][s.pawn[1]];
    for (int i = 0; i < qr::WS * qr::WS; i++) {
        if ((s.wallsH >> i) & 1ull) h ^= z.wallHKey[i];
        if ((s.wallsV >> i) & 1ull) h ^= z.wallVKey[i];
    }
    if (s.turn == 1) h ^= z.turnKey;
    s.hash = h;
}

// --- cell / wall notation ----------------------------------------------

inline std::string cellToAlg(int cell) {
    if (cell < 0 || cell >= qr::N * qr::N) return std::string("??");
    int r = qr::rowOf(cell), c = qr::colOf(cell);
    std::string out;
    out += (char)('a' + c);
    out += (char)('0' + (qr::N - r));
    return out;
}

// -1 when `s` is not a well formed cell token.
inline int algToCell(const std::string& s) {
    if (s.size() != 2) return -1;
    int c = s[0] - 'a';
    int rank = s[1] - '0';
    if (c < 0 || c >= qr::N) return -1;
    if (rank < 1 || rank > qr::N) return -1;
    return qr::cellIdx(qr::N - rank, c);
}

inline std::string wallToAlg(int orientation, int r, int c) {
    if (orientation < 0 || orientation > 1 || r < 0 || r >= qr::WS || c < 0 || c >= qr::WS)
        return std::string("???");
    std::string out;
    out += (char)('a' + c);
    out += (char)('0' + (qr::WS - r));
    out += (orientation == 0 ? 'h' : 'v');
    return out;
}

inline bool algToWall(const std::string& s, int& orientation, int& r, int& c) {
    if (s.size() != 3) return false;
    char o = s[2];
    if (o != 'h' && o != 'H' && o != 'v' && o != 'V') return false;
    int col = s[0] - 'a';
    int rank = s[1] - '0';
    if (col < 0 || col >= qr::WS) return false;
    if (rank < 1 || rank > qr::WS) return false;
    orientation = (o == 'h' || o == 'H') ? 0 : 1;
    r = qr::WS - rank;
    c = col;
    return true;
}

inline std::string moveNotation(bool isWall, int a, int b, int c) {
    return isWall ? wallToAlg(a, b, c) : cellToAlg(a);
}

inline std::string moveNotation(const qr::Move& m) {
    return moveNotation(m.isWall, m.a, m.b, m.c);
}

// Parses a single move token ("e5" or "c5h") into a Move. Returns false when
// the token is not well formed; it does NOT check legality.
inline bool parseMoveToken(const std::string& tok, qr::Move& out) {
    if (tok.size() == 2) {
        int cell = algToCell(tok);
        if (cell < 0) return false;
        out = qr::Move::pawn(cell);
        return true;
    }
    if (tok.size() == 3) {
        int o, r, c;
        if (!algToWall(tok, o, r, c)) return false;
        out = qr::Move::wall(o, r, c);
        return true;
    }
    return false;
}

// --- QFEN ---------------------------------------------------------------

inline std::string formatQFEN(const qr::State& s) {
    std::string out;
    out += cellToAlg(s.pawn[0]);
    out += ' ';
    out += cellToAlg(s.pawn[1]);
    out += ' ';
    out += std::to_string((int)s.wallsLeft[0]);
    out += ' ';
    out += std::to_string((int)s.wallsLeft[1]);
    out += ' ';
    out += std::to_string(s.turn);
    out += ' ';
    // Emitted H slots first then V slots, each in slot order, which is exactly
    // ascending (orientation, r, c) -- no explicit sort needed.
    std::string walls;
    for (int o = 0; o < 2; o++) {
        uint64_t bb = (o == 0) ? s.wallsH : s.wallsV;
        for (int i = 0; i < qr::WS * qr::WS; i++) {
            if (!((bb >> i) & 1ull)) continue;
            if (!walls.empty()) walls += ',';
            walls += wallToAlg(o, i / qr::WS, i % qr::WS);
        }
    }
    out += walls.empty() ? std::string("-") : walls;
    return out;
}

namespace detail {
inline std::vector<std::string> splitWhitespace(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else {
            cur += ch;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

inline bool parseSmallInt(const std::string& s, int& out) {
    if (s.empty() || s.size() > 3) return false;
    int v = 0;
    for (char ch : s) {
        if (ch < '0' || ch > '9') return false;
        v = v * 10 + (ch - '0');
    }
    out = v;
    return true;
}
}  // namespace detail

// Structural parse only: field count, token shapes, value ranges and duplicate
// wall slots. Semantic legality (both players reachable goals, pawn overlap,
// wall budget) is deliberately NOT checked here -- the position editor needs to
// hold temporarily illegal positions, and the callers that require legality run
// the validation bitmask separately.
//
// Returns false and leaves `out` untouched on any parse error.
inline bool parseQFEN(const std::string& text, qr::State& out) {
    std::vector<std::string> f = detail::splitWhitespace(text);
    if (f.size() < 5 || f.size() > 6) return false;

    int p0 = algToCell(f[0]);
    int p1 = algToCell(f[1]);
    if (p0 < 0 || p1 < 0) return false;

    int w0 = 0, w1 = 0, turn = 0;
    if (!detail::parseSmallInt(f[2], w0) || w0 > qr::WALLS_PER_PLAYER) return false;
    if (!detail::parseSmallInt(f[3], w1) || w1 > qr::WALLS_PER_PLAYER) return false;
    if (!detail::parseSmallInt(f[4], turn) || turn > 1) return false;

    uint64_t wh = 0, wv = 0;
    if (f.size() == 6 && f[5] != "-") {
        std::string tok;
        std::string field = f[5];
        field += ',';  // sentinel so the last token is flushed by the loop
        for (char ch : field) {
            if (ch != ',') { tok += ch; continue; }
            if (tok.empty()) return false;  // empty item, e.g. "c5h,,f4v"
            int o, r, c;
            if (!algToWall(tok, o, r, c)) return false;
            uint64_t bit = 1ull << qr::slotIdx(r, c);
            uint64_t& target = (o == 0) ? wh : wv;
            if (target & bit) return false;  // duplicate slot
            target |= bit;
            tok.clear();
        }
    }

    qr::State s;
    s.pawn[0] = (uint8_t)p0;
    s.pawn[1] = (uint8_t)p1;
    s.wallsH = wh;
    s.wallsV = wv;
    s.wallsLeft[0] = (int8_t)w0;
    s.wallsLeft[1] = (int8_t)w1;
    s.turn = turn;
    recomputeHash(s);
    out = s;
    return true;
}

// --- position validation ------------------------------------------------
//
// Bitmask shared by qr_edit_validate() and qr_set_qfen(); see GUI_PLAN.md.
constexpr int QVAL_P0_BLOCKED   = 1;
constexpr int QVAL_P1_BLOCKED   = 2;
constexpr int QVAL_PAWN_OVERLAP = 4;
constexpr int QVAL_TOO_MANY_WALLS = 8;
constexpr int QVAL_PAWN_ON_GOAL = 16;

inline int popcount64(uint64_t v) {
    int n = 0;
    while (v) { v &= v - 1; n++; }
    return n;
}

inline int validatePosition(const qr::State& s) {
    int mask = 0;
    if (!qr::hasPathToGoal(s.wallsH, s.wallsV, s.pawn[0], 0)) mask |= QVAL_P0_BLOCKED;
    if (!qr::hasPathToGoal(s.wallsH, s.wallsV, s.pawn[1], 1)) mask |= QVAL_P1_BLOCKED;
    if (s.pawn[0] == s.pawn[1]) mask |= QVAL_PAWN_OVERLAP;
    if (popcount64(s.wallsH) + popcount64(s.wallsV) > 2 * qr::WALLS_PER_PLAYER)
        mask |= QVAL_TOO_MANY_WALLS;
    if (qr::rowOf(s.pawn[0]) == qr::GOAL_ROW[0]) mask |= QVAL_PAWN_ON_GOAL;
    if (qr::rowOf(s.pawn[1]) == qr::GOAL_ROW[1]) mask |= QVAL_PAWN_ON_GOAL;
    return mask;
}

}  // namespace qgui
