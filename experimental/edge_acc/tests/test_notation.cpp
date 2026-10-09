// test_notation.cpp -- correctness suite for the P6 GUI engine surface
// (gui-premium.md): QFEN round-trip and import diagnostics, history
// navigation (goto/truncate), the multi-line analysis result shape and the
// editor validity bitmask.
//
// Includes ../gui_web/engine_wasm.cpp directly so the test reaches the
// anonymous-namespace helpers (writeQfen/qfenImportScratch) as well as the
// exported C surface. The file is plain C++ when __EMSCRIPTEN__ is off.
// Build: -O2 -std=c++17 -Isrc, no arguments, run from the repo root.
#include <cstdio>
#include <string>
#include <random>
#include "../gui_web/engine_wasm.cpp"

using namespace qr;

static int g_failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::printf("FAIL: %s (line %d)\n", msg, __LINE__);       \
            g_failures++;                                             \
        }                                                             \
    } while (0)

static std::string lastErrText() {
    char buf[160];
    qr_last_error(buf, (int)sizeof(buf));
    return std::string(buf);
}

// Plays n uniformly random legal plies through the live-game surface.
static void playRandom(int n, std::mt19937& rng) {
    for (int i = 0; i < n && qr_winner() == -1; i++) {
        int count = qr_legal_moves_count();
        if (count <= 0) return;
        int pick = (int)(rng() % (unsigned)count);
        if (qr_legal_move_is_wall(pick))
            CHECK(qr_apply_wall_move(qr_legal_move_a(pick), qr_legal_move_b(pick),
                                     qr_legal_move_c(pick)) == 1, "random wall apply");
        else
            CHECK(qr_apply_pawn_move(qr_legal_move_a(pick)) == 1, "random pawn apply");
    }
}

// Compares the live game against the scratch through the exported getters.
static bool liveMatchesScratch() {
    if (qr_turn() != qr_scr_turn()) return false;
    if (qr_pawn(0) != qr_scr_pawn(0) || qr_pawn(1) != qr_scr_pawn(1)) return false;
    if (qr_walls_left(0) != qr_scr_walls_left(0)) return false;
    if (qr_walls_left(1) != qr_scr_walls_left(1)) return false;
    for (int s = 0; s < WS * WS; s++) {
        if (qr_wall_h_bit(s) != qr_scr_wall_h_bit(s)) return false;
        if (qr_wall_v_bit(s) != qr_scr_wall_v_bit(s)) return false;
    }
    return true;
}

static void testQfenRoundTrip() {
    std::mt19937 rng(12345);
    qr_new_game();
    int checked = 0;
    for (int step = 0; step < 200 && checked < 300; step++) {
        // Export the live game, import into the scratch, compare states,
        // then re-export the scratch: both strings must be identical
        // (canonical form, plan section 16 acceptance item 15).
        char buf[512];
        int len = qr_qfen_export(buf, (int)sizeof(buf));
        CHECK(len > 0, "qfen export live");
        std::string qfen(buf);
        CHECK(qr_qfen_import_scratch(qfen.c_str()) == 0,
              ("import failed: " + qfen + " -> " + lastErrText()).c_str());
        CHECK(liveMatchesScratch(), "scratch matches live after round-trip");
        char buf2[512];
        qr_qfen_export_scratch(buf2, (int)sizeof(buf2));
        std::string qfen2(buf2);
        // The live export may carry the ply cursor as an optional 7th field;
        // the scratch export does not. Compare the shared 6 fields.
        auto sixFields = [](const std::string& q) {
            std::istringstream in(q);
            std::string out, w;
            for (int i = 0; i < 6 && (in >> w); i++) {
                if (i) out += ' ';
                out += w;
            }
            return out;
        };
        CHECK(sixFields(qfen) == sixFields(qfen2),
              ("canonical form differs:\n  " + qfen + "\n  " + qfen2).c_str());
        checked++;
        playRandom(1 + (int)(rng() % 5u), rng);
        if (qr_winner() != -1) qr_new_game();
    }
    CHECK(checked >= 100, "enough round-trip samples");
}

static void testQfenDiagnostics() {
    struct Case { const char* qfen; int wantCode; };
    const Case cases[] = {
        {"e2 e9 8 6 x5h 0", 1},      // bad file letter
        {"e2 e9 8 6 c6x 0", 1},      // bad orientation
        {"e2 e9 8 6", 1},            // missing fields
        {"e2 e9 8 6 c6h 2", 1},      // turn must be 0 or 1
        {"e2 e2 8 6 - 0", 2},        // pawns share a cell
        {"e2 e9 10 10 c6h 0", 2},    // more walls than the 20 that exist
        {"e2 e9 8 6 c6h c6h 0", 2},  // duplicate slot
        {"e2 e9 8 6 b1h b1v 0", 2},  // crossing walls
        {"e2 e9 8 6 a1h b1h 0", 2},  // colinear overlap
    };
    for (const Case& c : cases) {
        int code = qr_qfen_import_scratch(c.qfen);
        CHECK(code == c.wantCode,
              ("diagnostic for '" + std::string(c.qfen) + "' got code " +
               std::to_string(code) + ": " + lastErrText()).c_str());
    }

    // A walled-in player must fail the path check. A legal 2x2 box around
    // player 0 at e5: H slots (3,4)/(5,4), V slots (4,3)/(4,5).
    CHECK(qr_qfen_import_scratch("e5 e9 8 8 e4h e6h d5v f5v 0") == 2,
          "boxed-in player rejected");
    CHECK(lastErrText().find("player 0") != std::string::npos,
          "path diagnostic names player 0");

    // Whitespace/slash tolerance plus an optional ply count.
    CHECK(qr_qfen_import_scratch("e2   e9 8\t6 c6h/e4h f3v 1 14") == 0,
          ("tolerant import failed: " + lastErrText()).c_str());
    CHECK(qr_scr_pawn(0) == cellIdx(1, 4), "pawn0 parsed");
    CHECK(qr_scr_turn() == 1, "turn parsed");
}

static void testHistoryNavigation() {
    std::mt19937 rng(777);
    qr_new_game();
    playRandom(12, rng);
    int n = qr_ply_count();
    CHECK(n >= 12, "history recorded");
    CHECK(qr_cursor() == n, "cursor at end after play");

    int mid = n / 2;
    CHECK(qr_goto_ply(mid) == mid, "goto returns clamped ply");
    CHECK(qr_cursor() == mid, "cursor moved");
    CHECK(qr_ply_count() == n, "goto keeps the future for redo");

    // Replay the move recorded at ply `mid` from the ply state: the result
    // must equal the recorded successor state.
    qr_scratch_from_ply(mid);
    Move m = g_histMoves[(size_t)mid];
    CHECK(scratchApply(m) == 1, "recorded move still legal at its ply");
    CHECK(g_scratch.hash == g_histStates[(size_t)mid + 1].hash,
          "replayed successor hash matches history");

    CHECK(qr_truncate_history(mid) == mid, "truncate reports new length");
    CHECK(qr_ply_count() == mid, "future dropped");
    CHECK(qr_goto_ply(-3) == 0, "clamp below range");
    CHECK(qr_truncate_history(999999) == mid, "clamp above range keeps the game");

    // Repetition bookkeeping survives navigation.
    qr_new_game();
    playRandom(30, rng);
    qr_goto_ply(3);
    qr_truncate_history(3);
    CHECK(qr_ply_count() == 3, "truncate after goto");
}

static void testAnalysisShape() {
    qr_new_game();
    qr_scratch_from_live();
    int lines = qr_analyze(4, 200, 3);
    CHECK(lines >= 1 && lines <= 3, "analyze returned 1..3 lines");
    CHECK(qr_an_line_count() == lines, "line count getter");
    for (int i = 0; i < lines; i++) {
        int len = qr_an_line_len(i);
        CHECK(len >= 1, "every line starts with its root move");
        qr_scratch_from_live();   // replay this PV from the same root
        for (int j = 0; j < len; j++) {
            int packed = qr_an_line_move(i, j);
            bool isWall = (packed >> 24) & 1;
            int a = (packed >> 16) & 255, b = (packed >> 8) & 255, c = packed & 255;
            int ok = isWall ? qr_scr_apply_wall(a, b, c) : qr_scr_apply_pawn(a);
            CHECK(ok == 1, "pv move is legal in sequence");
            if (!ok) break;
        }
        if (i > 0)
            CHECK(qr_an_line_score(i - 1) >= qr_an_line_score(i),
                  "lines sorted by score desc");
    }
    CHECK(qr_an_nodes() > 0, "node count reported");
    CHECK(qr_an_depth() >= 1, "depth reported");

    // Analysis must not touch the live game.
    uint64_t before = g_state.hash;
    qr_analyze(4, 150, 2);
    CHECK(g_state.hash == before, "analysis leaves the live game untouched");
}

static void testEditorValidity() {
    qr_scratch_reset();
    CHECK(qr_edit_validity() == 0, "initial position valid");

    int p1orig = qr_scr_pawn(1);
    // The setter refuses to merge the pawns into one cell (the GUI shows a
    // toast); bitmask bit 0 is a defensive check for imported positions.
    CHECK(qr_edit_set_pawn(1, qr_scr_pawn(0)) == -3, "pawn merge refused");
    CHECK(qr_edit_validity() == 0, "refusal left position valid");
    CHECK(qr_edit_set_pawn(1, p1orig) == 0, "restore pawn");

    // Default budgets are 10/10; one placed wall pushes the total to 21.
    CHECK(qr_edit_set_wall(0, 3, 3, 1) == 1, "editor wall placed");
    CHECK((qr_edit_validity() & 16) != 0, "budget exceeded flagged");
    CHECK(qr_edit_apply() == 0, "apply refused while invalid");
    qr_edit_set_walls_left(0, 9);
    CHECK(qr_edit_validity() == 0, "budget restored");
    qr_edit_set_walls_left(0, 10);
    CHECK(qr_edit_set_wall(0, 3, 3, 0) == 0, "wall removed");

    // Physical conflicts refuse placement; path problems stay allowed and
    // are only reported by the bitmask.
    CHECK(qr_edit_set_wall(1, 3, 3, 1) == 1, "crossing-free v wall placed");
    CHECK(qr_edit_set_wall(0, 3, 3, 1) == -1, "crossing refused");
    qr_edit_set_wall(1, 3, 3, 0);

    CHECK(qr_edit_apply() == 1, "apply commits a valid position");
    CHECK(qr_ply_count() == 0 && qr_cursor() == 0, "applied game starts empty");
    CHECK(qr_walls_left(0) == 10 && qr_walls_left(1) == 10, "budgets carried over");
}

int main() {
    testQfenRoundTrip();
    testQfenDiagnostics();
    testHistoryNavigation();
    testAnalysisShape();
    testEditorValidity();
    if (g_failures == 0) std::printf("test_notation: ALL OK\n");
    else std::printf("test_notation: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
