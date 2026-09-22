#include <cstdio>
#include <fstream>
#include <string>

#include "../tools/selfplay/seed_positions.hpp"

static int failures = 0;

#define CHECK(condition, message)                                      \
    do {                                                               \
        if (!(condition)) {                                            \
            std::printf("FAIL: %s (line %d)\n", message, __LINE__);   \
            ++failures;                                                \
        }                                                              \
    } while (0)

int main() {
    const std::string path = "tests/tmp_selfplay_seed_positions.jsonl";
    {
        std::ofstream out(path);
        out << R"({"schema":"zquoridor.position.v1","id":"first","history":["e2","e8","e3"],"metadata":{"opponent":"claustrophobia"}})" << '\n';
        out << R"({"schema":"zquoridor.position.v1","id":"second","history":["e2","e8","e3","e7"],"metadata":{"opponent":"titanium"}})" << '\n';
    }

    std::string error;
    const auto seeds = qr::loadSelfPlaySeeds(path, error);
    CHECK(error.empty(), "valid seed file loads without an error");
    CHECK(seeds.size() == 2, "both seed positions are loaded");
    if (seeds.size() == 2) {
        CHECK(seeds[0].id == "first", "seed id is preserved");
        CHECK(seeds[0].state.turn == 1, "three-ply seed preserves side to move");
        CHECK(seeds[0].state.pawn[0] == qr::cellIdx(2, 4), "first pawn is replayed to e3");
        CHECK(seeds[0].repetition.size == 3, "repetition history includes every replayed ply");
        CHECK(seeds[1].state.turn == 0, "four-ply seed preserves side to move");
        CHECK(seeds[1].state.pawn[1] == qr::cellIdx(6, 4), "second pawn is replayed to e7");
    }

    std::remove(path.c_str());

    const std::string badPath = "tests/tmp_selfplay_seed_positions_bad.jsonl";
    {
        std::ofstream out(badPath);
        out << R"({"id":"bad","history":["a1"]})" << '\n';
    }
    error.clear();
    const auto badSeeds = qr::loadSelfPlaySeeds(badPath, error);
    CHECK(badSeeds.empty(), "illegal seed history is rejected");
    CHECK(!error.empty(), "illegal seed history reports an error");
    std::remove(badPath.c_str());

    if (failures == 0) std::printf("PASS: self-play seed positions\n");
    return failures == 0 ? 0 : 1;
}
