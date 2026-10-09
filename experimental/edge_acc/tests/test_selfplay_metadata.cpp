#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "../tools/selfplay/selfplay_metadata.hpp"

#define main selfplay_cli_main_for_test
#include "../tools/selfplay/selfplay_main.cpp"
#undef main

static int failures = 0;

#define CHECK(condition, message)                                      \
    do {                                                               \
        if (!(condition)) {                                            \
            std::printf("FAIL: %s (line %d)\n", message, __LINE__);   \
            ++failures;                                                \
        }                                                              \
    } while (0)

struct RootFixture {
    std::vector<float> visits;
    std::vector<float> q;
};

static std::string testPath(const char* name) {
    return std::string("tests/") + name;
}

static bool readMetadataFile(const std::string& path, size_t& count,
                             qr::TrainingMetaV1& first,
                             qr::TrainingMetaV1& last,
                             int expectedSourceClass = -1) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    count = 0;
    qr::TrainingMetaV1 record{};
    while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
        if (count == 0) first = record;
        last = record;
        if (expectedSourceClass >= 0) {
            CHECK(record.sourceClass == (uint8_t)expectedSourceClass,
                  "configured source class appears on every metadata row");
        }
        ++count;
    }
    return input.eof();
}

static bool countV3File(const std::string& path, size_t& count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    count = 0;
    qr::TrainingSample record{};
    while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) ++count;
    return input.eof();
}

static int runCli(std::vector<std::string> args) {
    std::vector<std::vector<char>> storage;
    std::vector<char*> argv;
    storage.reserve(args.size());
    argv.reserve(args.size() + 1);
    for (const std::string& arg : args) {
        storage.emplace_back(arg.begin(), arg.end());
        storage.back().push_back('\0');
        argv.push_back(storage.back().data());
    }
    argv.push_back(nullptr);
    try {
        return selfplay_cli_main_for_test((int)args.size(), argv.data());
    } catch (const std::exception&) {
        return 1;
    }
}

static void testCliMetadataGeneration() {
    const std::string outTemplate = testPath("tmp_meta_cli_{shard:03d}.bin");
    const std::string metaTemplate = testPath("tmp_meta_cli_{shard:03d}.meta");
    const std::string out0 = testPath("tmp_meta_cli_000.bin");
    const std::string out1 = testPath("tmp_meta_cli_001.bin");
    const std::string meta0 = testPath("tmp_meta_cli_000.meta");
    const std::string meta1 = testPath("tmp_meta_cli_001.meta");
    std::remove(out0.c_str());
    std::remove(out1.c_str());
    std::remove(meta0.c_str());
    std::remove(meta1.c_str());

    const int rc = runCli({"selfplay", "--no-mcab", "--heuristic", "--games", "2",
                           "--chunk-games", "1", "--threads", "1", "--time-ms", "1",
                           "--max-plies", "100", "--out", outTemplate,
                           "--meta-out", metaTemplate, "--meta-source-class", "37"});
    CHECK(rc == 0, "CLI metadata run succeeds");
    for (const std::string& path : {out0, out1, meta0, meta1}) {
        std::ifstream input(path, std::ios::binary);
        CHECK((bool)input, "chunked output path exists");
    }

    for (const std::string& pair : {std::string("0"), std::string("1")}) {
        const std::string outPath = testPath(("tmp_meta_cli_00" + pair + ".bin").c_str());
        const std::string metaPath = testPath(("tmp_meta_cli_00" + pair + ".meta").c_str());
        size_t v3Count = 0, metaCount = 0;
        qr::TrainingMetaV1 first{}, last{};
        CHECK(countV3File(outPath, v3Count), "V3 shard can be read");
        CHECK(readMetadataFile(metaPath, metaCount, first, last, 37),
              "metadata shard can be read");
        CHECK(v3Count > 0 && v3Count == metaCount,
              "V3 and metadata counts are aligned");
        CHECK(first.gameLength > 0 && first.pliesToEnd == first.gameLength,
              "first sample stores full game length and terminal distance");
        CHECK(last.pliesToEnd == 1, "last sample stores one ply to the result");
        CHECK(first.sourceClass == 37 && last.sourceClass == 37,
              "CLI source class propagates to every metadata row");
    }

    const std::string ambiguousOut = testPath("tmp_meta_ambiguous_{shard:03d}.bin");
    const std::string ambiguousMeta = testPath("tmp_meta_ambiguous.meta");
    const std::string ambiguousPrimary = testPath("tmp_meta_ambiguous_000.bin");
    std::remove(ambiguousPrimary.c_str());
    std::remove(ambiguousMeta.c_str());
    CHECK(runCli({"selfplay", "--games", "2", "--chunk-games", "1", "--out",
                  ambiguousOut, "--meta-out", ambiguousMeta}) != 0,
          "CLI rejects an ambiguous chunked metadata path");
    CHECK(!std::ifstream(ambiguousPrimary),
          "ambiguous metadata path does not create a primary output");

    const std::string missingDirMeta = testPath("missing_metadata_dir/meta.bin");
    const std::string missingPrimary = testPath("tmp_meta_open_failure.bin");
    std::remove(missingPrimary.c_str());
    CHECK(runCli({"selfplay", "--no-mcab", "--heuristic", "--games", "1",
                  "--threads", "1", "--time-ms", "1", "--max-plies", "1",
                  "--out", missingPrimary, "--meta-out", missingDirMeta}) != 0,
          "metadata open failure returns a nonzero status");
    CHECK(!std::ifstream(missingPrimary),
          "metadata open failure does not leave a primary output");

    CHECK(runCli({"selfplay", "--games", "1", "--out", missingPrimary,
                  "--meta-out", meta0, "--meta-source-class", "256"}) != 0,
          "CLI rejects a source class above 255");
    CHECK(runCli({"selfplay", "--games", "1", "--out", missingPrimary,
                  "--meta-out", meta0, "--meta-source-class", "-1"}) != 0,
          "CLI rejects a negative source class");

    for (const std::string& path : {out0, out1, meta0, meta1, ambiguousPrimary,
                                    ambiguousMeta, missingPrimary}) {
        std::remove(path.c_str());
    }
}

int main() {
    static_assert(sizeof(qr::TrainingMetaV1) == 20);

    RootFixture root{{3.0f, 1.0f}, {0.75f, 0.25f}};
    CHECK(std::fabs(qr::rootMeanValue(root) - 0.625f) < 1e-6f,
          "rootMeanValue computes a visit-weighted mean");

    qr::TrainingMetaV1 meta = qr::makeTrainingMeta(42, 0.625f, 3);
    meta.pliesToEnd = 3;
    meta.gameLength = 12;
    CHECK(meta.gameId == 42, "metadata stores the game id");
    CHECK(meta.pliesToEnd == 3, "metadata stores terminal distance");
    CHECK(meta.gameLength == 12, "metadata stores full game length");
    CHECK(qr::trainingMetaRootValueIsValid(meta), "finite root value is valid");

    const qr::TrainingMetaV1 missing = qr::makeTrainingMeta(
        43, std::numeric_limits<float>::quiet_NaN());
    CHECK((missing.qualityFlags & qr::TRAINING_META_ROOT_VALUE_MISSING) != 0,
          "missing root value sets the explicit quality flag");
    CHECK(std::isnan(missing.rootValue), "missing root value is NaN");
    CHECK(!qr::trainingMetaRootValueIsValid(missing),
          "missing root value is not valid");

    testCliMetadataGeneration();

    if (failures == 0) std::printf("PASS: self-play metadata\n");
    return failures == 0 ? 0 : 1;
}
