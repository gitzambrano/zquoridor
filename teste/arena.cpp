// arena.cpp -- Arena de verdade entre DUAS engines distintas.
//
// Por que este arquivo substituiu a versao antiga de arena.cpp (guardada
// em teste/arena_singleengine_old.cpp): a versao antiga compilava UM único binário (a partir só do ref1) e
// instanciava duas Negamax daquele MESMO binário -- ou seja, "Engine 2"
// no run_arena.py nunca era de fato o código do --ref2: o worktree do
// ref2 era criado e depois descartado sem nunca ser compilado. O
// confronto rodava, na prática, o engine do ref1 contra ele mesmo,
// disfarçado de A/B test.
//
// Este arquivo resolve isso incluindo os headers (rules.hpp/search.hpp)
// de CADA ref sob um namespace próprio (qr_e1 / qr_e2), via o truque de
// pré-processador `#define qr qr_eN` antes do #include -- como tudo em
// rules.hpp/search.hpp/cat.hpp/endgame_race.hpp/dsu.hpp vive dentro de
// `namespace qr { ... }`, a macro renomeia esse namespace no momento da
// expansão, e os dois conjuntos de símbolos (Negamax, State, Move, TT,
// etc.) convivem no mesmo processo sem colidir. Cada engine roda com o
// código-fonte REAL do ref correspondente.
//
// Ponte entre os dois "mundos": como State pode (em tese) ter layout
// diferente entre refs, nunca fazemos cast entre qr_e1::State e
// qr_e2::State. Mantemos DOIS estados paralelos (s1, s2), avançados pela
// MESMA sequência lógica de lances (Move é um POD trivial e estável:
// isWall/a/b/c), e usamos s1 como fonte-da-verdade para abertura
// aleatória, detecção de vencedor e detecção de empate por repetição do
// jogo real (exatamente o que arena.cpp fazia com seu único estado).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <string>
#include <cstdint>
#include <algorithm>

#ifndef ENGINE1_SEARCH_HPP
#error "compile com -DENGINE1_SEARCH_HPP=\"\\\"/caminho/absoluto/search.hpp\\\"\""
#endif
#ifndef ENGINE2_SEARCH_HPP
#error "compile com -DENGINE2_SEARCH_HPP=\"\\\"/caminho/absoluto/search.hpp\\\"\""
#endif

#define qr qr_e1
#include ENGINE1_SEARCH_HPP
#undef qr

#define qr qr_e2
#include ENGINE2_SEARCH_HPP
#undef qr

// =============================================================================
// CONFIG DEFAULTS -- edite aqui para mudar o comportamento padrao sem
// precisar passar flag nenhuma. Cada constante tem uma flag equivalente
// (ver o parsing em main() abaixo); a flag, quando passada, tem
// prioridade sobre a constante. Rodar `./arena.exe` sem argumento nenhum
// usa TODOS os valores daqui.
//
// E1_FORCE_HEURISTIC_DEFAULT / E2_FORCE_HEURISTIC_DEFAULT SEPARADOS (nao
// um unico liga/desliga global): permite configurar heuristica vs. NNUE
// no mesmo confronto so mudando estas duas linhas -- ex. E1=true,
// E2=false roda Engine 1 heuristica contra Engine 2 NNUE sem precisar de
// nenhuma flag de linha de comando. As flags --heuristic (as duas de
// uma vez), --e1-heuristic/--e2-heuristic (uma de cada vez) e
// --e1-nnue/--e2-nnue (caminho de pesos especifico) continuam disponiveis
// pra sobrepor isso por execucao, sem precisar recompilar.
// =============================================================================
constexpr int GAMES_DEFAULT = 40;
constexpr int TIME_MS_DEFAULT = 10;
constexpr int RANDOM_PLIES_DEFAULT = 4;
constexpr int REPORT_GAMES_DEFAULT = 0;
constexpr uint64_t SEED_DEFAULT = 42;
constexpr const char* BIN_FILE_DEFAULT = "";   // vazio = nao grava dataset .bin
constexpr bool INVERT_COLORS_DEFAULT = true;
// Gerar uma bateria de testes SEM NNUE (so heuristica, as duas engines):
// mude as duas constantes abaixo para true, OU passe --heuristic na
// linha de comando toda vez -- mesmo efeito, a flag so existe pra nao
// precisar editar/recompilar.
constexpr bool E1_FORCE_HEURISTIC_DEFAULT = false;
constexpr bool E2_FORCE_HEURISTIC_DEFAULT = false;

// Struct de amostra de treino local (independente das duas engines) --
// layout idêntico ao TrainingSample de selfplay.hpp/arena.cpp original.
#pragma pack(push, 1)
struct TrainingSample {
    uint8_t  ownPawn;
    uint8_t  oppPawn;
    uint64_t wallsH;
    uint64_t wallsV;
    int8_t   wallsLeftOwn;
    int8_t   wallsLeftOpp;
    int16_t  searchScore;
    int8_t   gameResult;
    uint16_t policyTarget;
    uint8_t  ownDist;
    uint8_t  oppDist;
    uint8_t  mover;         // 0/1 -- ver nota completa em selfplay.hpp
    int16_t  ownCatTotal;   // soma do calor de corredor (cat.hpp) do mover -- ver nota em selfplay.hpp
    int16_t  oppCatTotal;
};
#pragma pack(pop)
static_assert(sizeof(TrainingSample) == 32, "TrainingSample precisa ficar packed");

static inline qr_e2::Move toE2(const qr_e1::Move& m) {
    qr_e2::Move r; r.isWall = m.isWall; r.a = m.a; r.b = m.b; r.c = m.c; return r;
}

// Helpers SFINAE para tentar chamar setEvalMode / loadWeightsQuant se existirem no namespace qr_e1 / qr_e2
// (permite compilar arena entre a versao atual e refs Git antigas que nao tinham NNUE).
template <typename Dummy = void>
auto tryLoadWeightsE1(const std::string& path, int) -> decltype(qr_e1::loadWeightsQuant(path)) {
    return qr_e1::loadWeightsQuant(path);
}
inline bool tryLoadWeightsE1(const std::string&, ...) { return false; }

template <typename Dummy = void>
auto tryLoadWeightsE2(const std::string& path, int) -> decltype(qr_e2::loadWeightsQuant(path)) {
    return qr_e2::loadWeightsQuant(path);
}
inline bool tryLoadWeightsE2(const std::string&, ...) { return false; }

// Mesmo truque SFINAE para defaultNnueWeightsPath(): refs antigos (antes
// de NNUE virar default) não têm essa função em nnue.hpp, então o overload
// "..." devolve string vazia (== "sem default disponível para este ref",
// tratado abaixo como "essa engine fica heurística por não ter pesos").
template <typename Dummy = void>
auto tryDefaultPathE1(int) -> decltype(qr_e1::defaultNnueWeightsPath()) {
    return qr_e1::defaultNnueWeightsPath();
}
inline std::string tryDefaultPathE1(...) { return std::string(); }

template <typename Dummy = void>
auto tryDefaultPathE2(int) -> decltype(qr_e2::defaultNnueWeightsPath()) {
    return qr_e2::defaultNnueWeightsPath();
}
inline std::string tryDefaultPathE2(...) { return std::string(); }

template <typename Eng>
auto trySetEvalModeNnue(Eng& eng, int) -> decltype(eng.setEvalMode(Eng::EvalMode::NNUE), void()) {
    eng.setEvalMode(Eng::EvalMode::NNUE);
}
template <typename Eng>
void trySetEvalModeNnue(Eng&, ...) {}

static bool g_e1UseNnue = false;
static bool g_e2UseNnue = false;

// Um jogo completo. engine1PlayerIdx define quem (0=brancas,1=pretas) é
// o engine do ref1 nesta partida. Estado é mantido em paralelo nos dois
// namespaces; s1 é a referência canônica para regras de fim de jogo.
int playArenaGame(int engine1PlayerIdx, int timeMs, int randomPlies, std::mt19937_64& rng,
                   uint64_t& eng1NodesOut, uint64_t& eng2NodesOut,
                   double& eng1TimeOut, double& eng2TimeOut,
                   std::vector<TrainingSample>* samplesOut) {
    qr_e1::Negamax eng1;
    qr_e2::Negamax eng2;
    if (g_e1UseNnue) trySetEvalModeNnue(eng1, 0);
    if (g_e2UseNnue) trySetEvalModeNnue(eng2, 0);
    eng1.clearTT();
    eng2.clearTT();

    qr_e1::State s1 = qr_e1::initialState();
    qr_e2::State s2 = qr_e2::initialState();

    // Abertura aleatória: gerada a partir do ref1 (fonte-da-verdade das
    // regras) e replicada lance-a-lance no estado do ref2.
    for (int ply = 0; ply < randomPlies; ply++) {
        if (qr_e1::winner(s1) != -1) break;
        auto moves = qr_e1::legalMoves(s1);
        if (moves.empty()) break;
        std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
        qr_e1::Move m = moves[pick(rng)];
        s1 = qr_e1::applyMove(s1, m);
        s2 = qr_e2::applyMove(s2, toE2(m));
    }

    qr_e1::RepetitionTable realHistory;

    struct MoveRecord {
        qr_e1::State state;
        qr_e1::Move move;
        int moverTurn;
        int searchScore;
    };
    std::vector<MoveRecord> gameRecords;

    int winnerPlayer = -1;
    int maxPlies = 300;
    qr_e1::RepetitionTable hist1;
    qr_e2::RepetitionTable hist2;

    for (int ply = 0; ply < maxPlies; ply++) {
        int w = qr_e1::winner(s1);
        if (w != -1) { winnerPlayer = w; break; }

        if (realHistory.count(s1.hash) >= 2) { winnerPlayer = -1; break; }

        int currentTurn = s1.turn;
        qr_e1::Move mChosen;

        if (currentTurn == engine1PlayerIdx) {
            qr_e1::SearchStats st;
            auto t0 = std::chrono::high_resolution_clock::now();
            mChosen = eng1.chooseMove(s1, 40, timeMs, st, hist1);
            auto t1 = std::chrono::high_resolution_clock::now();
            eng1TimeOut += std::chrono::duration<double>(t1 - t0).count();
            eng1NodesOut += st.nodes;
            if (samplesOut) gameRecords.push_back({s1, mChosen, currentTurn, st.score});
        } else {
            qr_e2::SearchStats st;
            auto t0 = std::chrono::high_resolution_clock::now();
            qr_e2::Move m2 = eng2.chooseMove(s2, 40, timeMs, st, hist2);
            auto t1 = std::chrono::high_resolution_clock::now();
            eng2TimeOut += std::chrono::duration<double>(t1 - t0).count();
            eng2NodesOut += st.nodes;
            mChosen = qr_e1::Move{m2.isWall, m2.a, m2.b, m2.c};
            if (samplesOut) gameRecords.push_back({s1, mChosen, currentTurn, st.score});
        }

        realHistory.push(s1.hash);
        hist1.push(s1.hash);
        hist2.push(s2.hash);
        s1 = qr_e1::applyMove(s1, mChosen);
        s2 = qr_e2::applyMove(s2, toE2(mChosen));
    }

    if (winnerPlayer == -1 && qr_e1::winner(s1) != -1) winnerPlayer = qr_e1::winner(s1);

    if (samplesOut && winnerPlayer != -1) {
        for (const auto& rec : gameRecords) {
            int mover = rec.moverTurn, opp = 1 - rec.moverTurn;
            // Mesmo espelho de perspectiva de selfplay.hpp (nnue.hpp,
            // 2026-08) -- sem isso este --bin-file gerava dados em
            // coordenada CRUA, incompatível com o que selfplay.hpp grava
            // hoje (silenciosamente: mesmo dtype/tamanho de arquivo,
            // corrompe o treino se misturado).
            TrainingSample ts{};
            ts.ownPawn = (uint8_t)qr_e1::mirroredPawnCell(rec.state.pawn[mover], mover);
            ts.oppPawn = (uint8_t)qr_e1::mirroredPawnCell(rec.state.pawn[opp], mover);
            ts.wallsH = qr_e1::mirrorWallBitboard(rec.state.wallsH, mover);
            ts.wallsV = qr_e1::mirrorWallBitboard(rec.state.wallsV, mover);
            ts.wallsLeftOwn = rec.state.wallsLeft[mover];
            ts.wallsLeftOpp = rec.state.wallsLeft[opp];
            ts.searchScore = (int16_t)rec.searchScore;
            ts.gameResult = (mover == winnerPlayer) ? (int8_t)1 : (int8_t)-1;
            ts.policyTarget = qr_e1::moveToPolicyIndex(qr_e1::mirrorMoveForPerspective(rec.move, mover));
            ts.ownDist = (uint8_t)std::min(255, qr_e1::shortestPathLen(rec.state.wallsH, rec.state.wallsV, rec.state.pawn[mover], mover));
            ts.oppDist = (uint8_t)std::min(255, qr_e1::shortestPathLen(rec.state.wallsH, rec.state.wallsV, rec.state.pawn[opp], opp));
            ts.mover = (uint8_t)mover;
            {
                qr_e1::CorridorHeat ownHeat = qr_e1::computeCorridorHeat(rec.state.wallsH, rec.state.wallsV, rec.state.pawn[mover], mover);
                qr_e1::CorridorHeat oppHeat = qr_e1::computeCorridorHeat(rec.state.wallsH, rec.state.wallsV, rec.state.pawn[opp], opp);
                int ownSum = 0, oppSum = 0;
                for (int i = 0; i < qr_e1::N * qr_e1::N; i++) { ownSum += ownHeat.heat[i]; oppSum += oppHeat.heat[i]; }
                ts.ownCatTotal = (int16_t)ownSum;
                ts.oppCatTotal = (int16_t)oppSum;
            }
            samplesOut->push_back(ts);
        }
    }

    return winnerPlayer;
}

int main(int argc, char* argv[]) {
    int totalGames = GAMES_DEFAULT;
    int timeMs = TIME_MS_DEFAULT;
    int randomPlies = RANDOM_PLIES_DEFAULT;
    int reportGames = REPORT_GAMES_DEFAULT;
    uint64_t seed = SEED_DEFAULT;
    std::string binFilePath = BIN_FILE_DEFAULT;
    bool invertColors = INVERT_COLORS_DEFAULT;
    std::string e1NnuePath = "";
    std::string e2NnuePath = "";
    bool e1Explicit = false, e2Explicit = false;
    bool forceHeuristic = false;
    bool e1ForceHeuristic = E1_FORCE_HEURISTIC_DEFAULT, e2ForceHeuristic = E2_FORCE_HEURISTIC_DEFAULT;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--games") == 0 && i + 1 < argc) totalGames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--time") == 0 && i + 1 < argc) timeMs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--random-plies") == 0 && i + 1 < argc) randomPlies = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (std::strcmp(argv[i], "--report-games") == 0 && i + 1 < argc) reportGames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bin-file") == 0 && i + 1 < argc) binFilePath = argv[++i];
        else if (std::strcmp(argv[i], "--e1-nnue") == 0 && i + 1 < argc) { e1NnuePath = argv[++i]; e1Explicit = true; e1ForceHeuristic = false; }
        else if (std::strcmp(argv[i], "--e2-nnue") == 0 && i + 1 < argc) { e2NnuePath = argv[++i]; e2Explicit = true; e2ForceHeuristic = false; }
        else if (std::strcmp(argv[i], "--heuristic") == 0) forceHeuristic = true;
        else if (std::strcmp(argv[i], "--e1-heuristic") == 0) e1ForceHeuristic = true;
        else if (std::strcmp(argv[i], "--e2-heuristic") == 0) e2ForceHeuristic = true;
        else if (std::strcmp(argv[i], "--e1-nnue-default") == 0) e1ForceHeuristic = false;
        else if (std::strcmp(argv[i], "--e2-nnue-default") == 0) e2ForceHeuristic = false;
        else if (std::strcmp(argv[i], "--no-invert") == 0) invertColors = false;
    }

    // NNUE é o default das duas engines. Se --e1-nnue/--e2-nnue não foram
    // passados explicitamente, tenta o caminho default de cada ref (via
    // SFINAE -- refs antigos sem NNUE devolvem string vazia e ficam
    // heurísticos, sem erro). --heuristic desliga isso por completo nas
    // duas (debug/histórico/comparação com a heurística antiga).
    // --e1-heuristic/--e2-heuristic desligam só uma engine de cada vez --
    // é o que permite medir a força de jogo NNUE vs heurística num mesmo
    // confronto (--e1-heuristic sozinho: Engine 1 heurística, Engine 2
    // NNUE default).
    if (!forceHeuristic) {
        if (!e1ForceHeuristic && e1NnuePath.empty()) e1NnuePath = tryDefaultPathE1(0);
        if (!e2ForceHeuristic && e2NnuePath.empty()) e2NnuePath = tryDefaultPathE2(0);
    }
    if (forceHeuristic || e1ForceHeuristic) e1NnuePath.clear();
    if (forceHeuristic || e2ForceHeuristic) e2NnuePath.clear();

    if (!e1NnuePath.empty()) {
        if (!tryLoadWeightsE1(e1NnuePath, 0)) {
            if (e1Explicit) {
                std::fprintf(stderr, "[arena] ERRO: falha ao carregar pesos NNUE para Engine 1 de '%s'\n", e1NnuePath.c_str());
            } else {
                std::fprintf(stderr, "[arena] aviso: pesos NNUE default nao encontrados para Engine 1 ('%s') -- usando heuristica\n", e1NnuePath.c_str());
            }
        } else {
            g_e1UseNnue = true;
            std::fprintf(stderr, "[arena] Engine 1 usando NNUE ('%s')\n", e1NnuePath.c_str());
        }
    } else if (forceHeuristic || e1ForceHeuristic) {
        std::fprintf(stderr, "[arena] Engine 1: heuristica (via --heuristic/--e1-heuristic, ou E1_FORCE_HEURISTIC_DEFAULT no topo do arquivo)\n");
    }
    if (!e2NnuePath.empty()) {
        if (!tryLoadWeightsE2(e2NnuePath, 0)) {
            if (e2Explicit) {
                std::fprintf(stderr, "[arena] ERRO: falha ao carregar pesos NNUE para Engine 2 de '%s'\n", e2NnuePath.c_str());
            } else {
                std::fprintf(stderr, "[arena] aviso: pesos NNUE default nao encontrados para Engine 2 ('%s') -- usando heuristica\n", e2NnuePath.c_str());
            }
        } else {
            g_e2UseNnue = true;
            std::fprintf(stderr, "[arena] Engine 2 usando NNUE ('%s')\n", e2NnuePath.c_str());
        }
    } else if (forceHeuristic || e2ForceHeuristic) {
        std::fprintf(stderr, "[arena] Engine 2: heuristica (via --heuristic/--e2-heuristic, ou E2_FORCE_HEURISTIC_DEFAULT no topo do arquivo)\n");
    }


    if (totalGames % 2 != 0) totalGames++;
    int totalPairs = totalGames / 2;
    std::mt19937_64 rng(seed);

    int eng1Wins = 0, baseWins = 0, draws = 0;
    uint64_t eng1Nodes = 0, baseNodes = 0;
    double eng1TimeSec = 0.0, baseTimeSec = 0.0;
    int totalEng1Wins = 0, totalBaseWins = 0, totalDraws = 0;
    uint64_t totalEng1Nodes = 0, totalBaseNodes = 0;
    double totalEng1Time = 0.0, totalBaseTime = 0.0;
    int lastReportedGames = 0;

    std::vector<TrainingSample> allSamples;
    std::vector<TrainingSample>* samplesPtr = binFilePath.empty() ? nullptr : &allSamples;

    auto reportIfNeeded = [&](int gamesPlayed) {
        if (reportGames > 0 && (gamesPlayed - lastReportedGames >= reportGames)) {
            double eng1Nps = eng1TimeSec > 0 ? eng1Nodes / eng1TimeSec : 0.0;
            double baseNps = baseTimeSec > 0 ? baseNodes / baseTimeSec : 0.0;
            std::printf("PROGRESS_JSON:{\"games\":%d,\"candWins\":%d,\"baseWins\":%d,\"draws\":%d,\"candNodes\":%llu,\"baseNodes\":%llu,\"candTimeSec\":%.4f,\"baseTimeSec\":%.4f,\"candNps\":%.0f,\"baseNps\":%.0f}\n",
                        gamesPlayed - lastReportedGames, eng1Wins, baseWins, draws,
                        (unsigned long long)eng1Nodes, (unsigned long long)baseNodes,
                        eng1TimeSec, baseTimeSec, eng1Nps, baseNps);
            std::fflush(stdout);
            totalEng1Nodes += eng1Nodes; totalBaseNodes += baseNodes;
            totalEng1Time += eng1TimeSec; totalBaseTime += baseTimeSec;
            eng1Wins = 0; baseWins = 0; draws = 0;
            eng1Nodes = 0; baseNodes = 0; eng1TimeSec = 0.0; baseTimeSec = 0.0;
            lastReportedGames = gamesPlayed;
        }
    };

    if (invertColors) {
        for (int p = 0; p < totalPairs; p++) {
            std::mt19937_64 openingRng(rng());
            std::mt19937_64 openingRngCopy = openingRng;

            int resA = playArenaGame(0, timeMs, randomPlies, openingRng, eng1Nodes, baseNodes, eng1TimeSec, baseTimeSec, samplesPtr);
            if (resA == -1) { draws++; totalDraws++; }
            else if (resA == 0) { eng1Wins++; totalEng1Wins++; }
            else { baseWins++; totalBaseWins++; }

            int resB = playArenaGame(1, timeMs, randomPlies, openingRngCopy, eng1Nodes, baseNodes, eng1TimeSec, baseTimeSec, samplesPtr);
            if (resB == -1) { draws++; totalDraws++; }
            else if (resB == 1) { eng1Wins++; totalEng1Wins++; }
            else { baseWins++; totalBaseWins++; }

            reportIfNeeded((p + 1) * 2);
        }
    } else {
        for (int g = 0; g < totalGames; g++) {
            std::mt19937_64 openingRng(rng());
            int eng1Player = g % 2;
            int res = playArenaGame(eng1Player, timeMs, randomPlies, openingRng, eng1Nodes, baseNodes, eng1TimeSec, baseTimeSec, samplesPtr);
            if (res == -1) { draws++; totalDraws++; }
            else if (res == eng1Player) { eng1Wins++; totalEng1Wins++; }
            else { baseWins++; totalBaseWins++; }
            reportIfNeeded(g + 1);
        }
    }

    totalEng1Nodes += eng1Nodes; totalBaseNodes += baseNodes;
    totalEng1Time += eng1TimeSec; totalBaseTime += baseTimeSec;

    if (!binFilePath.empty() && !allSamples.empty()) {
        FILE* fp = std::fopen(binFilePath.c_str(), "wb");
        if (fp) { std::fwrite(allSamples.data(), sizeof(TrainingSample), allSamples.size(), fp); std::fclose(fp); }
    }

    double eng1Nps = totalEng1Time > 0 ? totalEng1Nodes / totalEng1Time : 0.0;
    double baseNps = totalBaseTime > 0 ? totalBaseNodes / totalBaseTime : 0.0;
    std::printf("RESULT_JSON:{\"candWins\":%d,\"baseWins\":%d,\"draws\":%d,\"candNodes\":%llu,\"baseNodes\":%llu,\"candTimeSec\":%.4f,\"baseTimeSec\":%.4f,\"candNps\":%.0f,\"baseNps\":%.0f}\n",
                totalEng1Wins, totalBaseWins, totalDraws,
                (unsigned long long)totalEng1Nodes, (unsigned long long)totalBaseNodes,
                totalEng1Time, totalBaseTime, eng1Nps, baseNps);
    std::fflush(stdout);
    return 0;
}
