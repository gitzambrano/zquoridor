#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <string>
#include "rules.hpp"
#include "search.hpp"

using namespace qr;

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <string>
#include "rules.hpp"
#include "search.hpp"
#include "selfplay.hpp"

using namespace qr;

// Executa uma única partida e salva amostras se binFp for não-nulo.
int playArenaGame(int engine1PlayerIdx, int timeMs, const State& startState, uint64_t& eng1NodesOut, uint64_t& eng2NodesOut, double& eng1TimeOut, double& eng2TimeOut, std::vector<TrainingSample>* samplesOut = nullptr) {
    State s = startState;
    Negamax eng1;
    Negamax eng2;

    // Garantir zeramento e isolamento completo da TT entre partidas
    eng1.clearTT();
    eng2.clearTT();

    RepetitionTable realHistory;
    realHistory.push(s.hash);

    struct MoveRecord {
        State state;
        Move move;
        int moverTurn;
        int searchScore;
    };
    std::vector<MoveRecord> gameRecords;

    int winnerPlayer = -1;
    int maxPlies = 300;
    for (int ply = 0; ply < maxPlies; ply++) {
        int w = winner(s);
        if (w != -1) {
            winnerPlayer = w;
            break;
        }

        if (realHistory.count(s.hash) >= 3) {
            winnerPlayer = -1; // Empate por repetição
            break;
        }

        SearchStats st;
        Move m;
        auto t0 = std::chrono::high_resolution_clock::now();
        int currentTurn = s.turn;

        if (s.turn == engine1PlayerIdx) {
            m = eng1.chooseMove(s, 40, timeMs, st, realHistory);
            auto t1 = std::chrono::high_resolution_clock::now();
            eng1TimeOut += std::chrono::duration<double>(t1 - t0).count();
            eng1NodesOut += st.nodes;
        } else {
            RepetitionTable emptyHistory;
            m = eng2.chooseMove(s, 40, timeMs, st, emptyHistory);
            auto t1 = std::chrono::high_resolution_clock::now();
            eng2TimeOut += std::chrono::duration<double>(t1 - t0).count();
            eng2NodesOut += st.nodes;
        }

        if (samplesOut) {
            gameRecords.push_back({s, m, currentTurn, st.score});
        }

        s = applyMove(s, m);
        realHistory.push(s.hash);
    }

    if (winnerPlayer == -1 && winner(s) != -1) {
        winnerPlayer = winner(s);
    }

    // Se solicitado gravação de .bin e o jogo teve um vencedor (não empate / limite)
    if (samplesOut && winnerPlayer != -1) {
        for (const auto& rec : gameRecords) {
            TrainingSample ts{};
            ts.ownPawn = rec.state.pawn[rec.moverTurn];
            ts.oppPawn = rec.state.pawn[1 - rec.moverTurn];
            ts.wallsH = rec.state.wallsH;
            ts.wallsV = rec.state.wallsV;
            ts.wallsLeftOwn = rec.state.wallsLeft[rec.moverTurn];
            ts.wallsLeftOpp = rec.state.wallsLeft[1 - rec.moverTurn];
            ts.searchScore = (int16_t)rec.searchScore;
            ts.gameResult = (rec.moverTurn == winnerPlayer) ? (int8_t)1 : (int8_t)-1;
            ts.policyTarget = moveToPolicyIndex(rec.move);
            ts.ownDist = (uint8_t)std::min(255, shortestPathLen(rec.state.wallsH, rec.state.wallsV, rec.state.pawn[rec.moverTurn], rec.moverTurn));
            ts.oppDist = (uint8_t)std::min(255, shortestPathLen(rec.state.wallsH, rec.state.wallsV, rec.state.pawn[1 - rec.moverTurn], 1 - rec.moverTurn));
            samplesOut->push_back(ts);
        }
    }

    return winnerPlayer;
}

State generateRandomOpening(int randomPlies, std::mt19937_64& rng) {
    State s = initialState();
    for (int ply = 0; ply < randomPlies; ply++) {
        if (winner(s) != -1) break;
        auto moves = legalMoves(s);
        if (moves.empty()) break;
        std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
        s = applyMove(s, moves[pick(rng)]);
    }
    return s;
}

int main(int argc, char* argv[]) {
    int totalGames = 40;
    int timeMs = 10;
    int randomPlies = 4;
    int reportGames = 0;
    uint64_t seed = 42;

    std::string binFilePath = "";

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--games") == 0 && i + 1 < argc) {
            totalGames = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--time") == 0 && i + 1 < argc) {
            timeMs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--random-plies") == 0 && i + 1 < argc) {
            randomPlies = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (std::strcmp(argv[i], "--report-games") == 0 && i + 1 < argc) {
            reportGames = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bin-file") == 0 && i + 1 < argc) {
            binFilePath = argv[++i];
        }
    }

    if (totalGames % 2 != 0) {
        totalGames++;
    }
    int totalPairs = totalGames / 2;

    std::mt19937_64 rng(seed);

    int eng1Wins = 0;
    int eng2Wins = 0;
    int draws = 0;
    uint64_t eng1Nodes = 0;
    uint64_t eng2Nodes = 0;
    double eng1TimeSec = 0.0;
    double eng2TimeSec = 0.0;

    int lastReportedGames = 0;
    std::vector<TrainingSample> allSamples;
    std::vector<TrainingSample>* samplesPtr = binFilePath.empty() ? nullptr : &allSamples;

    for (int p = 0; p < totalPairs; p++) {
        State openingState = generateRandomOpening(randomPlies, rng);

        // Jogo A: Engine 1 = Jogador 0 (Brancas), Engine 2 = Jogador 1 (Pretas)
        int resA = playArenaGame(0, timeMs, openingState, eng1Nodes, eng2Nodes, eng1TimeSec, eng2TimeSec, samplesPtr);
        if (resA == -1) draws++;
        else if (resA == 0) eng1Wins++;
        else eng2Wins++;

        // Jogo B: Engine 2 = Jogador 0 (Brancas), Engine 1 = Jogador 1 (Pretas) na MESMA abertura
        int resB = playArenaGame(1, timeMs, openingState, eng1Nodes, eng2Nodes, eng1TimeSec, eng2TimeSec, samplesPtr);
        if (resB == -1) draws++;
        else if (resB == 1) eng1Wins++;
        else eng2Wins++;

        int gamesPlayed = (p + 1) * 2;
        if (reportGames > 0 && (gamesPlayed - lastReportedGames >= reportGames)) {
            double eng1Nps = eng1TimeSec > 0 ? eng1Nodes / eng1TimeSec : 0.0;
            double eng2Nps = eng2TimeSec > 0 ? eng2Nodes / eng2TimeSec : 0.0;
            std::printf("PROGRESS_JSON:{\"games\":%d,\"candWins\":%d,\"baseWins\":%d,\"draws\":%d,\"candNodes\":%llu,\"baseNodes\":%llu,\"candTimeSec\":%.4f,\"baseTimeSec\":%.4f,\"candNps\":%.0f,\"baseNps\":%.0f}\n",
                        gamesPlayed - lastReportedGames, eng1Wins, eng2Wins, draws,
                        (unsigned long long)eng1Nodes, (unsigned long long)eng2Nodes,
                        eng1TimeSec, eng2TimeSec, eng1Nps, eng2Nps);
            std::fflush(stdout);
            
            eng1Wins = 0; eng2Wins = 0; draws = 0;
            eng1Nodes = 0; eng2Nodes = 0;
            eng1TimeSec = 0.0; eng2TimeSec = 0.0;
            lastReportedGames = gamesPlayed;
        }
    }

    if (!binFilePath.empty() && !allSamples.empty()) {
        FILE* fp = std::fopen(binFilePath.c_str(), "wb");
        if (fp) {
            std::fwrite(allSamples.data(), sizeof(TrainingSample), allSamples.size(), fp);
            std::fclose(fp);
        }
    }

    double eng1Nps = eng1TimeSec > 0 ? eng1Nodes / eng1TimeSec : 0.0;
    double eng2Nps = eng2TimeSec > 0 ? eng2Nodes / eng2TimeSec : 0.0;

    std::printf("RESULT_JSON:{\"candWins\":%d,\"baseWins\":%d,\"draws\":%d,\"candNodes\":%llu,\"baseNodes\":%llu,\"candTimeSec\":%.4f,\"baseTimeSec\":%.4f,\"candNps\":%.0f,\"baseNps\":%.0f}\n",
                eng1Wins, eng2Wins, draws,
                (unsigned long long)eng1Nodes, (unsigned long long)eng2Nodes,
                eng1TimeSec, eng2TimeSec, eng1Nps, eng2Nps);
    std::fflush(stdout);

    return 0;
}

