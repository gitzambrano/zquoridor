#include <cstdio>
#include <chrono>
#include <vector>
#include <random>
#include "rules.hpp"
#include "search.hpp"

using namespace qr;

int playOneArenaGame(int newEnginePlayerIdx, std::mt19937_64& rng, int timeMs) {
    State s = initialState();
    Negamax newEngine;
    Negamax oldEngine;

    // Abertura aleatória curta (2 plies) para variedade
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (int ply = 0; ply < 2; ply++) {
        auto moves = legalMoves(s);
        std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
        s = applyMove(s, moves[pick(rng)]);
    }

    RepetitionTable realHistory;
    realHistory.push(s.hash);

    int maxPlies = 300;
    for (int ply = 2; ply < maxPlies; ply++) {
        int w = winner(s);
        if (w != -1) return w;

        if (realHistory.count(s.hash) >= 3) {
            return -1; // Empate por repetição
        }

        SearchStats st;
        Move m;
        if (s.turn == newEnginePlayerIdx) {
            // Nova engine (com memória e contempt)
            m = newEngine.chooseMove(s, 40, timeMs, st, realHistory);
        } else {
            // Antiga engine (sem memória, passa tabela vazia)
            RepetitionTable emptyHistory;
            m = oldEngine.chooseMove(s, 40, timeMs, st, emptyHistory);
        }

        s = applyMove(s, m);
        realHistory.push(s.hash);
    }
    return -1; // Limite de plies excedido (empate)
}

int main() {
    std::printf("=== Iniciando Arena: Engine Nova (Com Contempt + Memorizacao) vs Engine Antiga ===\n");
    std::mt19937_64 rng(42);
    int timeMs = 5; // 5ms por lance para rapidez

    int games = 100;
    int newWins = 0;
    int oldWins = 0;
    int draws = 0;

    for (int i = 0; i < games; i++) {
        int newPlayer = (i % 2); // alternando cores
        int res = playOneArenaGame(newPlayer, rng, timeMs);
        if (res == -1) {
            draws++;
        } else if (res == newPlayer) {
            newWins++;
        } else {
            oldWins++;
        }
        if ((i + 1) % 10 == 0) {
            std::printf("  Progresso: %d/%d jogos concluidos. Nova: %d | Antiga: %d | Empates: %d\n",
                        i + 1, games, newWins, oldWins, draws);
            std::fflush(stdout);
        }
    }

    std::printf("\n=== RESULTADO FINAL ===\n");
    std::printf("Nova Engine:  %d vitorias (%.1f%%)\n", newWins, 100.0 * newWins / games);
    std::printf("Antiga Engine: %d vitorias (%.1f%%)\n", oldWins, 100.0 * oldWins / games);
    std::printf("Empates:       %d (%.1f%%)\n", draws, 100.0 * draws / games);
    return 0;
}
