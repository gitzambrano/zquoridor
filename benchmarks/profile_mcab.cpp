// profile_mcab -- sampling profiler sobre o caminho de PRODUCAO:
// McabRunner (MCTS hibrido) + NNUE, 200ms/lance em posicoes de meio-jogo.
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <atomic>
#include <windows.h>
#include "rules.hpp"
#include "search.hpp"
#include "../src/mcab.hpp"

using namespace qr;
using clockT = std::chrono::steady_clock;
// compat com refs antigos: usa push(h, irreversible) se existir
template <class T>
auto pushCompat(T& t, uint64_t h, bool irrev, int) -> decltype(t.push(h, irrev), void()) { t.push(h, irrev); }
template <class T>
void pushCompat(T& t, uint64_t h, bool, long) { t.push(h); }

struct RipSample { DWORD64 addr; int count; };
static RipSample g_samples[4096];
static volatile long g_sampleCount = 0;
static std::atomic<bool> g_profiling{true};
static DWORD64 g_base = 0;
static DWORD64 g_imageBase = 0;

static void recordRip(DWORD64 rip) {
    DWORD64 symAddr = g_imageBase + (rip - g_base);
    long n = g_sampleCount;
    for (long i = 0; i < n; i++) {
        if (g_samples[i].addr == symAddr) { g_samples[i].count++; return; }
    }
    if (n < 4096) { g_samples[n] = {symAddr, 1}; g_sampleCount = n + 1; }
}

static DWORD WINAPI sampler(LPVOID param) {
    HANDLE h = (HANDLE)param;
    CONTEXT ctx;
    while (g_profiling.load(std::memory_order_relaxed)) {
        if (SuspendThread(h) != (DWORD)-1) {
            ctx.ContextFlags = CONTEXT_CONTROL;
            if (GetThreadContext(h, &ctx)) recordRip(ctx.Rip);
            ResumeThread(h);
        }
        Sleep(1);
    }
    return 0;
}

int main(int argc, char** argv) {
    loadWeightsQuant("data/nnue/nnue_weights_int8.bin");

    HMODULE hMod = GetModuleHandleA(nullptr);
    g_base = (DWORD64)hMod;
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)hMod;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)((BYTE*)hMod + dos->e_lfanew);
    g_imageBase = nt->OptionalHeader.ImageBase;

    HANDLE hSelf;
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(),
                    GetCurrentProcess(), &hSelf, 0, FALSE, DUPLICATE_SAME_ACCESS);
    CreateThread(nullptr, 0, sampler, hSelf, 0, nullptr);

    std::mt19937 rng(777);
    auto t0 = clockT::now();
    long long moves = 0;
    long long totalNodes = 0;
    while (clockT::now() - t0 < std::chrono::seconds(30)) {
        // posicao de meio-jogo com historico realista
        State s = initialState();
        RepetitionTable hist;
        hist.markRoot();
        for (int p = 0; p < 24; p++) {
            if (winner(s) != -1) break;
            auto ms_ = legalMoves(s);
            MoveList candPawn;
            for (size_t i = 0; i < ms_.size(); i++) if (!ms_[i].isWall) candPawn.push_back(ms_[i]);
            const MoveList& src = !candPawn.empty() ? candPawn : ms_;
            s = applyMove(s, src[rng() % src.size()]);
            pushCompat(hist, s.hash, false, 0);
        }
        if (winner(s) != -1) break;

        Negamax eng;
        eng.setEvalMode(Negamax::EvalMode::NNUE);
        eng.setPolicyOrderingEnabled(true);
        mcab::McabRunner<Negamax, State, Move, MoveList, AccPair, RepetitionTable, SearchStats> runner;
        SearchStats st;
        mcab::McabStats mst;
        Move m = runner.choose(eng, s, 40, 200, st, hist, &mst);
        (void)m;
        totalNodes += mst.nodesExpanded;
        moves++;
    }

    g_profiling.store(false);
    Sleep(50);
    printf("[prof] base=%llx imageBase=%llx lances=%lld nosExpandidos=%lld\n", (unsigned long long)g_base,
           (unsigned long long)g_imageBase, moves, totalNodes);
    long n = g_sampleCount;
    for (long i = 0; i < n; i++)
        for (long j = i + 1; j < n; j++)
            if (g_samples[j].count > g_samples[i].count) { RipSample tmp = g_samples[i]; g_samples[i] = g_samples[j]; g_samples[j] = tmp; }
    int total = 0;
    for (long i = 0; i < n; i++) total += g_samples[i].count;
    printf("amostras=%d\n", total);
    for (long i = 0; i < n && i < 40; i++)
        printf("0x%llx %6d %5.1f%%\n", (unsigned long long)g_samples[i].addr,
               g_samples[i].count, 100.0 * g_samples[i].count / total);
    return 0;
}
