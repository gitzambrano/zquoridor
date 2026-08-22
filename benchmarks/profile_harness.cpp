// profile_harness -- sampling profiler para Windows.
//
// CUIDADO COM DEADLOCK (loader lock): o sampler NUNCA chama APIs que
// pegam lock (GetModuleHandleA/printf/etc.) enquanto a main thread pode
// estar SUSPENSA segurando esse mesmo lock -- se isso acontecer o
// processo trava pra sempre (pego em producao nesta sessao). Base e
// ImageBase sao cacheados ANTES do CreateThread; o sampler so le
// memoria propria, suspende, pega RIP e resume.
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <atomic>
#include <windows.h>
#include "rules.hpp"
#include "search.hpp"

using namespace qr;
using clockT = std::chrono::steady_clock;

struct RipSample { DWORD64 addr; int count; };
static RipSample g_samples[4096];
static volatile long g_sampleCount = 0;
static std::atomic<bool> g_profiling{true};

// cacheados ANTES de iniciar o sampler
static DWORD64 g_base = 0;
static DWORD64 g_imageBase = 0;

static void recordRip(DWORD64 rip) {
    DWORD64 symAddr = g_imageBase + (rip - g_base);
    long n = g_sampleCount;
    for (long i = 0; i < n; i++) {
        if (g_samples[i].addr == symAddr) { g_samples[i].count++; return; }
    }
    if (n < 4096) {
        g_samples[n] = {symAddr, 1};
        g_sampleCount = n + 1;
    }
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

    // cache de base + ImageBase preferido ANTES do sampler existir
    HMODULE hMod = GetModuleHandleA(nullptr);
    g_base = (DWORD64)hMod;
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)hMod;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)((BYTE*)hMod + dos->e_lfanew);
    g_imageBase = nt->OptionalHeader.ImageBase;

    HANDLE hSelf;
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(),
                    GetCurrentProcess(), &hSelf, 0, FALSE, DUPLICATE_SAME_ACCESS);
    CreateThread(nullptr, 0, sampler, hSelf, 0, nullptr);

    // ---- carga idêntica ao bench_repthist (modo NNUE) ----
    std::mt19937 rng(20260822);
    auto t0 = clockT::now();
    uint64_t totalNodes = 0;
    for (int g = 0; g < 6 && clockT::now() - t0 < std::chrono::seconds(25); g++) {
        State s = initialState();
        RepetitionTable hist;
        hist.markRoot();
        int warm = 24 + 8 * g;
        for (int p = 0; p < warm; p++) {
            if (winner(s) != -1) break;
            auto moves = legalMoves(s);
            if (moves.empty()) break;
            MoveList candPawn, candAny;
            for (size_t i = 0; i < moves.size(); i++) {
                State ns = applyMove(s, moves[i]);
                hist.push(ns.hash, false);
                bool rep = hist.isRepetitionDraw(ns.hash);
                hist.pop();
                if (rep) continue;
                candAny.push_back(moves[i]);
                if (!moves[i].isWall) candPawn.push_back(moves[i]);
            }
            const MoveList& src = !candPawn.empty() ? candPawn : (!candAny.empty() ? candAny : moves);
            State ns = applyMove(s, src[rng() % src.size()]);
            hist.push(ns.hash, false);
            s = ns;
        }
        if (winner(s) != -1) continue;
        Negamax eng;
        eng.setEvalMode(Negamax::EvalMode::NNUE);
        eng.setPolicyOrderingEnabled(true);
        SearchStats st2;
        RepetitionTable rt = hist;
        rt.markRoot();
        eng.testNegamaxKeepTT(s, 7, -SCORE_INF, SCORE_INF, st2);
        totalNodes += st2.nodes;
    }

    g_profiling.store(false);
    Sleep(50);
    printf("[prof] base=%llx imageBase=%llx nos=%llu\n", (unsigned long long)g_base,
           (unsigned long long)g_imageBase, (unsigned long long)totalNodes);
    // ordena por contagem desc
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
