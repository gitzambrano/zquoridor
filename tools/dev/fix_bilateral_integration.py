#!/usr/bin/env python3
from pathlib import Path


def replace(path, old, new, count=1):
    p=Path(path); s=p.read_text()
    if new in s and old not in s: return
    if old not in s: raise SystemExit(f'anchor missing in {path}: {old[:120]!r}')
    p.write_text(s.replace(old,new,count))

# Float writer still used the pre-v2 hidden size.
replace('src/nnue.hpp',
'''        for (auto& row : wv1_wl) std::fwrite(row.data(), sizeof(float), 32, f);
        std::fwrite(bv1_wl.data(), sizeof(float), 32, f);
        std::fwrite(wv2_wl.data(), sizeof(float), 32, f);''',
'''        for (auto& row : wv1_wl) std::fwrite(row.data(), sizeof(float), VALUE_HIDDEN, f);
        std::fwrite(bv1_wl.data(), sizeof(float), VALUE_HIDDEN, f);
        std::fwrite(wv2_wl.data(), sizeof(float), VALUE_HIDDEN, f);''')

# A bilateral network must record evalNNUE from both perspectives. Using the
# one-accumulator compatibility helper evaluates (acc,acc), which is not the
# production value function and would poison a later self-play generation.
replace('tools/selfplay/selfplay.hpp',
'''        {
            AccumulatorQuant accMover = buildAccumulatorQuant(s, s.turn);
            double probMoverWins = (double)nnueWinProbQuant(accMover);
            evalWhiteProb = (s.turn == 0) ? probMoverWins : (1.0 - probMoverWins);
            if (mcTemperaturePly) {
                forwardPolicyQuant(accMover, policyOut);
            }
        }''',
'''        {
            AccPair evalPair = buildAccPairRoot(s, nullptr);
            double probMoverWins = (double)nnueWinProbQuant(evalPair.acc[s.turn], evalPair.acc[1 - s.turn]);
            evalWhiteProb = (s.turn == 0) ? probMoverWins : (1.0 - probMoverWins);
            if (mcTemperaturePly) {
                forwardPolicyQuant(evalPair.acc[s.turn], policyOut);
            }
        }''')

# Arena binary-sample output is always computed from qr_e1. Use both current
# qr_e1 accumulators so generated samples match production bilateral eval.
replace('tools/arena/arena.cpp',
'''template <typename Dummy = void>
auto tryNnueWinProbE1(const qr_e1::State& s, int mover, int)
    -> decltype(qr_e1::nnueWinProbQuant(qr_e1::buildAccumulatorQuant(s, mover))) {
    return qr_e1::nnueWinProbQuant(qr_e1::buildAccumulatorQuant(s, mover));
}
inline float tryNnueWinProbE1(const qr_e1::State&, int, ...) { return 0.5f; }''',
'''template <typename Dummy = void>
auto tryNnueWinProbE1(const qr_e1::State& s, int mover, int)
    -> decltype(qr_e1::nnueWinProbQuant(qr_e1::buildAccumulatorQuant(s, mover))) {
    auto own = qr_e1::buildAccumulatorQuant(s, mover);
    auto opp = qr_e1::buildAccumulatorQuant(s, 1 - mover);
    return qr_e1::nnueWinProbQuant(own, opp);
}
inline float tryNnueWinProbE1(const qr_e1::State&, int, ...) { return 0.5f; }''')

print('bilateral integration fixes applied')
