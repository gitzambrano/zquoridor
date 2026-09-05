#!/usr/bin/env python3
"""Convert bilateral value v2 into an exactly antisymmetric shared scorer v3."""
from pathlib import Path


def replace(path, old, new, count=1):
    p = Path(path)
    s = p.read_text()
    if new in s and old not in s:
        return
    if old not in s:
        raise SystemExit(f"anchor missing in {path}: {old[:120]!r}")
    p.write_text(s.replace(old, new, count))

# ---------------------------------------------------------------------------
# C++: same 512-input compute budget, 32-wide scorer evaluated in both orders.
# ---------------------------------------------------------------------------
p = 'src/nnue.hpp'
replace(p, 'constexpr int VALUE_HIDDEN = 64;', 'constexpr int VALUE_HIDDEN = 32;')
replace(p,
'''// Bilateral WL head: both perspective accumulators -> 64 -> 1.''',
'''// Antisymmetric bilateral WL scorer: shared F(own,opp), with
    // V(own,opp)=F(own,opp)-F(opp,own). Exact zero-sum symmetry by construction.''')

old_float = '''inline float forwardValueWL(const Accumulator& own, const Accumulator& opp) {
    std::array<float, VALUE_HIDDEN> h{};
    auto& W = weights();
    for (int i = 0; i < HIDDEN; i++) {
        const float aOwn = screlu(own.v[i]);
        const float aOpp = screlu(opp.v[i]);
        for (int j = 0; j < VALUE_HIDDEN; j++) {
            h[j] += aOwn * W.wv1_wl[i][j];
            h[j] += aOpp * W.wv1_wl[HIDDEN + i][j];
        }
    }
    float out = W.bv2_wl;
    for (int j = 0; j < VALUE_HIDDEN; j++) {
        const float hj = clippedRelu(h[j] + W.bv1_wl[j]);
        out += hj * W.wv2_wl[j];
    }
    return out;
}
'''
new_float = '''inline float forwardValueOrdered(const Accumulator& own, const Accumulator& opp) {
    std::array<float, VALUE_HIDDEN> h{};
    auto& W = weights();
    for (int i = 0; i < HIDDEN; i++) {
        const float aOwn = screlu(own.v[i]);
        const float aOpp = screlu(opp.v[i]);
        for (int j = 0; j < VALUE_HIDDEN; j++) {
            h[j] += aOwn * W.wv1_wl[i][j];
            h[j] += aOpp * W.wv1_wl[HIDDEN + i][j];
        }
    }
    float out = W.bv2_wl;
    for (int j = 0; j < VALUE_HIDDEN; j++) {
        const float hj = clippedRelu(h[j] + W.bv1_wl[j]);
        out += hj * W.wv2_wl[j];
    }
    return out;
}

inline float forwardValueWL(const Accumulator& own, const Accumulator& opp) {
    return forwardValueOrdered(own, opp) - forwardValueOrdered(opp, own);
}
'''
replace(p, old_float, new_float)

old_quant = '''inline float forwardValueHeadQuant(const AccumulatorQuant& own, const AccumulatorQuant& opp,
                                    const std::array<std::array<int8_t, VALUE_HIDDEN>, VALUE_INPUT>& wv1,
                                    const std::array<int32_t, VALUE_HIDDEN>& bv1,
                                    const std::array<int8_t, VALUE_HIDDEN>& wv2,
                                    int32_t bv2) {
    auto& W = weightsQuant();
    alignas(32) std::array<uint8_t, VALUE_INPUT> a{};
    for (int i = 0; i < HIDDEN; i++) {
        a[i] = screluQuant(own.v[i], W.QA);
        a[HIDDEN + i] = screluQuant(opp.v[i], W.QA);
    }

    std::array<int32_t, VALUE_HIDDEN> h{};
    const int8_t* wv1f = &wv1[0][0];
    for (int i = 0; i < VALUE_INPUT; i++) {
        const int32_t ai = a[i];
        const int8_t* row = wv1f + (size_t)i * VALUE_HIDDEN;
        for (int j = 0; j < VALUE_HIDDEN; j++) h[j] += ai * (int32_t)row[j];
    }
    const int64_t QAQB = (int64_t)W.QA * (int64_t)W.QB;
    std::array<int32_t, VALUE_HIDDEN> hj{};
    for (int j = 0; j < VALUE_HIDDEN; j++) {
        int64_t hv = (int64_t)h[j] + (int64_t)bv1[j];
        if (hv < 0) hv = 0;
        if (hv > QAQB) hv = QAQB;
        hj[j] = (int32_t)hv;
    }
    int64_t out = bv2;
    for (int j = 0; j < VALUE_HIDDEN; j++) out += (int64_t)hj[j] * (int64_t)wv2[j];
    const int64_t denom = QAQB * (int64_t)W.QB;
    return (float)((double)out / (double)denom);
}
'''
new_quant = '''inline int64_t forwardValueOrderedQuantRaw(const AccumulatorQuant& own, const AccumulatorQuant& opp,
                                             const std::array<std::array<int8_t, VALUE_HIDDEN>, VALUE_INPUT>& wv1,
                                             const std::array<int32_t, VALUE_HIDDEN>& bv1,
                                             const std::array<int8_t, VALUE_HIDDEN>& wv2,
                                             int32_t bv2) {
    auto& W = weightsQuant();
    alignas(32) std::array<uint8_t, VALUE_INPUT> a{};
    for (int i = 0; i < HIDDEN; i++) {
        a[i] = screluQuant(own.v[i], W.QA);
        a[HIDDEN + i] = screluQuant(opp.v[i], W.QA);
    }
    std::array<int32_t, VALUE_HIDDEN> h{};
    const int8_t* wv1f = &wv1[0][0];
    for (int i = 0; i < VALUE_INPUT; i++) {
        const int32_t ai = a[i];
        const int8_t* row = wv1f + (size_t)i * VALUE_HIDDEN;
        for (int j = 0; j < VALUE_HIDDEN; j++) h[j] += ai * (int32_t)row[j];
    }
    const int64_t QAQB = (int64_t)W.QA * (int64_t)W.QB;
    int64_t out = bv2;
    for (int j = 0; j < VALUE_HIDDEN; j++) {
        int64_t hv = (int64_t)h[j] + (int64_t)bv1[j];
        if (hv < 0) hv = 0;
        if (hv > QAQB) hv = QAQB;
        out += hv * (int64_t)wv2[j];
    }
    return out;
}

inline float forwardValueHeadQuant(const AccumulatorQuant& own, const AccumulatorQuant& opp,
                                    const std::array<std::array<int8_t, VALUE_HIDDEN>, VALUE_INPUT>& wv1,
                                    const std::array<int32_t, VALUE_HIDDEN>& bv1,
                                    const std::array<int8_t, VALUE_HIDDEN>& wv2,
                                    int32_t bv2) {
    auto& W = weightsQuant();
    const int64_t ab = forwardValueOrderedQuantRaw(own, opp, wv1, bv1, wv2, bv2);
    const int64_t ba = forwardValueOrderedQuantRaw(opp, own, wv1, bv1, wv2, bv2);
    const int64_t denom = (int64_t)W.QA * (int64_t)W.QB * (int64_t)W.QB;
    return (float)((double)(ab - ba) / (double)denom);
}
'''
replace(p, old_quant, new_quant)

# v3 should record bilateral evals correctly in future generations too.
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

# ---------------------------------------------------------------------------
# PyTorch: same shared ordered scorer, evaluated in both orders and subtracted.
# ---------------------------------------------------------------------------
p = 'training/train_nnue.py'
replace(p, 'VALUE_HIDDEN = 64', 'VALUE_HIDDEN = 32')
old_forward = '''    def forward(self, x: torch.Tensor, x_opp: torch.Tensor):
        a = screlu(self.fc1(x))
        a_opp = screlu(self.fc1(x_opp))
        h_wl = clipped_relu(self.value1_wl(torch.cat((a, a_opp), dim=-1)))
        value_wl = self.value2_wl(h_wl).squeeze(-1)
        policy_logits = self.policy(a)
        return value_wl, policy_logits
'''
new_forward = '''    def forward(self, x: torch.Tensor, x_opp: torch.Tensor):
        a = screlu(self.fc1(x))
        a_opp = screlu(self.fc1(x_opp))
        def ordered(lhs, rhs):
            h = clipped_relu(self.value1_wl(torch.cat((lhs, rhs), dim=-1)))
            return self.value2_wl(h).squeeze(-1)
        value_wl = ordered(a, a_opp) - ordered(a_opp, a)
        policy_logits = self.policy(a)
        return value_wl, policy_logits
'''
replace(p, old_forward, new_forward)

# Quantizer uses VALUE_HIDDEN for all offsets, so only dimension changes.
replace('training/quantize_nnue.py', 'VALUE_HIDDEN = 64', 'VALUE_HIDDEN = 32')

print('antisymmetric bilateral v3 applied')
