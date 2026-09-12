#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def replace(path, old, new, count=None):
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    n = text.count(old)
    if n == 0:
        raise RuntimeError(f"{path}: pattern not found: {old[:100]!r}")
    if count is not None and n != count:
        raise RuntimeError(f"{path}: expected {count} matches, found {n}: {old[:100]!r}")
    p.write_text(text.replace(old, new), encoding="utf-8")


def regex_replace(path, pattern, repl, count=1):
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    out, n = re.subn(pattern, repl, text, count=count, flags=re.S)
    if n != count:
        raise RuntimeError(f"{path}: expected {count} regex matches, found {n}: {pattern[:100]!r}")
    p.write_text(out, encoding="utf-8")


# -----------------------------------------------------------------------------
# C++ NNUE. Keep the sparse transformer shared. Concatenate the two perspective
# accumulators only at the head input. This preserves incremental updates.
# -----------------------------------------------------------------------------
replace(
    "src/nnue.hpp",
    "constexpr int HIDDEN = 256;\nconstexpr int POLICY_OUT",
    "constexpr int HIDDEN = 256;\nconstexpr int HEAD_INPUT = 2 * HIDDEN;  // own and opponent accumulator activations\nconstexpr int POLICY_OUT",
    1,
)
replace("src/nnue.hpp", "std::array<std::array<float, 32>, HIDDEN> wv1_wl;", "std::array<std::array<float, 32>, HEAD_INPUT> wv1_wl;", 1)
replace("src/nnue.hpp", "std::vector<std::array<float, HIDDEN>> wp;", "std::vector<std::array<float, HEAD_INPUT>> wp;", 1)
replace("src/nnue.hpp", "std::fread(row.data(), sizeof(float), HIDDEN, f) == (size_t)HIDDEN;\n        bp.assign", "std::fread(row.data(), sizeof(float), HEAD_INPUT, f) == (size_t)HEAD_INPUT;\n        bp.assign", 1)
replace("src/nnue.hpp", "std::fwrite(row.data(), sizeof(float), HIDDEN, f);\n        std::fwrite(bp.data()", "std::fwrite(row.data(), sizeof(float), HEAD_INPUT, f);\n        std::fwrite(bp.data()", 1)

regex_replace(
    "src/nnue.hpp",
    r"inline float forwardValueWL\(const Accumulator& acc\) \{.*?\n\}\n\ninline void forwardPolicy\(const Accumulator& acc, std::array<float, POLICY_OUT>& out\) \{.*?\n\}",
    '''inline float forwardValueWL(const Accumulator& own, const Accumulator& opp) {
    std::array<float, 32> h{};
    auto& W = weights();
    for (int i = 0; i < HIDDEN; i++) {
        float aOwn = screlu(own.v[i]);
        float aOpp = screlu(opp.v[i]);
        for (int j = 0; j < 32; j++) {
            h[j] += aOwn * W.wv1_wl[i][j];
            h[j] += aOpp * W.wv1_wl[HIDDEN + i][j];
        }
    }
    float out = W.bv2_wl;
    for (int j = 0; j < 32; j++) {
        float hj = clippedRelu(h[j] + W.bv1_wl[j]);
        out += hj * W.wv2_wl[j];
    }
    return out;
}

inline void forwardPolicy(const Accumulator& own, const Accumulator& opp,
                          std::array<float, POLICY_OUT>& out) {
    auto& W = weights();
    std::array<float, HEAD_INPUT> a;
    for (int i = 0; i < HIDDEN; i++) {
        a[i] = screlu(own.v[i]);
        a[HIDDEN + i] = screlu(opp.v[i]);
    }
    for (int o = 0; o < POLICY_OUT; o++) {
        float s = W.bp[o];
        auto& row = W.wp[o];
        for (int i = 0; i < HEAD_INPUT; i++) s += a[i] * row[i];
        out[o] = s;
    }
}''',
)

replace("src/nnue.hpp", "std::array<std::array<int8_t, 32>, HIDDEN> wv1_wl{};", "std::array<std::array<int8_t, 32>, HEAD_INPUT> wv1_wl{};", 1)
replace("src/nnue.hpp", "std::vector<std::array<int8_t, HIDDEN>> wp;", "std::vector<std::array<int8_t, HEAD_INPUT>> wp;", 1)
replace("src/nnue.hpp", "+ (long)HIDDEN * 32 * sizeof(int8_t)                       // wv1_wl", "+ (long)HEAD_INPUT * 32 * sizeof(int8_t)                   // wv1_wl", 1)
replace("src/nnue.hpp", "+ (long)POLICY_OUT * HIDDEN * sizeof(int8_t)               // wp", "+ (long)POLICY_OUT * HEAD_INPUT * sizeof(int8_t)           // wp", 1)
replace("src/nnue.hpp", "std::fread(row.data(), sizeof(int8_t), HIDDEN, f) == (size_t)HIDDEN;\n        bp.assign", "std::fread(row.data(), sizeof(int8_t), HEAD_INPUT, f) == (size_t)HEAD_INPUT;\n        bp.assign", 1)

regex_replace(
    "src/nnue.hpp",
    r"inline float forwardValueHeadQuant\(const AccumulatorQuant& acc,\n                                    const std::array<std::array<int8_t, 32>, HIDDEN>& wv1,.*?\n\}\n\n// forwardValueWLQuant",
    '''inline float forwardValueHeadQuant(const AccumulatorQuant& own,
                                    const AccumulatorQuant& opp,
                                    const std::array<std::array<int8_t, 32>, HEAD_INPUT>& wv1,
                                    const std::array<int32_t, 32>& bv1,
                                    const std::array<int8_t, 32>& wv2,
                                    int32_t bv2) {
    auto& W = weightsQuant();
    alignas(32) std::array<uint8_t, HEAD_INPUT> a;
    for (int i = 0; i < HIDDEN; i++) {
        a[i] = screluQuant(own.v[i], W.QA);
        a[HIDDEN + i] = screluQuant(opp.v[i], W.QA);
    }

    std::array<int32_t, 32> h{};
    const int8_t* wv1f = &wv1[0][0];
    for (int i = 0; i < HEAD_INPUT; i++) {
        const int32_t ai = a[i];
        const int8_t* row = wv1f + (size_t)i * 32;
        for (int j = 0; j < 32; j++) h[j] += ai * (int32_t)row[j];
    }
    int64_t QAQB = (int64_t)W.QA * (int64_t)W.QB;
    std::array<int32_t, 32> hj{};
    for (int j = 0; j < 32; j++) {
        int64_t hv = (int64_t)h[j] + (int64_t)bv1[j];
        if (hv < 0) hv = 0;
        if (hv > QAQB) hv = QAQB;
        hj[j] = (int32_t)hv;
    }
    int64_t out = bv2;
    for (int j = 0; j < 32; j++) out += (int64_t)hj[j] * (int64_t)wv2[j];
    int64_t denom = QAQB * (int64_t)W.QB;
    return (float)((double)out / (double)denom);
}

// forwardValueWLQuant''',
)
replace(
    "src/nnue.hpp",
    "inline float forwardValueWLQuant(const AccumulatorQuant& acc) {\n    auto& W = weightsQuant();\n    return forwardValueHeadQuant(acc, W.wv1_wl, W.bv1_wl, W.wv2_wl, W.bv2_wl);\n}",
    "inline float forwardValueWLQuant(const AccumulatorQuant& own, const AccumulatorQuant& opp) {\n    auto& W = weightsQuant();\n    return forwardValueHeadQuant(own, opp, W.wv1_wl, W.bv1_wl, W.wv2_wl, W.bv2_wl);\n}",
    1,
)
replace(
    "src/nnue.hpp",
    "inline float nnueWinProbQuant(const AccumulatorQuant& acc) {\n    float logit = forwardValueWLQuant(acc);",
    "inline float nnueWinProbQuant(const AccumulatorQuant& own, const AccumulatorQuant& opp) {\n    float logit = forwardValueWLQuant(own, opp);",
    1,
)
regex_replace(
    "src/nnue.hpp",
    r"inline void forwardPolicyQuant\(const AccumulatorQuant& acc, std::array<float, POLICY_OUT>& out\) \{.*?\n\}\n\n// =========================================================================",
    '''inline void forwardPolicyQuant(const AccumulatorQuant& own, const AccumulatorQuant& opp,
                               std::array<float, POLICY_OUT>& out) {
    auto& W = weightsQuant();
    alignas(32) std::array<uint8_t, HEAD_INPUT> a;
    for (int i = 0; i < HIDDEN; i++) {
        a[i] = screluQuant(own.v[i], W.QA);
        a[HIDDEN + i] = screluQuant(opp.v[i], W.QA);
    }

    const int outputs = (own.ownWallsLeftBucket == 0) ? (N * N) : POLICY_OUT;
    if (outputs < POLICY_OUT) out.fill(0.f);

    const double qaqb = (double)((int64_t)W.QA * (int64_t)W.QB);
    for (int o = 0; o < outputs; o++) {
        const int8_t* row = W.wp[o].data();
        const uint8_t* av = a.data();
        int32_t s = W.bp[o];
        for (int i = 0; i < HEAD_INPUT; i++)
            s += (int32_t)av[i] * (int32_t)row[i];
        out[o] = (float)((double)s / qaqb);
    }
}

// =========================================================================''',
)
replace(
    "src/nnue.hpp",
    "inline int nnueEvalInt(const AccPair& ap, int side) {\n    float logit = forwardValueWLQuant(ap.acc[side]);",
    "inline int nnueEvalInt(AccPair& ap, int side, PlayerPathCacheTable* xtable = nullptr) {\n    resolvePending(ap, side, xtable);\n    resolvePending(ap, 1 - side, xtable);\n    float logit = forwardValueWLQuant(ap.acc[side], ap.acc[1 - side]);",
    1,
)

# -----------------------------------------------------------------------------
# Search. Full-head inference needs both perspectives resolved at the node.
# -----------------------------------------------------------------------------
replace("src/search.hpp", "forwardPolicyQuant(local.acc[root.turn], pArr);", "forwardPolicyQuant(local.acc[root.turn], local.acc[1 - root.turn], pArr);", 1)
replace("src/search.hpp", "nnueEvalInt(*curAcc, side)", "nnueEvalInt(*curAcc, side, &xdistCache)")
replace("src/search.hpp", "nnueEvalInt(*curAcc, s.turn)", "nnueEvalInt(*curAcc, s.turn, &xdistCache)")
replace(
    "src/search.hpp",
    "            forwardPolicyQuant(curAcc->acc[side], policyArr);",
    "            resolvePending(*curAcc, 1 - side, &xdistCache);\n            forwardPolicyQuant(curAcc->acc[side], curAcc->acc[1 - side], policyArr);",
    1,
)

# -----------------------------------------------------------------------------
# Self-play diagnostics and MC temperature policy. Build both views explicitly.
# -----------------------------------------------------------------------------
replace(
    "tools/selfplay/selfplay.hpp",
    "            AccumulatorQuant accMover = buildAccumulatorQuant(s, s.turn);\n            double probMoverWins = (double)nnueWinProbQuant(accMover);",
    "            AccumulatorQuant accMover = buildAccumulatorQuant(s, s.turn);\n            AccumulatorQuant accOpp = buildAccumulatorQuant(s, 1 - s.turn);\n            double probMoverWins = (double)nnueWinProbQuant(accMover, accOpp);",
    1,
)
replace("tools/selfplay/selfplay.hpp", "forwardPolicyQuant(accMover, policyOut);", "forwardPolicyQuant(accMover, accOpp, policyOut);", 1)

# -----------------------------------------------------------------------------
# Trainer. Derive the opponent canonical view from each mover-canonical sample.
# The dataset format does not change.
# -----------------------------------------------------------------------------
replace("training/train_nnue.py", "HIDDEN = 256\nPOLICY_OUT", "HIDDEN = 256\nHEAD_INPUT = 2 * HIDDEN\nPOLICY_OUT", 1)
replace("training/train_nnue.py", "self.value1_wl = nn.Linear(HIDDEN, 32)", "self.value1_wl = nn.Linear(HEAD_INPUT, 32)", 1)
replace("training/train_nnue.py", "self.policy = nn.Linear(HIDDEN, POLICY_OUT)", "self.policy = nn.Linear(HEAD_INPUT, POLICY_OUT)", 1)
replace(
    "training/train_nnue.py",
    "    def forward(self, x: torch.Tensor):\n        acc = self.fc1(x)\n        a = screlu(acc)\n        h_wl = clipped_relu(self.value1_wl(a))",
    '''    @staticmethod
    def opponent_view(x: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x)
        y[:, 0:81] = x[:, 81:162].reshape(-1, N, N).flip(1).reshape(-1, 81)
        y[:, 81:162] = x[:, 0:81].reshape(-1, N, N).flip(1).reshape(-1, 81)
        y[:, 162:226] = x[:, 162:226].reshape(-1, WS, WS).flip(1).reshape(-1, 64)
        y[:, 226:290] = x[:, 226:290].reshape(-1, WS, WS).flip(1).reshape(-1, 64)
        y[:, 290:290 + DIST_BUCKETS] = x[:, 290 + DIST_BUCKETS:290 + 2 * DIST_BUCKETS]
        y[:, 290 + DIST_BUCKETS:290 + 2 * DIST_BUCKETS] = x[:, 290:290 + DIST_BUCKETS]
        wl = 290 + 2 * DIST_BUCKETS
        y[:, wl:wl + WALLS_LEFT_BUCKETS] = x[:, wl + WALLS_LEFT_BUCKETS:wl + 2 * WALLS_LEFT_BUCKETS]
        y[:, wl + WALLS_LEFT_BUCKETS:wl + 2 * WALLS_LEFT_BUCKETS] = x[:, wl:wl + WALLS_LEFT_BUCKETS]
        return y

    def forward(self, x: torch.Tensor):
        own = screlu(self.fc1(x))
        opp = screlu(self.fc1(self.opponent_view(x)))
        a = torch.cat((own, opp), dim=1)
        h_wl = clipped_relu(self.value1_wl(a))''',
    1,
)
replace("training/train_nnue.py", "return dict(num_features=NUM_FEATURES, hidden=HIDDEN, policy_out=POLICY_OUT,\n                qa=args.qa, qb=args.qb)", "return dict(num_features=NUM_FEATURES, hidden=HIDDEN, head_input=HEAD_INPUT, full_accumulator=True,\n                policy_out=POLICY_OUT, qa=args.qa, qb=args.qb)", 1)
replace("training/train_nnue.py", "elems_per_sample = NUM_FEATURES + HIDDEN + 2 * 32 + POLICY_OUT", "elems_per_sample = NUM_FEATURES + 2 * HIDDEN + HEAD_INPUT + 2 * 32 + POLICY_OUT", 1)
replace("training/train_nnue.py", "assert wv1_wl.shape == (HIDDEN, 32)", "assert wv1_wl.shape == (HEAD_INPUT, 32)", 1)
replace("training/train_nnue.py", "assert wp.shape == (POLICY_OUT, HIDDEN)", "assert wp.shape == (POLICY_OUT, HEAD_INPUT)", 1)

# -----------------------------------------------------------------------------
# Quantizer. The sparse transformer stays 354x256. Only head matrices widen.
# -----------------------------------------------------------------------------
replace("training/quantize_nnue.py", "HIDDEN = 256\nPOLICY_OUT", "HIDDEN = 256\nHEAD_INPUT = 2 * HIDDEN\nPOLICY_OUT", 1)
replace("training/quantize_nnue.py", "head_floats = HIDDEN * 32 + 32 + 32 + 1", "head_floats = HEAD_INPUT * 32 + 32 + 32 + 1", 1)
replace("training/quantize_nnue.py", "tail = POLICY_OUT * HIDDEN + POLICY_OUT", "tail = POLICY_OUT * HEAD_INPUT + POLICY_OUT", 1)
replace("training/quantize_nnue.py", "count=HIDDEN * 32).reshape(HIDDEN, 32)", "count=HEAD_INPUT * 32).reshape(HEAD_INPUT, 32)", 1)
replace("training/quantize_nnue.py", "count=POLICY_OUT * HIDDEN).reshape(POLICY_OUT, HIDDEN)", "count=POLICY_OUT * HEAD_INPUT).reshape(POLICY_OUT, HEAD_INPUT)", 1)

# -----------------------------------------------------------------------------
# Update the architecture record. Keep production weights untouched on main.
# -----------------------------------------------------------------------------
status = ROOT / "status.md"
with status.open("a", encoding="utf-8") as f:
    f.write("\n\n### 2026-09-12 - Full accumulator experiment\n\n")
    f.write("The `exp/full-accumulator` branch widens the NNUE head input from 256 to 512 activations. ")
    f.write("The first 256 activations use the side-to-move perspective. The second 256 activations use the opponent perspective. ")
    f.write("Both views use the same 354 to 256 sparse transformer. Search keeps both views incremental through `AccPair`. ")
    f.write("The value and policy heads now consume both views. The dataset layout does not change because the trainer derives the opponent view from each canonical sample. ")
    f.write("The weight layout changes. Therefore, this branch rejects the current production weights until a full-accumulator network is trained and quantized. ")
    f.write("Do not promote this architecture to `main` before float and quantized parity tests pass and an arena shows a strength gain over the current TD-S-Head production network.\n")

print("Full accumulator patch applied.")
