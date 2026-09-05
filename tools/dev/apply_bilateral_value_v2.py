#!/usr/bin/env python3
"""Apply the bilateral-value NNUE experiment.

The accumulator transform stays shared and incremental. The policy head keeps
using the mover accumulator. The WL head consumes both player perspectives:
[SCReLU(acc_side), SCReLU(acc_opponent)] -> 64 -> 1.
"""
from pathlib import Path


def repl(path: str, old: str, new: str, count: int = 1):
    p = Path(path)
    s = p.read_text()
    if new in s and old not in s:
        return
    if old not in s:
        raise SystemExit(f"patch anchor missing in {path}: {old[:100]!r}")
    s = s.replace(old, new, count)
    p.write_text(s)


# ---------------------------------------------------------------------------
# C++ architecture and quantized inference.
# ---------------------------------------------------------------------------
p = "src/nnue.hpp"
repl(p,
"constexpr int HIDDEN = 256;\nconstexpr int POLICY_OUT = N * N + WS * WS * 2;             // 81 destino peão + 128 muro = 209",
"constexpr int HIDDEN = 256;\nconstexpr int VALUE_INPUT = HIDDEN * 2;\nconstexpr int VALUE_HIDDEN = 64;\nconstexpr int POLICY_OUT = N * N + WS * WS * 2;             // 81 destino peão + 128 muro = 209")

repl(p,
"    // cabeça de RESULTADO (WL): HIDDEN -> 32 -> 1 (logit único, sem empate)\n    std::array<std::array<float, 32>, HIDDEN> wv1_wl;\n    std::array<float, 32> bv1_wl{};\n    std::array<float, 32> wv2_wl{};",
"    // Bilateral WL head: both perspective accumulators -> 64 -> 1.\n    std::array<std::array<float, VALUE_HIDDEN>, VALUE_INPUT> wv1_wl;\n    std::array<float, VALUE_HIDDEN> bv1_wl{};\n    std::array<float, VALUE_HIDDEN> wv2_wl{};")

repl(p,
"        for (auto& row : wv1_wl) for (auto& v : row) v = d1(rng);\n        for (auto& v : wv2_wl) v = d1(rng);",
"        for (auto& row : wv1_wl) for (auto& v : row) v = d1(rng);\n        for (auto& v : wv2_wl) v = d1(rng);")

repl(p,
"        for (auto& row : wv1_wl) ok = ok && std::fread(row.data(), sizeof(float), 32, f) == 32;\n        ok = ok && std::fread(bv1_wl.data(), sizeof(float), 32, f) == 32;\n        ok = ok && std::fread(wv2_wl.data(), sizeof(float), 32, f) == 32;",
"        for (auto& row : wv1_wl) ok = ok && std::fread(row.data(), sizeof(float), VALUE_HIDDEN, f) == (size_t)VALUE_HIDDEN;\n        ok = ok && std::fread(bv1_wl.data(), sizeof(float), VALUE_HIDDEN, f) == (size_t)VALUE_HIDDEN;\n        ok = ok && std::fread(wv2_wl.data(), sizeof(float), VALUE_HIDDEN, f) == (size_t)VALUE_HIDDEN;")

old_float = '''inline float forwardValueWL(const Accumulator& acc) {
    std::array<float, 32> h{};
    auto& W = weights();
    for (int i = 0; i < HIDDEN; i++) {
        float a = screlu(acc.v[i]);
        for (int j = 0; j < 32; j++) h[j] += a * W.wv1_wl[i][j];
    }
    float out = W.bv2_wl;
    for (int j = 0; j < 32; j++) {
        float hj = clippedRelu(h[j] + W.bv1_wl[j]);
        out += hj * W.wv2_wl[j];
    }
    return out;
}
'''
new_float = '''inline float forwardValueWL(const Accumulator& own, const Accumulator& opp) {
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

// Compatibility helper for tools that probe one accumulator only. Production
// search uses the bilateral overload through nnueEvalInt(AccPair,...).
inline float forwardValueWL(const Accumulator& acc) { return forwardValueWL(acc, acc); }
'''
repl(p, old_float, new_float)

repl(p,
"    std::array<std::array<int8_t, 32>, HIDDEN> wv1_wl{}; // escala QB\n    std::array<int32_t, 32> bv1_wl{};                      // escala QA*QB\n    std::array<int8_t, 32> wv2_wl{};                       // escala QB",
"    std::array<std::array<int8_t, VALUE_HIDDEN>, VALUE_INPUT> wv1_wl{}; // escala QB\n    std::array<int32_t, VALUE_HIDDEN> bv1_wl{};                      // escala QA*QB\n    std::array<int8_t, VALUE_HIDDEN> wv2_wl{};                       // escala QB")

repl(p,
"            + (long)HIDDEN * 32 * sizeof(int8_t)                       // wv1_wl\n            + 32 * sizeof(int32_t)                                     // bv1_wl\n            + 32 * sizeof(int8_t)                                      // wv2_wl",
"            + (long)VALUE_INPUT * VALUE_HIDDEN * sizeof(int8_t)        // wv1_wl\n            + VALUE_HIDDEN * sizeof(int32_t)                            // bv1_wl\n            + VALUE_HIDDEN * sizeof(int8_t)                             // wv2_wl")

repl(p,
"        for (auto& row : wv1_wl) ok = ok && std::fread(row.data(), sizeof(int8_t), 32, f) == 32;\n        ok = ok && std::fread(bv1_wl.data(), sizeof(int32_t), 32, f) == 32;\n        ok = ok && std::fread(wv2_wl.data(), sizeof(int8_t), 32, f) == 32;",
"        for (auto& row : wv1_wl) ok = ok && std::fread(row.data(), sizeof(int8_t), VALUE_HIDDEN, f) == (size_t)VALUE_HIDDEN;\n        ok = ok && std::fread(bv1_wl.data(), sizeof(int32_t), VALUE_HIDDEN, f) == (size_t)VALUE_HIDDEN;\n        ok = ok && std::fread(wv2_wl.data(), sizeof(int8_t), VALUE_HIDDEN, f) == (size_t)VALUE_HIDDEN;")

start = Path(p).read_text().index("inline float forwardValueHeadQuant(")
end = Path(p).read_text().index("// forwardValueWLQuant", start)
s = Path(p).read_text()
new_quant_head = '''inline float forwardValueHeadQuant(const AccumulatorQuant& own, const AccumulatorQuant& opp,
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
Path(p).write_text(s[:start] + new_quant_head + s[end:])

repl(p,
'''inline float forwardValueWLQuant(const AccumulatorQuant& acc) {
    auto& W = weightsQuant();
    return forwardValueHeadQuant(acc, W.wv1_wl, W.bv1_wl, W.wv2_wl, W.bv2_wl);
}
''',
'''inline float forwardValueWLQuant(const AccumulatorQuant& own, const AccumulatorQuant& opp) {
    auto& W = weightsQuant();
    return forwardValueHeadQuant(own, opp, W.wv1_wl, W.bv1_wl, W.wv2_wl, W.bv2_wl);
}
inline float forwardValueWLQuant(const AccumulatorQuant& acc) { return forwardValueWLQuant(acc, acc); }
''')

repl(p,
'''inline float nnueWinProbQuant(const AccumulatorQuant& acc) {
    float logit = forwardValueWLQuant(acc);
    return 1.0f / (1.0f + std::exp(-logit));
}
''',
'''inline float nnueWinProbQuant(const AccumulatorQuant& own, const AccumulatorQuant& opp) {
    float logit = forwardValueWLQuant(own, opp);
    return 1.0f / (1.0f + std::exp(-logit));
}
inline float nnueWinProbQuant(const AccumulatorQuant& acc) { return nnueWinProbQuant(acc, acc); }
''')

old_child = '''    // Perspectiva de quem joga em `child` -- precisa estar pronta já.
    resolvePending(parent, opp, xtable);
    child.acc[opp] = parent.acc[opp];
    child.pending[opp] = false;
    updateAccumulatorForMoveQuant(child.acc[opp], /*viewerIsMover=*/false, before, m, xtable);
    // Perspectiva de quem jogou -- adia. parent.acc[mover] já está
    // garantidamente resolvida (invariante da struct: é a perspectiva de
    // s.turn no nó de `parent`, sempre eager).
    child.acc[mover] = parent.acc[mover];
    child.pending[mover] = true;
    child.pendBefore[mover] = before;
    child.pendMove[mover] = m;
    child.pendViewerIsMover[mover] = true;
'''
new_child = '''    // Bilateral value inference needs both perspectives at every evaluated
    // leaf. Keep both accumulators eager in this experiment; the arena also
    // measures the NPS cost of removing the old one-perspective lazy update.
    resolvePending(parent, opp, xtable);
    resolvePending(parent, mover, xtable);
    child.acc[opp] = parent.acc[opp];
    child.acc[mover] = parent.acc[mover];
    child.pending[opp] = false;
    child.pending[mover] = false;
    updateAccumulatorForMoveQuant(child.acc[opp], /*viewerIsMover=*/false, before, m, xtable);
    updateAccumulatorForMoveQuant(child.acc[mover], /*viewerIsMover=*/true, before, m, xtable);
'''
repl(p, old_child, new_child)

repl(p,
'''inline int nnueEvalInt(const AccPair& ap, int side) {
    float logit = forwardValueWLQuant(ap.acc[side]);
    return (int)std::lround(logit * (float)NNUE_EVAL_SCALE);
}
''',
'''inline int nnueEvalInt(const AccPair& ap, int side) {
    float logit = forwardValueWLQuant(ap.acc[side], ap.acc[1 - side]);
    return (int)std::lround(logit * (float)NNUE_EVAL_SCALE);
}
''')

# ---------------------------------------------------------------------------
# PyTorch trainer: shared transformer, bilateral WL head, unilateral policy.
# ---------------------------------------------------------------------------
p = "training/train_nnue.py"
repl(p,
"HIDDEN = 256\nPOLICY_OUT = N * N + WS * WS * 2                                # 209",
"HIDDEN = 256\nVALUE_INPUT = HIDDEN * 2\nVALUE_HIDDEN = 64\nPOLICY_OUT = N * N + WS * WS * 2                                # 209")

repl(p,
'''        self.fc1 = nn.Linear(NUM_FEATURES, HIDDEN)
        self.value1_wl = nn.Linear(HIDDEN, 32)
        self.value2_wl = nn.Linear(32, 1)
        self.policy = nn.Linear(HIDDEN, POLICY_OUT)

    def forward(self, x: torch.Tensor):
        acc = self.fc1(x)
        a = screlu(acc)
        h_wl = clipped_relu(self.value1_wl(a))
        value_wl = self.value2_wl(h_wl).squeeze(-1)
        policy_logits = self.policy(a)
        return value_wl, policy_logits
''',
'''        self.fc1 = nn.Linear(NUM_FEATURES, HIDDEN)
        self.value1_wl = nn.Linear(VALUE_INPUT, VALUE_HIDDEN)
        self.value2_wl = nn.Linear(VALUE_HIDDEN, 1)
        self.policy = nn.Linear(HIDDEN, POLICY_OUT)

    def forward(self, x: torch.Tensor, x_opp: torch.Tensor):
        a = screlu(self.fc1(x))
        a_opp = screlu(self.fc1(x_opp))
        h_wl = clipped_relu(self.value1_wl(torch.cat((a, a_opp), dim=-1)))
        value_wl = self.value2_wl(h_wl).squeeze(-1)
        policy_logits = self.policy(a)
        return value_wl, policy_logits
''')

anchor = '''    x[np.arange(n), wl_base + own_wl_bucket] = 1.0
    x[np.arange(n), wl_base + WALLS_LEFT_BUCKETS + opp_wl_bucket] = 1.0

    # wl_target'''
insert = '''    x[np.arange(n), wl_base + own_wl_bucket] = 1.0
    x[np.arange(n), wl_base + WALLS_LEFT_BUCKETS + opp_wl_bucket] = 1.0

    # Second canonical perspective. The sample is already canonical for the
    # mover, so opponent perspective is a vertical board mirror plus an
    # own/opp swap. No additional information is stored in the shard.
    x_opp = np.zeros_like(x)
    own_p = chunk["own_pawn"].astype(np.int64)
    opp_p = chunk["opp_pawn"].astype(np.int64)
    mir_own_p = (N - 1 - (opp_p // N)) * N + (opp_p % N)
    mir_opp_p = (N - 1 - (own_p // N)) * N + (own_p % N)
    x_opp[np.arange(n), mir_own_p] = 1.0
    x_opp[np.arange(n), 81 + mir_opp_p] = 1.0
    x_opp[:, 162:162 + 64] = bits_h.reshape(n, WS, WS)[:, ::-1, :].reshape(n, 64)
    x_opp[:, 162 + 64:162 + 128] = bits_v.reshape(n, WS, WS)[:, ::-1, :].reshape(n, 64)
    x_opp[np.arange(n), 290 + opp_bucket] = 1.0
    x_opp[np.arange(n), 290 + DIST_BUCKETS + own_bucket] = 1.0
    x_opp[np.arange(n), wl_base + opp_wl_bucket] = 1.0
    x_opp[np.arange(n), wl_base + WALLS_LEFT_BUCKETS + own_wl_bucket] = 1.0

    # wl_target'''
repl(p, anchor, insert)

repl(p,
'''        "x": torch.from_numpy(x).to(device, non_blocking=True),
        "wl_target":''',
'''        "x": torch.from_numpy(x).to(device, non_blocking=True),
        "x_opp": torch.from_numpy(x_opp).to(device, non_blocking=True),
        "wl_target":''')

repl(p,
'''                "x": t["x"][start:end],
                "wl_target":''',
'''                "x": t["x"][start:end],
                "x_opp": t["x_opp"][start:end],
                "wl_target":''')

# Both train and validation loops use the same call spelling.
path = Path(p)
s = path.read_text()
s = s.replace('value_wl, policy_logits = model(t["x"])', 'value_wl, policy_logits = model(t["x"], t["x_opp"])')
path.write_text(s)

repl(p,
"    assert wv1_wl.shape == (HIDDEN, 32)",
"    assert wv1_wl.shape == (VALUE_INPUT, VALUE_HIDDEN)")

# Include value dimensions in checkpoint fingerprint so old states cannot resume.
repl(p,
'''    return dict(num_features=NUM_FEATURES, hidden=HIDDEN, policy_out=POLICY_OUT,
                qa=args.qa, qb=args.qb)''',
'''    return dict(num_features=NUM_FEATURES, hidden=HIDDEN, value_input=VALUE_INPUT,
                value_hidden=VALUE_HIDDEN, policy_out=POLICY_OUT, qa=args.qa, qb=args.qb)''')

# ---------------------------------------------------------------------------
# Quantizer: new value-head dimensions, policy unchanged.
# ---------------------------------------------------------------------------
p = "training/quantize_nnue.py"
repl(p,
"HIDDEN = 256\nPOLICY_OUT = N * N + WS * WS * 2  # 209",
"HIDDEN = 256\nVALUE_INPUT = HIDDEN * 2\nVALUE_HIDDEN = 64\nPOLICY_OUT = N * N + WS * WS * 2  # 209")

repl(p,
'''    head_floats = HIDDEN * 32 + 32 + 32 + 1  # wv1 + bv1 + wv2 + bv2 de UMA cabeça 256->32->1
    base = NUM_FEATURES * HIDDEN + HIDDEN + head_floats
    tail = POLICY_OUT * HIDDEN + POLICY_OUT
    expected_new = base + tail                # sem cabeça auxiliar (2026-08+)
    expected_old = base + head_floats + tail   # com cabeça auxiliar (formato antigo)
    expected_new_bytes = expected_new * 4
    expected_old_bytes = expected_old * 4
    actual_bytes = os.path.getsize(path)
    if actual_bytes not in (expected_new_bytes, expected_old_bytes):
''',
'''    head_floats = VALUE_INPUT * VALUE_HIDDEN + VALUE_HIDDEN + VALUE_HIDDEN + 1
    base = NUM_FEATURES * HIDDEN + HIDDEN + head_floats
    tail = POLICY_OUT * HIDDEN + POLICY_OUT
    expected_new = base + tail
    expected_new_bytes = expected_new * 4
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_new_bytes:
''')

# Remove old-format specific wording/branch by replacing the whole error + flag anchor.
s = Path(p).read_text()
s = s.replace('f"({expected_new_bytes} bytes / {expected_new} floats, sem cabeça auxiliar) nem com "\n            f"o formato antigo ({expected_old_bytes} bytes / {expected_old} floats, com cabeça "\n            f"auxiliar) para NUM_FEATURES={NUM_FEATURES} -- verifique se o arquivo foi gerado "',
              'f"({expected_new_bytes} bytes / {expected_new} floats) para NUM_FEATURES={NUM_FEATURES} -- verifique se o arquivo foi gerado "')
s = s.replace('    is_old_format = (actual_bytes == expected_old_bytes)\n', '')
s = s.replace('''        def read_head():
            wv1 = np.fromfile(f, dtype="<f4", count=HIDDEN * 32).reshape(HIDDEN, 32)
            bv1 = np.fromfile(f, dtype="<f4", count=32)
            wv2 = np.fromfile(f, dtype="<f4", count=32)
            bv2 = np.fromfile(f, dtype="<f4", count=1)[0]
            return wv1, bv1, wv2, bv2

        wv1_wl, bv1_wl, wv2_wl, bv2_wl = read_head()
        if is_old_format:
            read_head()  # cabeça auxiliar antiga -- lê pra avançar o cursor, descarta
            print(f"'{path}' está no formato antigo (com cabeça auxiliar) -- "
                  f"cabeça auxiliar ignorada, só w1/b1/cabeça WL/policy são quantizados.")
''', '''        wv1_wl = np.fromfile(f, dtype="<f4", count=VALUE_INPUT * VALUE_HIDDEN).reshape(VALUE_INPUT, VALUE_HIDDEN)
        bv1_wl = np.fromfile(f, dtype="<f4", count=VALUE_HIDDEN)
        wv2_wl = np.fromfile(f, dtype="<f4", count=VALUE_HIDDEN)
        bv2_wl = np.fromfile(f, dtype="<f4", count=1)[0]
''')
Path(p).write_text(s)

print("bilateral value v2 patch applied")
