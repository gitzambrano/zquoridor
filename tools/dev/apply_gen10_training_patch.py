#!/usr/bin/env python3
"""Apply the Gen10 visit-policy and wall-poor value-training patch.

This is an idempotent repository migration helper used by the remote
experiment workflow. It exists so the changes can be built and tested on a
real GitHub runner before the helper is removed in branch cleanup.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def edit(path, old, new):
    p = ROOT / path
    text = p.read_text(encoding="utf-8-sig")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"anchor not found in {path}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1) Self-play record: append sparse top-8 root visit distribution.
edit(
    "tools/selfplay/selfplay.hpp",
    "    int16_t  oppCatTotal;   // soma do calor de corredor do oponente\n};\n#pragma pack(pop)\nstatic_assert(sizeof(TrainingSample) == 32,\n    \"TrainingSample precisa ficar packed/sem padding -- o layout é lido direto por numpy no treino\");",
    "    int16_t  oppCatTotal;   // soma do calor de corredor do oponente\n"
    "    uint16_t policyTopIdx[8];   // top-8 policy indices by MCAB root visits, mirrored\n"
    "    uint16_t policyTopProb[8];  // top-8 visit probabilities, normalized to sum 65535\n"
    "};\n#pragma pack(pop)\nstatic_assert(sizeof(TrainingSample) == 64,\n"
    "    \"TrainingSample must stay packed; Python reads the 64-byte layout directly\");"
)
edit(
    "tools/selfplay/selfplay.hpp",
    "        TrainingSample rec;\n        int mover = s.turn, opp = 1 - s.turn;",
    "        TrainingSample rec{};\n        int mover = s.turn, opp = 1 - s.turn;"
)
edit(
    "tools/selfplay/selfplay.hpp",
    "        rec.policyTarget = moveToPolicyIndex(mirrorMoveForPerspective(chosen, mover));\n        rec.ownDist =",
    "        rec.policyTarget = moveToPolicyIndex(mirrorMoveForPerspective(chosen, mover));\n"
    "        // Preserve the information produced by the MCAB teacher instead of\n"
    "        // collapsing ~20k simulations to one one-hot move label. A stale tree\n"
    "        // is rejected by hash, which also masks epsilon/shallow and exact\n"
    "        // no-wall fallback plies from the visit-policy loss.\n"
    "        if (cfg.mcabParams.enabled) {\n"
    "            const auto* rootNode = mcabRunner.search.rootNodeForInspection();\n"
    "            if (rootNode && rootNode->expanded && rootNode->state.hash == s.hash) {\n"
    "                struct VisitEdge { size_t edge; float visits; };\n"
    "                std::vector<VisitEdge> ranked;\n"
    "                size_t nm = std::min(rootNode->moves.size(), rootNode->N.size());\n"
    "                ranked.reserve(nm);\n"
    "                for (size_t i = 0; i < nm; ++i)\n"
    "                    if (rootNode->N[i] > 0.f) ranked.push_back({i, rootNode->N[i]});\n"
    "                std::stable_sort(ranked.begin(), ranked.end(),\n"
    "                    [](const VisitEdge& a, const VisitEdge& b) { return a.visits > b.visits; });\n"
    "                if (ranked.size() > 8) ranked.resize(8);\n"
    "                double sumVisits = 0.0;\n"
    "                for (const auto& e : ranked) sumVisits += (double)e.visits;\n"
    "                uint32_t assigned = 0;\n"
    "                for (size_t j = 0; j < ranked.size() && sumVisits > 0.0; ++j) {\n"
    "                    const Move& vm = rootNode->moves[ranked[j].edge];\n"
    "                    rec.policyTopIdx[j] = moveToPolicyIndex(mirrorMoveForPerspective(vm, mover));\n"
    "                    uint16_t q = (uint16_t)std::floor(65535.0 * ranked[j].visits / sumVisits);\n"
    "                    rec.policyTopProb[j] = q;\n"
    "                    assigned += q;\n"
    "                }\n"
    "                if (!ranked.empty() && sumVisits > 0.0)\n"
    "                    rec.policyTopProb[0] = (uint16_t)(rec.policyTopProb[0] + (65535u - assigned));\n"
    "            }\n"
    "        }\n"
    "        rec.ownDist ="
)

# 2) Arena keeps byte-for-byte compatible records. Arena positions have no
# teacher visit target, so the appended arrays stay zero-initialized.
edit(
    "tools/arena/arena.cpp",
    "    int16_t  oppCatTotal;\n};\n#pragma pack(pop)\nstatic_assert(sizeof(TrainingSample) == 32, \"TrainingSample precisa ficar packed\");",
    "    int16_t  oppCatTotal;\n"
    "    uint16_t policyTopIdx[8];\n"
    "    uint16_t policyTopProb[8];\n"
    "};\n#pragma pack(pop)\nstatic_assert(sizeof(TrainingSample) == 64, \"TrainingSample must stay packed\");"
)
# Zero initialization at all common declaration spellings.
p = ROOT / "tools/arena/arena.cpp"
t = p.read_text(encoding="utf-8-sig")
t = t.replace("TrainingSample rec;", "TrainingSample rec{};")
p.write_text(t, encoding="utf-8")

# 3) Python reader: current format becomes additive 64-byte V3; keep V2 32.
p = ROOT / "training/read_selfplay.py"
t = p.read_text(encoding="utf-8-sig")
if "SAMPLE_DTYPE_V2" not in t:
    start = t.index("# --- formato atual (32 bytes/amostra, 2026-08+) ------------------------------")
    end = t.index("# Campos comuns entre SAMPLE_DTYPE_LEGACY", start)
    block = t[start:end]
    block = block.replace("SAMPLE_DTYPE = np.dtype([", "SAMPLE_DTYPE_V2 = np.dtype([", 1)
    block = block.replace("assert SAMPLE_DTYPE.itemsize == 32", "assert SAMPLE_DTYPE_V2.itemsize == 32", 1)
    block += "\n# --- formato V3 (64 bytes/amostra): V2 + top-8 MCAB visit policy -----------\n"
    block += "SAMPLE_DTYPE = np.dtype(SAMPLE_DTYPE_V2.descr + [\n"
    block += "    (\"policy_top_idx\",  \"<u2\", (8,)),\n"
    block += "    (\"policy_top_prob\", \"<u2\", (8,)),\n"
    block += "])\nassert SAMPLE_DTYPE.itemsize == 64\n\n"
    t = t[:start] + block + t[end:]

    marker = "def _upcast_legacy(arr27: np.ndarray) -> np.ndarray:\n"
    idx = t.index(marker)
    # insert V2 upcaster after legacy function by anchoring next section
    nextsec = t.index("# Bucket one-hot", idx)
    up = "\ndef _upcast_v2(arr32: np.ndarray) -> np.ndarray:\n"
    up += "    \"\"\"Upcast the 32-byte canonical format to the 64-byte V3 layout.\"\"\"\n"
    up += "    out = np.zeros(len(arr32), dtype=SAMPLE_DTYPE)\n"
    up += "    for field in SAMPLE_DTYPE_V2.names:\n        out[field] = arr32[field]\n"
    up += "    return out\n\n"
    t = t[:nextsec] + up + t[nextsec:]

    # Replace format detector/sniffer/load block wholesale.
    d0 = t.index("def _detect_format_by_size")
    d1 = t.index("def expand_data_paths", d0)
    detector = '''def _valid_probe(arr, v3=False):
    if len(arr) == 0:
        return False
    if int(arr["own_pawn"].max()) > 80 or int(arr["policy_target"].max()) > 208:
        return False
    if not v3:
        return True
    probs = arr["policy_top_prob"].astype(np.uint64)
    sums = probs.sum(axis=1)
    if not np.all((sums == 0) | (sums == 65535)):
        return False
    idx = arr["policy_top_idx"]
    return not np.any((probs > 0) & (idx > 208))


def _detect_format(path: str):
    size = os.path.getsize(path)
    candidates = []
    if size % 64 == 0:
        candidates.append((SAMPLE_DTYPE, 64, True))
    if size % 32 == 0:
        candidates.append((SAMPLE_DTYPE_V2, 32, False))
    if size % 27 == 0:
        candidates.append((SAMPLE_DTYPE_LEGACY, 27, False))
    for dtype, itemsize, is_v3 in candidates:
        n = size // itemsize
        if n == 0:
            continue
        probe = np.memmap(path, dtype=dtype, mode="r", shape=(n,))[:min(n, 64)]
        if _valid_probe(probe, is_v3):
            return dtype, itemsize
    raise ValueError(f"self-play file has no supported 27/32/64-byte layout: {path}")


def load_selfplay(path: str, quiet: bool = False) -> np.ndarray:
    """Load 27, 32, or 64-byte self-play and return the 64-byte V3 dtype."""
    dtype, itemsize = _detect_format(path)
    n = os.path.getsize(path) // itemsize
    if n == 0:
        return np.empty(0, dtype=SAMPLE_DTYPE)
    raw = np.memmap(path, dtype=dtype, mode="r", shape=(n,))
    if itemsize == 27:
        if not quiet:
            sys.stderr.write(f"[read_selfplay] legacy 27-byte format: {path}\\n")
        return _upcast_legacy(raw)
    if itemsize == 32:
        return _upcast_v2(raw)
    return raw


def _is_legacy_file(path: str) -> bool:
    if os.path.getsize(path) == 0:
        return False
    _, itemsize = _detect_format(path)
    return itemsize == 27


'''
    t = t[:d0] + detector + t[d1:]
    # update obvious prose and 32-byte wording without relying on it for behavior
    t = t.replace("sempre devolve SAMPLE_DTYPE (32", "sempre devolve SAMPLE_DTYPE (64")
    t = t.replace("upcast automatico para 32 bytes aplicado", "upcast automatico para 64 bytes aplicado")
    p.write_text(t, encoding="utf-8")

# 4) Trainer: soft sparse visit target + targeted WL weighting.
p = ROOT / "training/train_nnue.py"
t = p.read_text(encoding="utf-8-sig")
if "policy_top_prob" not in t:
    t = t.replace(
        'def to_chunk_tensors(chunk: np.ndarray, k_chunk: np.ndarray, device, pw_chunk: np.ndarray = None,\n                     pr_chunk: np.ndarray = None, wl_gamma: float = 1.0):',
        'def to_chunk_tensors(chunk: np.ndarray, k_chunk: np.ndarray, device, pw_chunk: np.ndarray = None,\n                     pr_chunk: np.ndarray = None, wl_gamma: float = 1.0, wall_poor_wl_weight: float = 1.0):'
    )
    t = t.replace(
        '    return {\n        "x": torch.from_numpy(x).to(device, non_blocking=True),\n        "wl_target": torch.from_numpy(wl_target).to(device, non_blocking=True),\n        "policy_target": torch.from_numpy(chunk["policy_target"].astype(np.int64)).to(device, non_blocking=True),\n        "policy_w": torch.from_numpy(pw).to(device, non_blocking=True),\n    }',
        '    risk = ((chunk["walls_left_own"] == 0) & (chunk["walls_left_opp"] >= 4)) | \\\n           ((chunk["walls_left_own"] <= 2) & ((chunk["walls_left_opp"] - chunk["walls_left_own"]) >= 4) & \\\n            ((chunk["own_dist"].astype(np.int16) + 2) <= chunk["opp_dist"].astype(np.int16)))\n'
        '    wl_w = np.where(risk, np.float32(wall_poor_wl_weight), np.float32(1.0)).astype(np.float32)\n'
        '    top_idx = chunk["policy_top_idx"].astype(np.int64)\n'
        '    top_prob = chunk["policy_top_prob"].astype(np.float32) / np.float32(65535.0)\n'
        '    visit_valid = (chunk["policy_top_prob"][:, 0] > 0).astype(np.float32)\n'
        '    pw = pw * visit_valid\n'
        '    return {\n'
        '        "x": torch.from_numpy(x).to(device, non_blocking=True),\n'
        '        "wl_target": torch.from_numpy(wl_target).to(device, non_blocking=True),\n'
        '        "wl_w": torch.from_numpy(wl_w).to(device, non_blocking=True),\n'
        '        "policy_target": torch.from_numpy(chunk["policy_target"].astype(np.int64)).to(device, non_blocking=True),\n'
        '        "policy_top_idx": torch.from_numpy(top_idx).to(device, non_blocking=True),\n'
        '        "policy_top_prob": torch.from_numpy(top_prob).to(device, non_blocking=True),\n'
        '        "policy_w": torch.from_numpy(pw).to(device, non_blocking=True),\n'
        '    }'
    )
    t = t.replace(
        'def iter_gpu_batches(chunk: np.ndarray, k_chunk: np.ndarray, device, batch_size: int, gpu_chunk_size: int,\n                     pw_chunk: np.ndarray = None, pr_chunk: np.ndarray = None, wl_gamma: float = 1.0):',
        'def iter_gpu_batches(chunk: np.ndarray, k_chunk: np.ndarray, device, batch_size: int, gpu_chunk_size: int,\n                     pw_chunk: np.ndarray = None, pr_chunk: np.ndarray = None, wl_gamma: float = 1.0,\n                     wall_poor_wl_weight: float = 1.0):'
    )
    t = t.replace(
        '        t = to_chunk_tensors(sub, k_sub, device, pw_sub, pr_sub, wl_gamma)',
        '        t = to_chunk_tensors(sub, k_sub, device, pw_sub, pr_sub, wl_gamma, wall_poor_wl_weight)'
    )
    t = t.replace(
        '                "wl_target": t["wl_target"][start:end],\n                "policy_target": t["policy_target"][start:end],\n                "policy_w": t["policy_w"][start:end],',
        '                "wl_target": t["wl_target"][start:end],\n'
        '                "wl_w": t["wl_w"][start:end],\n'
        '                "policy_target": t["policy_target"][start:end],\n'
        '                "policy_top_idx": t["policy_top_idx"][start:end],\n'
        '                "policy_top_prob": t["policy_top_prob"][start:end],\n'
        '                "policy_w": t["policy_w"][start:end],'
    )
    old_loss = '''def weighted_policy_loss(policy_logits, policy_t, pw):
    """Cross entropy of the policy head, weighted for each sample.

    A weight of 0 removes the sample from the policy loss. The same sample
    still trains the value head, because this weight never touches
    `loss_outcome`. This is what excludes the montecarlo opening plies,
    whose recorded policy target is a move that the policy head itself
    sampled, with no search behind it.

    When every weight in the batch is 0, the numerator is also 0, so the
    term is 0 and adds no gradient. The clamp only avoids a division by
    zero, it never changes a real value.
    """
    per = F.cross_entropy(policy_logits, policy_t, reduction="none")
    return (per * pw).sum() / pw.sum().clamp(min=1e-8)
'''
    new_loss = '''def weighted_policy_loss(policy_logits, policy_top_idx, policy_top_prob, pw):
    """Sparse soft cross entropy against the MCAB top-8 visit distribution."""
    logp = F.log_softmax(policy_logits, dim=-1)
    gathered = torch.gather(logp, 1, policy_top_idx)
    per = -(gathered * policy_top_prob).sum(dim=-1)
    return (per * pw).sum() / pw.sum().clamp(min=1e-8)


def weighted_outcome_loss(value_logits, target, weight):
    per = F.binary_cross_entropy_with_logits(value_logits, target, reduction="none")
    return (per * weight).sum() / weight.sum().clamp(min=1e-8)
'''
    if old_loss not in t:
        raise SystemExit("weighted_policy_loss anchor missing")
    t = t.replace(old_loss, new_loss, 1)
    # Replace eval/train loss call sites globally.
    t = t.replace('loss_outcome = F.binary_cross_entropy_with_logits(value_wl, t["wl_target"])',
                  'loss_outcome = weighted_outcome_loss(value_wl, t["wl_target"], t["wl_w"])')
    t = t.replace('loss_policy = weighted_policy_loss(policy_logits, policy_t, pw)',
                  'loss_policy = weighted_policy_loss(policy_logits, t["policy_top_idx"], t["policy_top_prob"], pw)')
    # Thread wall-poor weight through iterator calls in eval and train.
    t = t.replace('pr_chunk, wl_gamma):', 'pr_chunk, wl_gamma, wall_poor_wl_weight):')
    t = t.replace('pr_chunk, wl_gamma):\n            policy_t', 'pr_chunk, wl_gamma, wall_poor_wl_weight):\n            policy_t')
    # run_eval signature/callers: default keeps compatibility.
    t = t.replace('w_outcome, w_policy, pw_by_sample=None, pr_by_sample=None, wl_gamma=1.0):',
                  'w_outcome, w_policy, pw_by_sample=None, pr_by_sample=None, wl_gamma=1.0, wall_poor_wl_weight=1.0):')
    # parser option
    anchor = '    g_loss.add_argument("--wl-gamma", type=float, default=WL_GAMMA_DEFAULT,\n                        help=f"desconto do alvo WL pela duracao ate o fim da partida "\n                             f"(padrao: {WL_GAMMA_DEFAULT}). 1.0 = desligado, alvo = so o lambda.\")\n'
    if anchor not in t:
        raise SystemExit("wl-gamma parser anchor missing")
    t = t.replace(anchor, anchor + '    g_loss.add_argument("--wall-poor-wl-weight", type=float, default=1.0,\n                        help="relative WL loss weight for wall-poor race positions")\n', 1)
    # Make iterator calls use args where available. The simple textual pattern
    # appears in train; eval retains default unless run_eval passes it.
    t = t.replace('pr_chunk, args.wl_gamma):', 'pr_chunk, args.wl_gamma, args.wall_poor_wl_weight):')
    # run_eval call sites end with args.wl_gamma in current trainer.
    t = t.replace('pr_by_sample, args.wl_gamma)', 'pr_by_sample, args.wl_gamma, args.wall_poor_wl_weight)')
    p.write_text(t, encoding="utf-8")

print("Gen10 training patch applied")
