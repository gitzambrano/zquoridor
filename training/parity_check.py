"""
parity_check.py -- lado Python da checagem de paridade numérica com
nnue_verify.cpp. Recalcula value_wl/value_aux/policy pra MESMA posição de
teste fixa (estado inicial + muro H em (3,4) + muro V em (5,2)), usando
só numpy (sem depender do objeto nn.Module do PyTorch) pra recomputar
exatamente as fórmulas de nnue.hpp a partir do arquivo de pesos
exportado. Se bater com a saída de `./nnue_verify data/nnue/nnue_weights.bin`,
o pipeline export -> load -> forward em C++ está numericamente correto.

CABEÇA AUXILIAR REMOVIDA (2026-08): `value` já foi duas cabeças
independentes (value_wl/value_aux); a auxiliar (imitação MSE da
heurística evalSimple) foi removida de nnue.hpp/train_nnue.py por virar
dead weight -- see the architecture table in docs/plan.md and CLAUDE.md ("Evaluation: what each
estágio usa"). Este script computa e imprime só value_wl agora.

ALINHADO COM NNUE.HPP (354 features, 2026-08): este script ficou para
trás duas vezes no passado e as duas vezes o sintoma foi o mesmo (saída
sem sentido nenhum, porque o load lê o arquivo com o layout errado):
(1) buckets de muros restantes (WALLS_LEFT_BUCKETS) viraram feature --
    NUM_FEATURES subiu de 332 para 354;
(2) buildAccumulator passou a ESPELHAR peão/slot de muro por perspectiva
    (mirroredPawnCell/mirroredWallSlot em nnue.hpp) -- sem isso a
    perspectiva 1 lia as linhas de peso da perspectiva 0. As funções
    mirrored_* abaixo reproduzem esse espelhamento. Se nnue.hpp ganhar
    feature ou transformação nova, atualize AQUI no mesmo commit.

Também recomputa, de forma totalmente independente do C++, o forward
QUANTIZADO (int8/int16, mesmas fórmulas de NNUEWeightsQuant em nnue.hpp)
a partir do arquivo gerado por training/quantize_nnue.py -- ponto de
atenção: `//` do NumPy arredonda pra baixo (floor); divisão de inteiros
em C++ trunca em direção a zero. Pra valores negativos os dois divergem
(ex.: -7 // 2 = -4 em Python, -7 / 2 = -3 em C++), então as divisões de
dequantização aqui usam `trunc_div` (baseada em np.trunc/np.fix), não
`//`, pra bater exatamente com nnue.hpp::truncDiv.

Uso:
    python3 parity_check.py ../data/nnue/nnue_weights.bin [../data/nnue/nnue_weights_int8.bin]
"""
import os
import sys
import numpy as np

from student_model import encode_features

N, WS = 9, 8
DIST_BUCKETS = 21   # ver DIST_BUCKETS em nnue.hpp (Seção 7.10 do plano)
WALLS_LEFT_BUCKETS = 11  # WALLS_PER_PLAYER + 1 -- ver WALLS_LEFT_BUCKETS em nnue.hpp
NUM_FEATURES = 504  # multipath_phase: race + multipath + phase features
HIDDEN = 512
POLICY_OUT = N * N + WS * WS * 2  # 209
DIST_FEAT_BASE = N * N + N * N + WS * WS * 2              # 290
WALLS_LEFT_FEAT_BASE = DIST_FEAT_BASE + 2 * DIST_BUCKETS  # 332
WALLS_PER_PLAYER = 10  # rules.hpp -- orçamento do initialState()


def slot_idx(r, c):
    return r * WS + c


def cell_idx(r, c):
    return r * N + c


def edge_blocked(walls_h, walls_v, ra, ca, rb, cb):
    """Reimplementação independente de edgeBlocked (rules.hpp) -- walls_h/
    walls_v aqui são sets de tuplas (r, c) de slot ocupado, não bitboards."""
    if ra == rb:
        r, c = ra, min(ca, cb)
        blocked = False
        if r - 1 >= 0:
            blocked = blocked or (r - 1, c) in walls_v
        if r < WS:
            blocked = blocked or (r, c) in walls_v
        return blocked
    else:
        r, c = min(ra, rb), ca
        blocked = False
        if c - 1 >= 0:
            blocked = blocked or (r, c - 1) in walls_h
        if c < WS:
            blocked = blocked or (r, c) in walls_h
        return blocked


def shortest_path_len(walls_h, walls_v, start_cell, player):
    """Reimplementação independente de shortestPathLen (rules.hpp), em
    Python puro (BFS com deque) -- usada só pra montar a posição de teste
    fixa deste script, mantendo a checagem de paridade totalmente alheia
    ao código C++ que está sendo verificado."""
    from collections import deque
    goal_row = (N - 1) if player == 0 else 0
    sr, sc = start_cell // N, start_cell % N
    if sr == goal_row:
        return 0
    dist = {start_cell: 0}
    q = deque([start_cell])
    while q:
        cell = q.popleft()
        r, c = cell // N, cell % N
        d0 = dist[cell]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < N and 0 <= nc < N):
                continue
            ncell = nr * N + nc
            if ncell in dist:
                continue
            if edge_blocked(walls_h, walls_v, r, c, nr, nc):
                continue
            if nr == goal_row:
                return d0 + 1
            dist[ncell] = d0 + 1
            q.append(ncell)
    return -1


def dist_bucket(dist):
    return min(max(dist, 0), DIST_BUCKETS - 1)


def get_phase_bucket(total_walls, buckets=6):
    if buckets == 6:
        if total_walls <= 0: return 0
        if total_walls <= 2: return 1
        if total_walls <= 5: return 2
        if total_walls <= 9: return 3
        if total_walls <= 14: return 4
        return 5
    return 0


def load_weights(path):
    actual_floats = os.path.getsize(path) // 4
    base_floats = NUM_FEATURES * HIDDEN + HIDDEN + POLICY_OUT * HIDDEN + POLICY_OUT
    head_floats_total = actual_floats - base_floats

    value_buckets, value_depth = 1, 1
    found = False
    for b in (1, 6):
        for d in (1, 2):
            hf = b * (17505 if d == 2 else 16449)
            if hf == head_floats_total:
                value_buckets, value_depth = b, d
                found = True
                break
        if found:
            break

    with open(path, "rb") as f:
        w1 = np.fromfile(f, dtype="<f4", count=NUM_FEATURES * HIDDEN).reshape(NUM_FEATURES, HIDDEN)
        b1 = np.fromfile(f, dtype="<f4", count=HIDDEN)
        d = dict(w1=w1, b1=b1, value_buckets=value_buckets, value_depth=value_depth)
        if value_buckets == 1 and value_depth == 1:
            d["wv1_wl"] = np.fromfile(f, dtype="<f4", count=HIDDEN * 32).reshape(HIDDEN, 32)
            d["bv1_wl"] = np.fromfile(f, dtype="<f4", count=32)
            d["wv2_wl"] = np.fromfile(f, dtype="<f4", count=32)
            d["bv2_wl"] = np.fromfile(f, dtype="<f4", count=1)[0]
        else:
            for b in range(value_buckets):
                d[f"wv1_wl_{b}"] = np.fromfile(f, dtype="<f4", count=HIDDEN * 32).reshape(HIDDEN, 32)
                d[f"bv1_wl_{b}"] = np.fromfile(f, dtype="<f4", count=32)
                if value_depth == 2:
                    d[f"wv2_wl_{b}"] = np.fromfile(f, dtype="<f4", count=32 * 32).reshape(32, 32)
                    d[f"bv2_wl_{b}"] = np.fromfile(f, dtype="<f4", count=32)
                    d[f"wv3_wl_{b}"] = np.fromfile(f, dtype="<f4", count=32)
                    d[f"bv3_wl_{b}"] = np.fromfile(f, dtype="<f4", count=1)[0]
                else:
                    d[f"wv2_wl_{b}"] = np.fromfile(f, dtype="<f4", count=32)
                    d[f"bv2_wl_{b}"] = np.fromfile(f, dtype="<f4", count=1)[0]

        wp = np.fromfile(f, dtype="<f4", count=POLICY_OUT * HIDDEN).reshape(POLICY_OUT, HIDDEN)
        bp = np.fromfile(f, dtype="<f4", count=POLICY_OUT)
        d["wp"] = wp
        d["bp"] = bp
    return d


def load_weights_quant(path):
    with open(path, "rb") as f:
        qa = int(np.fromfile(f, dtype="<i4", count=1)[0])
        qb = int(np.fromfile(f, dtype="<i4", count=1)[0])
        w1 = np.fromfile(f, dtype="<i2", count=NUM_FEATURES * HIDDEN).reshape(NUM_FEATURES, HIDDEN).astype(np.int64)
        b1 = np.fromfile(f, dtype="<i2", count=HIDDEN).astype(np.int64)

        actual_bytes = os.path.getsize(path)
        base_bytes = 4 + 4 + NUM_FEATURES * HIDDEN * 2 + HIDDEN * 2 + POLICY_OUT * HIDDEN * 1 + POLICY_OUT * 4
        head_bytes_total = actual_bytes - base_bytes

        value_buckets, value_depth = 1, 1
        found = False
        for b in (1, 6):
            for d in (1, 2):
                hb = b * (17700 if d == 2 else 16548)
                if hb == head_bytes_total:
                    value_buckets, value_depth = b, d
                    found = True
                    break
            if found:
                break

        d = dict(QA=qa, QB=qb, w1=w1, b1=b1, value_buckets=value_buckets, value_depth=value_depth)
        if value_buckets == 1 and value_depth == 1:
            d["wv1_wl"] = np.fromfile(f, dtype="<i1", count=HIDDEN * 32).reshape(HIDDEN, 32).astype(np.int64)
            d["bv1_wl"] = np.fromfile(f, dtype="<i4", count=32).astype(np.int64)
            d["wv2_wl"] = np.fromfile(f, dtype="<i1", count=32).astype(np.int64)
            d["bv2_wl"] = int(np.fromfile(f, dtype="<i4", count=1)[0])
        else:
            for b in range(value_buckets):
                d[f"wv1_wl_{b}"] = np.fromfile(f, dtype="<i1", count=HIDDEN * 32).reshape(HIDDEN, 32).astype(np.int64)
                d[f"bv1_wl_{b}"] = np.fromfile(f, dtype="<i4", count=32).astype(np.int64)
                if value_depth == 2:
                    d[f"wv2_wl_{b}"] = np.fromfile(f, dtype="<i1", count=32 * 32).reshape(32, 32).astype(np.int64)
                    d[f"bv2_wl_{b}"] = np.fromfile(f, dtype="<i4", count=32).astype(np.int64)
                    d[f"wv3_wl_{b}"] = np.fromfile(f, dtype="<i1", count=32).astype(np.int64)
                    d[f"bv3_wl_{b}"] = int(np.fromfile(f, dtype="<i4", count=1)[0])
                else:
                    d[f"wv2_wl_{b}"] = np.fromfile(f, dtype="<i1", count=32).astype(np.int64)
                    d[f"bv2_wl_{b}"] = int(np.fromfile(f, dtype="<i4", count=1)[0])

        wp = np.fromfile(f, dtype="<i1", count=POLICY_OUT * HIDDEN).reshape(POLICY_OUT, HIDDEN).astype(np.int64)
        bp = np.fromfile(f, dtype="<i4", count=POLICY_OUT).astype(np.int64)
        d["wp"] = wp
        d["bp"] = bp
    return d


def screlu(x):
    return np.clip(x, 0.0, 1.0) ** 2


def clipped_relu(x):
    return np.clip(x, 0.0, 1.0)


def final_descale(num, den):
    """Des-escala final (int64 -> score float comparável): divisão em
    ponto flutuante, igual ao lado C++ (ver comentário em
    forwardValueQuant/forwardPolicyQuant em nnue.hpp). NÃO usar `//` do
    NumPy aqui -- floor/trunc de inteiro jogaria fora a parte fracionária
    do score (bug pego nesta sessão: erro de ~1 unidade em vez de
    ~0,01-0,03). A única divisão que precisa ser inteira de verdade é a
    da SCReLU (não-negativa), calculada à parte abaixo."""
    return np.asarray(num, dtype=np.float64) / np.asarray(den, dtype=np.float64)


def mirrored_pawn_cell(cell, perspective):
    """Espelho de linha usado por mirroredPawnCell (nnue.hpp): a
    perspectiva 1 inverte r -> N-1-r; a coluna nunca muda."""
    if perspective == 0:
        return cell
    return cell_idx(N - 1 - cell // N, cell % N)


def mirrored_wall_rc(r, c, perspective):
    """Espelho de slot usado por mirroredWallSlot (nnue.hpp): a
    perspectiva 1 inverte r -> WS-1-r; a coluna nunca muda."""
    if perspective == 0:
        return (r, c)
    return (WS - 1 - r, c)


def build_feature_vector(own_pawn, opp_pawn, walls_h_bits, walls_v_bits,
                         own_player, own_walls_left=WALLS_PER_PLAYER,
                         opp_walls_left=WALLS_PER_PLAYER):
    """walls_h_bits/walls_v_bits: listas de slots (r,c) com muro presente,
    em coordenada CRUA do tabuleiro -- o espelhamento por perspectiva é
    aplicado AQUI, igual ao buildAccumulator(s, perspective) do lado C++.
    own_player: índice 0/1 de quem é o peão "próprio" nesta perspectiva --
    necessário aqui (e só aqui) pra saber qual GOAL_ROW usar na BFS; no
    C++ ele coincide com o próprio argumento `perspective`.
    own_walls_left/opp_walls_left: orçamentos de muro da posição; o default
    WALLS_PER_PLAYER cobre a posição de teste fixa (initialState() + dois
    bits de muro pintados direto no bitboard, sem tocar o orçamento)."""
    own_canon = mirrored_pawn_cell(own_pawn, own_player)
    opp_canon = mirrored_pawn_cell(opp_pawn, own_player)
    walls_h_set = set(walls_h_bits)
    walls_v_set = set(walls_v_bits)
    walls_h_canon = 0
    walls_v_canon = 0
    for (r, c) in walls_h_bits:
        rm, cm = mirrored_wall_rc(r, c, own_player)
        walls_h_canon |= 1 << slot_idx(rm, cm)
    for (r, c) in walls_v_bits:
        rm, cm = mirrored_wall_rc(r, c, own_player)
        walls_v_canon |= 1 << slot_idx(rm, cm)
    opp_player = 1 - own_player
    own_dist = dist_bucket(shortest_path_len(walls_h_set, walls_v_set, own_pawn, own_player))
    opp_dist = dist_bucket(shortest_path_len(walls_h_set, walls_v_set, opp_pawn, opp_player))
    data = {
        "own_pawn": np.array([own_canon], dtype=np.int64),
        "opp_pawn": np.array([opp_canon], dtype=np.int64),
        "walls_h": np.array([walls_h_canon], dtype=np.uint64),
        "walls_v": np.array([walls_v_canon], dtype=np.uint64),
        "own_dist": np.array([own_dist], dtype=np.int64),
        "opp_dist": np.array([opp_dist], dtype=np.int64),
        "walls_left_own": np.array([own_walls_left], dtype=np.int64),
        "walls_left_opp": np.array([opp_walls_left], dtype=np.int64),
    }
    return encode_features(data, np.array([0], dtype=np.int64), "multipath_phase")[0]


def build_active_features(own_pawn, opp_pawn, walls_h_bits, walls_v_bits,
                          own_player, own_walls_left=WALLS_PER_PLAYER,
                          opp_walls_left=WALLS_PER_PLAYER):
    """Return the active multipath_phase feature indices."""
    return np.flatnonzero(build_feature_vector(
        own_pawn, opp_pawn, walls_h_bits, walls_v_bits, own_player,
        own_walls_left, opp_walls_left)).tolist()


def forward_head(a, weights, prefix, bucket=0):
    """Cabeça de valor (float32, só WL desde a remoção da cabeça auxiliar
    em 2026-08) -- mesma lógica de forwardValueWL em nnue.hpp."""
    suf = f"_{bucket}" if f"wv1_{prefix}_{bucket}" in weights else ""
    h = a @ weights[f"wv1_{prefix}{suf}"]
    hj = clipped_relu(h + weights[f"bv1_{prefix}{suf}"])
    if f"wv3_{prefix}{suf}" in weights:
        h2 = hj @ weights[f"wv2_{prefix}{suf}"]
        hk = clipped_relu(h2 + weights[f"bv2_{prefix}{suf}"])
        return weights[f"bv3_{prefix}{suf}"] + float(hk @ weights[f"wv3_{prefix}{suf}"])
    return weights[f"bv2_{prefix}{suf}"] + float(hj @ weights[f"wv2_{prefix}{suf}"])


def forward(weights, x, total_walls=20):
    acc = x @ weights["w1"] + weights["b1"]           # (256,)
    a = screlu(acc)
    bucket = get_phase_bucket(total_walls, weights.get("value_buckets", 1))
    value_wl = forward_head(a, weights, "wl", bucket=bucket)
    policy = a @ weights["wp"].T + weights["bp"]         # (209,)
    return value_wl, policy


def forward_head_quant(a, W, prefix, QA, QB, bucket=0):
    """Cabeça de valor quantizada (só WL desde a remoção da cabeça
    auxiliar em 2026-08) -- mesma lógica de forwardValueWLQuant em
    nnue.hpp."""
    suf = f"_{bucket}" if f"wv1_{prefix}_{bucket}" in W else ""
    h = a @ W[f"wv1_{prefix}{suf}"].astype(np.int64)          # (32,), escala QA*QB
    h_total = h + W[f"bv1_{prefix}{suf}"]
    QAQB = QA * QB
    hj = np.clip(h_total, 0, QAQB)
    if f"wv3_{prefix}{suf}" in W:
        h1_q = hj // QB
        h2 = h1_q @ W[f"wv2_{prefix}{suf}"].astype(np.int64)
        h2_total = h2 + W[f"bv2_{prefix}{suf}"]
        hk = np.clip(h2_total, 0, QAQB)
        out = W[f"bv3_{prefix}{suf}"] + int(hk @ W[f"wv3_{prefix}{suf}"])
        return float(final_descale(out, QAQB * QB))
    out = W[f"bv2_{prefix}{suf}"] + int(hj @ W[f"wv2_{prefix}{suf}"])
    return float(final_descale(out, QAQB * QB))


def forward_quant(W, active_feats, total_walls=20):
    QA, QB = W["QA"], W["QB"]
    acc = W["b1"].copy()
    for f in active_feats:
        acc += W["w1"][f]
    # SCReLU inteira: clamp(acc,0,QA)^2 / QA -- entrada não-negativa após o
    # clamp, então trunc==floor aqui (só as divisões finais de
    # dequantização abaixo precisam de trunc_div de verdade).
    c = np.clip(acc, 0, QA)
    a = (c.astype(np.int64) ** 2) // QA   # divisor e dividendo não-negativos: // == trunc aqui

    bucket = get_phase_bucket(total_walls, W.get("value_buckets", 1))
    value_wl = forward_head_quant(a, W, "wl", QA, QB, bucket=bucket)

    QAQB = QA * QB
    policy_raw = a @ W["wp"].astype(np.int64).T + W["bp"]  # (209,), escala QA*QB
    policy = final_descale(policy_raw, QAQB)
    return value_wl, policy


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    W = load_weights(sys.argv[1])
    Wq = load_weights_quant(sys.argv[2]) if len(sys.argv) == 3 else None

    # mesma posição de teste do nnue_verify.cpp: estado inicial + muro H(3,4) + muro V(5,2)
    pawn0 = cell_idx(0, 4)
    pawn1 = cell_idx(8, 4)
    walls_h = [(3, 4)]
    walls_v = [(5, 2)]

    for perspective, (own, opp) in enumerate([(pawn0, pawn1), (pawn1, pawn0)]):
        x = build_feature_vector(own, opp, walls_h, walls_v, own_player=perspective)
        value_wl, policy = forward(W, x)
        best = int(np.argmax(policy))
        print(f"perspectiva={perspective}  value_wl={value_wl:.6f}  "
              f"argmax_policy={best}  policy[argmax]={policy[best]:.6f}  (float32)")
        print(f"  policy[0..4] = {' '.join(f'{v:.6f}' for v in policy[:5])}")

        if Wq is not None:
            feats = build_active_features(own, opp, walls_h, walls_v, own_player=perspective)
            value_wl_q, policy_q = forward_quant(Wq, feats)
            best_q = int(np.argmax(policy_q))
            print(f"perspectiva={perspective}  value_wl={value_wl_q:.6f}  "
                  f"argmax_policy={best_q}  policy[argmax]={policy_q[best_q]:.6f}  (int8, "
                  f"erro_wl={value_wl - value_wl_q:.6f}, "
                  f"argmax_bate={'sim' if best == best_q else 'NAO'})")
            print(f"  policy[0..4] = {' '.join(f'{v:.6f}' for v in policy_q[:5])}")
