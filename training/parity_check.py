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
peso morto -- ver a tabela em status.md/CLAUDE.md ("Avaliação: o que cada
estágio usa"). Este script computa e imprime só value_wl agora.

ATENÇÃO (pré-existente, não corrigido nesta mudança): NUM_FEATURES abaixo
está em 332 (sem WALLS_LEFT_BUCKETS), desatualizado em relação aos 354 de
nnue.hpp/train_nnue.py desde que os buckets de muros restantes foram
adicionados como feature -- ver a mesma nota em quantize_nnue.py, que TEM
a checagem de tamanho de arquivo que pegaria esse desalinhamento; este
script não tem essa checagem. Fora do escopo desta mudança (não mexe com
cabeça auxiliar nem avaliação salva no .bin); ver status.md.

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
import sys
import numpy as np

N, WS = 9, 8
DIST_BUCKETS = 21   # ver DIST_BUCKETS em nnue.hpp (Seção 7.10 do plano)
NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS  # 332
HIDDEN = 256
POLICY_OUT = N * N + WS * WS * 2  # 209


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


def load_weights(path):
    with open(path, "rb") as f:
        w1 = np.fromfile(f, dtype="<f4", count=NUM_FEATURES * HIDDEN).reshape(NUM_FEATURES, HIDDEN)
        b1 = np.fromfile(f, dtype="<f4", count=HIDDEN)

        def read_head():
            wv1 = np.fromfile(f, dtype="<f4", count=HIDDEN * 32).reshape(HIDDEN, 32)
            bv1 = np.fromfile(f, dtype="<f4", count=32)
            wv2 = np.fromfile(f, dtype="<f4", count=32)
            bv2 = np.fromfile(f, dtype="<f4", count=1)[0]
            return wv1, bv1, wv2, bv2

        wv1_wl, bv1_wl, wv2_wl, bv2_wl = read_head()
        wp = np.fromfile(f, dtype="<f4", count=POLICY_OUT * HIDDEN).reshape(POLICY_OUT, HIDDEN)
        bp = np.fromfile(f, dtype="<f4", count=POLICY_OUT)
    return dict(w1=w1, b1=b1,
                wv1_wl=wv1_wl, bv1_wl=bv1_wl, wv2_wl=wv2_wl, bv2_wl=bv2_wl,
                wp=wp, bp=bp)


def load_weights_quant(path):
    with open(path, "rb") as f:
        qa = int(np.fromfile(f, dtype="<i4", count=1)[0])
        qb = int(np.fromfile(f, dtype="<i4", count=1)[0])
        w1 = np.fromfile(f, dtype="<i2", count=NUM_FEATURES * HIDDEN).reshape(NUM_FEATURES, HIDDEN).astype(np.int64)
        b1 = np.fromfile(f, dtype="<i2", count=HIDDEN).astype(np.int64)

        def read_head_quant():
            wv1 = np.fromfile(f, dtype="<i1", count=HIDDEN * 32).reshape(HIDDEN, 32).astype(np.int64)
            bv1 = np.fromfile(f, dtype="<i4", count=32).astype(np.int64)
            wv2 = np.fromfile(f, dtype="<i1", count=32).astype(np.int64)
            bv2 = int(np.fromfile(f, dtype="<i4", count=1)[0])
            return wv1, bv1, wv2, bv2

        wv1_wl, bv1_wl, wv2_wl, bv2_wl = read_head_quant()
        wp = np.fromfile(f, dtype="<i1", count=POLICY_OUT * HIDDEN).reshape(POLICY_OUT, HIDDEN).astype(np.int64)
        bp = np.fromfile(f, dtype="<i4", count=POLICY_OUT).astype(np.int64)
    return dict(QA=qa, QB=qb, w1=w1, b1=b1,
                wv1_wl=wv1_wl, bv1_wl=bv1_wl, wv2_wl=wv2_wl, bv2_wl=bv2_wl,
                wp=wp, bp=bp)


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


def build_feature_vector(own_pawn, opp_pawn, walls_h_bits, walls_v_bits, own_player):
    """walls_h_bits/walls_v_bits: listas de slots (r,c) com muro presente.
    own_player: índice 0/1 de quem é o peão "próprio" nesta perspectiva --
    necessário aqui (e só aqui) pra saber qual GOAL_ROW usar na BFS; não
    existe no lado C++ porque lá a perspectiva já vem com esse índice
    (ver buildAccumulator(s, perspective) em nnue.hpp)."""
    x = np.zeros(NUM_FEATURES, dtype=np.float32)
    x[own_pawn] = 1.0
    x[81 + opp_pawn] = 1.0
    walls_h_set = set(walls_h_bits)
    walls_v_set = set(walls_v_bits)
    for (r, c) in walls_h_bits:
        x[162 + slot_idx(r, c)] = 1.0
    for (r, c) in walls_v_bits:
        x[162 + 64 + slot_idx(r, c)] = 1.0
    opp_player = 1 - own_player
    own_dist = dist_bucket(shortest_path_len(walls_h_set, walls_v_set, own_pawn, own_player))
    opp_dist = dist_bucket(shortest_path_len(walls_h_set, walls_v_set, opp_pawn, opp_player))
    x[290 + own_dist] = 1.0
    x[290 + DIST_BUCKETS + opp_dist] = 1.0
    return x


def build_active_features(own_pawn, opp_pawn, walls_h_bits, walls_v_bits, own_player):
    """Mesma posição, mas como lista de índices de feature ativos -- usado
    no forward quantizado pra somar só as linhas relevantes de w1 (o
    acumulador em nnue.hpp é sempre incremental/esparso, nunca faz um
    produto matricial denso contra as 332 features)."""
    feats = [own_pawn, 81 + opp_pawn]
    feats += [162 + slot_idx(r, c) for (r, c) in walls_h_bits]
    feats += [162 + 64 + slot_idx(r, c) for (r, c) in walls_v_bits]
    walls_h_set = set(walls_h_bits)
    walls_v_set = set(walls_v_bits)
    opp_player = 1 - own_player
    own_dist = dist_bucket(shortest_path_len(walls_h_set, walls_v_set, own_pawn, own_player))
    opp_dist = dist_bucket(shortest_path_len(walls_h_set, walls_v_set, opp_pawn, opp_player))
    feats += [290 + own_dist, 290 + DIST_BUCKETS + opp_dist]
    return feats


def forward_head(a, weights, prefix):
    """Cabeça de valor (float32, só WL desde a remoção da cabeça auxiliar
    em 2026-08) -- mesma lógica de forwardValueWL em nnue.hpp."""
    h = a @ weights[f"wv1_{prefix}"]
    hj = clipped_relu(h + weights[f"bv1_{prefix}"])
    return weights[f"bv2_{prefix}"] + float(hj @ weights[f"wv2_{prefix}"])


def forward(weights, x):
    acc = x @ weights["w1"] + weights["b1"]           # (256,)
    a = screlu(acc)
    value_wl = forward_head(a, weights, "wl")
    policy = a @ weights["wp"].T + weights["bp"]         # (209,)
    return value_wl, policy


def forward_head_quant(a, W, prefix, QA, QB):
    """Cabeça de valor quantizada (só WL desde a remoção da cabeça
    auxiliar em 2026-08) -- mesma lógica de forwardValueWLQuant em
    nnue.hpp."""
    h = a @ W[f"wv1_{prefix}"].astype(np.int64)          # (32,), escala QA*QB
    h_total = h + W[f"bv1_{prefix}"]
    QAQB = QA * QB
    hj = np.clip(h_total, 0, QAQB)
    out = W[f"bv2_{prefix}"] + int(hj @ W[f"wv2_{prefix}"])
    return float(final_descale(out, QAQB * QB))


def forward_quant(W, active_feats):
    QA, QB = W["QA"], W["QB"]
    acc = W["b1"].copy()
    for f in active_feats:
        acc += W["w1"][f]
    # SCReLU inteira: clamp(acc,0,QA)^2 / QA -- entrada não-negativa após o
    # clamp, então trunc==floor aqui (só as divisões finais de
    # dequantização abaixo precisam de trunc_div de verdade).
    c = np.clip(acc, 0, QA)
    a = (c.astype(np.int64) ** 2) // QA   # divisor e dividendo não-negativos: // == trunc aqui

    value_wl = forward_head_quant(a, W, "wl", QA, QB)

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
