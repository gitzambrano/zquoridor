"""
quantize_nnue.py -- quantização int8/int16 (Seção 7.8 do plano),
espelhando exatamente `NNUEWeightsQuant::loadFromFile` em nnue.hpp.

Lê um arquivo de pesos float32 no layout de `NNUEWeights::loadFromFile`
(o que `train_nnue.py`/`export_weights` grava: w1, b1, wv1_wl, bv1_wl,
wv2_wl, bv2_wl, wv1_aux, bv1_aux, wv2_aux, bv2_aux, wp, bp -- duas cabeças
de valor, ver nota "MUDANÇA DESTA SESSÃO" abaixo) e escreve um segundo
arquivo com:

    cabeçalho: QA (int32), QB (int32)
    w1      (354x256, int16, escala QA)
    b1      (256,     int16, escala QA)
    wv1_wl  (256x32,  int8,  escala QB)
    bv1_wl  (32,      int32, escala QA*QB)
    wv2_wl  (32,      int8,  escala QB)
    bv2_wl  (1,       int32, escala QA*QB*QB)
    wv1_aux (256x32,  int8,  escala QB)
    bv1_aux (32,      int32, escala QA*QB)
    wv2_aux (32,      int8,  escala QB)
    bv2_aux (1,       int32, escala QA*QB*QB)
    wp      (209x256, int8,  escala QB)
    bp      (209,     int32, escala QA*QB)

MUDANÇA DESTA SESSÃO -- quantization-aware training (QAT): antes, QA era
fixo em 255 mas QB era DINÂMICO, calculado depois do treino a partir do
maior peso em valor absoluto entre as camadas de cabeça (só arredondando
no final, sem nada durante o treino garantindo que os pesos coubessem no
range). Agora QA **e** QB são constantes FIXAS decididas antes do treino
(`QA_DEFAULT`/`QB_DEFAULT` abaixo, as MESMAS usadas em `nnue.hpp` e em
`train_nnue.py`/`train_nnue_numpy.py`), e o treino aplica um
WeightClipper a cada passo do otimizador que trava os pesos dentro do
range representável nessas escalas -- este script deixou de "decidir" a
escala e passou a só arredondar pesos que, por construção, já deveriam
caber. `compute_qb_posthoc` (renomeada, mantida só para depuração/pesos
legados sem clipper) ainda existe caso se precise quantizar um arquivo
antigo sem QAT; o caminho `quantize_file` default usa as constantes
fixas. O formato do cabeçalho do .bin NÃO mudou (ainda QA/QB int32 no
início) -- só o valor de QB deixou de variar por arquivo.

O aviso de saturação abaixo (`quantize_int8_saturating`/
`quantize_int16_saturating`) continua ativo como rede de segurança: com
QAT funcionando, não deveria disparar nunca; se disparar, é sinal de que
o arquivo de pesos não passou pelo clipper (ex. treino antigo, ou --qa/
--qb passados aqui divergentes dos usados no treino).

Uso normal (sem argumentos -- usa defaults data/nnue/nnue_weights.bin -> data/nnue/nnue_weights_int8.bin):
    python3 quantize_nnue.py

Uso com argumentos customizados (para quantizar qualquer checkpoint .bin especifico):
    python3 quantize_nnue.py data/checkpoints/best_20260808_..._ep0010.bin data/nnue/nnue_weights_int8.bin
"""
import os
import sys
import numpy as np

N, WS = 9, 8
DIST_BUCKETS = 21  # ver DIST_BUCKETS em nnue.hpp (Seção 7.10 do plano)
# WALLS_LEFT_BUCKETS: feature nova de 2026-08 -- ver nota completa em
# WALLS_LEFT_BUCKETS/NUM_FEATURES em nnue.hpp. Muros restantes de cada
# jogador (0..WALLS_PER_PLAYER=10), one-hot, 11 buckets por lado.
WALLS_PER_PLAYER = 10
WALLS_LEFT_BUCKETS = WALLS_PER_PLAYER + 1  # 11
# CUIDADO: NUM_FEATURES está duplicado em TRÊS lugares (nnue.hpp,
# train_nnue.py, aqui) -- não existe hoje uma fonte única compartilhada
# entre C++ e os dois scripts Python. Mudar a arquitetura exige atualizar
# os três em conjunto (é exatamente o que aconteceu de errado uma vez:
# este arquivo ficou com NUM_FEATURES=332 depois de nnue.hpp/train_nnue.py
# já terem ido para 354, e como load_float_weights() só validava a
# contagem de floats LIDA contra essa mesma constante desatualizada -- uma
# checagem tautológica, nunca contra o tamanho real do arquivo em disco --
# o script "funcionava" silenciosamente enquanto lia tudo a partir de
# w1 em diante nos offsets ERRADOS. Ver a checagem de tamanho real do
# arquivo em load_float_weights() abaixo, adicionada para isso não se
# repetir de novo).
NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS + 2 * WALLS_LEFT_BUCKETS  # 354
HIDDEN = 256
POLICY_OUT = N * N + WS * WS * 2  # 209

# QAT: constantes fixas -- devem bater com QA_DEFAULT/QB_DEFAULT em
# nnue.hpp e train_nnue.py.
QA_DEFAULT = 255
QB_DEFAULT = 64
QA = QA_DEFAULT  # nome curto mantido por compatibilidade com uso anterior deste módulo

# Caminhos default (editar aqui, ou passar via linha de comando --
# `python3 quantize_nnue.py` sem argumento nenhum usa os dois abaixo).
# Mesma convenção de OUT_DEFAULT em train_nnue.py.
SRC_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nnue", "nnue_weights.bin")
DST_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nnue", "nnue_weights_int8.bin")
INT8_MAX = 127
INT16_MAX = 32767

_HEAD_FIELDS = ("wv1_wl", "bv1_wl", "wv2_wl", "bv2_wl",
                "wv1_aux", "bv1_aux", "wv2_aux", "bv2_aux")


def load_float_weights(path):
    import os
    expected = (NUM_FEATURES * HIDDEN + HIDDEN
                + 2 * (HIDDEN * 32 + 32 + 32 + 1)
                + POLICY_OUT * HIDDEN + POLICY_OUT)
    expected_bytes = expected * 4  # float32
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"'{path}' tem {actual_bytes} bytes, esperado {expected_bytes} "
            f"({expected} floats) para NUM_FEATURES={NUM_FEATURES} -- "
            f"verifique se o arquivo foi gerado com a MESMA arquitetura "
            f"(NUM_FEATURES) deste script; ver nota sobre a constante "
            f"NUM_FEATURES estar duplicada em três lugares, no topo do "
            f"arquivo. NÃO era checado antes (só comparava a contagem de "
            f"floats lida contra a mesma constante usada pra ler -- "
            f"tautológico, nunca pegava esse tipo de desalinhamento).")
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
        wv1_aux, bv1_aux, wv2_aux, bv2_aux = read_head()
        wp = np.fromfile(f, dtype="<f4", count=POLICY_OUT * HIDDEN).reshape(POLICY_OUT, HIDDEN)
        bp = np.fromfile(f, dtype="<f4", count=POLICY_OUT)
    got = (w1.size + b1.size
           + wv1_wl.size + bv1_wl.size + wv2_wl.size + 1
           + wv1_aux.size + bv1_aux.size + wv2_aux.size + 1
           + wp.size + bp.size)
    if got != expected:
        raise ValueError(f"arquivo de pesos truncado/incompleto (ou no layout antigo de cabeça "
                          f"única -- ver nota de MUDANÇA DESTA SESSÃO no topo do arquivo): "
                          f"esperado {expected} floats, leu {got}")
    return dict(w1=w1, b1=b1,
                wv1_wl=wv1_wl, bv1_wl=bv1_wl, wv2_wl=wv2_wl, bv2_wl=bv2_wl,
                wv1_aux=wv1_aux, bv1_aux=bv1_aux, wv2_aux=wv2_aux, bv2_aux=bv2_aux,
                wp=wp, bp=bp)


def compute_qb_posthoc(W, margin=0.98):
    """QB dinâmico pós-hoc (comportamento ANTIGO de compute_qb, mantido só
    para depuração de pesos legados sem WeightClipper): floor(127 /
    max_abs) entre as cinco matrizes de cabeça (wv1_wl, wv2_wl, wv1_aux,
    wv2_aux, wp) juntas. O caminho normal (QAT) não chama esta função --
    usa QB_DEFAULT fixo, o mesmo aplicado pelo clipper durante o treino.
    """
    max_abs = max(np.abs(W[k]).max() for k in ("wv1_wl", "wv2_wl", "wv1_aux", "wv2_aux", "wp"))
    if max_abs < 1e-8:
        return 1  # pesos degenerados (ex. rede não treinada) -- evita divisão por zero
    qb = int(np.floor((INT8_MAX / max_abs) * margin))
    return max(1, qb)


def quantize_int16_saturating(x, scale, name):
    q = np.round(x * scale)
    n_sat = int(np.sum((q < -INT16_MAX - 1) | (q > INT16_MAX)))
    if n_sat > 0:
        pct = 100.0 * n_sat / q.size
        print(f"  aviso: {name} saturou {n_sat}/{q.size} pesos ({pct:.1f}%) no range int16 -- "
              f"clamping (com QAT isso não deveria acontecer -- indica pesos de um treino sem "
              f"WeightClipper, ou --qa divergente do usado no treino)")
    q = np.clip(q, -INT16_MAX - 1, INT16_MAX)
    return q.astype(np.int16)


def quantize_int8_saturating(x, scale, name):
    q = np.round(x * scale)
    n_sat = int(np.sum((q < -INT8_MAX - 1) | (q > INT8_MAX)))
    if n_sat > 0:
        pct = 100.0 * n_sat / q.size
        print(f"  aviso: {name} saturou {n_sat}/{q.size} pesos ({pct:.1f}%) no range int8 -- "
              f"clamping (com QAT isso não deveria acontecer -- indica pesos de um treino sem "
              f"WeightClipper, ou --qb divergente do usado no treino)")
    q = np.clip(q, -INT8_MAX - 1, INT8_MAX)
    return q.astype(np.int8)


def quantize(W, qa=QA_DEFAULT, qb=QB_DEFAULT):
    """QAT: qa/qb são fixos por default (mesmas constantes do treino). Se
    qb=None for passado explicitamente, cai no cálculo pós-hoc antigo
    (compute_qb_posthoc) -- só para pesos legados sem clipper."""
    if qb is None:
        qb = compute_qb_posthoc(W)
    max_abs_heads = max(np.abs(W[k]).max() for k in ("wv1_wl", "wv2_wl", "wv1_aux", "wv2_aux", "wp"))
    print(f"QA={qa}  QB={qb}  (max_abs entre wv1_wl/wv2_wl/wv1_aux/wv2_aux/wp = {max_abs_heads:.4f})")

    w1_q = quantize_int16_saturating(W["w1"], qa, "w1")
    b1_q = quantize_int16_saturating(W["b1"], qa, "b1")

    def quantize_head(prefix):
        wv1_q = quantize_int8_saturating(W[f"wv1_{prefix}"], qb, f"wv1_{prefix}")
        bv1_q = np.round(W[f"bv1_{prefix}"] * qa * qb).astype(np.int32)
        wv2_q = quantize_int8_saturating(W[f"wv2_{prefix}"], qb, f"wv2_{prefix}")
        bv2_q = np.int32(round(float(W[f"bv2_{prefix}"]) * qa * qb * qb))
        return wv1_q, bv1_q, wv2_q, bv2_q

    wv1_wl_q, bv1_wl_q, wv2_wl_q, bv2_wl_q = quantize_head("wl")
    wv1_aux_q, bv1_aux_q, wv2_aux_q, bv2_aux_q = quantize_head("aux")

    wp_q = quantize_int8_saturating(W["wp"], qb, "wp")
    bp_q = np.round(W["bp"] * qa * qb).astype(np.int32)

    return dict(QA=qa, QB=qb, w1=w1_q, b1=b1_q,
                wv1_wl=wv1_wl_q, bv1_wl=bv1_wl_q, wv2_wl=wv2_wl_q, bv2_wl=bv2_wl_q,
                wv1_aux=wv1_aux_q, bv1_aux=bv1_aux_q, wv2_aux=wv2_aux_q, bv2_aux=bv2_aux_q,
                wp=wp_q, bp=bp_q)


def write_quantized(Q, path):
    import os
    dir_name = os.path.dirname(os.path.abspath(path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.array([Q["QA"]], dtype="<i4").tobytes())
        f.write(np.array([Q["QB"]], dtype="<i4").tobytes())
        f.write(np.ascontiguousarray(Q["w1"]).astype("<i2").tobytes())
        f.write(np.ascontiguousarray(Q["b1"]).astype("<i2").tobytes())
        for prefix in ("wl", "aux"):
            f.write(np.ascontiguousarray(Q[f"wv1_{prefix}"]).astype("<i1").tobytes())
            f.write(np.ascontiguousarray(Q[f"bv1_{prefix}"]).astype("<i4").tobytes())
            f.write(np.ascontiguousarray(Q[f"wv2_{prefix}"]).astype("<i1").tobytes())
            f.write(np.array([Q[f"bv2_{prefix}"]], dtype="<i4").tobytes())
        f.write(np.ascontiguousarray(Q["wp"]).astype("<i1").tobytes())
        f.write(np.ascontiguousarray(Q["bp"]).astype("<i4").tobytes())


def quantize_file(src_path, dst_path, qa=QA_DEFAULT, qb=QB_DEFAULT):
    """Ponto de entrada reusável por train_nnue.py/train_nnue_numpy.py --
    treino já deixa a versão quantizada pronta automaticamente ao final
    do export float32, passando os mesmos qa/qb usados pelo clipper."""
    W = load_float_weights(src_path)
    Q = quantize(W, qa=qa, qb=qb)
    write_quantized(Q, dst_path)
    return Q


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", nargs="?", default=SRC_DEFAULT,
                    help=f"pesos float32 de entrada (padrao: {SRC_DEFAULT})")
    p.add_argument("dst", nargs="?", default=DST_DEFAULT,
                    help=f"pesos int8/int16 quantizados de saida (padrao: {DST_DEFAULT})")
    p.add_argument("--qa", type=int, default=QA_DEFAULT, help=f"escala QA (padrao: {QA_DEFAULT})")
    p.add_argument("--qb", type=int, default=QB_DEFAULT, help=f"escala QB (padrao: {QB_DEFAULT})")
    args = p.parse_args()
    Q = quantize_file(args.src, args.dst, qa=args.qa, qb=args.qb)
    print(f"pesos quantizados gravados em {args.dst}")
