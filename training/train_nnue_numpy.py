"""
train_nnue_numpy.py -- treino da NNUE do zquoridor em NumPy puro (forward
e backward manuais, Adam/AdamW implementado a mao). Fallback funcional de
train_nnue.py para ambientes sem PyTorch; mesma arquitetura, mesma loss
combinada e mesmo layout de export binario (ver `NNUEWeights::loadFromFile`
em nnue.hpp).

Recursos:
  - Varios arquivos .bin em --data (lista separada por virgula, diretorio,
    glob, ou --data repetido), com leitura preguicosa via MultiFileSelfPlay
    (read_selfplay.py) -- os shards nunca sao concatenados inteiros em RAM.
  - Split treino/validacao por fracao (--val-split) ou --val-data explicito.
  - Treino em chunks dimensionados por um orcamento de RAM (--ram-budget-gb),
    para datasets maiores que a memoria disponivel.
  - Weight decay desacoplado (estilo AdamW, so em pesos, nunca em bias) com
    annealing (--weight-decay, --weight-decay-min, --wd-schedule).
  - LR schedule com warmup (--lr-schedule, --warmup-epochs, --lr-min,
    --step-size, --step-gamma, --exp-gamma).
  - Early stopping com melhor checkpoint salvo automaticamente
    (--early-stop, --patience, --min-delta, --monitor, --ckpt-dir); ao
    final, os pesos exportados sao os do melhor epoch (nao os do ultimo),
    a menos que --no-restore-best seja passado.
  - Plots de convergencia/validacao em PNG (--plot-dir).
  - Quantization-aware training (QAT) com QA/QB fixos (--qa/--qb), iguais
    aos usados em nnue.hpp e quantize_nnue.py.

Todos os defaults abaixo (secao DEFAULT CONFIG) valem como "flags no
cabecalho do arquivo": editar as constantes muda o comportamento padrao
sem precisar passar nada na linha de comando; qualquer flag de linha de
comando sobrescreve a constante correspondente.

Exemplos:
    python3 train_nnue_numpy.py --data ../data/selfplay_*.bin \
        --out ../data/nnue/nnue_weights.bin --plot-dir ../data/plots

    python3 train_nnue_numpy.py --data a.bin,b.bin --init-from prev.bin \
        --epochs 80 --weight-decay 2e-4 --wd-schedule cosine \
        --early-stop --patience 10 --monitor val_loss \
        --out ../data/nnue/nnue_weights.bin --ckpt-dir ../data/checkpoints
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_selfplay import SAMPLE_DTYPE, DIST_BUCKETS, load_multi_selfplay  # noqa: E402
from quantize_nnue import quantize_file  # noqa: E402

# ============================== DEFAULT CONFIG ==============================
# Editar aqui muda o default sem precisar de flag; toda entrada tem uma
# flag de linha de comando correspondente que sobrescreve o valor abaixo.
EPOCHS_DEFAULT = 60
BATCH_SIZE_DEFAULT = 512
SEED_DEFAULT = 0
VAL_SPLIT_DEFAULT = 0.1

LR_DEFAULT = 1e-3
LR_MIN_DEFAULT = 1e-5
LR_SCHEDULE_DEFAULT = "cosine"        # none | step | exponential | cosine
WARMUP_EPOCHS_DEFAULT = 2
STEP_SIZE_DEFAULT = 10
STEP_GAMMA_DEFAULT = 0.5
EXP_GAMMA_DEFAULT = 0.97

# Weight decay desacoplado (estilo AdamW/Stockfish-nnue-pytorch): aplicado
# so as matrizes de peso (nunca aos bias), e "annealed" (reduzido) ao
# longo do treino -- comeca regularizando mais forte enquanto a rede ainda
# esta mudando bastante, e afrouxa perto da convergencia para nao atrapalhar
# o ajuste fino dos ultimos epochs.
WEIGHT_DECAY_DEFAULT = 1e-4
WEIGHT_DECAY_MIN_DEFAULT = 0.0
WD_SCHEDULE_DEFAULT = "cosine"        # none | constant | linear | cosine

EARLY_STOP_DEFAULT = True
PATIENCE_DEFAULT = 8
MIN_DELTA_DEFAULT = 1e-4
MONITOR_DEFAULT = "val_loss"          # val_loss | val_outcome | val_score | val_policy | val_policy_acc

RAM_BUDGET_GB_DEFAULT = 32.0
RAM_CHUNK_FRACTION_DEFAULT = 0.25     # fracao do orcamento de RAM usada como buffer de shuffle

W_SCORE_DEFAULT = 0.3
W_OUTCOME_DEFAULT = 1.0
W_POLICY_DEFAULT = 1.0
QA_DEFAULT = 255
QB_DEFAULT = 64
LOG_EVERY_DEFAULT = 1
# =============================================================================

N, WS = 9, 8
NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS   # 332
HIDDEN = 256
POLICY_OUT = N * N + WS * WS * 2                                # 209
VALUE_SCALE = 200.0

INT8_MAX = 127
INT16_MAX = 32767


def screlu(x):
    c = np.clip(x, 0.0, 1.0)
    return c * c


def clipped_relu(x):
    return np.clip(x, 0.0, 1.0)


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --- features -------------------------------------------------------------
def to_batch(batch):
    n = len(batch)
    x = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    x[np.arange(n), batch["own_pawn"]] = 1.0
    x[np.arange(n), 81 + batch["opp_pawn"]] = 1.0
    wh = batch["walls_h"].astype(np.uint64)
    wv = batch["walls_v"].astype(np.uint64)
    bits_h = ((wh[:, None] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.float32)
    bits_v = ((wv[:, None] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.float32)
    x[:, 162:162 + 64] = bits_h
    x[:, 162 + 64:162 + 128] = bits_v
    own_bucket = np.minimum(batch["own_dist"].astype(np.int64), DIST_BUCKETS - 1)
    opp_bucket = np.minimum(batch["opp_dist"].astype(np.int64), DIST_BUCKETS - 1)
    x[np.arange(n), 290 + own_bucket] = 1.0
    x[np.arange(n), 290 + DIST_BUCKETS + opp_bucket] = 1.0
    score_t = batch["search_score"].astype(np.float32) / VALUE_SCALE
    result_t = (batch["game_result"].astype(np.float32) + 1.0) / 2.0
    policy_t = batch["policy_target"].astype(np.int64)
    return x, score_t, result_t, policy_t


# --- pesos ------------------------------------------------------------------
def init_weights(rng):
    def linear_init(fan_in, fan_out):
        bound = 1.0 / np.sqrt(fan_in)
        w = rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype(np.float32)
        b = rng.uniform(-bound, bound, size=(fan_out,)).astype(np.float32)
        return w, b

    def head_init():
        wv1, bv1 = linear_init(HIDDEN, 32)
        wv2_full, bv2_full = linear_init(32, 1)
        return wv1, bv1, wv2_full.reshape(-1), bv2_full

    w1, b1 = linear_init(NUM_FEATURES, HIDDEN)
    wv1_wl, bv1_wl, wv2_wl, bv2_wl = head_init()
    wv1_aux, bv1_aux, wv2_aux, bv2_aux = head_init()
    wp_t, bp = linear_init(HIDDEN, POLICY_OUT)
    wp = wp_t.T.copy()  # armazenado como (POLICY_OUT, HIDDEN), igual ao export
    return dict(w1=w1, b1=b1,
                wv1_wl=wv1_wl, bv1_wl=bv1_wl, wv2_wl=wv2_wl, bv2_wl=bv2_wl,
                wv1_aux=wv1_aux, bv1_aux=bv1_aux, wv2_aux=wv2_aux, bv2_aux=bv2_aux,
                wp=wp, bp=bp)


def load_weights_bin(path):
    with open(path, "rb") as f:
        w1 = np.fromfile(f, dtype="<f4", count=NUM_FEATURES * HIDDEN).reshape(NUM_FEATURES, HIDDEN)
        b1 = np.fromfile(f, dtype="<f4", count=HIDDEN)

        def read_head():
            wv1 = np.fromfile(f, dtype="<f4", count=HIDDEN * 32).reshape(HIDDEN, 32)
            bv1 = np.fromfile(f, dtype="<f4", count=32)
            wv2 = np.fromfile(f, dtype="<f4", count=32)
            bv2 = np.fromfile(f, dtype="<f4", count=1)
            return wv1, bv1, wv2, bv2

        wv1_wl, bv1_wl, wv2_wl, bv2_wl = read_head()
        wv1_aux, bv1_aux, wv2_aux, bv2_aux = read_head()
        wp = np.fromfile(f, dtype="<f4", count=POLICY_OUT * HIDDEN).reshape(POLICY_OUT, HIDDEN)
        bp = np.fromfile(f, dtype="<f4", count=POLICY_OUT)
    return dict(w1=w1.copy(), b1=b1.copy(),
                wv1_wl=wv1_wl.copy(), bv1_wl=bv1_wl.copy(), wv2_wl=wv2_wl.copy(), bv2_wl=bv2_wl.copy(),
                wv1_aux=wv1_aux.copy(), bv1_aux=bv1_aux.copy(), wv2_aux=wv2_aux.copy(), bv2_aux=bv2_aux.copy(),
                wp=wp.copy(), bp=bp.copy())


def export_weights(W, path):
    dir_name = os.path.dirname(os.path.abspath(path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.ascontiguousarray(W["w1"], dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(W["b1"], dtype=np.float32).tobytes())
        for head in ("wl", "aux"):
            f.write(np.ascontiguousarray(W[f"wv1_{head}"], dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(W[f"bv1_{head}"], dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(W[f"wv2_{head}"], dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(W[f"bv2_{head}"], dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(W["wp"], dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(W["bp"], dtype=np.float32).tobytes())


def clip_weights_inplace(W, qa=QA_DEFAULT, qb=QB_DEFAULT):
    """QAT: trava w1/b1 no range representavel em int16 (escala qa) e as
    matrizes de peso das cabecas/politica no range representavel em int8
    (escala qb), in-place, a cada passo do otimizador."""
    w1_max = INT16_MAX / qa
    head_max = INT8_MAX / qb
    np.clip(W["w1"], -w1_max, w1_max, out=W["w1"])
    np.clip(W["b1"], -w1_max, w1_max, out=W["b1"])
    for key in ("wv1_wl", "wv2_wl", "wv1_aux", "wv2_aux", "wp"):
        np.clip(W[key], -head_max, head_max, out=W[key])


# --- otimizador: Adam com weight decay desacoplado (AdamW) -----------------
class AdamW:
    """Adam com weight decay desacoplado (Loshchilov & Hutter, 2019): o
    termo de decay e aplicado direto no parametro, fora do gradiente e dos
    momentos de primeira/segunda ordem, entao nao interage com a
    normalizacao adaptativa do Adam. Aplicado somente as matrizes de peso
    (WEIGHT_KEYS), nunca aos biases -- pratica padrao (ver AdamW original e
    nnue-pytorch do Stockfish)."""

    WEIGHT_KEYS = {"w1", "wv1_wl", "wv2_wl", "wv1_aux", "wv2_aux", "wp"}

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        bias1 = 1.0 - self.b1 ** self.t
        bias2 = 1.0 - self.b2 ** self.t
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / bias1
            vhat = self.v[k] / bias2
            if self.weight_decay > 0.0 and k in self.WEIGHT_KEYS:
                params[k] -= self.lr * self.weight_decay * params[k]
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# --- schedules ---------------------------------------------------------------
def lr_at_epoch(epoch, epochs, base_lr, min_lr, schedule,
                 warmup_epochs=0, step_size=10, step_gamma=0.5, exp_gamma=0.97):
    """epoch e 1-indexado. `schedule` em {none, step, exponential, cosine}."""
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    e = epoch - warmup_epochs
    total = max(1, epochs - warmup_epochs)
    if schedule == "none":
        return base_lr
    if schedule == "step":
        return max(min_lr, base_lr * (step_gamma ** (e // max(1, step_size))))
    if schedule == "exponential":
        return max(min_lr, base_lr * (exp_gamma ** e))
    if schedule == "cosine":
        frac = min(e, total) / total
        cos = 0.5 * (1.0 + np.cos(np.pi * frac))
        return min_lr + (base_lr - min_lr) * cos
    raise ValueError(f"lr-schedule desconhecido: {schedule}")


def wd_at_epoch(epoch, epochs, base_wd, min_wd, schedule):
    """epoch e 1-indexado. `schedule` em {none, constant, linear, cosine}."""
    if schedule == "none":
        return 0.0
    if schedule == "constant" or epochs <= 1:
        return base_wd
    frac = (epoch - 1) / max(1, epochs - 1)
    if schedule == "linear":
        return base_wd + (min_wd - base_wd) * frac
    if schedule == "cosine":
        cos = 0.5 * (1.0 + np.cos(np.pi * frac))
        return min_wd + (base_wd - min_wd) * cos
    raise ValueError(f"wd-schedule desconhecido: {schedule}")


# --- early stopping ----------------------------------------------------------
class EarlyStopper:
    """Monitora uma metrica de validacao a cada epoch; guarda uma copia dos
    pesos sempre que a metrica melhora e sinaliza quando parar apos
    `patience` epochs seguidos sem melhora >= `min_delta`."""

    MINIMIZE = {"val_loss", "val_outcome", "val_score", "val_policy"}
    MAXIMIZE = {"val_policy_acc"}

    def __init__(self, monitor="val_loss", patience=8, min_delta=1e-4, enabled=True):
        if monitor not in self.MINIMIZE | self.MAXIMIZE:
            raise ValueError(f"monitor desconhecido: {monitor}")
        self.monitor = monitor
        self.mode = "min" if monitor in self.MINIMIZE else "max"
        self.patience = max(1, patience)
        self.min_delta = min_delta
        self.enabled = enabled
        self.best = None
        self.best_epoch = 0
        self.best_state = None
        self.num_bad_epochs = 0

    def _improved(self, value):
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value, epoch, weights):
        improved = self._improved(value)
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.best_state = {k: v.copy() for k, v in weights.items()}
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        should_stop = self.enabled and self.num_bad_epochs >= self.patience
        return improved, should_stop


# --- orcamento de RAM: tamanho de chunk para o buffer de shuffle -----------
def compute_chunk_size(n_total, ram_budget_gb, fraction=RAM_CHUNK_FRACTION_DEFAULT,
                        min_size=20_000, max_size=5_000_000):
    """Numero de posicoes carregadas por vez (como structs crus + features
    densas) durante o treino, escolhido para que o buffer de shuffle caiba
    dentro de `fraction` do orcamento de RAM informado. Cada posicao ocupa
    SAMPLE_DTYPE.itemsize bytes crus mais, no pior caso, NUM_FEATURES*4
    bytes se a matriz densa do chunk inteiro for materializada de uma vez."""
    budget_bytes = ram_budget_gb * (1024 ** 3) * fraction
    bytes_per_sample = SAMPLE_DTYPE.itemsize + NUM_FEATURES * 4
    size = int(budget_bytes // max(1, bytes_per_sample))
    size = max(min_size, min(max_size, size))
    return min(size, max(1, n_total))


def iter_chunks(n, chunk_size, rng):
    """Gera arrays de indices embaralhados, em blocos de ate `chunk_size`,
    cobrindo `0..n-1` uma vez (uma epoch)."""
    perm = rng.permutation(n)
    for start in range(0, n, chunk_size):
        yield perm[start:start + chunk_size]


# --- forward / backward -------------------------------------------------------
def forward(W, x):
    acc = x @ W["w1"] + W["b1"]
    a = screlu(acc)

    h_wl_pre = a @ W["wv1_wl"] + W["bv1_wl"]
    h_wl = clipped_relu(h_wl_pre)
    value_wl = h_wl @ W["wv2_wl"] + W["bv2_wl"][0]

    h_aux_pre = a @ W["wv1_aux"] + W["bv1_aux"]
    h_aux = clipped_relu(h_aux_pre)
    value_aux = h_aux @ W["wv2_aux"] + W["bv2_aux"][0]

    policy_logits = a @ W["wp"].T + W["bp"]
    cache = dict(x=x, acc=acc, a=a,
                 h_wl_pre=h_wl_pre, h_wl=h_wl, value_wl=value_wl,
                 h_aux_pre=h_aux_pre, h_aux=h_aux, value_aux=value_aux)
    return value_wl, value_aux, policy_logits, cache


def backward(W, cache, policy_logits, score_t, result_t, policy_t,
             w_score, w_outcome, w_policy):
    n = cache["x"].shape[0]
    a = cache["a"]
    grads = {}

    value_wl = cache["value_wl"]
    sig = 1.0 / (1.0 + np.exp(-value_wl))
    dvalue_wl = w_outcome * (sig - result_t) / n

    h_wl = cache["h_wl"]
    grads["bv2_wl"] = np.array([dvalue_wl.sum()], dtype=np.float32)
    grads["wv2_wl"] = (h_wl * dvalue_wl[:, None]).sum(axis=0).astype(np.float32)
    dh_wl = dvalue_wl[:, None] * W["wv2_wl"][None, :]
    mask_wl = (cache["h_wl_pre"] > 0) & (cache["h_wl_pre"] < 1)
    dh_wl_pre = dh_wl * mask_wl
    grads["bv1_wl"] = dh_wl_pre.sum(axis=0).astype(np.float32)
    grads["wv1_wl"] = (a.T @ dh_wl_pre).astype(np.float32)
    da_from_wl = dh_wl_pre @ W["wv1_wl"].T

    value_aux = cache["value_aux"]
    value_aux_scaled = value_aux / VALUE_SCALE
    dvalue_aux_scaled = w_score * 2.0 * (value_aux_scaled - score_t) / n
    dvalue_aux = dvalue_aux_scaled / VALUE_SCALE

    h_aux = cache["h_aux"]
    grads["bv2_aux"] = np.array([dvalue_aux.sum()], dtype=np.float32)
    grads["wv2_aux"] = (h_aux * dvalue_aux[:, None]).sum(axis=0).astype(np.float32)
    dh_aux = dvalue_aux[:, None] * W["wv2_aux"][None, :]
    mask_aux = (cache["h_aux_pre"] > 0) & (cache["h_aux_pre"] < 1)
    dh_aux_pre = dh_aux * mask_aux
    grads["bv1_aux"] = dh_aux_pre.sum(axis=0).astype(np.float32)
    grads["wv1_aux"] = (a.T @ dh_aux_pre).astype(np.float32)
    da_from_aux = dh_aux_pre @ W["wv1_aux"].T

    probs = softmax(policy_logits)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), policy_t] = 1.0
    dpolicy_logits = w_policy * (probs - onehot) / n
    grads["bp"] = dpolicy_logits.sum(axis=0).astype(np.float32)
    grads["wp"] = (dpolicy_logits.T @ a).astype(np.float32)
    da_from_policy = dpolicy_logits @ W["wp"]

    da = da_from_wl + da_from_aux + da_from_policy
    acc = cache["acc"]
    mask_acc = (acc > 0) & (acc < 1)
    dacc = da * 2.0 * np.clip(acc, 0.0, 1.0) * mask_acc

    x = cache["x"]
    grads["b1"] = dacc.sum(axis=0).astype(np.float32)
    grads["w1"] = (x.T @ dacc).astype(np.float32)

    return grads


def compute_losses(value_wl, value_aux, policy_logits, score_t, result_t, policy_t):
    sig = 1.0 / (1.0 + np.exp(-value_wl))
    eps = 1e-7
    loss_outcome = float(-np.mean(result_t * np.log(sig + eps) + (1 - result_t) * np.log(1 - sig + eps)))
    value_aux_scaled = value_aux / VALUE_SCALE
    loss_score = float(np.mean((value_aux_scaled - score_t) ** 2))
    probs = softmax(policy_logits)
    n = len(policy_t)
    loss_policy = float(-np.mean(np.log(probs[np.arange(n), policy_t] + eps)))
    acc_policy = float(np.mean(policy_logits.argmax(axis=1) == policy_t))
    return loss_score, loss_outcome, loss_policy, acc_policy


# --- avaliacao em chunks (limitado por RAM) ---------------------------------
def run_eval(ds, indices, batch_size, chunk_size, W, w_score, w_outcome, w_policy, rng_unused=None):
    total = dict(loss=0.0, score=0.0, outcome=0.0, policy=0.0, correct=0)
    n_items = len(indices)
    for start in range(0, n_items, chunk_size):
        chunk_idx = indices[start:start + chunk_size]
        chunk = ds[chunk_idx]
        for bs in range(0, len(chunk), batch_size):
            batch = chunk[bs:bs + batch_size]
            x, score_t, result_t, policy_t = to_batch(batch)
            value_wl, value_aux, policy_logits, _ = forward(W, x)
            ls, lo, lp, _ = compute_losses(value_wl, value_aux, policy_logits, score_t, result_t, policy_t)
            loss = w_outcome * lo + w_score * ls + w_policy * lp
            nb = len(batch)
            total["loss"] += loss * nb
            total["score"] += ls * nb
            total["outcome"] += lo * nb
            total["policy"] += lp * nb
            total["correct"] += int((policy_logits.argmax(axis=1) == policy_t).sum())
    for k in ("loss", "score", "outcome", "policy"):
        total[k] /= max(1, n_items)
    total["policy_acc"] = total["correct"] / max(1, n_items)
    return total


# --- plots --------------------------------------------------------------------
def save_plots(history, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    epochs = history["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    def plot_pair(ax, key, title):
        ax.plot(epochs, history[f"train_{key}"], label="treino")
        ax.plot(epochs, history[f"val_{key}"], label="validacao")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plot_pair(axes[0, 0], "loss", "Loss total")
    plot_pair(axes[0, 1], "outcome", "Loss resultado (BCE)")
    plot_pair(axes[0, 2], "score", "Loss score aux (MSE)")
    plot_pair(axes[1, 0], "policy", "Loss politica (CE)")
    plot_pair(axes[1, 1], "policy_acc", "Acuracia da politica")

    ax = axes[1, 2]
    ax.plot(epochs, history["lr"], color="tab:orange", label="learning rate")
    ax.set_ylabel("learning rate")
    ax.set_xlabel("epoch")
    ax2 = ax.twinx()
    ax2.plot(epochs, history["wd"], color="tab:blue", label="weight decay")
    ax2.set_ylabel("weight decay")
    ax.set_title("Schedules")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(plot_dir, "training_curves.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --- treino -------------------------------------------------------------------
def run(args):
    train_paths, train_ds = load_multi_selfplay(args.data)
    print(f"dados de treino: {len(train_paths)} arquivo(s), {len(train_ds):,} posicoes")
    for p, n in train_ds.sizes():
        print(f"  - {p}: {n:,} posicoes")

    rng = np.random.default_rng(args.seed)

    if args.val_data:
        val_paths, val_ds = load_multi_selfplay(args.val_data)
        print(f"dados de validacao (explicitos): {len(val_paths)} arquivo(s), {len(val_ds):,} posicoes")
        train_idx = np.arange(len(train_ds))
        val_idx = np.arange(len(val_ds))
    else:
        n_total = len(train_ds)
        n_val = max(1, int(n_total * args.val_split))
        perm = rng.permutation(n_total)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        val_ds = train_ds
        print(f"split treino/validacao: {len(train_idx):,} / {len(val_idx):,} (val-split={args.val_split})")

    if args.init_from:
        W = load_weights_bin(args.init_from)
        print(f"pesos iniciais carregados de {args.init_from} (continuando treino)")
    else:
        W = init_weights(np.random.default_rng(42))
        print("pesos iniciais aleatorios (treino do zero)")

    clip_weights_inplace(W, qa=args.qa, qb=args.qb)
    print(f"QAT: QA={args.qa} QB={args.qb} (pesos travados a cada passo)")

    opt = AdamW(W, lr=args.lr, weight_decay=args.weight_decay)

    chunk_size = compute_chunk_size(len(train_idx), args.ram_budget_gb, args.ram_chunk_fraction)
    print(f"orcamento de RAM: {args.ram_budget_gb:.1f} GB -> chunk_size={chunk_size:,} posicoes/bloco")

    stopper = EarlyStopper(monitor=args.monitor, patience=args.patience,
                            min_delta=args.min_delta, enabled=args.early_stop)

    ckpt_dir = args.ckpt_dir
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    history = {k: [] for k in (
        "epoch", "train_loss", "val_loss", "train_outcome", "val_outcome",
        "train_score", "val_score", "train_policy", "val_policy",
        "train_policy_acc", "val_policy_acc", "lr", "wd",
    )}

    t0 = time.time()
    stopped_early = False
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        lr = lr_at_epoch(epoch, args.epochs, args.lr, args.lr_min, args.lr_schedule,
                          args.warmup_epochs, args.step_size, args.step_gamma, args.exp_gamma)
        wd = wd_at_epoch(epoch, args.epochs, args.weight_decay, args.weight_decay_min, args.wd_schedule)
        opt.lr = lr
        opt.weight_decay = wd

        tr_total = dict(loss=0.0, score=0.0, outcome=0.0, policy=0.0, correct=0)
        n_train_items = len(train_idx)
        for chunk_idx in iter_chunks(n_train_items, chunk_size, rng):
            global_idx = train_idx[chunk_idx]
            chunk = train_ds[global_idx]
            perm_local = rng.permutation(len(chunk))
            for start in range(0, len(perm_local), args.batch_size):
                batch_idx = perm_local[start:start + args.batch_size]
                batch = chunk[batch_idx]
                x, score_t, result_t, policy_t = to_batch(batch)
                value_wl, value_aux, policy_logits, cache = forward(W, x)
                ls, lo, lp, _ = compute_losses(value_wl, value_aux, policy_logits, score_t, result_t, policy_t)
                loss = args.w_outcome * lo + args.w_score * ls + args.w_policy * lp
                grads = backward(W, cache, policy_logits, score_t, result_t, policy_t,
                                  args.w_score, args.w_outcome, args.w_policy)
                opt.step(W, grads)
                clip_weights_inplace(W, qa=args.qa, qb=args.qb)
                nb = len(batch)
                tr_total["loss"] += loss * nb
                tr_total["score"] += ls * nb
                tr_total["outcome"] += lo * nb
                tr_total["policy"] += lp * nb
                tr_total["correct"] += int((policy_logits.argmax(axis=1) == policy_t).sum())
        for k in ("loss", "score", "outcome", "policy"):
            tr_total[k] /= max(1, n_train_items)
        tr_total["policy_acc"] = tr_total["correct"] / max(1, n_train_items)

        va = run_eval(val_ds, val_idx, args.batch_size, chunk_size, W,
                      args.w_score, args.w_outcome, args.w_policy)

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_total["loss"])
        history["val_loss"].append(va["loss"])
        history["train_outcome"].append(tr_total["outcome"])
        history["val_outcome"].append(va["outcome"])
        history["train_score"].append(tr_total["score"])
        history["val_score"].append(va["score"])
        history["train_policy"].append(tr_total["policy"])
        history["val_policy"].append(va["policy"])
        history["train_policy_acc"].append(tr_total["policy_acc"])
        history["val_policy_acc"].append(va["policy_acc"])
        history["lr"].append(lr)
        history["wd"].append(wd)

        monitored = {"val_loss": va["loss"], "val_outcome": va["outcome"],
                     "val_score": va["score"], "val_policy": va["policy"],
                     "val_policy_acc": va["policy_acc"]}[args.monitor]
        improved, should_stop = stopper.step(monitored, epoch, W)

        if ckpt_dir and improved:
            export_weights(stopper.best_state, os.path.join(ckpt_dir, "best.bin"))

        if epoch % args.log_every == 0 or epoch == args.epochs or improved or should_stop:
            star = " *" if improved else "  "
            print(f"epoch {epoch:3d}/{args.epochs} | lr={lr:.2e} wd={wd:.2e} | "
                  f"treino: loss={tr_total['loss']:.4f} (score={tr_total['score']:.4f} "
                  f"outcome={tr_total['outcome']:.4f} policy={tr_total['policy']:.4f} "
                  f"acc={tr_total['policy_acc']:.3f}) | "
                  f"val: loss={va['loss']:.4f} (score={va['score']:.4f} outcome={va['outcome']:.4f} "
                  f"policy={va['policy']:.4f} acc={va['policy_acc']:.3f}){star} | "
                  f"{time.time() - t0:.0f}s")

        if should_stop:
            print(f"\nearly stopping: sem melhora em '{args.monitor}' por {stopper.patience} epochs "
                  f"(melhor epoch={stopper.best_epoch}, {args.monitor}={stopper.best:.4f})")
            stopped_early = True
            break

    if args.early_stop and not args.no_restore_best and stopper.best_state is not None:
        print(f"\nrestaurando pesos do melhor epoch ({stopper.best_epoch}, "
              f"{args.monitor}={stopper.best:.4f}) para export final")
        W = stopper.best_state
    elif stopper.best_state is not None:
        print(f"\nmelhor epoch registrado: {stopper.best_epoch} ({args.monitor}={stopper.best:.4f}); "
              f"export final usa os pesos do ultimo epoch (--no-restore-best)")

    export_weights(W, args.out)
    print(f"pesos exportados para {args.out}")

    if ckpt_dir:
        export_weights(W, os.path.join(ckpt_dir, "last.bin"))

    if not args.no_quantize:
        quant_path = args.quant_out or _default_quant_path(args.out)
        print(f"quantizando int8/int16 -> {quant_path} (QA={args.qa} QB={args.qb})")
        quantize_file(args.out, quant_path, qa=args.qa, qb=args.qb)
        print(f"pesos quantizados gravados em {quant_path}")

    if args.plot_dir:
        plot_path = save_plots(history, args.plot_dir)
        print(f"plots de convergencia salvos em {plot_path}")

    print(f"\nresumo: {last_epoch} epoch(s) executados"
          f"{' (parada antecipada)' if stopped_early else ''}, "
          f"melhor {args.monitor}={stopper.best:.4f} no epoch {stopper.best_epoch}, "
          f"tempo total {time.time() - t0:.0f}s")
    return W, history


def _default_quant_path(out_path):
    if out_path.endswith(".bin"):
        return out_path[:-len(".bin")] + "_int8.bin"
    return out_path + "_int8.bin"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g_data = p.add_argument_group("dados")
    g_data.add_argument("--data", action="append", required=True,
                         help="arquivo(s) .bin de self-play; aceita lista separada por virgula, "
                              "diretorio, glob, ou a flag repetida varias vezes")
    g_data.add_argument("--val-data", action="append", default=None,
                         help="arquivo(s) .bin usados so para validacao (mesmas regras de --data); "
                              "se omitido, a validacao vem de uma fracao de --data (--val-split)")
    g_data.add_argument("--val-split", type=float, default=VAL_SPLIT_DEFAULT,
                         help=f"fracao de --data reservada para validacao quando --val-data nao "
                              f"e passado (default {VAL_SPLIT_DEFAULT})")
    g_data.add_argument("--init-from", default=None, help="pesos .bin existentes p/ continuar o treino")
    g_data.add_argument("--seed", type=int, default=SEED_DEFAULT)

    g_opt = p.add_argument_group("otimizacao")
    g_opt.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    g_opt.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    g_opt.add_argument("--lr", type=float, default=LR_DEFAULT)
    g_opt.add_argument("--lr-min", type=float, default=LR_MIN_DEFAULT)
    g_opt.add_argument("--lr-schedule", choices=["none", "step", "exponential", "cosine"],
                        default=LR_SCHEDULE_DEFAULT)
    g_opt.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS_DEFAULT)
    g_opt.add_argument("--step-size", type=int, default=STEP_SIZE_DEFAULT,
                        help="epochs por degrau em --lr-schedule=step")
    g_opt.add_argument("--step-gamma", type=float, default=STEP_GAMMA_DEFAULT)
    g_opt.add_argument("--exp-gamma", type=float, default=EXP_GAMMA_DEFAULT)

    g_wd = p.add_argument_group("weight decay (annealing)")
    g_wd.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY_DEFAULT,
                       help="weight decay desacoplado inicial (estilo AdamW), so em pesos")
    g_wd.add_argument("--weight-decay-min", type=float, default=WEIGHT_DECAY_MIN_DEFAULT,
                       help="valor para o qual o weight decay converge ao final do treino")
    g_wd.add_argument("--wd-schedule", choices=["none", "constant", "linear", "cosine"],
                       default=WD_SCHEDULE_DEFAULT)

    g_es = p.add_argument_group("early stopping")
    g_es.add_argument("--early-stop", dest="early_stop", action="store_true", default=EARLY_STOP_DEFAULT)
    g_es.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    g_es.add_argument("--patience", type=int, default=PATIENCE_DEFAULT)
    g_es.add_argument("--min-delta", type=float, default=MIN_DELTA_DEFAULT)
    g_es.add_argument("--monitor", choices=["val_loss", "val_outcome", "val_score",
                                             "val_policy", "val_policy_acc"],
                       default=MONITOR_DEFAULT)
    g_es.add_argument("--no-restore-best", action="store_true",
                       help="exporta os pesos do ultimo epoch em vez dos do melhor epoch")
    g_es.add_argument("--ckpt-dir", default=None,
                       help="diretorio para salvar best.bin (atualizado a cada melhora) e last.bin")

    g_mem = p.add_argument_group("orcamento de memoria")
    g_mem.add_argument("--ram-budget-gb", type=float, default=RAM_BUDGET_GB_DEFAULT,
                        help=f"orcamento de RAM usado para dimensionar o chunk de treino "
                             f"(default {RAM_BUDGET_GB_DEFAULT} GB)")
    g_mem.add_argument("--ram-chunk-fraction", type=float, default=RAM_CHUNK_FRACTION_DEFAULT,
                        help="fracao do orcamento de RAM reservada ao buffer de shuffle")

    g_loss = p.add_argument_group("pesos de loss / QAT")
    g_loss.add_argument("--w-score", type=float, default=W_SCORE_DEFAULT)
    g_loss.add_argument("--w-outcome", type=float, default=W_OUTCOME_DEFAULT)
    g_loss.add_argument("--w-policy", type=float, default=W_POLICY_DEFAULT)
    g_loss.add_argument("--qa", type=int, default=QA_DEFAULT)
    g_loss.add_argument("--qb", type=int, default=QB_DEFAULT)

    g_out = p.add_argument_group("saida")
    g_out.add_argument("--out", required=True, help="caminho de saida dos pesos treinados (.bin)")
    g_out.add_argument("--no-quantize", action="store_true")
    g_out.add_argument("--quant-out", default=None)
    g_out.add_argument("--plot-dir", default=None,
                        help="diretorio para salvar plots de convergencia/validacao (PNG)")
    g_out.add_argument("--log-every", type=int, default=LOG_EVERY_DEFAULT)

    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
