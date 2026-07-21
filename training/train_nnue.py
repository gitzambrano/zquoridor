"""
train_nnue.py -- treino da NNUE do zquoridor em PyTorch, a partir dos
dados de self-play gerados pelo harness C++ (selfplay). Espelho exato da
arquitetura em nnue.hpp:

  acumulador: Linear(332, 256)                    -> w1, b1
  ativacao do acumulador: SCReLU  (clip(x,0,1)^2)
  cabeca de RESULTADO (WL, sem empate):
      Linear(256, 32) -> ClippedReLU -> Linear(32, 1)   -> wv1_wl,bv1_wl,wv2_wl,bv2_wl
  cabeca AUXILIAR (imitacao da heuristica evalSimple):
      Linear(256, 32) -> ClippedReLU -> Linear(32, 1)   -> wv1_aux,bv1_aux,wv2_aux,bv2_aux
  policy head: Linear(256, 209), aplicado direto na saida do SCReLU        -> wp, bp

Quantization-aware training (QAT): QA/QB sao constantes fixas (--qa/--qb,
mesmos valores usados em nnue.hpp/quantize_nnue.py) e um WeightClipper
(nos moldes do nnue-pytorch do Stockfish) e aplicado a cada passo do
otimizador, travando os pesos dentro do range representavel em int8
(cabecas) / int16 (acumulador).

Recursos:
  - Varios arquivos .bin em --data (lista separada por virgula, diretorio,
    glob, ou --data repetido), via MultiFileSelfPlay (read_selfplay.py) --
    os shards nunca sao concatenados inteiros em RAM.
  - Split treino/validacao por fracao (--val-split) ou --val-data explicito.
  - Treino em chunks dimensionados por orcamentos de RAM e VRAM
    (--ram-budget-gb, --vram-budget-gb; default calibrado para 32GB RAM /
    6GB VRAM). --batch-size e --chunk-size aceitam "auto" (default) para
    serem calculados a partir desses orcamentos, ou um inteiro explicito.
  - AdamW com weight decay desacoplado (so em pesos, nunca em bias) e
    annealing (--weight-decay, --weight-decay-min, --wd-schedule).
  - LR schedule com warmup (--lr-schedule, --warmup-epochs, --lr-min,
    --step-size, --step-gamma, --exp-gamma).
  - Early stopping com melhor checkpoint salvo automaticamente
    (--early-stop, --patience, --min-delta, --monitor, --ckpt-dir); ao
    final, os pesos exportados sao os do melhor epoch (nao os do ultimo),
    a menos que --no-restore-best seja passado.
  - Plots de convergencia/validacao em PNG (--plot-dir).

Todos os defaults abaixo (secao DEFAULT CONFIG) valem como "flags no
cabecalho do arquivo": editar as constantes muda o comportamento padrao
sem precisar passar nada na linha de comando; qualquer flag de linha de
comando sobrescreve a constante correspondente.

Exemplos:
    python3 train_nnue.py --data ../data/selfplay_*.bin \
        --out ../data/nnue/nnue_weights.bin --plot-dir ../data/plots

    python3 train_nnue.py --data a.bin,b.bin --init-from prev.bin \
        --epochs 80 --weight-decay 2e-4 --wd-schedule cosine \
        --vram-budget-gb 6 --ram-budget-gb 32 \
        --early-stop --patience 10 --monitor val_loss \
        --out ../data/nnue/nnue_weights.bin --ckpt-dir ../data/checkpoints
"""
import argparse
import copy
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_selfplay import SAMPLE_DTYPE, DIST_BUCKETS, load_multi_selfplay  # noqa: E402
from quantize_nnue import quantize_file  # noqa: E402

# ============================== DEFAULT CONFIG ==============================
# Editar aqui muda o default sem precisar de flag; toda entrada tem uma
# flag de linha de comando correspondente que sobrescreve o valor abaixo.
DATA_DEFAULT = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "selfplay")
]
OUT_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nnue", "nnue_weights.bin")

EPOCHS_DEFAULT = 60
BATCH_SIZE_DEFAULT = "auto"          # inteiro, ou "auto" (calculado a partir de --vram-budget-gb)
SEED_DEFAULT = 0
VAL_SPLIT_DEFAULT = 0.1

LR_DEFAULT = 1e-3
LR_MIN_DEFAULT = 1e-5
LR_SCHEDULE_DEFAULT = "cosine"        # none | step | exponential | cosine
WARMUP_EPOCHS_DEFAULT = 2
STEP_SIZE_DEFAULT = 10
STEP_GAMMA_DEFAULT = 0.5
EXP_GAMMA_DEFAULT = 0.97

# Weight decay desacoplado (AdamW real do torch, decoupled por construcao):
# aplicado so as matrizes de peso (nunca aos bias), e "annealed" ao longo
# do treino -- regulariza mais forte no inicio, afrouxa perto do fim para
# nao atrapalhar o ajuste fino dos ultimos epochs (mesma postura do
# nnue-pytorch do Stockfish, que tambem reduz regularizacao com o tempo).
WEIGHT_DECAY_DEFAULT = 1e-4
WEIGHT_DECAY_MIN_DEFAULT = 0.0
WD_SCHEDULE_DEFAULT = "cosine"        # none | constant | linear | cosine

EARLY_STOP_DEFAULT = True
PATIENCE_DEFAULT = 8
MIN_DELTA_DEFAULT = 1e-4
MONITOR_DEFAULT = "val_loss"          # val_loss | val_outcome | val_score | val_policy | val_policy_acc

# Orcamentos de memoria usados para calcular batch-size/chunk-size "auto".
# Os defaults abaixo foram escolhidos para uma maquina tipica de
# desenvolvimento com 6GB de VRAM e 32GB de RAM (ver compute_auto_batch_size
# / compute_chunk_size para a formula e as margens de seguranca).
VRAM_BUDGET_GB_DEFAULT = 6.0
RAM_BUDGET_GB_DEFAULT = 32.0
RAM_CHUNK_FRACTION_DEFAULT = 0.25     # fracao do orcamento de RAM usada como buffer de shuffle
CHUNK_SIZE_DEFAULT = "auto"           # inteiro, ou "auto" (calculado a partir de --ram-budget-gb)

W_SCORE_DEFAULT = 0.3
W_OUTCOME_DEFAULT = 1.0
W_POLICY_DEFAULT = 1.0
QA_DEFAULT = 255
QB_DEFAULT = 64
LOG_EVERY_DEFAULT = 1
DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"
# =============================================================================

N, WS = 9, 8
NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS   # 332
HIDDEN = 256
POLICY_OUT = N * N + WS * WS * 2                                # 209
VALUE_SCALE = 200.0

INT8_MAX = 127
INT16_MAX = 32767


# --- modelo: espelho exato de nnue.hpp --------------------------------------
def screlu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0) ** 2


def clipped_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


class QuoridorNNUE(nn.Module):
    """Duas cabecas de valor independentes: value1_wl/value2_wl (resultado,
    WL) e value1_aux/value2_aux (imitacao da heuristica). Cada uma tem seu
    proprio bottleneck 256->32->1, sem nada compartilhado alem do
    acumulador `a`."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, HIDDEN)
        self.value1_wl = nn.Linear(HIDDEN, 32)
        self.value2_wl = nn.Linear(32, 1)
        self.value1_aux = nn.Linear(HIDDEN, 32)
        self.value2_aux = nn.Linear(32, 1)
        self.policy = nn.Linear(HIDDEN, POLICY_OUT)

    def forward(self, x: torch.Tensor):
        acc = self.fc1(x)
        a = screlu(acc)
        h_wl = clipped_relu(self.value1_wl(a))
        value_wl = self.value2_wl(h_wl).squeeze(-1)
        h_aux = clipped_relu(self.value1_aux(a))
        value_aux = self.value2_aux(h_aux).squeeze(-1)
        policy_logits = self.policy(a)
        return value_wl, value_aux, policy_logits


class WeightClipper:
    """QAT: aplicado a cada passo do otimizador. Trava cada matriz de pesos
    dentro do range representavel na escala fixa correspondente -- int16 a
    escala QA para o acumulador, int8 a escala QB para as cabecas."""

    def __init__(self, qa: float = QA_DEFAULT, qb: float = QB_DEFAULT):
        self.qa = qa
        self.qb = qb

    def __call__(self, model: "QuoridorNNUE"):
        w1_max = INT16_MAX / self.qa
        head_max = INT8_MAX / self.qb
        with torch.no_grad():
            model.fc1.weight.clamp_(-w1_max, w1_max)
            model.fc1.bias.clamp_(-w1_max, w1_max)
            for layer in (model.value1_wl, model.value2_wl,
                          model.value1_aux, model.value2_aux, model.policy):
                layer.weight.clamp_(-head_max, head_max)


# --- features -----------------------------------------------------------------
def to_batch_tensors(batch: np.ndarray, device):
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

    return {
        "x": torch.from_numpy(x).to(device, non_blocking=True),
        "search_score": torch.from_numpy(batch["search_score"].astype(np.float32)).to(device, non_blocking=True),
        "game_result": torch.from_numpy(batch["game_result"].astype(np.float32)).to(device, non_blocking=True),
        "policy_target": torch.from_numpy(batch["policy_target"].astype(np.int64)).to(device, non_blocking=True),
    }


# --- orcamento de memoria: auto batch-size (VRAM) / chunk-size (RAM) -------
def compute_auto_batch_size(vram_budget_gb, reserved_gb=1.0, min_bs=64, max_bs=16384):
    """Estimativa conservadora de quantas amostras cabem por batch dentro do
    orcamento de VRAM informado. `reserved_gb` cobre o contexto CUDA e os
    pesos/estados do otimizador (a rede e minuscula, isso domina o
    orcamento fixo). Para o restante, soma-se entrada + ativacoes de todas
    as camadas + gradientes, com fator de seguranca 4x para os buffers
    intermediarios que o autograd mantem vivos durante o backward."""
    elems_per_sample = NUM_FEATURES + HIDDEN + 2 * 32 + POLICY_OUT
    bytes_per_sample = elems_per_sample * 4 * 4  # float32, fwd+bwd, margem 4x
    usable = max(0.0, (vram_budget_gb - reserved_gb)) * (1024 ** 3)
    bs = int(usable // max(1, bytes_per_sample))
    return max(min_bs, min(max_bs, bs))


def compute_chunk_size(n_total, ram_budget_gb, fraction=RAM_CHUNK_FRACTION_DEFAULT,
                        min_size=20_000, max_size=5_000_000):
    """Numero de posicoes lidas por vez do dataset (structs crus) para
    formar o buffer de shuffle, escolhido para caber em `fraction` do
    orcamento de RAM informado."""
    budget_bytes = ram_budget_gb * (1024 ** 3) * fraction
    bytes_per_sample = SAMPLE_DTYPE.itemsize + NUM_FEATURES * 4
    size = int(budget_bytes // max(1, bytes_per_sample))
    size = max(min_size, min(max_size, size))
    return min(size, max(1, n_total))


def resolve_int_or_auto(value, auto_fn):
    if isinstance(value, str) and value.strip().lower() == "auto":
        return auto_fn()
    return int(value)


def iter_chunks(n, chunk_size, rng):
    perm = rng.permutation(n)
    for start in range(0, n, chunk_size):
        yield perm[start:start + chunk_size]


# --- schedules ----------------------------------------------------------------
def lr_at_epoch(epoch, epochs, base_lr, min_lr, schedule,
                 warmup_epochs=0, step_size=10, step_gamma=0.5, exp_gamma=0.97):
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


def apply_lr_wd(opt, lr, wd):
    for group in opt.param_groups:
        group["lr"] = lr
        if group.get("_is_weight_group", False):
            group["weight_decay"] = wd


# --- early stopping -------------------------------------------------------------
class EarlyStopper:
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

    def step(self, value, epoch, model):
        improved = self._improved(value)
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        should_stop = self.enabled and self.num_bad_epochs >= self.patience
        return improved, should_stop


# --- export no layout binario lido por nnue.hpp ------------------------------
def export_weights(model: QuoridorNNUE, path: str):
    model.eval()
    with torch.no_grad():
        w1 = model.fc1.weight.detach().cpu().numpy().T.astype(np.float32)
        b1 = model.fc1.bias.detach().cpu().numpy().astype(np.float32)

        def head_arrays(value1: nn.Linear, value2: nn.Linear):
            wv1 = value1.weight.detach().cpu().numpy().T.astype(np.float32)
            bv1 = value1.bias.detach().cpu().numpy().astype(np.float32)
            wv2 = value2.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
            bv2 = value2.bias.detach().cpu().numpy().astype(np.float32)
            return wv1, bv1, wv2, bv2

        wv1_wl, bv1_wl, wv2_wl, bv2_wl = head_arrays(model.value1_wl, model.value2_wl)
        wv1_aux, bv1_aux, wv2_aux, bv2_aux = head_arrays(model.value1_aux, model.value2_aux)
        wp = model.policy.weight.detach().cpu().numpy().astype(np.float32)
        bp = model.policy.bias.detach().cpu().numpy().astype(np.float32)

    assert w1.shape == (NUM_FEATURES, HIDDEN)
    assert wv1_wl.shape == (HIDDEN, 32)
    assert wv1_aux.shape == (HIDDEN, 32)
    assert wp.shape == (POLICY_OUT, HIDDEN)

    dir_name = os.path.dirname(os.path.abspath(path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "wb") as f:
        f.write(np.ascontiguousarray(w1).tobytes())
        f.write(np.ascontiguousarray(b1).tobytes())
        f.write(np.ascontiguousarray(wv1_wl).tobytes())
        f.write(np.ascontiguousarray(bv1_wl).tobytes())
        f.write(np.ascontiguousarray(wv2_wl).tobytes())
        f.write(np.ascontiguousarray(bv2_wl).tobytes())
        f.write(np.ascontiguousarray(wv1_aux).tobytes())
        f.write(np.ascontiguousarray(bv1_aux).tobytes())
        f.write(np.ascontiguousarray(wv2_aux).tobytes())
        f.write(np.ascontiguousarray(bv2_aux).tobytes())
        f.write(np.ascontiguousarray(wp).tobytes())
        f.write(np.ascontiguousarray(bp).tobytes())


def _default_quant_path(out_path: str) -> str:
    if out_path.endswith(".bin"):
        return out_path[: -len(".bin")] + "_int8.bin"
    return out_path + "_int8.bin"


# --- avaliacao em chunks (limitado por RAM/VRAM) -----------------------------
@torch.no_grad()
def run_eval(ds, indices, batch_size, chunk_size, model, device, w_score, w_outcome, w_policy):
    model.eval()
    total = dict(loss=0.0, score=0.0, outcome=0.0, policy=0.0, correct=0)
    n_items = len(indices)
    for start in range(0, n_items, chunk_size):
        chunk_idx = indices[start:start + chunk_size]
        chunk = ds[chunk_idx]
        for bs in range(0, len(chunk), batch_size):
            batch = chunk[bs:bs + batch_size]
            t = to_batch_tensors(batch, device)
            score_t = t["search_score"] / VALUE_SCALE
            result_t = (t["game_result"] + 1.0) / 2.0
            policy_t = t["policy_target"]

            value_wl, value_aux, policy_logits = model(t["x"])
            loss_outcome = F.binary_cross_entropy_with_logits(value_wl, result_t)
            loss_score = F.mse_loss(value_aux / VALUE_SCALE, score_t)
            loss_policy = F.cross_entropy(policy_logits, policy_t)
            loss = w_outcome * loss_outcome + w_score * loss_score + w_policy * loss_policy

            nb = len(batch)
            total["loss"] += loss.item() * nb
            total["score"] += loss_score.item() * nb
            total["outcome"] += loss_outcome.item() * nb
            total["policy"] += loss_policy.item() * nb
            total["correct"] += (policy_logits.argmax(dim=-1) == policy_t).sum().item()
    for k in ("loss", "score", "outcome", "policy"):
        total[k] /= max(1, n_items)
    total["policy_acc"] = total["correct"] / max(1, n_items)
    return total


# --- plots ----------------------------------------------------------------------
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


# --- treino ------------------------------------------------------------------
def train(args):
    train_paths, train_ds = load_multi_selfplay(args.data)
    print(f"dados de treino: {len(train_paths)} arquivo(s), {len(train_ds):,} posicoes")
    for p, n in train_ds.sizes():
        print(f"  - {p}: {n:,} posicoes")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

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

    device = torch.device(args.device)
    batch_size = resolve_int_or_auto(args.batch_size, lambda: compute_auto_batch_size(args.vram_budget_gb))
    chunk_size = resolve_int_or_auto(
        args.chunk_size, lambda: compute_chunk_size(len(train_idx), args.ram_budget_gb, args.ram_chunk_fraction))
    print(f"orcamento de VRAM: {args.vram_budget_gb:.1f} GB -> batch_size={batch_size:,}")
    print(f"orcamento de RAM: {args.ram_budget_gb:.1f} GB -> chunk_size={chunk_size:,} posicoes/bloco")
    print(f"device: {device}")

    model = QuoridorNNUE().to(device)
    if args.init_from:
        raw = _load_raw_weights(args.init_from)
        _load_into_model(model, raw)
        print(f"pesos iniciais carregados de {args.init_from} (continuando treino)")
    else:
        print("pesos iniciais aleatorios (treino do zero)")

    clipper = WeightClipper(qa=args.qa, qb=args.qb)
    clipper(model)
    print(f"QAT: QA={args.qa} QB={args.qb} (pesos travados a cada passo)")

    bias_params, weight_params = [], []
    for name, param in model.named_parameters():
        (bias_params if name.endswith(".bias") else weight_params).append(param)
    opt = torch.optim.AdamW([
        {"params": weight_params, "weight_decay": args.weight_decay, "_is_weight_group": True},
        {"params": bias_params, "weight_decay": 0.0, "_is_weight_group": False},
    ], lr=args.lr)

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
        apply_lr_wd(opt, lr, wd)

        model.train()
        tr_total = dict(loss=0.0, score=0.0, outcome=0.0, policy=0.0, correct=0)
        n_train_items = len(train_idx)
        for chunk_idx in iter_chunks(n_train_items, chunk_size, rng):
            global_idx = train_idx[chunk_idx]
            chunk = train_ds[global_idx]
            perm_local = rng.permutation(len(chunk))
            for start in range(0, len(perm_local), batch_size):
                batch_idx = perm_local[start:start + batch_size]
                batch = chunk[batch_idx]
                t = to_batch_tensors(batch, device)
                score_t = t["search_score"] / VALUE_SCALE
                result_t = (t["game_result"] + 1.0) / 2.0
                policy_t = t["policy_target"]

                value_wl, value_aux, policy_logits = model(t["x"])
                loss_outcome = F.binary_cross_entropy_with_logits(value_wl, result_t)
                loss_score = F.mse_loss(value_aux / VALUE_SCALE, score_t)
                loss_policy = F.cross_entropy(policy_logits, policy_t)
                loss = args.w_outcome * loss_outcome + args.w_score * loss_score + args.w_policy * loss_policy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                clipper(model)

                nb = len(batch)
                tr_total["loss"] += loss.item() * nb
                tr_total["score"] += loss_score.item() * nb
                tr_total["outcome"] += loss_outcome.item() * nb
                tr_total["policy"] += loss_policy.item() * nb
                tr_total["correct"] += (policy_logits.argmax(dim=-1) == policy_t).sum().item()
        for k in ("loss", "score", "outcome", "policy"):
            tr_total[k] /= max(1, n_train_items)
        tr_total["policy_acc"] = tr_total["correct"] / max(1, n_train_items)

        va = run_eval(val_ds, val_idx, batch_size, chunk_size, model, device,
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
        improved, should_stop = stopper.step(monitored, epoch, model)

        if ckpt_dir and improved:
            _export_state_dict(stopper.best_state, os.path.join(ckpt_dir, "best.bin"), device)

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
        model.load_state_dict(stopper.best_state)
    elif stopper.best_state is not None:
        print(f"\nmelhor epoch registrado: {stopper.best_epoch} ({args.monitor}={stopper.best:.4f}); "
              f"export final usa os pesos do ultimo epoch (--no-restore-best)")

    export_weights(model, args.out)
    print(f"pesos exportados para {args.out}")

    if ckpt_dir:
        export_weights(model, os.path.join(ckpt_dir, "last.bin"))

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
    return model, history


def _export_state_dict(state_dict, path, device):
    model = QuoridorNNUE().to(device)
    model.load_state_dict(state_dict)
    export_weights(model, path)


def _load_raw_weights(path):
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
    return dict(w1=w1, b1=b1, wv1_wl=wv1_wl, bv1_wl=bv1_wl, wv2_wl=wv2_wl, bv2_wl=bv2_wl,
                wv1_aux=wv1_aux, bv1_aux=bv1_aux, wv2_aux=wv2_aux, bv2_aux=bv2_aux, wp=wp, bp=bp)


def _load_into_model(model: QuoridorNNUE, raw: dict):
    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(raw["w1"].T.copy()))
        model.fc1.bias.copy_(torch.from_numpy(raw["b1"]))
        model.value1_wl.weight.copy_(torch.from_numpy(raw["wv1_wl"].T.copy()))
        model.value1_wl.bias.copy_(torch.from_numpy(raw["bv1_wl"]))
        model.value2_wl.weight.copy_(torch.from_numpy(raw["wv2_wl"].reshape(1, -1)))
        model.value2_wl.bias.copy_(torch.from_numpy(raw["bv2_wl"]))
        model.value1_aux.weight.copy_(torch.from_numpy(raw["wv1_aux"].T.copy()))
        model.value1_aux.bias.copy_(torch.from_numpy(raw["bv1_aux"]))
        model.value2_aux.weight.copy_(torch.from_numpy(raw["wv2_aux"].reshape(1, -1)))
        model.value2_aux.bias.copy_(torch.from_numpy(raw["bv2_aux"]))
        model.policy.weight.copy_(torch.from_numpy(raw["wp"]))
        model.policy.bias.copy_(torch.from_numpy(raw["bp"]))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g_data = p.add_argument_group("dados")
    g_data.add_argument("--data", action="append", default=None,
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
    g_opt.add_argument("--batch-size", default=BATCH_SIZE_DEFAULT,
                         help='inteiro, ou "auto" para calcular a partir de --vram-budget-gb (default "auto")')
    g_opt.add_argument("--lr", type=float, default=LR_DEFAULT)
    g_opt.add_argument("--lr-min", type=float, default=LR_MIN_DEFAULT)
    g_opt.add_argument("--lr-schedule", choices=["none", "step", "exponential", "cosine"],
                        default=LR_SCHEDULE_DEFAULT)
    g_opt.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS_DEFAULT)
    g_opt.add_argument("--step-size", type=int, default=STEP_SIZE_DEFAULT,
                        help="epochs por degrau em --lr-schedule=step")
    g_opt.add_argument("--step-gamma", type=float, default=STEP_GAMMA_DEFAULT)
    g_opt.add_argument("--exp-gamma", type=float, default=EXP_GAMMA_DEFAULT)
    g_opt.add_argument("--device", default=DEVICE_DEFAULT)

    g_wd = p.add_argument_group("weight decay (annealing)")
    g_wd.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY_DEFAULT,
                       help="weight decay desacoplado inicial (AdamW), so em pesos (nunca em bias)")
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

    g_mem = p.add_argument_group("orcamento de memoria (VRAM/RAM)")
    g_mem.add_argument("--vram-budget-gb", type=float, default=VRAM_BUDGET_GB_DEFAULT,
                        help=f"orcamento de VRAM usado para calcular --batch-size=auto "
                             f"(default {VRAM_BUDGET_GB_DEFAULT} GB)")
    g_mem.add_argument("--ram-budget-gb", type=float, default=RAM_BUDGET_GB_DEFAULT,
                        help=f"orcamento de RAM usado para calcular --chunk-size=auto "
                             f"(default {RAM_BUDGET_GB_DEFAULT} GB)")
    g_mem.add_argument("--ram-chunk-fraction", type=float, default=RAM_CHUNK_FRACTION_DEFAULT,
                        help="fracao do orcamento de RAM reservada ao buffer de shuffle")
    g_mem.add_argument("--chunk-size", default=CHUNK_SIZE_DEFAULT,
                        help='inteiro, ou "auto" para calcular a partir de --ram-budget-gb (default "auto")')

    g_loss = p.add_argument_group("pesos de loss / QAT")
    g_loss.add_argument("--w-score", type=float, default=W_SCORE_DEFAULT)
    g_loss.add_argument("--w-outcome", type=float, default=W_OUTCOME_DEFAULT)
    g_loss.add_argument("--w-policy", type=float, default=W_POLICY_DEFAULT)
    g_loss.add_argument("--qa", type=int, default=QA_DEFAULT)
    g_loss.add_argument("--qb", type=int, default=QB_DEFAULT)

    g_out = p.add_argument_group("saida")
    g_out.add_argument("--out", default=OUT_DEFAULT, help="caminho de saida dos pesos treinados (.bin)")
    g_out.add_argument("--no-quantize", action="store_true")
    g_out.add_argument("--quant-out", default=None)
    g_out.add_argument("--plot-dir", default=None,
                        help="diretorio para salvar plots de convergencia/validacao (PNG)")
    g_out.add_argument("--log-every", type=int, default=LOG_EVERY_DEFAULT)

    args = p.parse_args(argv)
    if not args.data:
        args.data = DATA_DEFAULT
    return args


if __name__ == "__main__":
    train(parse_args())
