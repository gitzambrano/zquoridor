"""
train_nnue.py -- treino da NNUE do zquoridor em PyTorch, a partir dos
dados de self-play gerados pelo harness C++ (selfplay). Espelho exato da
arquitetura em nnue.hpp:

  acumulador: Linear(354, 256)                    -> w1, b1
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
  - Gradient clipping (--grad-clip-norm) aplicado antes de cada optimizer.step,
    estabiliza o treino QAT (ideia portada do mixtrain.py).
  - Diagnostico de device no inicio do treino: nome da GPU, VRAM total, e
    pico de VRAM alocado ao final de cada epoch (torch.cuda.max_memory_allocated).
  - Transferencia CPU->GPU em blocos (--gpu-chunk-multiplier): cada chunk lido
    do disco e enviado a GPU em sub-blocos de `batch_size * multiplier`
    posicoes, e os batches de treino sao fatiados diretamente na VRAM --
    reduz bastante o numero de round-trips host->device comparado a
    transferir por batch.
  - Console verboso com progresso dentro do epoch (posicoes processadas,
    % concluido, loss corrente, ETA do epoch) alem do ETA do treino inteiro,
    throttled por tempo (--progress-every-secs).

GPU vs CPU -- UM parametro so: mude USE_GPU_DEFAULT (secao DEFAULT CONFIG
abaixo). True usa CUDA se disponivel (cai pra CPU sozinho se nao houver
GPU); False forca CPU mesmo com GPU disponivel. Nao existe um script
separado "sem torch" -- o torch roda em CPU tao bem quanto qualquer
implementacao numpy pura (e ganha o autograd de graca), entao um arquivo
so cobre os dois casos. --device continua aceito pra casos especiais
(ex. "cuda:1" numa maquina com varias GPUs), mas o normal e nem tocar
nisso.

CHECKPOINT / RESUME (estilo Zchezz) -- responde as duvidas de sempre:
  - "Cada treino comeca do zero?" Nao por padrao. A CADA epoch (nao so
    quando ha melhora) o estado COMPLETO de treino -- pesos, estado do
    otimizador (momentos do AdamW), epoch atual, historico de metricas,
    estado do early-stopper e os RNGs (numpy + torch + cuda) -- e gravado
    em `<ckpt-dir>/train_state_<data>_<hora>_ep<N>.pt` (NAO sobrescreve o
    epoch anterior -- fica com o historico completo, nomeado por
    data/horario/epoca; ver find_latest_checkpoint_suffix). Isso e
    diferente de best_..._ep<N>.bin/last_..._ep<N>.bin (que sao so os
    PESOS, no layout binario que nnue.hpp le): train_state_*.pt e o que
    permite retomar um treino como se ele nunca tivesse parado (mesmo
    optimizer momentum, mesmo ponto do LR/WD schedule, mesmo historico
    pros plots).
  - "Como ele parte de um checkpoint?" Automaticamente. Ao rodar de novo
    com o mesmo --ckpt-dir (o default ja aponta pra data/checkpoints), o
    script acha o train_state_*.pt de MAIOR epoca, confere se a "impressao
    digital" da arquitetura bate (NUM_FEATURES/HIDDEN/POLICY_OUT/QA/QB --
    se mudou a arquitetura ou a escala QAT, o checkpoint antigo e
    incompativel e e ignorado com aviso) e, se bater, carrega tudo e
    continua exatamente do epoch seguinte. --init-from soh entra em jogo
    quando NAO ha checkpoint de resume valido (ele so inicializa PESOS, o
    otimizador comeca zerado). --fresh ignora qualquer checkpoint e forca
    treino do zero mesmo que exista um. --resume-config aponta pra um
    train_config_*.json ESPECIFICO (nao precisa ser o mais recente -- da
    pra voltar pra uma epoca antiga de proposito) e reaproveita so os
    hiperparametros + os pesos que ele referencia, sem o otimizador.
  - "E se parar no meio?" Ctrl+C (SIGINT) e capturado: o handler salva o
    trio train_state/train_config/last daquele epoch IMEDIATAMENTE com os
    pesos do exato momento da interrupcao (nao espera terminar o epoch) e
    so entao encerra. Ao rodar de novo, esse epoch (que estava incompleto)
    e refeito do zero -- so o epoch em andamento se perde, nao o treino
    inteiro.
  - "E se eu rodar de novo DEPOIS que o treino ja terminou (bateu o teto de
    --epochs, ou parou por early stopping)?" Comeca um NOVO CICLO
    automaticamente a partir dos PESOS salvos (warm start) -- otimizador,
    LR/WD schedule e early-stopping reiniciados do zero, epoch volta a
    contar de 1. E o comportamento certo pro fluxo de bootstrapping (gera
    mais self-play com run_selfplay.py, retreina em cima da rede anterior,
    repete). Se quisesse continuar exatamente o MESMO ciclo (schedule
    contínuo), bastaria ter passado um --epochs maior desde o inicio.

Todos os defaults abaixo (secao DEFAULT CONFIG) valem como "flags no
cabecalho do arquivo": editar as constantes muda o comportamento padrao
sem precisar passar nada na linha de comando; qualquer flag de linha de
comando sobrescreve a constante correspondente. O uso normal e so
`python3 train_nnue.py`, sem nenhuma flag -- os defaults (incluindo
--data, --out e --ckpt-dir, todos resolvidos relativos a este arquivo) ja
sao bons o bastante pro dia a dia.

Exemplos (uso normal, sem flags):
    python3 train_nnue.py

Exemplos (casos especiais):
    python3 train_nnue.py --data a.bin,b.bin --epochs 80 \
        --weight-decay 2e-4 --wd-schedule cosine

    python3 train_nnue.py --fresh --init-from ../data/nnue/nnue_weights_legado.bin
"""
import argparse
import copy
import json
import os
import signal
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

# GPU vs CPU: o UNICO parametro que decide isso. True = usa CUDA se
# `torch.cuda.is_available()` (cai pra CPU sozinho, sem erro, se nao
# houver GPU visivel). False = forca CPU mesmo com GPU disponivel (util
# pra depurar ou comparar). --device continua aceito na CLI pra casos
# especiais (ex. "cuda:1"), mas normalmente nem se toca nisso.
USE_GPU_DEFAULT = True
DEVICE_DEFAULT = "cuda" if (USE_GPU_DEFAULT and torch.cuda.is_available()) else "cpu"

# --- checkpoint / resume (estilo Zchezz) ------------------------------------
# ckpt-dir agora tem default proprio (nao mais None): resume automatico so
# funciona se houver um diretorio consistente entre execucoes. Ver a secao
# "CHECKPOINT / RESUME" no docstring do topo do arquivo para o fluxo completo.
CKPT_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "checkpoints")
# CORRECAO: estava True, e --fresh usava action="store_true" SEM um
# "--no-fresh" para desligar -- ou seja, args.fresh era sempre True (passando
# --fresh ou nao, o valor era o mesmo), entao `resumed = None if args.fresh
# else try_load_train_state(...)` nunca chamava try_load_train_state: o
# resume automatico documentado acima ("Cada treino comeca do zero? Nao por
# padrao.") na pratica NUNCA acontecia. False + --no-fresh (par de
# --fresh, mesmo padrao de --early-stop/--no-early-stop abaixo) restaura o
# comportamento pretendido.
FRESH_DEFAULT = False   # True (ou --fresh) ignora qualquer checkpoint existente e comeca do zero

# Estabilizacao do treino QAT (ideia do mixtrain.py). 0 ou negativo desliga.
GRAD_CLIP_NORM_DEFAULT = 1.0

# Quantos batches sao agrupados numa unica transferencia host->device. O
# chunk (dimensionado por --ram-budget-gb) e lido do disco e convertido em
# features numpy normalmente, mas em vez de mandar pra GPU um batch de cada
# vez, agrupamos `gpu_chunk_multiplier` batches por transferencia -- os
# batches de treino sao depois fatiados direto na VRAM, sem novo round-trip.
GPU_CHUNK_MULTIPLIER_DEFAULT = 8

# Intervalo minimo (segundos) entre linhas de progresso dentro do epoch.
PROGRESS_EVERY_SECS_DEFAULT = 5.0
# =============================================================================

N, WS = 9, 8
# WALLS_LEFT_BUCKETS: feature nova de 2026-08 -- ver nota completa em
# WALLS_LEFT_BUCKETS/NUM_FEATURES em nnue.hpp. Muros restantes de cada
# jogador (0..WALLS_PER_PLAYER=10), one-hot, 11 buckets por lado.
WALLS_PER_PLAYER = 10   # deve bater com WALLS_PER_PLAYER em rules.hpp
WALLS_LEFT_BUCKETS = WALLS_PER_PLAYER + 1  # 11
# CUIDADO: NUM_FEATURES está duplicado em TRÊS lugares (nnue.hpp,
# quantize_nnue.py, aqui) -- não existe hoje uma fonte única compartilhada
# entre C++ e os dois scripts Python. Mudar a arquitetura exige atualizar
# os três em conjunto -- ver a mesma nota (com o incidente real que isso
# já causou) em quantize_nnue.py.
NUM_FEATURES = N * N + N * N + WS * WS * 2 + 2 * DIST_BUCKETS + 2 * WALLS_LEFT_BUCKETS  # 354
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
def to_chunk_tensors(chunk: np.ndarray, device):
    """Converte um bloco de TrainingSample (array estruturado numpy) para
    tensores densos e transfere tudo para `device` de uma vez so. Chamar
    isso uma vez por bloco grande (em vez de uma vez por batch) e o que
    permite ao `iter_gpu_batches` amortizar o custo de transferencia
    host->device entre varios batches.

    NOTA (2026-08): own_pawn/opp_pawn/walls_h/walls_v em `chunk` já vêm
    ESPELHADOS pra perspectiva canônica do mover (mirroredPawnCell/
    mirrorWallBitboard em selfplay.hpp, no momento da gravação) -- não é
    preciso (nem seria possível: o formato não registra a identidade
    física 0/1 do mover) espelhar de novo aqui. Datasets .bin gravados
    ANTES desta mudança de arquitetura têm essas 4 colunas em coordenada
    CRUA (sem espelho) -- misturar com dados novos é silencioso (mesmo
    dtype/tamanho de arquivo) e corrompe o treino sem erro nenhum; regere
    o dataset de self-play inteiro ao adotar esta versão."""
    n = len(chunk)
    x = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    x[np.arange(n), chunk["own_pawn"]] = 1.0
    x[np.arange(n), 81 + chunk["opp_pawn"]] = 1.0
    wh = chunk["walls_h"].astype(np.uint64)
    wv = chunk["walls_v"].astype(np.uint64)
    bits_h = ((wh[:, None] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.float32)
    bits_v = ((wv[:, None] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.float32)
    x[:, 162:162 + 64] = bits_h
    x[:, 162 + 64:162 + 128] = bits_v
    own_bucket = np.minimum(chunk["own_dist"].astype(np.int64), DIST_BUCKETS - 1)
    opp_bucket = np.minimum(chunk["opp_dist"].astype(np.int64), DIST_BUCKETS - 1)
    x[np.arange(n), 290 + own_bucket] = 1.0
    x[np.arange(n), 290 + DIST_BUCKETS + opp_bucket] = 1.0
    # muros restantes (feature nova, 2026-08 -- ver WALLS_LEFT_BUCKETS
    # acima e a nota equivalente em nnue.hpp). walls_left_own/opp já
    # existiam no formato (read_selfplay.py) desde antes desta mudança --
    # só não eram usados como feature de entrada nenhuma.
    wl_base = 290 + 2 * DIST_BUCKETS  # 332
    own_wl_bucket = np.clip(chunk["walls_left_own"].astype(np.int64), 0, WALLS_LEFT_BUCKETS - 1)
    opp_wl_bucket = np.clip(chunk["walls_left_opp"].astype(np.int64), 0, WALLS_LEFT_BUCKETS - 1)
    x[np.arange(n), wl_base + own_wl_bucket] = 1.0
    x[np.arange(n), wl_base + WALLS_LEFT_BUCKETS + opp_wl_bucket] = 1.0

    return {
        "x": torch.from_numpy(x).to(device, non_blocking=True),
        "search_score": torch.from_numpy(chunk["search_score"].astype(np.float32)).to(device, non_blocking=True),
        "game_result": torch.from_numpy(chunk["game_result"].astype(np.float32)).to(device, non_blocking=True),
        "policy_target": torch.from_numpy(chunk["policy_target"].astype(np.int64)).to(device, non_blocking=True),
    }


def iter_gpu_batches(chunk: np.ndarray, device, batch_size: int, gpu_chunk_size: int):
    """Percorre `chunk` em sub-blocos de ate `gpu_chunk_size` posicoes,
    transferindo cada sub-bloco para `device` uma unica vez (via
    `to_chunk_tensors`), e dentro dele fatia os batches de treino/avaliacao
    (`batch_size`) diretamente na VRAM -- sem nenhum novo round-trip
    host->device por batch. Em CPU o efeito e neutro (o "device" ja e a
    propria RAM), mas em GPU isso reduz o numero de transferencias por um
    fator de ~gpu_chunk_size/batch_size."""
    n = len(chunk)
    gpu_chunk_size = max(batch_size, gpu_chunk_size)
    for gstart in range(0, n, gpu_chunk_size):
        sub = chunk[gstart:gstart + gpu_chunk_size]
        t = to_chunk_tensors(sub, device)
        n_sub = len(sub)
        for start in range(0, n_sub, batch_size):
            end = min(start + batch_size, n_sub)
            yield {
                "x": t["x"][start:end],
                "search_score": t["search_score"][start:end],
                "game_result": t["game_result"][start:end],
                "policy_target": t["policy_target"][start:end],
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


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}h{m:02d}m"
    return f"{m:02d}m{s:02d}s"


def print_device_info(device):
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        print(f"device: {device}  |  GPU: {name}  |  VRAM total: {total_gb:.1f} GB")
    else:
        print(f"device: {device}  (CPU -- nenhuma GPU CUDA em uso; "
              f"passe --device cuda se esperava usar uma)")


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


# --- checkpoint / resume (estado COMPLETO de treino, estilo Zchezz) ---------
# Diferenca de best.bin/last.bin: aqueles sao so os PESOS, no layout binario
# que nnue.hpp le. train_state.pt carrega tambem otimizador, epoch, historico,
# early-stopper e RNGs -- e o que permite retomar o treino como se ele nunca
# tivesse parado, em vez de so reaproveitar os pesos como ponto de partida
# (isso ja existia via --init-from, mas reseta o momentum do AdamW e o
# progresso do LR/WD schedule).
def compute_fingerprint(args):
    """'Impressao digital' da arquitetura + escala QAT. Se um checkpoint
    salvo tiver uma impressao diferente da rodada atual (mudou HIDDEN,
    trocou QA/QB etc.), ele e incompativel e nao pode ser usado pra
    resume -- os tensores nem teriam o shape certo."""
    return dict(num_features=NUM_FEATURES, hidden=HIDDEN, policy_out=POLICY_OUT,
                qa=args.qa, qb=args.qb)


def _checkpoint_suffix(epoch):
    """Sufixo `{timestamp}_ep{epoch:04d}` usado em TODOS os arquivos de um
    mesmo save (train_state_<suffix>.pt, train_config_<suffix>.json,
    last_<suffix>.bin, best_<suffix>.bin quando há um novo melhor) --
    checkpoints não se sobrescrevem mais a cada epoch, ficam com histórico
    completo, nomeados por data/horário/época (pedido explícito). Calculado
    UMA VEZ por evento de save (no chamador, não dentro de cada função de
    save individual) -- os arquivos de um mesmo save precisam compartilhar
    o MESMO sufixo, senão fica impossível re-associar train_state/config/
    bin do mesmo momento (ver find_latest_checkpoint_suffix abaixo)."""
    return f"{time.strftime('%Y%m%d_%H%M%S')}_ep{epoch:04d}"


def find_latest_checkpoint_suffix(ckpt_dir):
    """Varre ckpt_dir por train_state_*_ep*.pt e devolve o sufixo do mais
    recente -- maior época primeiro; empate (pode acontecer se um save de
    emergência via Ctrl+C cair na mesma época de um save normal) resolvido
    pelo timestamp mais recente. None se não houver checkpoint nenhum."""
    import glob as _glob
    import re as _re
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for path in _glob.glob(os.path.join(ckpt_dir, "train_state_*_ep*.pt")):
        m = _re.search(r"train_state_(\d{8}_\d{6}_ep\d+)\.pt$", os.path.basename(path))
        if m:
            candidates.append(m.group(1))
    if not candidates:
        return None

    def _sort_key(suffix):
        ts, ep_part = suffix.rsplit("_ep", 1)
        return (int(ep_part), ts)

    candidates.sort(key=_sort_key)
    return candidates[-1]


def _train_state_path(ckpt_dir, suffix):
    return os.path.join(ckpt_dir, f"train_state_{suffix}.pt")


def _config_json_path(ckpt_dir, suffix):
    return os.path.join(ckpt_dir, f"train_config_{suffix}.json")


# Hiperparametros de OTIMIZACAO/loss/QAT que fazem sentido herdar de uma
# rodada anterior via --resume-config. Deliberadamente NAO inclui caminhos
# de I/O (--data, --val-data, --out, --ckpt-dir, --init-from, --resume-config,
# --fresh, --device, --plot-dir) nem flags de performance/console
# (--gpu-chunk-multiplier, --progress-every-secs, --log-every) -- esses sao
# especificos da MAQUINA/rodada atual, nao do "treino" em si, e copia-los
# cegamente de um JSON antigo tende a surpreender mais do que ajudar (ex.
# reaproveitar um --out de outra maquina).
_CONFIG_JSON_HYPERPARAM_KEYS = (
    "epochs", "batch_size", "seed", "val_split",
    "lr", "lr_min", "lr_schedule", "warmup_epochs", "step_size", "step_gamma", "exp_gamma",
    "weight_decay", "weight_decay_min", "wd_schedule",
    "early_stop", "patience", "min_delta", "monitor",
    "w_score", "w_outcome", "w_policy", "qa", "qb", "grad_clip_norm",
)


def save_config_json(ckpt_dir, args, fingerprint, epoch, best_metric, weights_path, suffix):
    """Grava train_config_<suffix>.json ao lado de train_state_<suffix>.pt:
    um snapshot HUMANO-LEGIVEL (json.dumps, sem torch/numpy) dos
    hiperparametros desta rodada + qual arquivo de pesos corresponde a
    este checkpoint. Nao substitui train_state.pt para resume EXATO (nao
    tem otimizador/RNG) -- e para (a) auditoria/registro de "com que
    config este .bin foi treinado" e (b) --resume-config, que reaproveita
    esses hiperparametros (nao o estado do otimizador) para comecar um
    ciclo novo de treino -- inclusive de uma epoca antiga especifica, nao
    só a mais recente, já que cada uma tem seu próprio arquivo agora."""
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "epoch": epoch,
        "best_metric": best_metric,
        "weights_path": os.path.abspath(weights_path) if weights_path else None,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hyperparams": {k: getattr(args, k) for k in _CONFIG_JSON_HYPERPARAM_KEYS if hasattr(args, k)},
    }
    path = _config_json_path(ckpt_dir, suffix)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)  # mesmo truque write-then-rename do train_state.pt


def load_config_json(path):
    """Le um train_config.json (de --resume-config OU do auto-detect em
    --ckpt-dir). Retorna None silenciosamente se o arquivo nao existe;
    lanca se existe mas esta corrompido/ilegivel (diferente do
    try_load_train_state, que so avisa em incompatibilidade de arquitetura
    -- aqui um JSON corrompido e sempre um erro do usuario apontando pro
    arquivo errado, vale falhar alto)."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_train_state(ckpt_dir, model, opt, epoch, epoch_completed, history, stopper, rng, fingerprint, suffix):
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = dict(
        fingerprint=fingerprint,
        epoch=epoch,
        epoch_completed=epoch_completed,
        model_state=model.state_dict(),
        opt_state=opt.state_dict(),
        history=history,
        stopper_best=stopper.best,
        stopper_best_epoch=stopper.best_epoch,
        stopper_best_state=stopper.best_state,
        stopper_num_bad_epochs=stopper.num_bad_epochs,
        numpy_rng_state=rng.bit_generator.state,
        torch_rng_state=torch.get_rng_state(),
        torch_cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    path = _train_state_path(ckpt_dir, suffix)
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # grava em arquivo temporario e renomeia -- evita
                                 # checkpoint corrompido se o processo morrer no meio da escrita


def try_load_train_state(ckpt_dir, fingerprint):
    """Localiza o checkpoint MAIS RECENTE em ckpt_dir (ver
    find_latest_checkpoint_suffix) e retorna o payload salvo se a
    arquitetura bater, senao None (silencioso se simplesmente nao existe
    checkpoint ainda; avisa se existe mas e incompativel)."""
    if not ckpt_dir:
        return None
    suffix = find_latest_checkpoint_suffix(ckpt_dir)
    if suffix is None:
        return None
    path = _train_state_path(ckpt_dir, suffix)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_fp = payload.get("fingerprint", {})
    if saved_fp != fingerprint:
        print(f"aviso: checkpoint de resume em {path} tem arquitetura/QAT diferente "
              f"da rodada atual ({saved_fp} != {fingerprint}) -- ignorando e comecando "
              f"do zero (apague os arquivos ou use um --ckpt-dir novo pra sumir com este aviso)")
        return None
    return payload


class _RunState:
    """Referencia mutavel compartilhada com o handler de SIGINT -- guarda so
    o que o handler precisa pra salvar um checkpoint de emergencia a
    qualquer momento, mesmo no meio de um epoch."""
    def __init__(self):
        self.model = None
        self.opt = None
        self.rng = None
        self.history = None
        self.stopper = None
        self.ckpt_dir = None
        self.fingerprint = None
        self.epoch = 0
        self.epoch_completed = False
        self.args = None  # p/ save_config_json (hiperparametros desta rodada)


_run_state = _RunState()
_ckpt_saved_on_interrupt = False


def _sigint_handler(signum, frame):
    """Ctrl+C: salva train_state_<suffix>.pt e last_<suffix>.bin com os
    pesos EXATOS do momento da interrupcao (nao espera o epoch terminar) e
    so entao encerra. Ao rodar de novo, o epoch interrompido e refeito --
    so ele se perde, nao o treino inteiro."""
    global _ckpt_saved_on_interrupt
    print("\n\n[Ctrl+C] interrompido -- salvando checkpoint de resume e pesos atuais...")
    rs = _run_state
    if rs.model is not None and rs.ckpt_dir and not _ckpt_saved_on_interrupt:
        _ckpt_saved_on_interrupt = True
        suffix = _checkpoint_suffix(rs.epoch)
        save_train_state(rs.ckpt_dir, rs.model, rs.opt, rs.epoch, rs.epoch_completed,
                          rs.history, rs.stopper, rs.rng, rs.fingerprint, suffix)
        last_path = os.path.join(rs.ckpt_dir, f"last_{suffix}.bin")
        export_weights(rs.model, last_path)
        if rs.args is not None:
            best_metric = rs.stopper.best if rs.stopper is not None else None
            save_config_json(rs.ckpt_dir, rs.args, rs.fingerprint, rs.epoch, best_metric, last_path, suffix)
        status = "completo" if rs.epoch_completed else "parcial -- sera refeito ao retomar"
        print(f"checkpoint salvo em {rs.ckpt_dir} (epoch {rs.epoch}, sufixo {suffix}, {status})")
    else:
        print("nada para salvar ainda (interrompido antes do 1o checkpoint).")
    print("saindo.")
    sys.exit(130)


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
def run_eval(ds, indices, batch_size, chunk_size, gpu_chunk_size, model, device,
             w_score, w_outcome, w_policy):
    model.eval()
    total = dict(loss=0.0, score=0.0, outcome=0.0, policy=0.0, correct=0)
    n_items = len(indices)
    for start in range(0, n_items, chunk_size):
        chunk_idx = indices[start:start + chunk_size]
        chunk = ds[chunk_idx]
        for t in iter_gpu_batches(chunk, device, batch_size, gpu_chunk_size):
            score_t = t["search_score"] / VALUE_SCALE
            result_t = (t["game_result"] + 1.0) / 2.0
            policy_t = t["policy_target"]

            value_wl, value_aux, policy_logits = model(t["x"])
            loss_outcome = F.binary_cross_entropy_with_logits(value_wl, result_t)
            loss_score = F.mse_loss(value_aux / VALUE_SCALE, score_t)
            loss_policy = F.cross_entropy(policy_logits, policy_t)
            loss = w_outcome * loss_outcome + w_score * loss_score + w_policy * loss_policy

            nb = len(t["x"])
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
    global _ckpt_saved_on_interrupt
    _ckpt_saved_on_interrupt = False
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
    print_device_info(device)
    batch_size = resolve_int_or_auto(args.batch_size, lambda: compute_auto_batch_size(args.vram_budget_gb))
    chunk_size = resolve_int_or_auto(
        args.chunk_size, lambda: compute_chunk_size(len(train_idx), args.ram_budget_gb, args.ram_chunk_fraction))
    gpu_chunk_size = batch_size * max(1, args.gpu_chunk_multiplier)
    print(f"orcamento de VRAM: {args.vram_budget_gb:.1f} GB -> batch_size={batch_size:,}")
    print(f"orcamento de RAM: {args.ram_budget_gb:.1f} GB -> chunk_size={chunk_size:,} posicoes/bloco")
    print(f"transferencia GPU em blocos de {gpu_chunk_size:,} posicoes "
          f"({args.gpu_chunk_multiplier}x o batch_size)")
    if args.grad_clip_norm and args.grad_clip_norm > 0:
        print(f"gradient clipping: max_norm={args.grad_clip_norm}")
    total_chunks_per_epoch = max(1, -(-len(train_idx) // chunk_size))
    print(f"chunks por epoch (RAM): ~{total_chunks_per_epoch}")

    ckpt_dir = args.ckpt_dir
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    fingerprint = compute_fingerprint(args)
    resumed = None if args.fresh else try_load_train_state(ckpt_dir, fingerprint)

    model = QuoridorNNUE().to(device)
    new_cycle = False
    if resumed is not None:
        model.load_state_dict(resumed["model_state"])
        prev_epoch = resumed["epoch"]
        prev_completed = resumed["epoch_completed"]
        prev_bad_epochs = resumed.get("stopper_num_bad_epochs", 0)
        hit_ceiling = prev_epoch >= args.epochs
        early_stopped = args.early_stop and prev_bad_epochs >= args.patience
        cycle_exhausted = prev_completed and (hit_ceiling or early_stopped)
        if cycle_exhausted:
            new_cycle = True
            print(f"pesos carregados de {_train_state_path(ckpt_dir, find_latest_checkpoint_suffix(ckpt_dir))} "
                  f"(ciclo anterior ja tinha terminado no epoch {prev_epoch}) -- "
                  f"iniciando um NOVO CICLO de treino a partir desses pesos "
                  f"(otimizador, LR/WD schedule e early-stopping reiniciados; "
                  f"e o comportamento certo pra continuar apos gerar mais self-play). "
                  f"Use --fresh pra tambem descartar os pesos e comecar aleatorio.")
        else:
            status = "completo" if prev_completed else "parcial (sera refeito)"
            print(f"RETOMANDO treino a partir de {_train_state_path(ckpt_dir, find_latest_checkpoint_suffix(ckpt_dir))} "
                  f"(ultimo epoch salvo: {prev_epoch}, {status}) -- mesmo otimizador, "
                  f"schedule e early-stopping de onde parou")
        if args.init_from:
            print(f"  (--init-from ignorado: ha um checkpoint valido em {ckpt_dir})")
    elif args.init_from:
        raw = _load_raw_weights(args.init_from)
        _load_into_model(model, raw)
        print(f"pesos iniciais carregados de {args.init_from} (continuando treino, "
              f"otimizador comeca do zero -- para retomar COM otimizador/schedule, "
              f"use o mesmo --ckpt-dir de uma rodada anterior em vez de --init-from)")
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

    history = {k: [] for k in (
        "epoch", "train_loss", "val_loss", "train_outcome", "val_outcome",
        "train_score", "val_score", "train_policy", "val_policy",
        "train_policy_acc", "val_policy_acc", "lr", "wd",
    )}

    start_epoch = 1
    if resumed is not None and not new_cycle:
        opt.load_state_dict(resumed["opt_state"])
        rng.bit_generator.state = resumed["numpy_rng_state"]
        torch.set_rng_state(resumed["torch_rng_state"])
        if resumed.get("torch_cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resumed["torch_cuda_rng_state"])
        history = resumed["history"]
        stopper.best = resumed["stopper_best"]
        stopper.best_epoch = resumed["stopper_best_epoch"]
        stopper.best_state = resumed["stopper_best_state"]
        stopper.num_bad_epochs = resumed["stopper_num_bad_epochs"]
        start_epoch = resumed["epoch"] + 1 if resumed["epoch_completed"] else resumed["epoch"]
    # new_cycle=True: fica tudo com o default (start_epoch=1, otimizador novo,
    # history/stopper zerados) -- so os PESOS do model vieram do checkpoint.

    # handler de Ctrl+C: salva um checkpoint de emergencia com o estado atual
    # (mesmo no meio de um epoch) antes de encerrar.
    _run_state.model = model
    _run_state.opt = opt
    _run_state.rng = rng
    _run_state.history = history
    _run_state.stopper = stopper
    _run_state.ckpt_dir = ckpt_dir
    _run_state.fingerprint = fingerprint
    _run_state.args = args
    signal.signal(signal.SIGINT, _sigint_handler)

    t0 = time.time()
    stopped_early = False
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        _run_state.epoch = epoch
        _run_state.epoch_completed = False
        lr = lr_at_epoch(epoch, args.epochs, args.lr, args.lr_min, args.lr_schedule,
                          args.warmup_epochs, args.step_size, args.step_gamma, args.exp_gamma)
        wd = wd_at_epoch(epoch, args.epochs, args.weight_decay, args.weight_decay_min, args.wd_schedule)
        apply_lr_wd(opt, lr, wd)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        tr_total = dict(loss=0.0, score=0.0, outcome=0.0, policy=0.0, correct=0)
        n_train_items = len(train_idx)
        epoch_start_time = time.time()
        last_progress_print = epoch_start_time
        positions_done = 0
        chunk_num = 0

        for chunk_idx in iter_chunks(n_train_items, chunk_size, rng):
            chunk_num += 1
            global_idx = train_idx[chunk_idx]
            chunk = train_ds[global_idx]
            chunk = chunk[rng.permutation(len(chunk))]  # embaralha uma vez por chunk

            for t in iter_gpu_batches(chunk, device, batch_size, gpu_chunk_size):
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
                if args.grad_clip_norm and args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                opt.step()
                clipper(model)

                nb = len(t["x"])
                tr_total["loss"] += loss.item() * nb
                tr_total["score"] += loss_score.item() * nb
                tr_total["outcome"] += loss_outcome.item() * nb
                tr_total["policy"] += loss_policy.item() * nb
                tr_total["correct"] += (policy_logits.argmax(dim=-1) == policy_t).sum().item()
                positions_done += nb

                now = time.time()
                if args.progress_every_secs > 0 and now - last_progress_print >= args.progress_every_secs:
                    last_progress_print = now
                    frac = min(1.0, positions_done / max(1, n_train_items))
                    elapsed = now - epoch_start_time
                    eta_epoch = elapsed / frac - elapsed if frac > 0 else 0.0
                    running_loss = tr_total["loss"] / max(1, positions_done)
                    running_acc = tr_total["correct"] / max(1, positions_done)
                    print(f"  epoch {epoch:3d}/{args.epochs}  chunk {chunk_num:>3}/{total_chunks_per_epoch} | "
                          f"{frac * 100:5.1f}% | {positions_done / 1e6:6.2f}M/{n_train_items / 1e6:.2f}M pos | "
                          f"loss={running_loss:.4f} acc={running_acc:.3f} | "
                          f"ETA epoch: {format_duration(eta_epoch)}")

        for k in ("loss", "score", "outcome", "policy"):
            tr_total[k] /= max(1, n_train_items)
        tr_total["policy_acc"] = tr_total["correct"] / max(1, n_train_items)

        va = run_eval(val_ds, val_idx, batch_size, chunk_size, gpu_chunk_size, model, device,
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

        # checkpoint de resume: gravado a CADA epoch (nao so quando melhora),
        # e' o que permite retomar depois de fechar o processo normalmente.
        # UM sufixo por epoch, reaproveitado nos 3 arquivos (+ best.bin
        # quando ha melhora) -- precisa ser o MESMO em todos pra poderem
        # ser re-associados depois (ver find_latest_checkpoint_suffix).
        suffix = _checkpoint_suffix(epoch)
        if ckpt_dir and improved:
            _export_state_dict(stopper.best_state, os.path.join(ckpt_dir, f"best_{suffix}.bin"), device)

        _run_state.epoch_completed = True
        if ckpt_dir:
            save_train_state(ckpt_dir, model, opt, epoch, True, history, stopper, rng, fingerprint, suffix)
            last_path = os.path.join(ckpt_dir, f"last_{suffix}.bin")
            export_weights(model, last_path)
            save_config_json(ckpt_dir, args, fingerprint, epoch, stopper.best, last_path, suffix)

        epoch_duration = time.time() - epoch_start_time
        avg_epoch_time = (time.time() - t0) / epoch
        eta_run = format_duration(avg_epoch_time * (args.epochs - epoch))
        vram_note = ""
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            vram_note = f" | pico VRAM: {peak_gb:.2f} GB"

        if epoch % args.log_every == 0 or epoch == args.epochs or improved or should_stop:
            star = " *" if improved else "  "
            print(f"epoch {epoch:3d}/{args.epochs} | lr={lr:.2e} wd={wd:.2e} | "
                  f"treino: loss={tr_total['loss']:.4f} (score={tr_total['score']:.4f} "
                  f"outcome={tr_total['outcome']:.4f} policy={tr_total['policy']:.4f} "
                  f"acc={tr_total['policy_acc']:.3f}) | "
                  f"val: loss={va['loss']:.4f} (score={va['score']:.4f} outcome={va['outcome']:.4f} "
                  f"policy={va['policy']:.4f} acc={va['policy_acc']:.3f}){star} | "
                  f"epoch: {format_duration(epoch_duration)} | total: {time.time() - t0:.0f}s | "
                  f"ETA treino: {eta_run}{vram_note}")

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
    g_data.add_argument("--init-from", default=None,
                         help="pesos .bin existentes p/ inicializar (so PESOS; ignorado se houver "
                              "um checkpoint de resume valido em --ckpt-dir -- ver --fresh)")
    g_data.add_argument("--seed", type=int, default=SEED_DEFAULT)

    g_opt = p.add_argument_group("otimizacao")
    g_opt.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    g_opt.add_argument("--batch-size", default=BATCH_SIZE_DEFAULT,
                         help='inteiro, ou "auto" para calcular a partir de --vram-budget-gb (default "auto")')
    g_opt.add_argument("--lr", type=str, default="auto",
                        help="learning rate inicial. 'auto' (default): herda do "
                             "train_config.json do checkpoint retomado (--ckpt-dir "
                             "auto-detectado ou --resume-config); se não há checkpoint/"
                             "config nenhum pra herdar, avisa e usa LR_DEFAULT "
                             f"({LR_DEFAULT}). Um valor numérico aqui SEMPRE sobrepõe "
                             "o herdado, mesmo em resume -- o schedule (--lr-schedule "
                             "etc.) recalcula a partir do novo valor a cada epoch, então "
                             "trocar o LR no meio de um treino retomado tem efeito "
                             "imediato (ver apply_lr_wd() no loop principal).")
    g_opt.add_argument("--lr-min", type=float, default=LR_MIN_DEFAULT)
    g_opt.add_argument("--lr-schedule", choices=["none", "step", "exponential", "cosine"],
                        default=LR_SCHEDULE_DEFAULT)
    g_opt.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS_DEFAULT)
    g_opt.add_argument("--step-size", type=int, default=STEP_SIZE_DEFAULT,
                        help="epochs por degrau em --lr-schedule=step")
    g_opt.add_argument("--step-gamma", type=float, default=STEP_GAMMA_DEFAULT)
    g_opt.add_argument("--exp-gamma", type=float, default=EXP_GAMMA_DEFAULT)
    g_opt.add_argument("--device", default=DEVICE_DEFAULT,
                        help="caso especial (ex. 'cuda:1'); normalmente mude USE_GPU_DEFAULT no "
                             f"topo do arquivo em vez desta flag (default resolvido: {DEVICE_DEFAULT})")

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

    g_ckpt = p.add_argument_group("checkpoint / resume")
    g_ckpt.add_argument("--ckpt-dir", default=CKPT_DIR_DEFAULT,
                         help=f"diretorio de checkpoints. Cada epoch grava um trio "
                              f"train_state_<data>_<hora>_ep<N>.pt / train_config_..._ep<N>.json / "
                              f"last_..._ep<N>.bin (+ best_..._ep<N>.bin quando ha melhora) -- "
                              f"nao sobrescreve os anteriores, fica com o historico completo. "
                              f"Resume automatico usa o de maior epoca (ver "
                              f"find_latest_checkpoint_suffix); passe vazio/None via codigo pra "
                              f"desligar checkpointing (default {CKPT_DIR_DEFAULT})")
    g_ckpt.add_argument("--fresh", dest="fresh", action="store_true", default=FRESH_DEFAULT,
                         help="ignora qualquer checkpoint de resume existente (o trio train_state/"
                              "train_config/last mais recente) em --ckpt-dir e comeca do zero (ou "
                              "de --init-from, se passado)")
    g_ckpt.add_argument("--no-fresh", dest="fresh", action="store_false",
                         help="forca resume automatico (default -- so precisa desta flag se "
                              "FRESH_DEFAULT tiver sido mudado pra True no topo do arquivo)")
    g_ckpt.add_argument("--resume-config", default=None,
                         help="caminho para um train_config.json de uma rodada anterior (de "
                              "QUALQUER --ckpt-dir, nao so o atual) -- usa os hiperparametros "
                              "de otimizacao dali como default desta rodada (qualquer flag passada "
                              "nesta chamada continua tendo prioridade) e usa os pesos que ele "
                              "aponta como --init-from, caso --init-from nao tenha sido passado. "
                              "Diferente do resume automatico via --ckpt-dir: nao traz otimizador/"
                              "epoch/historico, so pesos + config -- serve para comecar um ciclo "
                              "novo (ex. bootstrap em cima de self-play novo) com os mesmos "
                              "hiperparametros de um treino anterior, sem redigitar cada flag.")

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
    g_loss.add_argument("--grad-clip-norm", type=float, default=GRAD_CLIP_NORM_DEFAULT,
                         help=f"clip de norma do gradiente antes de cada optimizer.step; "
                              f"0 ou negativo desliga (default {GRAD_CLIP_NORM_DEFAULT})")

    g_perf = p.add_argument_group("performance / console")
    g_perf.add_argument("--gpu-chunk-multiplier", type=int, default=GPU_CHUNK_MULTIPLIER_DEFAULT,
                         help="quantos batches sao agrupados por transferencia host->device "
                              f"(default {GPU_CHUNK_MULTIPLIER_DEFAULT}x o batch_size)")
    g_perf.add_argument("--progress-every-secs", type=float, default=PROGRESS_EVERY_SECS_DEFAULT,
                         help="intervalo minimo entre linhas de progresso dentro do epoch; "
                              f"0 desliga (default {PROGRESS_EVERY_SECS_DEFAULT}s)")

    g_out = p.add_argument_group("saida")
    g_out.add_argument("--out", default=OUT_DEFAULT, help="caminho de saida dos pesos treinados (.bin)")
    g_out.add_argument("--no-quantize", action="store_true")
    g_out.add_argument("--quant-out", default=None)
    g_out.add_argument("--plot-dir", default=None,
                        help="diretorio para salvar plots de convergencia/validacao (PNG)")
    g_out.add_argument("--log-every", type=int, default=LOG_EVERY_DEFAULT)

    # --resume-config (ou auto-detect de <ckpt-dir>/train_config.json quando
    # --fresh nao foi passado): faz uma pre-leitura so de
    # --ckpt-dir/--fresh/--resume-config/--init-from para decidir se ha um
    # JSON pra aplicar, e se houver, reescreve os DEFAULTS do parser antes do
    # parse_args "de verdade" -- assim qualquer flag que o usuario passou
    # NESTA chamada continua ganhando (argparse so usa o default quando a
    # flag correspondente nao apareceu em argv).
    pre_args, _ = p.parse_known_args(argv)
    cfg_path = pre_args.resume_config
    if cfg_path is None and not pre_args.fresh and pre_args.ckpt_dir:
        latest_suffix = find_latest_checkpoint_suffix(pre_args.ckpt_dir)
        if latest_suffix is not None:
            auto_path = _config_json_path(pre_args.ckpt_dir, latest_suffix)
            if os.path.isfile(auto_path):
                cfg_path = auto_path
    hp_lr = None  # valor de lr herdado do JSON, se houver -- usado por --lr auto abaixo
    if cfg_path is not None:
        cfg = load_config_json(cfg_path)
        if cfg is not None:
            hp = {k: v for k, v in cfg.get("hyperparams", {}).items()
                  if k in _CONFIG_JSON_HYPERPARAM_KEYS}
            hp_lr = hp.get("lr")
            if hp:
                p.set_defaults(**hp)
                print(f"[config] hiperparametros herdados de {cfg_path} (epoch {cfg.get('epoch')}, "
                      f"best_metric={cfg.get('best_metric')}): {sorted(hp.keys())}")
            if pre_args.init_from is None and cfg.get("weights_path"):
                p.set_defaults(init_from=cfg["weights_path"])
                print(f"[config] --init-from herdado do JSON: {cfg['weights_path']}")

    args = p.parse_args(argv)
    if not args.data:
        args.data = DATA_DEFAULT

    # Resolve --lr auto (ver help do argumento acima). Resolvido aqui (não
    # deixado pra set_defaults sozinho) pra cobrir também o caso de alguém
    # digitar "--lr auto" explicitamente mesmo quando havia um hp_lr
    # herdável -- set_defaults só se aplica quando a flag NÃO aparece em
    # argv, então um "--lr auto" literal escondia o valor herdado sem essa
    # resolução explícita.
    if args.lr == "auto":
        if hp_lr is not None:
            args.lr = float(hp_lr)
            print(f"[lr] auto: herdado de {cfg_path} (lr={args.lr})")
        else:
            print(f"[lr] aviso: --lr auto pedido, mas nao ha checkpoint/config anterior "
                  f"em --ckpt-dir nem --resume-config -- comecando do zero com "
                  f"lr={LR_DEFAULT} (LR_DEFAULT)")
            args.lr = LR_DEFAULT
    else:
        args.lr = float(args.lr)

    return args


if __name__ == "__main__":
    train(parse_args())
