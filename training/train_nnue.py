"""
train_nnue.py -- treino da NNUE do zquoridor (PyTorch), a partir dos dados
de self-play gerados pelo harness C++. Espelho exato da arquitetura em
nnue.hpp:

  acumulador: Linear(354, 256)                    -> w1, b1
  SCReLU (clip(x,0,1)^2) na saida do acumulador
  cabeca RESULTADO (WL, sem empate, UNICA cabeca de valor -- a cabeca
  AUXILIAR de imitacao da heuristica evalSimple existiu ate 2026-08 e foi
  removida, ver nota em QuoridorNNUE mais abaixo):
      Linear(256, 32) -> ClippedReLU -> Linear(32, 1)   -> wv1_wl,bv1_wl,wv2_wl,bv2_wl
  policy head: Linear(256, 209), direto na saida do SCReLU                -> wp, bp

O alvo de treino da cabeca WL (2026-08+) e um BLEND por posicao entre o
resultado REAL do jogo e a avaliacao que a propria NNUE (de quando o
self-play rodou) deu aquela posicao, gravada em TrainingSample::evalNNUE
(ver read_selfplay.py/selfplay.hpp):
    wl_target = k * game_result_prob + (1 - k) * ev_prob
"k" e configuravel POR FONTE em DATA_SOURCES_DEFAULT/--data-sources (ver
nota la); k=1.0 ignora evalNNUE por completo (comportamento antigo, e o
que qualquer fonte de dado anterior a esta mudanca de formato deve usar).

QAT (quantization-aware training): QA/QB (--qa/--qb, mesmos valores de
nnue.hpp/quantize_nnue.py) definem a escala fixa; um WeightClipper (estilo
nnue-pytorch do Stockfish) trava os pesos no range representavel em int8
(cabecas) / int16 (acumulador) a cada passo do otimizador.
Uso normal, sem flags nenhuma -- os defaults (data, out, ckpt-dir, LR,
batch/chunk size "auto" etc., todos na secao DEFAULT CONFIG abaixo) ja
cobrem o dia a dia:
    python3 train_nnue.py
 
Casos especiais:
    python3 train_nnue.py --data a.bin,b.bin --epochs 80 \
        --weight-decay 2e-4 --wd-schedule cosine
    python3 train_nnue.py --fresh --init-from ../data/nnue/nnue_weights_legado.bin
 
GPU vs CPU: um unico parametro, USE_GPU_DEFAULT (secao DEFAULT CONFIG).
True = usa CUDA se disponivel, cai pra CPU sozinho se nao houver. False =
forca CPU mesmo com GPU disponivel. --device so pra casos especiais
(ex. "cuda:1" com varias GPUs).
 
CHECKPOINT / RESUME -- fluxo em 4 cenarios (o script imprime qual deles se
aplica logo no inicio de cada execucao):
 
  1. TREINO DO ZERO -- sem --ckpt-dir com checkpoint previo e sem
     --init-from: pesos aleatorios, epoch 1.
 
  2. TREINO A PARTIR DE PESOS EXISTENTES (--init-from arquivo.bin) -- so os
     PESOS sao carregados; otimizador/LR-schedule/early-stopping comecam do
     zero. Ignorado se ja houver um checkpoint valido em --ckpt-dir.
 
  3. RETOMANDO TREINO INTERROMPIDO -- --ckpt-dir tem um checkpoint de um
     treino que nao terminou (Ctrl+C ou --epochs ainda nao atingido).
     Restaura pesos, otimizador (momentos do AdamW), RNGs e historico
     EXATAMENTE do ultimo epoch salvo -- continua como se nunca tivesse
     parado. Se o epoch salvo foi interrompido no meio, ele e refeito do
     inicio (so ele, nao o treino inteiro).
 
  4. NOVO CICLO (bootstrap sobre self-play novo) -- --ckpt-dir tem um
     checkpoint de um treino que JA TERMINOU (--epochs atingido ou early
     stopping). So os PESOS sao reaproveitados (warm start); otimizador,
     LR/WD schedule e early-stopping reiniciam do zero, epoch volta a
     contar de 1. E o fluxo normal ao gerar uma nova geracao de self-play
     com a rede anterior e re-treinar em cima dela. Como o otimizador
     reinicia, o LR do treino do zero (1e-3) é agressivo demais e
     arriscaria afastar pesos ja convergidos -- por isso este cenario usa
     automaticamente um LR baixo de fine-tuning (--new-cycle-lr, default
     NEW_CYCLE_LR_DEFAULT) em vez de LR_DEFAULT, a menos que --lr numerico
     seja passado explicitamente.
 
  Arquivos gravados em --ckpt-dir, com CICLO+EPOCH no nome (ver CICLO
  abaixo) -- a cada epoch o arquivo do epoch anterior DESTE MESMO CICLO e
  apagado (so a versao mais recente do ciclo ATUAL fica no disco), mas
  arquivos de ciclos ANTERIORES (self-play de geracoes passadas) nunca sao
  apagados automaticamente -- e por isso que o ciclo entra no nome:
    - train_state_<ciclo>_ep<NNNN>.pt   estado completo p/ resume (pesos,
                                         otimizador, RNGs, historico,
                                         early-stopper, o proprio <ciclo>)
    - ckpt_<ciclo>_ep<NNNN>.bin          pesos (formato nnue.hpp) do MELHOR
                                         epoch do ciclo ate agora (val,
                                         --monitor) -- nao existe mais
                                         last.bin separado; se o epoch atual
                                         nao melhorou, o arquivo e regravado
                                         mesmo assim so pra atualizar o
                                         numero do epoch/timestamp no nome,
                                         com os MESMOS pesos de antes
    - train_config_<ciclo>_ep<NNNN>.json  hiperparametros da rodada, legivel
                                         por humano, mesmo epoch/ciclo do bin
  <ciclo> = data-hora (AAAAMMDD-HHMMSS) do INICIO do ciclo de treino atual
  (nao muda a cada epoch, so quando um ciclo novo comeca -- ver CICLO
  abaixo). Ctrl+C grava esses mesmos 3 arquivos com o estado exato da
  interrupcao antes de encerrar, seguindo a mesma regra de apagar so o
  epoch anterior do ciclo atual.

  Resume automatico: ao iniciar, o script procura em --ckpt-dir o arquivo
  train_state_*_ep*.pt mais recente (maior <ciclo>, e a maior epoch dentro
  dele) -- nao ha mais nome fixo. Isso cobre tanto retomar o ciclo atual
  quanto, se --ckpt-dir acumulou sobras de ciclos antigos, sempre pegar a
  mais nova.

  CICLO -- quando um <ciclo> novo comeca (timestamp novo no nome dos
  arquivos) vs. quando o MESMO <ciclo> continua (mesmo timestamp,
  sobrescrevendo o epoch anterior):
    - Resume de treino interrompido (cenario 3 abaixo): MESMO <ciclo> do
      checkpoint retomado.
    - Novo ciclo por bootstrap sobre self-play novo (cenario 4 abaixo):
      <ciclo> NOVO -- o ultimo arquivo do ciclo anterior fica no disco
      como registro historico daquela geracao (nao e apagado).
    - Treino do zero ou --init-from (cenarios 1 e 2): <ciclo> novo.
 
  A pasta final de export (--out, default data/nnue/) SO e atualizada
  quando o treino termina por completo (limite de --epochs ou early
  stopping) -- nunca durante o treino nem no Ctrl+C. Ao terminar, exporta
  os pesos do MELHOR epoch (nao do ultimo, a menos que --no-restore-best)
  e roda a quantizacao automatica pra int8/int16.
"""
import argparse
import copy
import glob
import json
import os
import re
import signal
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_selfplay import SAMPLE_DTYPE, DIST_BUCKETS, EV_SCALE, load_multi_selfplay, expand_data_paths  # noqa: E402
from quantize_nnue import quantize_file  # noqa: E402
 
# ============================== DEFAULT CONFIG ==============================
# Editar aqui muda o default sem precisar de flag; toda entrada tem uma flag
# de linha de comando correspondente que sobrescreve o valor abaixo.
 
# --- dados de treino ---------------------------------------------------------
# DATA_DEFAULT: usado so se DATA_SOURCES_DEFAULT (abaixo) estiver vazia.
DATA_DEFAULT = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "")
]
DATA_ROOT_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
 
# DATA_SOURCES_DEFAULT: lista de {"path": <pasta>, "frac": <0.0-1.0>} --
# mistura VARIAS geracoes de self-play com pesos diferentes (estilo
# mixtrain.py), em vez de treinar so em cima de DATA_DEFAULT/--data.
#   - "path" resolvido em ordem: como escrito -> DATA_ROOT/selfplay/<path>
#     -> DATA_ROOT/arena/<path> -> DATA_ROOT/<path>. A primeira que existir
#     "ganha" (resolve_data_source_path()).
#   - "frac": fracao de POSICOES (nao arquivos) sorteada aleatoriamente
#     (reprodutivel via --seed) daquela pasta. 1.0 = usa tudo, 0.0 = ignora.
#   - "k" (2026-08, opcional, default 1.0): peso do resultado REAL do jogo
#     vs. a avaliacao da propria NNUE gravada no .bin (TrainingSample::
#     evalNNUE, ver read_selfplay.py/selfplay.hpp) na hora de montar o alvo
#     de treino da cabeca WL, POR POSICAO:
#         wl_target = k * game_result_prob + (1 - k) * ev_prob
#     (ambos convertidos pra perspectiva do MOVER antes de misturar -- ver
#     to_chunk_tensors). k=1.0 (default) = comportamento antigo, ignora
#     ev_prob por completo -- e o que qualquer .bin gravado ANTES desta
#     mudanca de formato precisa usar (o campo nesse offset la e uma
#     heuristica antiga, nao uma avaliacao de rede; k=1.0 zera a
#     contribuicao dele sem precisar detectar o formato). k<1.0 e
#     "bootstrapping": deixa a propria NNUE (de quando o self-play rodou)
#     opinar sobre posicoes especificas de uma partida, em vez de so
#     herdar o resultado final igualmente pra toda posicao da partida --
#     tende a convergir mais rapido, mas em excesso pode fazer a rede
#     nova so reforcar os vieses da rede que gerou os dados. Ajuste por
#     tentativa; nao ha default "certo" alem de k=1.0 pra dado antigo.
#   - Pastas nao podem se sobrepor entre fontes (erro se um .bin cair em duas).
# Motivacao: manter uma fatia pequena de geracoes antigas evita que a rede
# "esqueca" padroes antigos e drifte para o otimo local do gen mais recente.
# Vazia = comportamento antigo, direto de DATA_DEFAULT/--data sem amostragem.
DATA_SOURCES_DEFAULT = [
      {"path": "selfplay/gen1", "frac": 0.1, "k": 1.0},
      {"path": "selfplay/gen2", "frac": 0.2, "k": 1.0},
      {"path": "selfplay/gen3", "frac": 1.0, "k": 1.0},
      {"path": "selfplay/gen4", "frac": 1.0, "k": 0.7},
      {"path": "selfplay/gen5-epsilon", "frac": 1.0, "k": 0.7},
      {"path": "selfplay/gen5-montecarlo", "frac": 1.0, "k": 0.65},
      {"path": "arena/gen1",    "frac": 0.3, "k": 1.0},
      {"path": "arena/gen2",    "frac": 0.2, "k": 1.0},
]
 
OUT_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nnue", "nnue_weights.bin")
 
# --- treino / otimizacao -----------------------------------------------------
EPOCHS_DEFAULT = 120
BATCH_SIZE_DEFAULT = "auto"           # inteiro, ou "auto" (via --vram-budget-gb)
SEED_DEFAULT = 0
VAL_SPLIT_DEFAULT = 0.1               # fracao dos dados reservada p/ validacao
 
LR_DEFAULT = 1e-3                     # LR inicial para treino do zero
LR_MIN_DEFAULT = 1e-6                 # piso do LR no fim do schedule
LR_SCHEDULE_DEFAULT = "cosine"        # none | step | exponential | cosine
WARMUP_EPOCHS_DEFAULT = 2
 
# LR usado automaticamente no cenario "NOVO CICLO" (bootstrap sobre self-play
# novo, otimizador reiniciado -- ver docstring do topo). LR_DEFAULT (pensado
# p/ treino do zero) e alto demais aqui: afastaria pesos ja convergidos do
# otimo local em vez de so refina-los. So entra em uso quando --lr fica em
# "auto"; --lr numerico explicito sempre tem prioridade.
NEW_CYCLE_LR_DEFAULT = 1e-5
STEP_SIZE_DEFAULT = 10                # epochs por degrau em --lr-schedule=step
STEP_GAMMA_DEFAULT = 0.5
EXP_GAMMA_DEFAULT = 0.97
 
# Weight decay desacoplado (AdamW), so nas matrizes de peso (nunca em bias),
# com annealing: mais forte no inicio, afrouxa perto do fim pra nao atrapalhar
# o ajuste fino dos ultimos epochs (mesma postura do nnue-pytorch do Stockfish).
WEIGHT_DECAY_DEFAULT = 1e-4
WEIGHT_DECAY_MIN_DEFAULT = 0.0
WD_SCHEDULE_DEFAULT = "cosine"        # none | constant | linear | cosine
 
EARLY_STOP_DEFAULT = True
PATIENCE_DEFAULT = 8                  # epochs sem melhora antes de parar
MIN_DELTA_DEFAULT = 1e-4              # melhora minima p/ contar como "melhorou"
MONITOR_DEFAULT = "val_loss"          # val_loss | val_outcome | val_policy | val_policy_acc
 
# --- orcamento de memoria (batch/chunk size "auto") --------------------------
# Defaults calibrados para uma maquina de dev tipica: 6GB VRAM / 32GB RAM.
VRAM_BUDGET_GB_DEFAULT = 6.0
RAM_BUDGET_GB_DEFAULT = 32.0
RAM_CHUNK_FRACTION_DEFAULT = 0.25     # fracao da RAM usada como buffer de shuffle
CHUNK_SIZE_DEFAULT = "auto"           # inteiro, ou "auto" (via --ram-budget-gb)
 
# Tetos de seguranca para batch_size/chunk_size "auto" -- limitam o valor
# calculado a partir dos orcamentos acima, independente do quanto VRAM/RAM
# permitiriam. Ajuste via --batch-size-max/--chunk-size-max se o orcamento
# real permitir mais (ou se picos de memoria fora da formula --
# fragmentacao do allocator CUDA, overhead do dataloader -- exigirem menos).
BATCH_SIZE_MAX_DEFAULT = 131_072
CHUNK_SIZE_MAX_DEFAULT = 20_000_000
 
# --- pesos de loss / QAT / logging -------------------------------------------
W_OUTCOME_DEFAULT = 1.0               # peso da loss da cabeca de resultado (BCE)
W_POLICY_DEFAULT = 1.0                # peso da loss de policy (CE)
# Opening plies excluded from the POLICY loss (they still train the value
# head). In montecarlo mode selfplay runs no search during the temperature
# window: it samples the move from the policy head itself and records that
# sampled move as the policy target (selfplay.hpp:474 and selfplay.hpp:533).
# Training on those targets teaches the policy head to imitate its own
# output, so this masks them out. The value equals MC_OBVIOUS_PLIES +
# MC_TEMP_DECAY_PLIES in tools/selfplay/run_selfplay.py (6 + 20). Measured
# on gen6: approximately 39.8 percent of all records fall in this window.
# Set to 0 to train on every ply, which is the behavior before 2026-08-26.
POLICY_OPENING_PLIES_DEFAULT = 26
QA_DEFAULT = 255                      # escala QAT do acumulador (int16)
QB_DEFAULT = 64                       # escala QAT das cabecas (int8)
LOG_EVERY_DEFAULT = 1                 # a cada quantos epochs imprime o resumo
PLOT_EVERY_EPOCHS_DEFAULT = 5         # a cada quantos epochs regrava o PNG
 
# GPU vs CPU: o UNICO parametro que decide isso. True = usa CUDA se
# disponivel (cai pra CPU sozinho, sem erro, se nao houver GPU). False =
# forca CPU mesmo com GPU disponivel (util p/ depurar/comparar). --device
# so pra casos especiais (ex. "cuda:1"); no dia a dia nem se toca nisso.
USE_GPU_DEFAULT = True
DEVICE_DEFAULT = "cuda" if (USE_GPU_DEFAULT and torch.cuda.is_available()) else "cpu"
 
# --- checkpoint / resume ------------------------------------------------------
# Ver a secao "CHECKPOINT / RESUME" no docstring do topo do arquivo para o
# fluxo completo (4 cenarios) e os 4 arquivos gravados em --ckpt-dir.
CKPT_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "checkpoints")
FRESH_DEFAULT = False   # True (ou --fresh) ignora qualquer checkpoint existente e comeca do zero
 
GRAD_CLIP_NORM_DEFAULT = 1.0          # clip de norma do gradiente; 0/negativo desliga
 
# Quantos batches sao agrupados por transferencia host->device: o chunk lido
# do disco e enviado a GPU em blocos de `batch_size * multiplier` posicoes de
# uma vez, e os batches de treino sao depois fatiados direto na VRAM.
GPU_CHUNK_MULTIPLIER_DEFAULT = 8
 
PROGRESS_EVERY_SECS_DEFAULT = 5.0     # intervalo minimo entre linhas de progresso no epoch
# A cada quantas atualizacoes de progresso uma linha fica fixa no scrollback
# (as outras so atualizam a linha viva via \r). Default 5s x 6 = 1 linha fixa/~30s.
PROGRESS_PERMANENT_EVERY_DEFAULT = 6
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
# VALUE_SCALE (200.0) removida 2026-08: so era usada pra normalizar a loss
# MSE da cabeca auxiliar (search_score/VALUE_SCALE), que nao existe mais --
# ver QuoridorNNUE. O merge WL/EV atual (wl_target em to_chunk_tensors) ja
# trabalha direto em probabilidade [0,1], sem precisar de escala nenhuma.
 
INT8_MAX = 127
INT16_MAX = 32767
 
 
# --- modelo: espelho exato de nnue.hpp --------------------------------------
def screlu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0) ** 2
 
 
def clipped_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)
 
 
class QuoridorNNUE(nn.Module):
    """Cabeca de valor unica (WL, resultado sem empate): value1_wl/value2_wl,
    bottleneck 256->32->1. A cabeca AUXILIAR de imitacao da heuristica
    (value1_aux/value2_aux) existiu ate 2026-08 e foi removida: era so
    scaffolding pra treinar enquanto o self-play nao vinha da propria NNUE
    -- desde que TrainingSample passou a gravar a avaliacao da propria rede
    (evalNNUE, ver read_selfplay.py/selfplay.hpp) em vez do score
    heuristico, ela virou peso morto (nunca foi consumida pela busca em
    nnue.hpp). Ver a tabela em status.md/CLAUDE.md ("Avaliacao: o que cada
    estagio usa") pro mapa completo."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, HIDDEN)
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
            for layer in (model.value1_wl, model.value2_wl, model.policy):
                layer.weight.clamp_(-head_max, head_max)


 
# --- features -----------------------------------------------------------------
def to_chunk_tensors(chunk: np.ndarray, k_chunk: np.ndarray, device, pw_chunk: np.ndarray = None):
    """Converte um bloco de TrainingSample (array estruturado numpy) para
    tensores densos e transfere tudo para `device` de uma vez so. Chamar
    isso uma vez por bloco grande (em vez de uma vez por batch) e o que
    permite ao `iter_gpu_batches` amortizar o custo de transferencia
    host->device entre varios batches.

    `k_chunk`: array float32 do MESMO tamanho/ORDEM de `chunk` (ja fatiado/
    permutado em paralelo pelo chamador -- ver k_by_source em
    load_training_population) com o "k" da fonte de cada posicao. Usado
    aqui pra montar wl_target = k*game_result_prob + (1-k)*ev_prob (ver
    nota no topo do arquivo e em DATA_SOURCES_DEFAULT).

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

    # wl_target: blend por posicao entre o resultado REAL do jogo (na
    # perspectiva do MOVER, igual sempre foi) e a avaliacao da propria NNUE
    # gravada no .bin (nnue_eval, perspectiva ABSOLUTA das brancas --
    # reprojetada pra perspectiva do mover via `mover` antes de misturar).
    # k=1 (fontes antigas) reduz isso a wl_target = result_prob_mover, ou
    # seja, o comportamento de sempre.
    result_prob_mover = (chunk["game_result"].astype(np.float32) + 1.0) / 2.0
    ev_white = chunk["nnue_eval"].astype(np.float32) / np.float32(EV_SCALE)
    mover = chunk["mover"].astype(np.float32)
    ev_mover = np.where(mover == 0.0, ev_white, 1.0 - ev_white).astype(np.float32)
    k = k_chunk.astype(np.float32)
    wl_target = k * result_prob_mover + (1.0 - k) * ev_mover

    if pw_chunk is None:
        pw = np.ones(len(chunk), dtype=np.float32)
    else:
        pw = pw_chunk.astype(np.float32)

    return {
        "x": torch.from_numpy(x).to(device, non_blocking=True),
        "wl_target": torch.from_numpy(wl_target).to(device, non_blocking=True),
        "policy_target": torch.from_numpy(chunk["policy_target"].astype(np.int64)).to(device, non_blocking=True),
        "policy_w": torch.from_numpy(pw).to(device, non_blocking=True),
    }


def iter_gpu_batches(chunk: np.ndarray, k_chunk: np.ndarray, device, batch_size: int, gpu_chunk_size: int,
                     pw_chunk: np.ndarray = None):
    """Percorre `chunk` (e `k_chunk`, MESMA ordem/tamanho) em sub-blocos de
    ate `gpu_chunk_size` posicoes, transferindo cada sub-bloco para
    `device` uma unica vez (via `to_chunk_tensors`), e dentro dele fatia os
    batches de treino/avaliacao (`batch_size`) diretamente na VRAM -- sem
    nenhum novo round-trip host->device por batch. Em CPU o efeito e
    neutro (o "device" ja e a propria RAM), mas em GPU isso reduz o numero
    de transferencias por um fator de ~gpu_chunk_size/batch_size."""
    n = len(chunk)
    gpu_chunk_size = max(batch_size, gpu_chunk_size)
    for gstart in range(0, n, gpu_chunk_size):
        sub = chunk[gstart:gstart + gpu_chunk_size]
        k_sub = k_chunk[gstart:gstart + gpu_chunk_size]
        pw_sub = None if pw_chunk is None else pw_chunk[gstart:gstart + gpu_chunk_size]
        t = to_chunk_tensors(sub, k_sub, device, pw_sub)
        n_sub = len(sub)
        for start in range(0, n_sub, batch_size):
            end = min(start + batch_size, n_sub)
            yield {
                "x": t["x"][start:end],
                "wl_target": t["wl_target"][start:end],
                "policy_target": t["policy_target"][start:end],
                "policy_w": t["policy_w"][start:end],
            }


# --- orcamento de memoria: auto batch-size (VRAM) / chunk-size (RAM) -------
def compute_auto_batch_size(vram_budget_gb, reserved_gb=1.0, min_bs=64, max_bs=BATCH_SIZE_MAX_DEFAULT):
    """Estimativa conservadora de quantas amostras cabem por batch dentro do
    orcamento de VRAM informado. `reserved_gb` cobre o contexto CUDA e os
    pesos/estados do otimizador (a rede e minuscula, isso domina o
    orcamento fixo). Para o restante, soma-se entrada + ativacoes de todas
    as camadas + gradientes, com fator de seguranca 4x para os buffers
    intermediarios que o autograd mantem vivos durante o backward.
    `max_bs` e so um teto de seguranca (ver --batch-size-max) -- nao deveria
    ser o fator dominante quando o orcamento de VRAM informado da folga."""
    elems_per_sample = NUM_FEATURES + HIDDEN + 2 * 32 + POLICY_OUT
    bytes_per_sample = elems_per_sample * 4 * 4  # float32, fwd+bwd, margem 4x
    usable = max(0.0, (vram_budget_gb - reserved_gb)) * (1024 ** 3)
    bs = int(usable // max(1, bytes_per_sample))
    return max(min_bs, min(max_bs, bs))
 
 
def compute_chunk_size(n_total, ram_budget_gb, fraction=RAM_CHUNK_FRACTION_DEFAULT,
                        min_size=20_000, max_size=CHUNK_SIZE_MAX_DEFAULT):
    """Numero de posicoes lidas por vez do dataset (structs crus) para
    formar o buffer de shuffle, escolhido para caber em `fraction` do
    orcamento de RAM informado. `max_size` e so um teto de seguranca (ver
    --chunk-size-max), nao deveria dominar sobre o orcamento de RAM real."""
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
    MINIMIZE = {"val_loss", "val_outcome", "val_policy"}
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
# Diferenca de ckpt_<ciclo>_ep<N>.bin: esse e so os PESOS, no layout binario
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
 
 
# --- caminhos versionados (<ciclo>_ep<epoch>) em --ckpt-dir ------------------
# Cada epoch grava um arquivo NOVO (nome muda: epoch sempre cresce dentro do
# mesmo ciclo); o do epoch anterior DO MESMO CICLO e apagado logo depois que
# o novo e gravado com sucesso (ver _prune_old_cycle_files). Arquivos de
# ciclos anteriores (<ciclo> diferente) nunca sao tocados aqui.
_CKPT_TIMESTAMP_FMT = "%Y%m%d-%H%M%S"
_TRAIN_STATE_RE = re.compile(r"^train_state_(?P<cycle>\d{8}-\d{6})_ep(?P<epoch>\d{4})\.pt$")


def new_cycle_id():
    """Timestamp (AAAAMMDD-HHMMSS) usado como <ciclo> ao comecar um treino
    do zero, um --init-from, ou um NOVO CICLO (bootstrap sobre self-play
    novo). Resume de treino interrompido reaproveita o <ciclo> salvo em vez
    de chamar isto -- ver train()."""
    return time.strftime(_CKPT_TIMESTAMP_FMT)


def _train_state_path(ckpt_dir, cycle_id, epoch):
    return os.path.join(ckpt_dir, f"train_state_{cycle_id}_ep{epoch:04d}.pt")


def _config_json_path(ckpt_dir, cycle_id, epoch):
    return os.path.join(ckpt_dir, f"train_config_{cycle_id}_ep{epoch:04d}.json")


def _ckpt_bin_path(ckpt_dir, cycle_id, epoch):
    return os.path.join(ckpt_dir, f"ckpt_{cycle_id}_ep{epoch:04d}.bin")


def _find_latest_train_state(ckpt_dir):
    """Procura em ckpt_dir o train_state_<ciclo>_ep<NNNN>.pt mais recente
    (maior <ciclo>, e dentro dele a maior epoch) -- substitui o antigo nome
    fixo train_state.pt. Retorna o caminho, ou None se nao houver nenhum."""
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for path in glob.glob(os.path.join(ckpt_dir, "train_state_*_ep*.pt")):
        m = _TRAIN_STATE_RE.match(os.path.basename(path))
        if m:
            candidates.append((m.group("cycle"), int(m.group("epoch")), path))
    if not candidates:
        return None
    candidates.sort()  # strings AAAAMMDD-HHMMSS ordenam cronologicamente
    return candidates[-1][2]


def _prune_old_cycle_files(old_paths, new_paths):
    """Apaga os arquivos do epoch anterior (mesmo ciclo) depois que os novos
    ja foram gravados com sucesso -- nunca ficamos sem checkpoint valido no
    disco no meio da troca. old_paths/new_paths sao dicts {chave: caminho};
    so apaga o que realmente mudou de nome (evita apagar o proprio arquivo
    recem-gravado, ex. no 1o epoch de um ciclo onde old==None)."""
    for key, old_path in old_paths.items():
        new_path = new_paths.get(key)
        if old_path and old_path != new_path and os.path.isfile(old_path):
            os.remove(old_path)
 
 
# Hiperparametros de OTIMIZACAO/loss/QAT que fazem sentido herdar de uma
# rodada anterior via --resume-config. Deliberadamente NAO inclui caminhos
# de I/O (--data, --val-data, --out, --ckpt-dir, --init-from, --resume-config,
# --fresh, --device, --plot-dir) nem flags de performance/console
# (--gpu-chunk-multiplier, --progress-every-secs, --log-every) -- esses sao
# especificos da MAQUINA/rodada atual, nao do "treino" em si, e copia-los
# cegamente de um JSON antigo tende a surpreender mais do que ajudar (ex.
# reaproveitar um --out de outra maquina).
_CONFIG_JSON_HYPERPARAM_KEYS = (
    # "epochs" DELIBERADAMENTE fora daqui: e orcamento DESTA sessao (quantas
    # epocas VOCE quer rodar agora), nao uma "receita de treino" como
    # lr/qa/qb -- herdar silenciosamente o teto do ciclo anterior faz
    # --epochs maior que o antigo ser ignorado sem aviso se o usuario
    # esquecer de repetir a flag. O teto do ciclo ANTERIOR (pra decidir se
    # ele foi concluido) vem de target_epochs em train_state.pt, nao daqui
    # -- ver hit_ceiling em train().
    "batch_size", "seed", "val_split",
    "lr", "lr_min", "lr_schedule", "warmup_epochs", "step_size", "step_gamma", "exp_gamma",
    "weight_decay", "weight_decay_min", "wd_schedule",
    "early_stop", "patience", "min_delta", "monitor",
    "w_outcome", "w_policy", "qa", "qb", "grad_clip_norm",
)
 
 
def save_config_json(path, args, fingerprint, epoch, best_metric, weights_path):
    """Grava train_config_<ciclo>_ep<NNNN>.json ao lado do train_state e do
    ckpt.bin do mesmo epoch: snapshot HUMANO-LEGIVEL (json.dumps, sem
    torch/numpy) dos hiperparametros desta rodada + qual arquivo de pesos
    corresponde a este checkpoint. Nao substitui train_state.pt para resume
    EXATO (nao tem otimizador/RNG) -- serve para (a) auditoria de "com que
    config este .bin foi treinado" e (b) --resume-config, que reaproveita
    esses hiperparametros (nao o estado do otimizador) para comecar um
    ciclo novo de treino."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "epoch": epoch,
        "best_metric": best_metric,
        "weights_path": os.path.abspath(weights_path) if weights_path else None,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hyperparams": {k: getattr(args, k) for k in _CONFIG_JSON_HYPERPARAM_KEYS if hasattr(args, k)},
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)  # mesmo truque write-then-rename do train_state.pt
 
 
def load_config_json(path):
    """Le um train_config_<ciclo>_ep<NNNN>.json (de --resume-config OU do
    auto-detect em --ckpt-dir). Retorna None silenciosamente se o arquivo
    nao existe; lanca se existe mas esta corrompido/ilegivel (diferente do
    try_load_train_state, que so avisa em incompatibilidade de arquitetura
    -- aqui um JSON corrompido e sempre um erro do usuario apontando pro
    arquivo errado, vale falhar alto)."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def save_train_state(path, model, opt, cycle_id, epoch, epoch_completed, history, stopper, rng, fingerprint,
                      target_epochs=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(
        cycle_id=cycle_id,
        fingerprint=fingerprint,
        epoch=epoch,
        epoch_completed=epoch_completed,
        # teto de --epochs QUE ESTE CICLO ESTAVA VISANDO (nao o --epochs da
        # chamada que vai RETOMAR este checkpoint no futuro). Gravado aqui
        # pra decidir "cycle_exhausted" em train() sem depender do --epochs
        # da proxima chamada -- ver comentario em hit_ceiling. None em
        # checkpoints antigos (gravados antes deste campo existir); nesse
        # caso train() cai de volta pro comportamento antigo (compara com
        # args.epochs da chamada atual).
        target_epochs=target_epochs,
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
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # grava em arquivo temporario e renomeia -- evita
                                 # checkpoint corrompido se o processo morrer no meio da escrita
 
 
def try_load_train_state(ckpt_dir, fingerprint):
    """Acha o train_state_<ciclo>_ep<NNNN>.pt mais recente em ckpt_dir (ver
    _find_latest_train_state) e retorna o payload se a arquitetura bater,
    senao None (silencioso se nao houver nenhum ainda; avisa se existe mas
    e incompativel)."""
    if not ckpt_dir:
        return None
    path = _find_latest_train_state(ckpt_dir)
    if path is None:
        return None
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
        self.cycle_id = None
        self.epoch = 0
        self.epoch_completed = False
        self.args = None  # p/ save_config_json (hiperparametros desta rodada)
        self.device = None
        self.last_paths = {}  # {"state":..., "bin":..., "config":...} do ultimo
                               # checkpoint gravado NESTE ciclo -- usado pra apagar
                               # o do epoch anterior depois que o novo e gravado


def save_checkpoint_bundle(rs, epoch, epoch_completed):
    """Grava train_state + ckpt.bin (pesos do MELHOR epoch ate agora,
    --monitor) + train_config para (rs.cycle_id, epoch), e so entao apaga
    os 3 arquivos do epoch anterior DO MESMO CICLO (rs.last_paths) --
    nunca ficamos sem checkpoint valido no disco no meio da troca.

    So chamada com epoch COMPLETO: o fim normal de cada epoch (train())
    sempre passa epoch_completed=True; o handler de Ctrl+C (_sigint_handler)
    so chama isto na rara janela em que o epoch ja terminou mas o proximo
    ainda nao comecou -- uma interrupcao no MEIO de um epoch nao chama
    esta funcao (o epoch em andamento e descartado e refeito do zero,
    sem gravar nada, ver _sigint_handler). epoch_completed continua sendo
    parametro/gravado em train_state.pt por clareza/futuro-proofing, mas
    hoje sempre chega True aqui."""
    new_paths = dict(
        state=_train_state_path(rs.ckpt_dir, rs.cycle_id, epoch),
        bin=_ckpt_bin_path(rs.ckpt_dir, rs.cycle_id, epoch),
        config=_config_json_path(rs.ckpt_dir, rs.cycle_id, epoch),
    )
    save_train_state(new_paths["state"], rs.model, rs.opt, rs.cycle_id, epoch, epoch_completed,
                      rs.history, rs.stopper, rs.rng, rs.fingerprint,
                      target_epochs=rs.args.epochs if rs.args is not None else None)
    best_state = rs.stopper.best_state if rs.stopper.best_state is not None else rs.model.state_dict()
    _export_state_dict(best_state, new_paths["bin"], rs.device)
    if rs.args is not None:
        save_config_json(new_paths["config"], rs.args, rs.fingerprint, epoch, rs.stopper.best, new_paths["bin"])
    _prune_old_cycle_files(rs.last_paths, new_paths)
    rs.last_paths = new_paths
    return new_paths["bin"]


_run_state = _RunState()
_ckpt_saved_on_interrupt = False
 
 
def _sigint_handler(signum, frame):
    """Ctrl+C: se o epoch atual JA terminou (raro -- janela bem pequena
    entre o fim do epoch e o proximo checkpoint periodico), grava esse
    checkpoint completo normalmente. Caso contrario (interrupcao no MEIO
    do epoch, o caso comum), nao salva nada -- o epoch em andamento e
    descartado inteiro e sera refeito do zero na proxima vez, a partir do
    ultimo checkpoint COMPLETO ja em disco (que fica intacto, nunca
    tocado aqui). Salvar um checkpoint parcial no meio do epoch (pesos e
    otimizador so parcialmente atualizados por alguns batches) so pra
    reaproveitar aquele pedacinho de progresso nao compensa a
    complexidade/risco -- e mais simples e mais seguro sempre redoing o
    epoch inteiro."""
    global _ckpt_saved_on_interrupt
    print("\n\n[Ctrl+C] interrompido...")
    rs = _run_state
    if rs.model is not None and rs.ckpt_dir and rs.epoch_completed and not _ckpt_saved_on_interrupt:
        _ckpt_saved_on_interrupt = True
        bin_path = save_checkpoint_bundle(rs, rs.epoch, rs.epoch_completed)
        if rs.args is not None and rs.args.plot_dir and rs.history and len(rs.history.get("epoch", [])) > 0:
            save_plots(rs.history, rs.args.plot_dir)
            print(f"plots salvos em {rs.args.plot_dir}")
        print(f"checkpoint salvo em {rs.ckpt_dir} (epoch {rs.epoch}, completo) -- {bin_path}")
    else:
        print(f"epoch {rs.epoch if rs.model is not None else '?'} em andamento descartado -- "
              f"ultimo checkpoint completo em disco fica intacto; esse epoch e refeito do zero ao retomar.")
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
        wp = model.policy.weight.detach().cpu().numpy().astype(np.float32)
        bp = model.policy.bias.detach().cpu().numpy().astype(np.float32)

    assert w1.shape == (NUM_FEATURES, HIDDEN)
    assert wv1_wl.shape == (HIDDEN, 32)
    assert wp.shape == (POLICY_OUT, HIDDEN)

    dir_name = os.path.dirname(os.path.abspath(path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Layout tem que casar EXATAMENTE com NNUEWeights::loadFromFile em
    # nnue.hpp (sem cabeca auxiliar, 2026-08+): w1, b1, wv1_wl, bv1_wl,
    # wv2_wl, bv2_wl, wp, bp -- nesta ordem, sem cabecalho.
    with open(path, "wb") as f:
        f.write(np.ascontiguousarray(w1).tobytes())
        f.write(np.ascontiguousarray(b1).tobytes())
        f.write(np.ascontiguousarray(wv1_wl).tobytes())
        f.write(np.ascontiguousarray(bv1_wl).tobytes())
        f.write(np.ascontiguousarray(wv2_wl).tobytes())
        f.write(np.ascontiguousarray(bv2_wl).tobytes())
        f.write(np.ascontiguousarray(wp).tobytes())
        f.write(np.ascontiguousarray(bp).tobytes())

 
def _default_quant_path(out_path: str) -> str:
    if out_path.endswith(".bin"):
        return out_path[: -len(".bin")] + "_int8.bin"
    return out_path + "_int8.bin"
 
 
def weighted_policy_loss(policy_logits, policy_t, pw):
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


# --- avaliacao em chunks (limitado por RAM/VRAM) -----------------------------
@torch.no_grad()
def run_eval(ds, indices, k_by_source, batch_size, chunk_size, gpu_chunk_size, model, device,
             w_outcome, w_policy, pw_by_sample=None):
    model.eval()
    total = dict(loss=0.0, outcome=0.0, policy=0.0, correct=0, pw=0.0)
    n_items = len(indices)
    for start in range(0, n_items, chunk_size):
        chunk_idx = indices[start:start + chunk_size]
        chunk = ds[chunk_idx]
        k_chunk = k_by_source[chunk_idx]
        pw_chunk = None if pw_by_sample is None else pw_by_sample[chunk_idx]
        for t in iter_gpu_batches(chunk, k_chunk, device, batch_size, gpu_chunk_size, pw_chunk):
            policy_t = t["policy_target"]
            pw = t["policy_w"]

            value_wl, policy_logits = model(t["x"])
            loss_outcome = F.binary_cross_entropy_with_logits(value_wl, t["wl_target"])
            loss_policy = weighted_policy_loss(policy_logits, policy_t, pw)
            loss = w_outcome * loss_outcome + w_policy * loss_policy

            nb = len(t["x"])
            # The policy metrics are averaged over the samples that the
            # policy loss actually saw, so a masked opening ply does not
            # dilute them. The value metrics keep the full count.
            nbw = pw.sum().item()
            total["loss"] += loss.item() * nb
            total["outcome"] += loss_outcome.item() * nb
            total["policy"] += loss_policy.item() * nbw
            total["pw"] += nbw
            hit = (policy_logits.argmax(dim=-1) == policy_t).float()
            total["correct"] += (hit * pw).sum().item()
    for key in ("loss", "outcome"):
        total[key] /= max(1, n_items)
    total["policy"] /= max(1.0, total["pw"])
    total["policy_acc"] = total["correct"] / max(1.0, total["pw"])
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
    plot_pair(axes[0, 1], "outcome", "Loss resultado (BCE, wl_target)")
    plot_pair(axes[0, 2], "policy", "Loss politica (CE)")
    plot_pair(axes[1, 0], "policy_acc", "Acuracia da politica")
    axes[1, 1].axis("off")  # slot vago desde a remocao da cabeca auxiliar (2026-08)

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
 
 
# --- fontes de treino (pastas com fracao, estilo mixtrain.py) ---------------
def resolve_data_source_path(path: str, data_root: str) -> str:
    """Resolve o `path` de uma entrada de DATA_SOURCES/--data-sources (ver
    DEFAULT CONFIG no topo do arquivo). Tenta, nessa ordem: (1) o caminho
    como foi passado -- absoluto, ou relativo ao cwd; (2)
    <data_root>/selfplay/<path>; (3) <data_root>/arena/<path>; (4)
    <data_root>/<path> (raiz de data/). A primeira que existir (arquivo OU
    diretorio) e usada."""
    candidates = [
        path,
        os.path.join(data_root, "selfplay", path),
        os.path.join(data_root, "arena", path),
        os.path.join(data_root, path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    tried = "\n  - ".join(candidates)
    raise FileNotFoundError(f"[data-sources] '{path}' nao encontrado. Tentei:\n  - {tried}")
 
 
def load_training_population(data_arg, sources_arg, data_root, seed):
    """Monta a populacao de treino a partir de --data (fluxo antigo, uma
    lista/glob/diretorio achatada, sem amostragem) OU --data-sources/
    DATA_SOURCES_DEFAULT (fluxo novo: lista de {"path", "frac", "k"}, uma
    fracao [0,1] de POSICOES sorteada aleatoriamente por pasta, e um peso
    "k" por fonte para o blend WL/EV -- ver comentario em
    DATA_SOURCES_DEFAULT).

    Retorna (train_paths, train_ds, base_idx, k_by_source):
      - train_paths / train_ds: iguais ao retorno de load_multi_selfplay,
        cobrindo TODOS os arquivos .bin de TODAS as fontes envolvidas.
      - base_idx: array de indices GLOBAIS (posicoes dentro de train_ds) ja
        filtrado pelas fracoes por fonte -- e a populacao que train() usa
        para o split treino/validacao (--val-split fatia DENTRO de
        base_idx, nao do total bruto). Sem --data-sources, base_idx e
        simplesmente arange(len(train_ds)) (comportamento antigo,
        inalterado).
      - k_by_source: array float32 de tamanho len(train_ds) (MESMO
        indexador que train_ds/base_idx, nao so das posicoes selecionadas)
        com o "k" da fonte de cada posicao -- 1.0 em tudo quando nao ha
        --data-sources (comportamento antigo: ignora evalNNUE). Indexavel
        diretamente por qualquer global_idx tirado de base_idx.
    """
    if not sources_arg:
        train_paths, train_ds = load_multi_selfplay(data_arg)
        return train_paths, train_ds, np.arange(len(train_ds)), np.ones(len(train_ds), dtype=np.float32)

    rng = np.random.default_rng(seed)
    all_paths, source_file_counts, kept_sources = [], [], []
    for entry in sources_arg:
        raw_path = entry["path"]
        frac = float(entry.get("frac", 1.0))
        k = float(entry.get("k", 1.0))
        if not (0.0 <= k <= 1.0):
            raise ValueError(f"[data-sources] k invalido ({k}) para '{raw_path}' -- "
                              f"precisa estar em [0.0, 1.0] (1.0 = ignora evalNNUE, so resultado real)")
        if frac <= 0.0:
            print(f"[data-sources] ignorado (frac=0): {raw_path}")
            continue
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"[data-sources] frac invalido ({frac}) para '{raw_path}' -- "
                              f"precisa estar em (0.0, 1.0]")
        resolved = resolve_data_source_path(raw_path, data_root)
        src_paths = expand_data_paths(resolved)
        if not src_paths:
            raise ValueError(f"[data-sources] nenhum .bin encontrado em '{resolved}' (fonte '{raw_path}')")
        all_paths.extend(src_paths)
        source_file_counts.append(len(src_paths))
        kept_sources.append((raw_path, frac, k, resolved))

    if not all_paths:
        raise ValueError("[data-sources] todas as fontes tem frac=0 (ou a lista esta vazia) -- "
                          "nada para treinar")
    if len(all_paths) != len(set(all_paths)):
        dupes = sorted({p for p in all_paths if all_paths.count(p) > 1})
        raise ValueError(f"[data-sources] pastas de fontes se sobrepoem -- arquivo(s) repetido(s) "
                          f"em mais de uma fonte: {dupes}")

    train_paths, train_ds = load_multi_selfplay(all_paths)
    file_lens = [n for _, n in train_ds.sizes()]
    k_by_source = np.ones(len(train_ds), dtype=np.float32)

    selected_chunks = []
    idx_cursor = 0
    global_offset = 0
    print(f"fontes de treino ({len(kept_sources)} pasta(s), sorteio aleatorio por posicao, seed={seed}):")
    for (raw_path, frac, k, resolved), nfiles in zip(kept_sources, source_file_counts):
        n_source = sum(file_lens[idx_cursor:idx_cursor + nfiles])
        start = global_offset
        idx_cursor += nfiles
        global_offset += n_source
        k_by_source[start:start + n_source] = k
        if frac >= 1.0:
            sel = np.arange(start, start + n_source)
        else:
            n_take = max(1, round(n_source * frac))
            local = rng.choice(n_source, size=n_take, replace=False)
            sel = start + np.sort(local)
        selected_chunks.append(sel)
        print(f"  - {raw_path:24s} ({resolved}): {nfiles} arquivo(s), {n_source:,} posicoes, "
              f"frac={frac:.2f} k={k:.2f} -> {len(sel):,} selecionadas")

    base_idx = np.concatenate(selected_chunks)
    print(f"total (todas as fontes, pos-amostragem): {len(base_idx):,} / {global_offset:,} posicoes "
          f"({100.0 * len(base_idx) / max(1, global_offset):.1f}%)")
    return train_paths, train_ds, base_idx, k_by_source
 
 
# --- treino ------------------------------------------------------------------
def _print_banner(lines):
    width = max(70, max(len(l) for l in lines) + 4)
    print("=" * width)
    for l in lines:
        print(l)
    print("=" * width)
 
 
def train(args):
    global _ckpt_saved_on_interrupt
    _ckpt_saved_on_interrupt = False
 
    # --- MODO desta execucao (resume / novo ciclo / do zero / --init-from) --
    # Decidido e anunciado ANTES de qualquer outra coisa (mesmo antes de
    # carregar os dados), pra responder de cara "em que caso estamos" sem
    # precisar ler o resto do log. Ver os 4 cenarios no docstring do topo.
    ckpt_dir = args.ckpt_dir
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    fingerprint = compute_fingerprint(args)
    resumed = None if args.fresh else try_load_train_state(ckpt_dir, fingerprint)
    new_cycle = False
    cycle_id = None
    if resumed is not None:
        resumed_path = _train_state_path(ckpt_dir, resumed["cycle_id"], resumed["epoch"])
        prev_epoch = resumed["epoch"]
        prev_completed = resumed["epoch_completed"]
        prev_bad_epochs = resumed.get("stopper_num_bad_epochs", 0)
        # teto que o ciclo ANTERIOR de fato visava. Usa o valor gravado no
        # proprio train_state (target_epochs) -- NAO o args.epochs desta
        # chamada -- senao pedir mais epocas pro ciclo novo (ex. --epochs 120
        # depois de um ciclo que rodou ate 60) faz hit_ceiling dar False e o
        # codigo confunde "novo ciclo sobre selfplay novo" com "retomando
        # treino interrompido no meio", herdando otimizador/LR schedule do
        # ciclo antigo e rodando so as epocas 61..120 em vez de reiniciar.
        # Fallback pra args.epochs em checkpoints antigos sem esse campo.
        prev_target_epochs = resumed.get("target_epochs") or args.epochs
        hit_ceiling = prev_epoch >= prev_target_epochs
        early_stopped = args.early_stop and prev_bad_epochs >= args.patience
        cycle_exhausted = prev_completed and (hit_ceiling or early_stopped)
        if cycle_exhausted:
            new_cycle = True
            cycle_id = new_cycle_id()
            lr_auto = getattr(args, "lr_auto", False)
            if lr_auto:
                args.lr = args.new_cycle_lr
            lr_note = (f"  LR: {args.lr:.1e} (--new-cycle-lr, nao o LR_DEFAULT="
                       f"{LR_DEFAULT:.1e} de treino do zero)" if lr_auto else
                       f"  LR: {args.lr:.1e} (--lr explicito, sobrepondo --new-cycle-lr)")
            _print_banner([
                f"MODO: NOVO CICLO (bootstrap sobre self-play novo)",
                f"  ciclo anterior completo no epoch {prev_epoch} -- {resumed_path}",
                f"  novo ciclo: {cycle_id} (arquivos do ciclo anterior ficam no disco)",
                f"  reaproveitando so os PESOS; otimizador, LR/WD schedule e "
                f"early-stopping reiniciam do zero (epoch volta a contar de 1)",
                lr_note,
            ])
        else:
            cycle_id = resumed["cycle_id"]
            status = "completo" if prev_completed else "parcial -- sera refeito"
            _print_banner([
                f"MODO: RETOMANDO TREINO INTERROMPIDO",
                f"  ultimo epoch salvo: {prev_epoch} ({status}) -- {resumed_path}",
                f"  mesmo ciclo ({cycle_id}), otimizador, schedule e early-stopping de onde parou",
            ])
        if args.init_from:
            print(f"  (--init-from ignorado: ha um checkpoint valido em {ckpt_dir})")
    elif args.init_from:
        cycle_id = new_cycle_id()
        _print_banner([
            f"MODO: TREINO A PARTIR DE PESOS EXISTENTES",
            f"  pesos: {args.init_from}",
            f"  ciclo novo: {cycle_id}",
            f"  otimizador comeca do zero (p/ retomar COM otimizador/schedule, "
            f"use o mesmo --ckpt-dir de uma rodada anterior em vez de --init-from)",
        ])
    else:
        cycle_id = new_cycle_id()
        _print_banner([f"MODO: TREINO DO ZERO -- pesos aleatorios, epoch 1  (ciclo {cycle_id})"])
 
    sources = args.data_sources if args.data_sources else DATA_SOURCES_DEFAULT
    train_paths, train_ds, base_idx, k_by_source = load_training_population(
        args.data, sources, DATA_ROOT_DEFAULT, args.seed)
    if sources:
        print(f"dados de treino: {len(train_paths)} arquivo(s), {len(train_ds):,} posicoes no total "
              f"(--data-sources ativo -- ver amostragem por fonte acima)")
    else:
        print(f"dados de treino: {len(train_paths)} arquivo(s), {len(train_ds):,} posicoes")
        for p, n in train_ds.sizes():
            print(f"  - {p}: {n:,} posicoes")

    # Policy-loss mask: the opening plies of a montecarlo game carry a
    # policy target that the policy head itself sampled, with no search
    # behind it. Excluding them keeps the value head untouched.
    def build_policy_w(ds, label):
        if args.policy_opening_plies <= 0:
            return None
        ply = ds.ply_index()
        w = (ply >= args.policy_opening_plies).astype(np.float32)
        kept, total = float(w.sum()), float(len(w))
        print(f"mascara de policy ({label}): plies < {args.policy_opening_plies} excluidos da loss "
              f"de policy -- {total - kept:,.0f} de {total:,.0f} posicoes ({1 - kept / max(1.0, total):.1%}) "
              f"treinam so a cabeca de valor")
        if kept == 0:
            raise SystemExit("--policy-opening-plies removeu TODAS as posicoes da loss de policy")
        return w

    policy_w_by_sample = build_policy_w(train_ds, "treino")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.val_data:
        val_paths, val_ds = load_multi_selfplay(args.val_data)
        print(f"dados de validacao (explicitos): {len(val_paths)} arquivo(s), {len(val_ds):,} posicoes")
        train_idx = base_idx
        val_idx = np.arange(len(val_ds))
        # --val-data e um diretorio explicito, fora de --data-sources/
        # DATA_SOURCES_DEFAULT -- nao tem "k" associado. k=1.0 (ignora
        # evalNNUE, so resultado real) e o mais correto pra metrica de
        # validacao: mede a rede contra o que realmente aconteceu na
        # partida, nao contra a opiniao de uma NNUE anterior.
        val_k_by_source = np.ones(len(val_ds), dtype=np.float32)
        val_policy_w = build_policy_w(val_ds, "validacao")
    else:
        n_total = len(base_idx)
        n_val = max(1, int(n_total * args.val_split))
        perm = rng.permutation(n_total)
        val_idx, train_idx = base_idx[perm[:n_val]], base_idx[perm[n_val:]]
        val_ds = train_ds
        val_k_by_source = k_by_source
        # Same dataset as training, so the same per-sample mask applies.
        val_policy_w = policy_w_by_sample
        print(f"split treino/validacao: {len(train_idx):,} / {len(val_idx):,} (val-split={args.val_split})")

 
    device = torch.device(args.device)
    print_device_info(device)
    batch_size = resolve_int_or_auto(
        args.batch_size, lambda: compute_auto_batch_size(args.vram_budget_gb, max_bs=args.batch_size_max))
    chunk_size = resolve_int_or_auto(
        args.chunk_size, lambda: compute_chunk_size(len(train_idx), args.ram_budget_gb, args.ram_chunk_fraction,
                                                      max_size=args.chunk_size_max))
    gpu_chunk_size = batch_size * max(1, args.gpu_chunk_multiplier)
    print(f"orcamento de VRAM: {args.vram_budget_gb:.1f} GB -> batch_size={batch_size:,}")
    print(f"orcamento de RAM: {args.ram_budget_gb:.1f} GB -> chunk_size={chunk_size:,} posicoes/bloco")
    print(f"transferencia GPU em blocos de {gpu_chunk_size:,} posicoes "
          f"({args.gpu_chunk_multiplier}x o batch_size)")
    if args.grad_clip_norm and args.grad_clip_norm > 0:
        print(f"gradient clipping: max_norm={args.grad_clip_norm}")
    total_chunks_per_epoch = max(1, -(-len(train_idx) // chunk_size))
    print(f"chunks por epoch (RAM): ~{total_chunks_per_epoch}")
 
    model = QuoridorNNUE().to(device)
    if resumed is not None:
        # strict=False: um train_state.pt de ANTES de 2026-08 (fingerprint
        # compativel, mas com chaves value1_aux.*/value2_aux.* que a
        # arquitetura atual nao tem mais) nao pode ser retomado em modo
        # estrito. Avisa e ignora essas chaves -- ver mesma logica em
        # _export_state_dict.
        missing, unexpected = model.load_state_dict(resumed["model_state"], strict=False)
        if unexpected:
            print(f"[resume] aviso: {len(unexpected)} chave(s) do checkpoint ignoradas "
                  f"(formato antigo, provavelmente cabeca auxiliar removida): {sorted(unexpected)}")
        if missing:
            raise RuntimeError(f"[resume] checkpoint incompativel: faltam chaves que o modelo "
                                f"atual precisa: {sorted(missing)}")
    elif args.init_from:
        raw = _load_raw_weights(args.init_from)
        _load_into_model(model, raw)
 
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
        "train_policy", "val_policy",
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
    _run_state.cycle_id = cycle_id
    _run_state.args = args
    _run_state.device = device
    if resumed is not None and not new_cycle:
        # retomando o MESMO ciclo: os arquivos ja em disco (epoch anterior)
        # devem ser apagados quando o proximo epoch for salvo, entao ja
        # entram como "last_paths" -- se new_cycle=True, fica {} de proposito
        # (nao mexemos nos arquivos do ciclo anterior).
        _run_state.last_paths = dict(
            state=_train_state_path(ckpt_dir, cycle_id, resumed["epoch"]),
            bin=_ckpt_bin_path(ckpt_dir, cycle_id, resumed["epoch"]),
            config=_config_json_path(ckpt_dir, cycle_id, resumed["epoch"]),
        )
    else:
        _run_state.last_paths = {}
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
        tr_total = dict(loss=0.0, outcome=0.0, policy=0.0, correct=0, pw=0.0)
        n_train_items = len(train_idx)
        epoch_start_time = time.time()
        last_progress_print = epoch_start_time
        positions_done = 0
        chunk_num = 0
        progress_print_count = 0
        last_progress_was_inline = False

        for chunk_idx in iter_chunks(n_train_items, chunk_size, rng):
            chunk_num += 1
            global_idx = train_idx[chunk_idx]
            chunk = train_ds[global_idx]
            k_chunk = k_by_source[global_idx]
            pw_chunk = None if policy_w_by_sample is None else policy_w_by_sample[global_idx]
            perm = rng.permutation(len(chunk))  # embaralha uma vez por chunk
            chunk = chunk[perm]
            k_chunk = k_chunk[perm]
            if pw_chunk is not None:
                pw_chunk = pw_chunk[perm]

            for t in iter_gpu_batches(chunk, k_chunk, device, batch_size, gpu_chunk_size, pw_chunk):
                policy_t = t["policy_target"]
                pw = t["policy_w"]

                value_wl, policy_logits = model(t["x"])
                loss_outcome = F.binary_cross_entropy_with_logits(value_wl, t["wl_target"])
                loss_policy = weighted_policy_loss(policy_logits, policy_t, pw)
                loss = args.w_outcome * loss_outcome + args.w_policy * loss_policy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip_norm and args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                opt.step()
                clipper(model)

                nb = len(t["x"])
                nbw = pw.sum().item()
                tr_total["loss"] += loss.item() * nb
                tr_total["outcome"] += loss_outcome.item() * nb
                tr_total["policy"] += loss_policy.item() * nbw
                tr_total["pw"] += nbw
                hit = (policy_logits.argmax(dim=-1) == policy_t).float()
                tr_total["correct"] += (hit * pw).sum().item()
                positions_done += nb

                now = time.time()
                if args.progress_every_secs > 0 and now - last_progress_print >= args.progress_every_secs:
                    last_progress_print = now
 
                now = time.time()
                if args.progress_every_secs > 0 and now - last_progress_print >= args.progress_every_secs:
                    last_progress_print = now
                    progress_print_count += 1
                    frac = min(1.0, positions_done / max(1, n_train_items))
                    elapsed = now - epoch_start_time
                    eta_epoch = elapsed / frac - elapsed if frac > 0 else 0.0
                    running_loss = tr_total["loss"] / max(1, positions_done)
                    running_acc = tr_total["correct"] / max(1, positions_done)
                    line = (f"  epoch {epoch:3d}/{args.epochs}  {frac * 100:5.1f}% "
                            f"({positions_done / 1e6:.2f}M/{n_train_items / 1e6:.2f}M pos) | "
                            f"loss={running_loss:.4f} acc={running_acc:.3f} | "
                            f"ETA epoch: {format_duration(eta_epoch)}")
                    # Atualiza a mesma linha (\r) na maioria das vezes; a cada
                    # --progress-permanent-every atualizacoes, deixa uma linha
                    # fixa no scrollback (estilo mixtrain.py) pra ter historico
                    # visivel sem inundar o terminal com uma linha nova a cada
                    # poucos segundos.
                    permanent = (progress_print_count % max(1, args.progress_permanent_every) == 0)
                    if permanent:
                        print(line)
                        last_progress_was_inline = False
                    else:
                        print(line + " " * 8, end="\r", flush=True)
                        last_progress_was_inline = True
 
        for k in ("loss", "outcome"):
            tr_total[k] /= max(1, n_train_items)
        # Same rule as run_eval: the policy metrics are averaged over the
        # samples that the policy loss saw, not over every sample.
        tr_total["policy"] /= max(1.0, tr_total["pw"])
        tr_total["policy_acc"] = tr_total["correct"] / max(1.0, tr_total["pw"])

        if last_progress_was_inline:
            print()  # fecha a linha viva (\r) antes de qualquer print permanente do epoch

        va = run_eval(val_ds, val_idx, val_k_by_source, batch_size, chunk_size, gpu_chunk_size, model, device,
                      args.w_outcome, args.w_policy, val_policy_w)

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_total["loss"])
        history["val_loss"].append(va["loss"])
        history["train_outcome"].append(tr_total["outcome"])
        history["val_outcome"].append(va["outcome"])
        history["train_policy"].append(tr_total["policy"])
        history["val_policy"].append(va["policy"])
        history["train_policy_acc"].append(tr_total["policy_acc"])
        history["val_policy_acc"].append(va["policy_acc"])
        history["lr"].append(lr)
        history["wd"].append(wd)

        monitored = {"val_loss": va["loss"], "val_outcome": va["outcome"],
                     "val_policy": va["policy"],
                     "val_policy_acc": va["policy_acc"]}[args.monitor]

        improved, should_stop = stopper.step(monitored, epoch, model)
 
        # checkpoint de resume: gravado a CADA epoch (nao so quando melhora) --
        # e o que permite retomar depois de fechar o processo normalmente.
        # ckpt_<ciclo>_ep<NNNN>.bin sempre leva os pesos do MELHOR epoch ate
        # agora (stopper.best_state), mesmo quando o epoch atual nao
        # melhorou -- so o nome (epoch/ciclo) muda a cada chamada; o arquivo
        # do epoch anterior do mesmo ciclo e apagado logo em seguida (ver
        # save_checkpoint_bundle).
        _run_state.epoch_completed = True
        if ckpt_dir:
            bin_path = save_checkpoint_bundle(_run_state, epoch, True)
 
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_time = (time.time() - t0) / epoch
        eta_run = format_duration(avg_epoch_time * (args.epochs - epoch))
        vram_note = ""
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            vram_note = f"  vram={peak_gb:.2f}GB"
 
        if epoch % args.log_every == 0 or epoch == args.epochs or improved or should_stop:
            star = " *" if improved else ""
            print(f"epoch {epoch:3d}/{args.epochs}  lr={lr:.2e}  wd={wd:.2e}  "
                  f"time={format_duration(epoch_duration)}  total={format_duration(time.time() - t0)}  "
                  f"ETA={eta_run}{vram_note}{star}")
            print(f"  treino  loss={tr_total['loss']:.4f}  acc={tr_total['policy_acc']:.3f}  "
                  f"(outcome={tr_total['outcome']:.4f} policy={tr_total['policy']:.4f})")
            print(f"  valid   loss={va['loss']:.4f}  acc={va['policy_acc']:.3f}  "
                  f"(outcome={va['outcome']:.4f} policy={va['policy']:.4f})")
 
        if args.plot_dir and (epoch % args.plot_every_epochs == 0 or epoch == args.epochs or should_stop):
            save_plots(history, args.plot_dir)
 
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
    # nota: o ckpt_<ciclo>_ep<N>.bin em --ckpt-dir NAO e reescrito aqui -- ja
    # reflete os pesos do melhor epoch (gravado dentro do loop, a cada
    # epoch); export_weights(model, args.out) acima grava so em --out, o
    # arquivo final pedido pelo usuario, nunca em --ckpt-dir.
 
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
    # strict=False: um state_dict de ANTES de 2026-08 (com chaves
    # value1_aux.*/value2_aux.*, cabeca auxiliar ja removida de
    # QuoridorNNUE) nao pode ser carregado em modo estrito -- torch reclama
    # de "unexpected key(s)". Avisamos e seguimos, descartando essas
    # chaves; todas as chaves que o modelo NOVO precisa (fc1/value*_wl/
    # policy) continuam batendo normalmente.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[export] aviso: {len(unexpected)} chave(s) do checkpoint ignoradas "
              f"(formato antigo, provavelmente cabeca auxiliar removida): {sorted(unexpected)}")
    if missing:
        raise RuntimeError(f"[export] checkpoint incompativel: faltam chaves que o modelo atual "
                            f"precisa: {sorted(missing)}")
    export_weights(model, path)


def _load_raw_weights(path):
    """Le um .bin de pesos crus (float32, layout de NNUEWeights::loadFromFile
    em nnue.hpp) para uso com --init-from. Aceita dois formatos, detectados
    pelo TAMANHO do arquivo:
      - atual (2026-08+, sem cabeca auxiliar): w1,b1,wv1_wl,bv1_wl,wv2_wl,
        bv2_wl,wp,bp.
      - antigo (com cabeca auxiliar, pre-2026-08): mesma coisa + um bloco
        extra wv1_aux,bv1_aux,wv2_aux,bv2_aux logo apos a cabeca WL.
    No formato antigo, o bloco aux e lido (pra manter o cursor do arquivo
    no lugar certo) e DESCARTADO -- a cabeca auxiliar nao existe mais no
    modelo, entao nao ha onde copiar esses pesos, e isso e o esperado (nao
    um erro): --init-from com um .bin desse tipo deve funcionar sem
    intervencao, so ignorando silenciosamente (com aviso) os pesos que a
    arquitetura atual nao usa mais."""
    head_floats = HIDDEN * 32 + 32 + 32 + 1  # wv1 + bv1 + wv2 + bv2 de UMA cabeca 256->32->1
    base_floats = NUM_FEATURES * HIDDEN + HIDDEN + head_floats
    tail_floats = POLICY_OUT * HIDDEN + POLICY_OUT
    expected_new = (base_floats + tail_floats) * 4
    expected_old = (base_floats + head_floats + tail_floats) * 4  # + bloco da cabeca aux

    file_size = os.path.getsize(path)
    if file_size not in (expected_new, expected_old):
        raise ValueError(
            f"[init-from] '{path}' tem tamanho inesperado ({file_size} bytes) -- nao bate nem "
            f"com o formato atual ({expected_new} bytes, sem cabeca auxiliar) nem com o formato "
            f"antigo ({expected_old} bytes, com cabeca auxiliar). Verifique se o arquivo nao esta "
            f"truncado/corrompido ou se veio de uma versao de nnue.hpp com outra arquitetura.")
    is_old_format = (file_size == expected_old)

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
        if is_old_format:
            read_head()  # cabeca auxiliar antiga -- le pra avancar o cursor, descarta o resultado
            print(f"[init-from] '{path}' esta no formato antigo (com cabeca auxiliar) -- "
                  f"cabeca auxiliar ignorada, so os pesos de w1/b1/cabeca WL/policy sao usados.")
        wp = np.fromfile(f, dtype="<f4", count=POLICY_OUT * HIDDEN).reshape(POLICY_OUT, HIDDEN)
        bp = np.fromfile(f, dtype="<f4", count=POLICY_OUT)
    return dict(w1=w1, b1=b1, wv1_wl=wv1_wl, bv1_wl=bv1_wl, wv2_wl=wv2_wl, bv2_wl=bv2_wl, wp=wp, bp=bp)


def _load_into_model(model: QuoridorNNUE, raw: dict):
    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(raw["w1"].T.copy()))
        model.fc1.bias.copy_(torch.from_numpy(raw["b1"]))
        model.value1_wl.weight.copy_(torch.from_numpy(raw["wv1_wl"].T.copy()))
        model.value1_wl.bias.copy_(torch.from_numpy(raw["bv1_wl"]))
        model.value2_wl.weight.copy_(torch.from_numpy(raw["wv2_wl"].reshape(1, -1)))
        model.value2_wl.bias.copy_(torch.from_numpy(raw["bv2_wl"]))
        model.policy.weight.copy_(torch.from_numpy(raw["wp"]))
        model.policy.bias.copy_(torch.from_numpy(raw["bp"]))


 
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
 
    g_data = p.add_argument_group("dados")
    g_data.add_argument("--data", action="append", default=None,
                         help="arquivo(s) .bin de self-play; aceita lista separada por virgula, "
                              "diretorio, glob, ou a flag repetida varias vezes. Ignorado se "
                              "--data-sources (ou DATA_SOURCES_DEFAULT no topo do arquivo) "
                              "estiver configurado")
    g_data.add_argument("--data-sources", default=None,
                         help="JSON com lista de fontes por pasta + fracao de posicoes, "
                              "sobrepondo DATA_SOURCES_DEFAULT -- mesmo formato: "
                              '\'[{"path":"gen7","frac":1.0},{"path":"gen6","frac":0.3}]\'. '
                              "Uso normal e editar DATA_SOURCES_DEFAULT no topo do arquivo em "
                              "vez desta flag (ver comentario ali)")
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
    g_opt.add_argument("--new-cycle-lr", type=float, default=NEW_CYCLE_LR_DEFAULT,
                        help=f"LR usado no lugar de LR_DEFAULT quando um NOVO CICLO e detectado "
                             f"(checkpoint anterior completou um treino inteiro; resume so os "
                             f"pesos, ver cycle_exhausted em train()) e --lr ficou em 'auto' "
                             f"(default {NEW_CYCLE_LR_DEFAULT}; nunca sobrescreve um --lr "
                             f"numerico explicito)")
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
    g_es.add_argument("--monitor", choices=["val_loss", "val_outcome",
                                             "val_policy", "val_policy_acc"],
                       default=MONITOR_DEFAULT)
    g_es.add_argument("--no-restore-best", action="store_true",
                       help="exporta os pesos do ultimo epoch em vez dos do melhor epoch")
 
    g_ckpt = p.add_argument_group("checkpoint / resume")
    g_ckpt.add_argument("--ckpt-dir", default=CKPT_DIR_DEFAULT,
                         help=f"diretorio de checkpoints. A cada epoch grava "
                              f"train_state_<ciclo>_ep<N>.pt / ckpt_<ciclo>_ep<N>.bin / "
                              f"train_config_<ciclo>_ep<N>.json e apaga os do epoch anterior "
                              f"DO MESMO CICLO -- arquivos de ciclos anteriores ficam no disco. "
                              f"Resume automatico acha o train_state_*_ep*.pt mais recente "
                              f"sozinho; passe vazio/None via codigo pra desligar checkpointing "
                              f"(default {CKPT_DIR_DEFAULT})")
    g_ckpt.add_argument("--fresh", dest="fresh", action="store_true", default=FRESH_DEFAULT,
                         help="ignora qualquer checkpoint de resume existente em --ckpt-dir e "
                              "comeca do zero (ou de --init-from, se passado)")
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
    g_mem.add_argument("--batch-size-max", type=int, default=BATCH_SIZE_MAX_DEFAULT,
                        help=f"teto de seguranca para --batch-size=auto, independente do quanto "
                             f"--vram-budget-gb permitiria (default {BATCH_SIZE_MAX_DEFAULT})")
    g_mem.add_argument("--chunk-size-max", type=int, default=CHUNK_SIZE_MAX_DEFAULT,
                        help=f"teto de seguranca para --chunk-size=auto, independente do quanto "
                             f"--ram-budget-gb permitiria (default {CHUNK_SIZE_MAX_DEFAULT})")
 
    g_loss = p.add_argument_group("pesos de loss / QAT")
    g_loss.add_argument("--w-outcome", type=float, default=W_OUTCOME_DEFAULT)
    g_loss.add_argument("--w-policy", type=float, default=W_POLICY_DEFAULT)
    g_loss.add_argument("--policy-opening-plies", type=int, default=POLICY_OPENING_PLIES_DEFAULT,
                        help=f"plies iniciais excluidos da loss de POLICY, mantidos na de valor "
                             f"(padrao: {POLICY_OPENING_PLIES_DEFAULT}). 0 = treina a policy em "
                             f"todos os plies, comportamento anterior a 2026-08-26.")
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
                         help="intervalo minimo entre atualizacoes da linha viva de progresso "
                              f"dentro do epoch; 0 desliga (default {PROGRESS_EVERY_SECS_DEFAULT}s)")
    g_perf.add_argument("--progress-permanent-every", type=int, default=PROGRESS_PERMANENT_EVERY_DEFAULT,
                         help="a cada quantas atualizacoes de progresso uma linha fica fixa no "
                              "scrollback (as demais so atualizam a linha viva via \\r), estilo "
                              f"mixtrain.py (default {PROGRESS_PERMANENT_EVERY_DEFAULT})")
 
    g_out = p.add_argument_group("saida")
    g_out.add_argument("--out", default=OUT_DEFAULT, help="caminho de saida dos pesos treinados (.bin)")
    g_out.add_argument("--no-quantize", action="store_true")
    g_out.add_argument("--quant-out", default=None)
    g_out.add_argument("--plot-dir", default=None,
                        help="diretorio para salvar plots de convergencia/validacao (PNG). Regravado "
                             "periodicamente durante o treino (ver --plot-every-epochs), nao so no "
                             "final -- da pra abrir e acompanhar com o treino ainda rodando.")
    g_out.add_argument("--plot-every-epochs", type=int, default=PLOT_EVERY_EPOCHS_DEFAULT,
                        help="a cada quantos epochs o PNG de --plot-dir e regravado "
                             f"(default {PLOT_EVERY_EPOCHS_DEFAULT})")
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
        latest_state_path = _find_latest_train_state(pre_args.ckpt_dir)
        if latest_state_path is not None:
            m = _TRAIN_STATE_RE.match(os.path.basename(latest_state_path))
            auto_path = _config_json_path(pre_args.ckpt_dir, m.group("cycle"), int(m.group("epoch")))
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
                # "lr" fica de fora do set_defaults: seu valor ja foi
                # capturado em hp_lr acima e e aplicado mais abaixo via
                # a resolucao de --lr "auto" (que tambem decide entre
                # hp_lr e --new-cycle-lr quando um NOVO CICLO e detectado
                # em train()). Se "lr" entrar aqui, o set_defaults troca o
                # default de --lr (que e a string sentinela "auto") pelo
                # float herdado, e isso quebra a deteccao de
                # args.lr_auto logo abaixo -- o override de
                # --new-cycle-lr no cycle_exhausted nunca dispara.
                # "lr_min" tambem fica de fora: e relativo ao lr do ciclo
                # anterior (ex. lr_min=1e-5 com lr=5e-5). Num NOVO CICLO
                # com new_cycle_lr=1e-5, herdar lr_min=1e-5 faz
                # base_lr == lr_min e o schedule cosine fica constante em
                # 1e-5 para sempre. LR_MIN_DEFAULT (5e-6) e o valor correto
                # quando nao passado explicitamente.
                hp_for_defaults = {k: v for k, v in hp.items() if k not in ("lr", "lr_min")}
                if hp_for_defaults:
                    p.set_defaults(**hp_for_defaults)
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
    args.lr_auto = (args.lr == "auto")
    if args.lr_auto:
        if hp_lr is not None:
            args.lr = float(hp_lr)
            print(f"[lr] auto: herdado de {cfg_path} (lr={args.lr})")
        else:
            print(f"[lr] aviso: --lr auto pedido, mas nao ha checkpoint/config anterior "
                  f"em --ckpt-dir nem --resume-config -- comecando do zero com "
                  f"lr={LR_DEFAULT} (LR_DEFAULT)")
            args.lr = LR_DEFAULT
        # NOTA: se train() detectar um NOVO CICLO (checkpoint anterior com
        # treino completo, so pesos sendo retomados), args.lr e sobrescrito
        # de novo ali para --new-cycle-lr (ver cycle_exhausted) -- o valor
        # herdado/LR_DEFAULT acima e so o default pra treino "normal"
        # (do zero ou resume no meio de um epoch).
    else:
        args.lr = float(args.lr)
 
    if args.data_sources:
        try:
            args.data_sources = json.loads(args.data_sources)
        except json.JSONDecodeError as e:
            p.error(f"--data-sources: JSON invalido ({e})")
 
    return args
 
 
if __name__ == "__main__":
    train(parse_args())