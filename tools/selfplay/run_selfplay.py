"""
run_selfplay.py -- Orquestrador Python para geração de dados de self-play.

Edite as variáveis da seção CONFIG abaixo e execute:
    python3 run_selfplay.py

O script localiza automaticamente selfplay.exe relativo a este arquivo,
compila se não encontrar (requer g++ com MinGW-w64), e chama o executável
com os parâmetros configurados.

Cada chunk vira um arquivo .bin independente em DATA_DIR, nomeado como
selfplay_000.bin, selfplay_001.bin, etc. O train_nnue.py aceita passar
vários arquivos de uma vez com --data.

SIZING GUIDE (hardware: 32 GB RAM, 6 GB VRAM):
  - TrainingSample = 32 bytes; partida media ~120 posicoes -> ~3.8 KB/partida
  - CHUNK_GAMES = 2000 -> arquivo de ~7.7 MB (carrega inteiro em RAM sem stress)
  - TOTAL_GAMES = 20000 -> 10 chunks, ~77 MB total -> dataset sólido p/ inicio
  - Para treino com train_nnue.py: o default --ram-budget-gb 32 e
    --vram-budget-gb 6 já usam os 200k-posições de chunk size automaticamente.
"""

import argparse
import os
import subprocess
import sys
import time

# =============================================================================
# CONFIG -- edite estas variaveis conforme necessario
# =============================================================================

# --- Geração de partidas ---
TOTAL_GAMES   = 50000   # partidas totais a gerar nesta rodada
CHUNK_GAMES   = 3000    # partidas por arquivo .bin
                        # Cada chunk fica ~6.5 MB -- confortável pra 32 GB RAM

# --- Busca ---
MAX_DEPTH     = 50      # profundidade máxima do negamax (iterative deepening)
TIME_MS       = 40     # orçamento de tempo por lance em ms
                        # 200 ms = boa qualidade; reduza para 50-100 ms se quiser
                        # gerar muito volume rapidamente (em detrimento da força)
 
# --- Modo de geração ---
# "epsilon"    = modo antigo/original: fases de abertura por epsilon-greedy
#                (ver OPENING_PLIES*/EPSILON_* abaixo).
# "montecarlo" = modo novo: amostragem por temperatura (estilo AlphaZero)
#                sobre a cabeça de política da NNUE, desde o lance 1 (ver
#                MC_* abaixo). Mais rápido (mais partidas/minuto) porque a
#                abertura não faz busca nenhuma, só forward pass da política
#                -- ver nota completa em SelfPlayConfig::mcMode (selfplay.hpp).
# Os dois modos coexistem: os parâmetros de cada um ficam sempre disponíveis
# abaixo e nenhum apaga o outro -- só o MODE escolhido é passado como flag
# ativa (--mc-mode) pro binário nesta execução.
MODE = "montecarlo"   # "epsilon" ou "montecarlo"

# --- Abertura aleatória (modo "epsilon") ---
# Fase 1: lances iniciais (óbvios no Quoridor) com muito pouco ruído
OPENING_PLIES1   = 6       # lances 1 a N1 sujeitos a EPSILON_OPENING1
EPSILON_OPENING1 = 0.2     # baixo: não distorce os lances óbvios da abertura

# Fase 2: janela de exploração pesada para diversificar posições iniciais
OPENING_PLIES2   = 12    # lances N1+1 a N2 sujeitos a EPSILON_OPENING2
EPSILON_OPENING2 = 0.7    # alto: cria muita variedade de abertura (lance totalmente aleatório)
 
EPSILON_MIDGAME  = 0.01   # prob. de desvio no midgame: escolhe 2º ou 3º melhor lance (não totalmente aleatório)
                          # -- também usado como ruído residual do modo "montecarlo" após a janela de decaimento (ver MC_TEMP_DECAY_PLIES)

# --- Temperatura Monte Carlo/AlphaZero (modo "montecarlo") ---
# Softmax(logit da política / temperatura) sobre os lances  legais, em duas
# fases sucessivas, sem busca nenhuma enquanto alguma delas estiver ativa:
#   fase 1 "óbvios"  [0..MC_OBVIOUS_PLIES)                -> temperatura fixa baixa
#   fase 2 "opening" [MC_OBVIOUS_PLIES..+MC_TEMP_DECAY_PLIES) -> decai linearmente
#                                                              de MC_TEMP_OPENING a MC_TEMP_END
# Depois disso, mesmo comportamento do midgame do modo antigo
# (EPSILON_MIDGAME -> 2º/3º melhor lance; senão busca completa).
MC_OBVIOUS_PLIES    = 6      # nº de lances iniciais (obvios no Quoridor) com temperatura fixa e baixa
MC_TEMP_OBVIOUS      = 0.3  # temperatura da fase 1 (baixa -> quase argmax, pouca variancia de proposito)
MC_TEMP_OPENING     = 1.00   # temperatura no inicio da fase 2 (<1 afia -- mais perto do argmax da politica)
                             # 2026-08-25: era 1.00. Valores >1 achatam a softmax, entao o lance
                             # amostrado fica em media PIOR que o argmax da propria politica -- e
                             # esse lance vira o policyTarget gravado (selfplay.hpp:533), treinando
                             # a cabeca de politica a imitar uma versao degradada de si mesma.
MC_TEMP_END         = 0.12   # temperatura ao fim da fase 2 (<1 afia -- quase argmax)
MC_TEMP_DECAY_PLIES = 16     # nº de lances da fase 2 (logo apos MC_OBVIOUS_PLIES) sobre os quais a temperatura decai
 
# --- Segurança ---  
MAX_PLIES     = 300     # corte: partidas que não terminam são descartadas

# --- TT (transposition table) ---
# False (default) = as 2 cores dividem uma única engine/TT dentro da mesma
#   partida -- mais rápido para gerar dados de treino (menos memória de TT
#   por thread, aproveita transposições do lado oposto). É o padrão usual
#   em geração de self-play e o objetivo aqui é throughput.
# True = cada cor usa engine/TT própria e isolada, igual à arena
#   (teste/arena_dual.cpp). Deixa a geração ~2x mais cara em memória de TT
#   por thread; use quando o objetivo é comparar comportamento/taxa de
#   empate do selfplay com o da arena, não gerar dados de treino.
SEPARATE_TT   = False

# --- Paralelismo ---
THREADS       = 8       # 0 = auto (usa hardware_concurrency); ajuste se quiser
                        # reservar threads para outras tarefas

# --- Semente ---
SEED          = 150    # semente base do RNG; chunks subsequentes variam automaticamente

# --- Saída ---
# Use {shard:03d} para nomear os chunks automaticamente e {mode} para o
# modo desta execução ("epsilon" ou "montecarlo") -- separa os dois em
# pastas distintas automaticamente (data/selfplay/epsilon/... vs.
# data/selfplay/montecarlo/...) pra dar pra treinar misturando as duas
# fontes com pesos por-fonte (k diferentes) em train_nnue.py, sem que uma
# rodada sobrescreva/misture shards da outra sem querer.
OUT_TEMPLATE  = "data/selfplay/gen7-{mode}/selfplay_{shard:03d}.bin"

# --- Avaliação de folha (NNUE vs. heurística) ---
# NNUE é o default de avaliação deste binário desde 2026-08 (selfplay
# tenta carregar data/nnue/nnue_weights_int8.bin automaticamente; cai
# para evalSimple com aviso se o arquivo não existir) -- NAO É MAIS "None
# = heurístico, caminho = NNUE" como antes. Os dois controles agora são
# INDEPENDENTES:
#
# FORCE_HEURISTIC: o liga/desliga de verdade. True = ignora qualquer NNUE
#   e usa evalSimple sempre (equivalente a --heuristic na linha de
#   comando) -- é isto que você muda pra gerar uma bateria de testes
#   heurística pura. False (default) = tenta NNUE, cai pra heurístico
#   sozinho se não achar pesos.
FORCE_HEURISTIC = False
# 
# NNUE_WEIGHTS_PATH: só use se quiser apontar pra um arquivo de pesos
# DIFERENTE do default (data/nnue/nnue_weights_int8.bin). Deixe None pra
# usar o caminho default do binário.
NNUE_WEIGHTS_PATH = None  # ex: "data/nnue/nnue_weights_experimental.bin"

# POLICY_ORDERING: liga Negamax::setPolicyOrderingEnabled
# (prompt_policy_ordering.md) -- soma o logit da cabeça de política da
# NNUE como termo extra na ordenação de lances. Só tem efeito quando NNUE
# está de fato ativo (FORCE_HEURISTIC=False e pesos carregam com
# sucesso); com heurística é ignorado sem erro. Default True desde 2026-08
# (era False) -- mesmo default de search.hpp/selfplay.hpp/arena.cpp/wasm
# agora. Shards gerados antes dessa mudança NÃO são reprodutíveis bit-a-
# bit com o default atual; use --no-policy-order pra voltar ao
# comportamento antigo. Equivale a --policy-order/--no-policy-order na
# linha de comando.
POLICY_ORDERING = True
# Piso de profundidade (search.hpp: Negamax::setPolicyOrderingMinDepth) --
# forwardPolicyQuant custa ~5.8x mais que o eval de folha; sem este piso
# ele roda em todo no interno e derruba nos/s ~3x (medido em producao).
POLICY_ORDER_MIN_DEPTH = 3

# --- MCTS híbrido com alpha-beta (busca de produção) ---
# Regra deste bloco: None = "vazio" -- não manda flag nenhuma e o selfplay.exe
# usa o valor de PRODUÇÃO, que está anotado no comentário de cada campo (a
# fonte da verdade é mcab::McabParams, em src/mcab.hpp). Preencher um campo
# aqui sobrescreve só ele.
#
# Precedência: flag de linha de comando > constante aqui > produção.
#
# RESSALVA DE FAIXA: os +46.9 ±23.5 Elo do híbrido sobre alpha-beta puro foram
# medidos a 200ms/lance. Ele roda a ~1/9 dos nós/s do AB puro; com TIME_MS
# baixo (o caso comum aqui, para gerar volume) a troca pode não compensar, e
# isso NÃO foi medido. Se estiver gerando com TIME_MS bem abaixo de 200,
# considere MCAB = False e compare a força da rede resultante.
MCAB              = None   # None = default do binário (LIGADO) | True | False
MCAB_NODES        = None   # produção: 20000 nós de árvore por lance
MCAB_LEAF_DEPTH   = None   # produção: 0 -- plies de alpha-beta em cada folha (medido)
MCAB_LEAF_DEPTH_MAX = None # produção: 8 (teto quando MCAB_ADAPTIVE_LEAF_DEPTH)
MCAB_ADAPTIVE_LEAF_DEPTH = None  # produção: desligado
MCAB_CPUCT        = None   # produção: 1.5
MCAB_FPU          = None   # produção: 0.0 (medido)
MCAB_SCORE_SCALE  = None   # produção: 200.0 (= NNUE_EVAL_SCALE)
MCAB_TREE_REUSE   = None   # produção: ligado (reuso de subárvore entre lances)
MCAB_MAX_TREE_DEPTH = None # produção: 48
# Ruído de Dirichlet nos priors da raiz: LIGADO por default no selfplay (e
# desligado na arena) -- é o que dá diversidade de abertura aos dados de
# treino. Desligue só se quiser reproduzir shards antigos.
MCAB_ROOT_NOISE   = None   # produção (selfplay): ligado
MCAB_ROOT_NOISE_ALPHA   = None  # produção: 0.3
MCAB_ROOT_NOISE_EPSILON = None  # produção: 0.25
MCAB_ROOT_SELECT  = None   # produção: "visits" | "q" | "visits-then-q"
MCAB_CLEAR_TT_PER_MOVE = None   # produção: desligado (limpa a TT do AB a cada lance)
MCAB_PROGRESSIVE_WIDENING = None # produção: desligado
MCAB_WIDENING_INITIAL = None    # produção: 16
MCAB_WIDENING_COEFFICIENT = None # produção: 2.0
MCAB_WIDENING_EXPONENT = None   # produção: 0.5

# =============================================================================
# PARÂMETROS DE BUSCA (search.hpp) -- mesma regra: None = produção
# =============================================================================
# Todos os knobs runtime de Negamax, os mesmos que tools/spsa/tune_spsa.cpp
# otimiza. None = não manda flag nenhuma e o binário fica no valor de PRODUÇÃO,
# anotado no comentário de cada linha (fonte da verdade: src/search.hpp).
#
# Precedência: flag de linha de comando > constante aqui > produção.
#
# Cuidado ao gerar dados de treino com qualquer um destes fora do default: os
# shards ficam com uma distribuição de posições diferente da de produção, e
# nada no arquivo .bin registra qual configuração os gerou -- anote no nome da
# pasta de saída (OUT_TEMPLATE) quando fugir dos valores de produção.
CONTEMPT             = None   # produção: -30 (score de empate, em centi-lances)
POLICY_ORDER_SCALE   = None   # produção: 400 (escala do logit da política na ordenação)
CAT_SCORE_SCALE      = None   # produção: 2 (peso do calor CAT vs. política em orderWallMoves)
LMR_MIN_DEPTH        = None   # produção: 3 (profundidade mínima para o LMR reduzir)
LMR_MIN_MOVE_INDEX   = None   # produção: 3 (1-based: 1º lance é PVS, reduz a partir do 3º)
LMR_DIVISOR          = None   # produção: 2.25 (divisor da fórmula de redução)
CAT_HOT_CM           = None   # produção: 150 (calor CAT que marca o lance como tático, pula LMR)
CAT_COLD_CM          = None   # produção: 30 (calor abaixo do qual o LMR reduz +1)
WALL_BFS_ORDER_MAX_PLY = None # produção: 2 (último ply com ordenação de muro por BFS)
QS_CRITICAL_BFS_DELTA  = None # produção: 2 (delta de BFS que torna o muro crítico na quiescência)
QUIESCENCE           = None   # produção: ligada -- True/False liga/desliga a quiescência de muro
LMR_PVS              = None   # produção: ligado -- True/False liga/desliga LMR+PVS

# Tabela usada pelo parse_args/monta-comando abaixo: (flag, constante, tipo).
# Um knob novo em search_tuning.hpp só precisa de uma linha aqui.
SEARCH_TUNING_KNOBS = [
    ("--contempt",               "contempt",               CONTEMPT,             int),
    ("--policy-order-scale",     "policy_order_scale",     POLICY_ORDER_SCALE,   int),
    ("--cat-score-scale",        "cat_score_scale",        CAT_SCORE_SCALE,      int),
    ("--lmr-min-depth",          "lmr_min_depth",          LMR_MIN_DEPTH,        int),
    ("--lmr-min-move-index",     "lmr_min_move_index",     LMR_MIN_MOVE_INDEX,   int),
    ("--lmr-divisor",            "lmr_divisor",            LMR_DIVISOR,          float),
    ("--cat-hot-cm",             "cat_hot_cm",             CAT_HOT_CM,           int),
    ("--cat-cold-cm",            "cat_cold_cm",            CAT_COLD_CM,          int),
    ("--wall-bfs-order-max-ply", "wall_bfs_order_max_ply", WALL_BFS_ORDER_MAX_PLY, int),
    ("--qs-critical-bfs-delta",  "qs_critical_bfs_delta",  QS_CRITICAL_BFS_DELTA,  int),
]
# Liga/desliga: o binário tem as duas formas (--X e --no-X).
SEARCH_TUNING_FLAGS = [
    ("quiescence", "quiescence", QUIESCENCE),
    ("lmr-pvs",    "lmr_pvs",    LMR_PVS),
]

# =============================================================================
# INTERNALS -- normalmente não é necessário editar abaixo desta linha
# =============================================================================

def find_project_root():
    """Sobe na hierarquia de diretórios até encontrar a pasta 'src'."""
    here = os.path.abspath(os.path.dirname(__file__))
    candidate = here
    for _ in range(5):
        if os.path.isdir(os.path.join(candidate, "src")):
            return candidate
        candidate = os.path.dirname(candidate)
    return here  # fallback

def find_selfplay_exe(root):
    """Retorna o caminho do executável selfplay (Windows ou Linux)."""
    for name in ("selfplay.exe", "selfplay"):
        path = os.path.join(root, "bin", name)
        if os.path.isfile(path):
            return path
    return None

def compile_selfplay(root):
    """Tenta compilar selfplay usando o build script adequado."""
    import platform
    if platform.system() == "Windows":
        bat = os.path.join(root, "build", "build_selfplay.bat")
        if os.path.isfile(bat):
            print("[run_selfplay] selfplay.exe nao encontrado. Compilando...")
            ret = subprocess.call(["cmd", "/c", bat], cwd=root)
            return ret == 0
    else:
        sh = os.path.join(root, "build", "build_selfplay.sh")
        if os.path.isfile(sh):
            print("[run_selfplay] selfplay nao encontrado. Compilando...")
            ret = subprocess.call(["bash", sh], cwd=root)
            return ret == 0
    return False

def next_free_shard(root, template):
    """Retorna o menor índice de shard que ainda não existe em disco."""
    shard = 0
    while True:
        path = os.path.join(root, template.format(shard=shard))
        if not os.path.exists(path):
            return shard
        shard += 1


def parse_args():
    """CLI opcional -- toda flag aqui tem a constante correspondente na
    seção CONFIG acima como default. Rodar `python3 run_selfplay.py` sem
    argumento nenhum usa 100% das constantes do arquivo; qualquer flag
    passada aqui sobrepõe só aquela constante para esta execução, sem
    precisar editar/recompilar nada. Mesmo padrão já usado em
    selfplay_main.cpp (FORCE_HEURISTIC_DEFAULT/--heuristic/--nnue) e em
    tools/arena/run_arena.py."""
    p = argparse.ArgumentParser(description="Orquestrador de self-play (config no topo do arquivo ou via flags)")
    p.add_argument("--games", type=int, default=TOTAL_GAMES, help=f"partidas totais (padrao: {TOTAL_GAMES})")
    p.add_argument("--chunk-games", type=int, default=CHUNK_GAMES, help=f"partidas por arquivo .bin (padrao: {CHUNK_GAMES})")
    p.add_argument("--depth", type=int, default=MAX_DEPTH, help=f"profundidade maxima (padrao: {MAX_DEPTH})")
    p.add_argument("--time-ms", type=int, default=TIME_MS, help=f"ms por lance (padrao: {TIME_MS})")
    p.add_argument("--mode", choices=["epsilon", "montecarlo"], default=MODE,
                    help=f"modo de geracao (padrao: {MODE}). 'epsilon' = fases de abertura "
                         "epsilon-greedy (comportamento original). 'montecarlo' = amostragem "
                         "por temperatura estilo AlphaZero sobre a politica da NNUE, desde o "
                         "lance 1, sem busca na abertura (mais partidas/minuto).")
    p.add_argument("--opening-plies", type=int, default=OPENING_PLIES1)
    p.add_argument("--epsilon", type=float, default=EPSILON_OPENING1)
    p.add_argument("--opening-plies2", type=int, default=OPENING_PLIES2)
    p.add_argument("--epsilon-opening2", type=float, default=EPSILON_OPENING2)
    p.add_argument("--epsilon-midgame", type=float, default=EPSILON_MIDGAME,
                    help="tambem usado como ruido residual pos-decaimento no modo montecarlo")
    p.add_argument("--mc-obvious-plies", type=int, default=MC_OBVIOUS_PLIES,
                    help=f"nº de lances iniciais com temperatura fixa baixa, modo montecarlo (padrao: {MC_OBVIOUS_PLIES})")
    p.add_argument("--mc-temp-obvious", type=float, default=MC_TEMP_OBVIOUS,
                    help=f"temperatura da janela de lances obvios, modo montecarlo (padrao: {MC_TEMP_OBVIOUS})")
    p.add_argument("--mc-temp-opening", type=float, default=MC_TEMP_OPENING,
                    help=f"temperatura no inicio da fase de decaimento, modo montecarlo (padrao: {MC_TEMP_OPENING})")
    p.add_argument("--mc-temp-end", type=float, default=MC_TEMP_END,
                    help=f"temperatura ao fim da janela de decaimento, modo montecarlo (padrao: {MC_TEMP_END})")
    p.add_argument("--mc-temp-decay-plies", type=int, default=MC_TEMP_DECAY_PLIES,
                    help=f"numero de lances sobre os quais a temperatura decai, modo montecarlo (padrao: {MC_TEMP_DECAY_PLIES})")
    p.add_argument("--max-plies", type=int, default=MAX_PLIES)
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default=OUT_TEMPLATE, help=f"template de saida (padrao: {OUT_TEMPLATE})")
    p.add_argument("--separate-tt", dest="separate_tt", action="store_true", default=SEPARATE_TT)
    p.add_argument("--shared-tt", dest="separate_tt", action="store_false", help="sobrepoe SEPARATE_TT=True do arquivo")
    # NNUE vs. heuristica -- ver nota completa em FORCE_HEURISTIC acima.
    p.add_argument("--heuristic", dest="force_heuristic", action="store_true", default=FORCE_HEURISTIC,
                    help=f"forca avaliacao heuristica, ignorando NNUE (padrao: {FORCE_HEURISTIC})")
    p.add_argument("--nnue", dest="force_heuristic", action="store_false",
                    help="sobrepoe FORCE_HEURISTIC=True do arquivo, forcando NNUE default sem editar/recompilar")
    p.add_argument("--nnue-weights", default=NNUE_WEIGHTS_PATH, help="caminho alternativo de pesos NNUE (padrao: o default do binario)")
    p.add_argument("--policy-order", dest="policy_order", action="store_true", default=POLICY_ORDERING,
                    help=f"liga a ordenacao de lances assistida pela cabeca de politica da NNUE (padrao: {POLICY_ORDERING}); sem efeito com --heuristic")
    p.add_argument("--no-policy-order", dest="policy_order", action="store_false",
                    help="sobrepoe POLICY_ORDERING=True do arquivo, desligando sem editar/recompilar")
    p.add_argument("--policy-order-min-depth", type=int, default=POLICY_ORDER_MIN_DEPTH,
                    help=f"piso de profundidade da ordenacao por politica (padrao: {POLICY_ORDER_MIN_DEPTH}); sem efeito se --policy-order/POLICY_ORDERING estiver desligado")
    # MCTS hibrido. default=None em tudo => "nao passa flag, o binario decide";
    # o valor do bloco CONFIG entra depois, em resolve_mcab().
    p.add_argument("--mcab", dest="mcab", action="store_true", default=None,
                    help="forca o MCTS hibrido LIGADO (ja e o default do binario)")
    p.add_argument("--no-mcab", dest="mcab", action="store_false",
                    help="forca alpha-beta PURO (util com --time-ms baixo: o hibrido so foi medido a 200ms/lance)")
    p.add_argument("--mcab-nodes", type=int, default=None, help="nos de arvore por lance (producao: 20000)")
    p.add_argument("--mcab-leaf-depth", type=int, default=None, help="plies de alpha-beta em cada folha (producao: 0)")
    p.add_argument("--mcab-leaf-depth-max", type=int, default=None, help="teto de leaf-depth no modo adaptativo (producao: 8)")
    p.add_argument("--mcab-adaptive-leaf-depth", dest="mcab_adaptive_leaf_depth", action="store_true", default=None,
                    help="escala leaf-depth com as visitas do no pai (producao: desligado)")
    p.add_argument("--mcab-cpuct", type=float, default=None, help="constante de exploracao do PUCT (producao: 1.5)")
    p.add_argument("--mcab-fpu", type=float, default=None, help="first-play-urgency: Q(pai) - X (producao: 0.0)")
    p.add_argument("--mcab-score-scale", type=float, default=None, help="escala do sigmoide score->Q (producao: 200)")
    p.add_argument("--mcab-no-tree-reuse", dest="mcab_tree_reuse", action="store_false", default=None,
                    help="desliga o reuso de subarvore entre lances (producao: ligado)")
    p.add_argument("--mcab-no-root-noise", dest="mcab_root_noise", action="store_false", default=None,
                    help="desliga o ruido de Dirichlet na raiz (producao no selfplay: ligado)")
    p.add_argument("--mcab-root-noise-alpha", type=float, default=None, help="alfa da Dirichlet (producao: 0.3)")
    p.add_argument("--mcab-root-noise-epsilon", type=float, default=None, help="peso do ruido no prior (producao: 0.25)")
    p.add_argument("--mcab-max-tree-depth", type=int, default=None, help="pilha de acumuladores NNUE da arvore (producao: 48)")
    p.add_argument("--mcab-root-select", choices=["visits", "q", "visits-then-q"], default=None,
                    help="criterio de escolha do lance na raiz (producao: visits)")
    p.add_argument("--mcab-clear-tt-per-move", dest="mcab_clear_tt_per_move", action="store_true", default=None,
                    help="limpa a TT do alpha-beta a cada lance (producao: desligado)")
    p.add_argument("--mcab-progressive-widening", dest="mcab_progressive_widening", action="store_true", default=None,
                    help="liga progressive widening (producao: desligado)")
    p.add_argument("--mcab-no-progressive-widening", dest="mcab_progressive_widening", action="store_false",
                    help="desliga progressive widening")
    p.add_argument("--mcab-widening-initial", type=int, default=None, help="prefixo inicial (producao: 16)")
    p.add_argument("--mcab-widening-coefficient", type=float, default=None, help="coeficiente c (producao: 2.0)")
    p.add_argument("--mcab-widening-exponent", type=float, default=None, help="expoente alpha (producao: 0.5)")
    # Parametros de busca (search.hpp). default=None em tudo => "nao manda
    # flag"; a constante do bloco CONFIG entra depois, na montagem do comando.
    for flag, dest, _const, tipo in SEARCH_TUNING_KNOBS:
        p.add_argument(flag, dest=dest, type=tipo, default=None,
                        help=f"parametro de busca (vazio = valor de producao, ver {dest.upper()} no topo do arquivo)")
    for flag, dest, _const in SEARCH_TUNING_FLAGS:
        p.add_argument(f"--{flag}", dest=dest, action="store_true", default=None,
                        help=f"liga {flag} (producao: ligado)")
        p.add_argument(f"--no-{flag}", dest=dest, action="store_false",
                        help=f"desliga {flag}")
    return p.parse_args()


def main():
    args = parse_args()
    root = find_project_root()
    exe  = find_selfplay_exe(root)

    if exe is None:
        ok = compile_selfplay(root)
        exe = find_selfplay_exe(root)
        if exe is None:
            print("ERRO: nao foi possivel encontrar ou compilar selfplay.", file=sys.stderr)
            print("  Execute manualmente: build\\build_selfplay.bat", file=sys.stderr)
            sys.exit(1)

    # Resolve {mode} no template ANTES da logica de {shard:03d} (next_free_shard/
    # formatShardPath so conhecem o placeholder de shard) -- troca simples de
    # string, {mode} nunca aparece de fato no .bin final.
    args.out = args.out.replace("{mode}", args.mode)

    # Garante que o diretório de saída existe.
    out_dir = os.path.dirname(os.path.join(root, args.out.split("{")[0]))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Detecta o próximo shard livre para não sobrescrever dados existentes.
    start_shard = next_free_shard(root, args.out)
    if start_shard > 0:
        print(f"[run_selfplay] {start_shard} shard(s) existente(s) detectado(s); "
              f"iniciando a partir do shard {start_shard:03d}.")

    # Monta o path de saída relativo à raiz do projeto.
    out_full = os.path.join(root, args.out).replace("\\", "/")

    # Monta os argumentos do executável.
    cmd = [
        exe,
        "--games",           str(args.games),
        "--chunk-games",     str(args.chunk_games),
        "--depth",           str(args.depth),
        "--time-ms",         str(args.time_ms),
        "--opening-plies",   str(args.opening_plies),
        "--epsilon",         str(args.epsilon),
        "--opening-plies2",  str(args.opening_plies2),
        "--epsilon-opening2", str(args.epsilon_opening2),
        "--epsilon-midgame", str(args.epsilon_midgame),
        "--max-plies",       str(args.max_plies),
    ]
    if args.mode == "montecarlo":
        cmd += [
            "--mc-mode",
            "--mc-obvious-plies",    str(args.mc_obvious_plies),
            "--mc-temp-obvious",     str(args.mc_temp_obvious),
            "--mc-temp-opening",     str(args.mc_temp_opening),
            "--mc-temp-end",         str(args.mc_temp_end),
            "--mc-temp-decay-plies", str(args.mc_temp_decay_plies),
        ]
    cmd += [
        "--seed",            str(args.seed + start_shard),   # semente varia por sessão
        "--start-shard",     str(start_shard),
        "--out",             out_full,
    ]
    if args.threads > 0:
        cmd += ["--threads", str(args.threads)]
    if args.separate_tt:
        cmd += ["--separate-tt"]
    if args.force_heuristic:
        cmd += ["--heuristic"]
    elif args.nnue_weights:
        # Converte para caminho absoluto relativo à raiz do projeto
        nnue_abs = os.path.join(root, args.nnue_weights)
        cmd += ["--nnue-weights", nnue_abs.replace("\\", "/")]
    # 2026-08: selfplay_main agora nasce com policy ordering LIGADA por
    # default -- manda o "--no-policy-order" explicito quando o usuario
    # quer desligar, ou o default do binario (ligado) prevalece.
    if args.policy_order:
        cmd += ["--policy-order", "--policy-order-min-depth", str(args.policy_order_min_depth)]
    else:
        cmd += ["--no-policy-order"]

    # MCTS hibrido: CLI (ja em args) > bloco CONFIG > default do binario.
    # Nada resolvido (None nos dois) => nenhuma flag e mandada.
    def mcab_val(attr, constante):
        v = getattr(args, attr)
        return constante if v is None else v

    mcab_on = mcab_val("mcab", MCAB)
    if mcab_on is True:
        cmd += ["--mcab"]
    elif mcab_on is False:
        cmd += ["--no-mcab"]
    for attr, constante, flag in (
        ("mcab_nodes",              MCAB_NODES,              "--mcab-nodes"),
        ("mcab_leaf_depth",         MCAB_LEAF_DEPTH,         "--mcab-leaf-depth"),
        ("mcab_leaf_depth_max",     MCAB_LEAF_DEPTH_MAX,     "--mcab-leaf-depth-max"),
        ("mcab_cpuct",              MCAB_CPUCT,              "--mcab-cpuct"),
        ("mcab_fpu",                MCAB_FPU,                "--mcab-fpu"),
        ("mcab_score_scale",        MCAB_SCORE_SCALE,        "--mcab-score-scale"),
        ("mcab_root_noise_alpha",   MCAB_ROOT_NOISE_ALPHA,   "--mcab-root-noise-alpha"),
        ("mcab_root_noise_epsilon", MCAB_ROOT_NOISE_EPSILON, "--mcab-root-noise-epsilon"),
        ("mcab_max_tree_depth",     MCAB_MAX_TREE_DEPTH,     "--mcab-max-tree-depth"),
        ("mcab_widening_initial",   MCAB_WIDENING_INITIAL,   "--mcab-widening-initial"),
        ("mcab_widening_coefficient", MCAB_WIDENING_COEFFICIENT, "--mcab-widening-coefficient"),
        ("mcab_widening_exponent",  MCAB_WIDENING_EXPONENT,  "--mcab-widening-exponent"),
    ):
        v = mcab_val(attr, constante)
        if v is not None:
            cmd += [flag, str(v)]
    # Flags sem valor (so existem na forma "liga" ou "desliga" no binario).
    if mcab_val("mcab_adaptive_leaf_depth", MCAB_ADAPTIVE_LEAF_DEPTH) is True:
        cmd += ["--mcab-adaptive-leaf-depth"]
    if mcab_val("mcab_tree_reuse", MCAB_TREE_REUSE) is False:
        cmd += ["--mcab-no-tree-reuse"]
    if mcab_val("mcab_root_noise", MCAB_ROOT_NOISE) is False:
        cmd += ["--mcab-no-root-noise"]
    if mcab_val("mcab_root_select", MCAB_ROOT_SELECT) is not None:
        cmd += ["--mcab-root-select", str(mcab_val("mcab_root_select", MCAB_ROOT_SELECT))]
    if mcab_val("mcab_clear_tt_per_move", MCAB_CLEAR_TT_PER_MOVE) is True:
        cmd += ["--mcab-clear-tt-per-move"]
    if mcab_val("mcab_progressive_widening", MCAB_PROGRESSIVE_WIDENING) is True:
        cmd += ["--mcab-progressive-widening"]
    elif mcab_val("mcab_progressive_widening", MCAB_PROGRESSIVE_WIDENING) is False:
        cmd += ["--mcab-no-progressive-widening"]

    # Parametros de busca (search.hpp): mesma regra -- CLI > CONFIG > producao.
    # Nada resolvido => nenhuma flag, e o binario fica no valor de producao.
    tuning_ativos = []
    for flag, dest, constante, _tipo in SEARCH_TUNING_KNOBS:
        v = mcab_val(dest, constante)
        if v is not None:
            cmd += [flag, str(v)]
            tuning_ativos.append(f"{flag.lstrip('-')}={v}")
    for flag, dest, constante in SEARCH_TUNING_FLAGS:
        v = mcab_val(dest, constante)
        if v is True:
            cmd += [f"--{flag}"]
            tuning_ativos.append(f"{flag}=on")
        elif v is False:
            cmd += [f"--no-{flag}"]
            tuning_ativos.append(f"{flag}=off")

    print("=" * 60)
    print(f"[run_selfplay] Iniciando geração de dados")
    print(f"  Executável  : {exe}")
    print(f"  Partidas    : {args.games} total / {args.chunk_games} por chunk")
    print(f"  Busca       : depth<={args.depth}, {args.time_ms} ms/lance")
    print(f"  Modo        : {args.mode}")
    if args.mode == "montecarlo":
        print(f"                obvios=[lances 1..{args.mc_obvious_plies}] temp={args.mc_temp_obvious} (fixa, quase argmax)")
        print(f"                opening=[lances {args.mc_obvious_plies+1}..{args.mc_obvious_plies+args.mc_temp_decay_plies}] temp=[{args.mc_temp_opening}..{args.mc_temp_end}) (decaindo, sem busca)")
        print(f"                pos-decaimento: midgame eps={args.epsilon_midgame} (2º/3º melhor lance)")
    else:
        print(f"  Abertura    : fase1=[lances 1..{args.opening_plies}] eps={args.epsilon} (lance aleatório)")
        print(f"                fase2=[lances {args.opening_plies+1}..{args.opening_plies2}] eps={args.epsilon_opening2} (lance aleatório)")
        print(f"                midgame eps={args.epsilon_midgame} (2º/3º melhor lance)")
    print(f"  Avaliação   : {'heurística (evalSimple) -- forçada' if args.force_heuristic else 'NNUE (default do binário), com fallback automático para heurística se os pesos não existirem'}")
    print(f"  Ordenação politica NNUE: {'LIGADA (default, min-depth=' + str(args.policy_order_min_depth) + ')' if args.policy_order else 'desligada (--no-policy-order)'}{'  [sem efeito -- heuristica forcada]' if args.policy_order and args.force_heuristic else ''}")
    # Só aparece quando algo foi tirado do valor de produção -- silêncio aqui
    # significa "busca em produção", que é o caso normal.
    if tuning_ativos:
        print(f"  Busca (fora do default): {', '.join(tuning_ativos)}")
    print(f"  Threads     : {args.threads or 'auto'}")
    print(f"  TT          : {'separada por cor' if args.separate_tt else 'compartilhada entre as 2 cores (default)'}")
    print(f"  Shard início: {start_shard:03d}")
    print(f"  Saída       : {out_full}")
    print("=" * 60)
    print()

    t0 = time.time()
    ret = subprocess.call(cmd, cwd=root)
    elapsed = time.time() - t0

    print()
    if ret == 0:
        print(f"[run_selfplay] Concluído em {elapsed:.1f} s")
        print()
        print("Próximos passos -- treinar a NNUE com os chunks gerados:")
        print(f"  cd training")
        print(f"  python3 train_nnue.py \\")
        print(f"      --data ../data/selfplay/*.bin \\")
        print(f"      --out ../data/nnue/nnue_weights.bin \\")
        print(f"      --plot-dir ../data/nnue/plots")
    else:
        print(f"[run_selfplay] ERRO: selfplay terminou com código {ret}", file=sys.stderr)
        sys.exit(ret)

if __name__ == "__main__":
    main()
