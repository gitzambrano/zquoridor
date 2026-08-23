#!/usr/bin/env python3
import os
import sys
import json
import shutil
import argparse
import subprocess
import tempfile
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================================================================
# CONFIGURAÇÕES PADRÃO DO BENCHMARK (Altere aqui se preferir não usar CLI)
# ==============================================================================

# CONFIGURAÇÃO DE REFERÊNCIAS DO GIT PARA O CONFRONTO:
# - Se definir como None ou "", o script pega a VERSÃO ATUAL DA MÁQUINA (incluindo alterações não comitadas).
# - Exemplos de valores suportados:
#     GIT_REF1 = None          -> Versão local atual com alterações não comitadas
#     GIT_REF2 = "main"        -> Branch main do Git
#     GIT_REF2 = "v1.0"        -> Tag v1.0 do Git
#     GIT_REF2 = "HEAD"        -> Último commit da branch atual
#     GIT_REF2 = "HEAD~3"      -> 3 commits atrás
#     GIT_REF2 = "minha-branch"-> Outra branch
GIT_REF1 = None               # None = versão local não comitada (ou passe string de ref git)
GIT_REF2 = 'main'             # Ref Git base para o confronto (ex: 'main', 'v1.0', 'HEAD')

INVERT_COLORS = True          # Se True, joga cada abertura 2x invertendo as cores (par). Se False, joga apenas 1x por abertura.
CREATE_BIN = True             # Se True, salva os dados das partidas em data/arena/ no formato .bin de treino
GAMES = 1000                  # Quantidade total de jogos
REPORT_GAMES = 50             # Atualiza e imprime o relatório parcial a cada N jogos concluídos (default 50)
TIME_MS = 150                 # Tempo de pensamento por lance em milissegundos
THREADS = 14                  # Número de núcleos / processos em paralelo (default 14)
RANDOM_OPENING_PLIES = 4      # Quantidade de lances aleatórios na abertura
SEED = 8589                   # Semente aleatória
COMPILER = "g++"              # Compilador C++
CXX_FLAGS = "-O3 -std=c++17"  # Flags de compilação

# --- Avaliação de folha (NNUE vs. heurística) ---
# NNUE é o default de avaliação em arena.exe (tenta carregar
# data/nnue/nnue_weights_int8.bin automaticamente; cai para evalSimple
# com aviso se o arquivo não existir). As DUAS constantes abaixo são
# SEPARADAS por engine -- permite configurar heurística vs. NNUE no mesmo
# confronto só editando aqui, sem precisar de flag nenhuma (ex.:
# E1_HEURISTIC=True, E2_HEURISTIC=False roda Engine 1 heurística contra
# Engine 2 NNUE). --e1-heuristic/--e2-heuristic/--heuristic na linha de
# comando sobrepõem isto por execução, sem precisar editar o arquivo.
E1_HEURISTIC = False
E2_HEURISTIC = False
# --- Ordenacao assistida por politica NNUE (prompt_policy_ordering.md) ---
# Liga Negamax::setPolicyOrderingEnabled em cada engine -- só tem efeito
# quando aquela engine está de fato em NNUE (heurística ignora sem erro,
# ver arena.cpp). Mesmo padrão separado por engine que E1_HEURISTIC/
# E2_HEURISTIC acima: edite aqui para configurar sem flag nenhuma, ou use
# --policy-order/--e1-policy-order/--e2-policy-order/--e1-no-policy-order/
# --e2-no-policy-order na linha de comando para sobrepor por execução.
# Default True nas duas desde 2026-08 (mesmo default de arena.cpp/
# selfplay/search.hpp agora: NNUE default -> policy head default).
E1_POLICY_ORDER = True
E2_POLICY_ORDER = True
# Piso de profundidade (search.hpp: Negamax::setPolicyOrderingMinDepth) --
# forwardPolicyQuant custa ~5.8x mais que o eval de folha; sem este piso
# ele roda em todo no interno e derruba nos/s ~3x (medido em producao).
# Mesmo default (3) de search.hpp/selfplay/arena.cpp.
E1_POLICY_ORDER_MIN_DEPTH = 3
E2_POLICY_ORDER_MIN_DEPTH = 3
# =============================================================================
# CONFIG DE ENGINE -- tudo por lado (E1_ = --ref1, E2_ = --ref2)
# =============================================================================
# Regra dos dois blocos abaixo: None = "vazio" -- não manda flag nenhuma e o
# arena.exe usa o valor de PRODUÇÃO, anotado no comentário de cada linha (a
# fonte da verdade são mcab::McabParams em src/mcab.hpp e os defaults de
# src/search.hpp; os comentários aqui são cópia de leitura). Preencher um
# campo sobrescreve só ele, só naquele lado.
#
# Precedência: flag --e1-/--e2- > flag global > constante aqui > produção.
#
# Tudo é por engine porque o ponto da arena é comparar duas configurações com
# UM parâmetro diferente -- muito menos variância do que medir cada
# configuração contra uma baseline separada. Toda flag também existe na forma
# global (--mcab-cpuct, --contempt), que vale para as duas engines.
#
# Um --refX anterior a uma feature (MCab, ou um setter novo de search.hpp) não
# tem o método correspondente: o arena compila mesmo assim (dispatch SFINAE em
# src/mcab.hpp e src/search_tuning.hpp), avisa quando é o caso, e aquele lado
# simplesmente ignora o knob.

# --- MCTS híbrido com alpha-beta (busca de produção) ---
# O híbrido vem LIGADO. Ressalva de faixa: os +46.9 ±23.5 Elo sobre alpha-beta
# puro foram medidos a 200ms/lance. Ele roda a ~1/9 dos nós/s do AB puro, então
# em controles de tempo bem mais curtos a troca provavelmente inverte -- para
# medir AB puro de um lado, use False (ou --e1-no-mcab/--e2-no-mcab).
E1_MCAB = None                     # None = default do binário (LIGADO) | True | False
E2_MCAB = None                     # idem
E1_MCAB_NODES = None               # produção: 20000 nós de árvore por lance
E2_MCAB_NODES = None               # (a 200ms quem limita é o tempo, não isto)
E1_MCAB_LEAF_DEPTH = None          # produção: 0 -- plies de alpha-beta em cada folha.
E2_MCAB_LEAF_DEPTH = None          # 0 = só NNUE + quiescência de muro; a Fase 8 mediu
                                   # 4 como catastrófico e 0 como o único ponto que
                                   # bate o AB puro a 200ms
E1_MCAB_LEAF_DEPTH_MAX = None      # produção: 8 -- teto de leaf-depth; só tem efeito
E2_MCAB_LEAF_DEPTH_MAX = None      # com ADAPTIVE_LEAF_DEPTH ligado
E1_MCAB_ADAPTIVE_LEAF_DEPTH = None # produção: desligado (não implementado na Fase 1).
E2_MCAB_ADAPTIVE_LEAF_DEPTH = None # True liga; False é o próprio default
E1_MCAB_CPUCT = None               # produção: 1.5 -- constante de exploração do PUCT
E2_MCAB_CPUCT = None
E1_MCAB_FPU = None                 # produção: 0.0 -- First Play Urgency: Q(pai) - fpu
E2_MCAB_FPU = None                 # nos filhos não visitados (0.0 mediu +24.4 vs. 0.1)
E1_MCAB_SCORE_SCALE = None         # produção: 200.0 -- escala do sigmoide score->Q
E2_MCAB_SCORE_SCALE = None         # em scoreToQ (= NNUE_EVAL_SCALE)
E1_MCAB_ROOT_SELECT = None         # produção: "visits" | "q" | "visits-then-q"
E2_MCAB_ROOT_SELECT = None
E1_MCAB_BACKUP = None              # produção: "avg"; modo antigo: "minimax"
E2_MCAB_BACKUP = None
E1_MCAB_PROGRESSIVE_WIDENING = None # produção: False; experimento: True
E2_MCAB_PROGRESSIVE_WIDENING = None # idem
E1_MCAB_WIDENING_INITIAL = None    # produção: 16; só tem efeito com progressive widening
E2_MCAB_WIDENING_INITIAL = None
E1_MCAB_WIDENING_COEFFICIENT = None # produção: 2.0 em k=16+2*N^0.5
E2_MCAB_WIDENING_COEFFICIENT = None
E1_MCAB_WIDENING_EXPONENT = None   # produção: 0.5
E2_MCAB_WIDENING_EXPONENT = None
E1_MCAB_TREE_REUSE = None          # produção: ligado -- reusa a subárvore do lance
E2_MCAB_TREE_REUSE = None          # anterior. False desliga
E1_MCAB_CLEAR_TT_PER_MOVE = None   # produção: desligado -- True limpa a TT do
E2_MCAB_CLEAR_TT_PER_MOVE = None   # alpha-beta a cada lance
E1_MCAB_MAX_TREE_DEPTH = None      # produção: 48 -- dimensiona mcabAccStack
E2_MCAB_MAX_TREE_DEPTH = None
E1_MCAB_EQUIV_MODE = None          # produção: desligado -- True força nodeBudget=0
E2_MCAB_EQUIV_MODE = None          # (modo equivalência com o alpha-beta puro)

# --- Parâmetros de busca (search.hpp) ---
# Os mesmos knobs que tools/spsa/tune_spsa.cpp otimiza. Expostos aqui para dar
# para MEDIR em Elo o candidato que o SPSA propôs -- antes disso era preciso
# editar search.hpp e recompilar. O uso típico é preencher só o lado E1 e
# deixar E2 vazio: aí a partida é "candidato vs. produção" com o resto igual.
E1_CONTEMPT = None                 # produção: -30 (score de empate, em centi-lances)
E2_CONTEMPT = None
E1_POLICY_ORDER_SCALE = None       # produção: 400 (escala do logit da política
E2_POLICY_ORDER_SCALE = None       # somado na ordenação de lances)
E1_CAT_SCORE_SCALE = None          # produção: 2 (peso do calor CAT vs. política
E2_CAT_SCORE_SCALE = None          # em orderWallMoves)
E1_LMR_MIN_DEPTH = None            # produção: 3 (profundidade mínima p/ LMR reduzir)
E2_LMR_MIN_DEPTH = None
E1_LMR_MIN_MOVE_INDEX = None       # produção: 3 (1-based; o 1º lance é PVS)
E2_LMR_MIN_MOVE_INDEX = None
E1_LMR_DIVISOR = None              # produção: 2.25 (divisor da fórmula de redução)
E2_LMR_DIVISOR = None
E1_CAT_HOT_CM = None               # produção: 150 (calor CAT que marca o lance como
E2_CAT_HOT_CM = None               # tático e faz o LMR pular a redução)
E1_CAT_COLD_CM = None              # produção: 30 (calor abaixo do qual reduz +1)
E2_CAT_COLD_CM = None
E1_WALL_BFS_ORDER_MAX_PLY = None   # produção: 2 (último ply com ordenação de muro
E2_WALL_BFS_ORDER_MAX_PLY = None   # por BFS)
E1_QS_CRITICAL_BFS_DELTA = None    # produção: 2 (delta de BFS que torna o muro
E2_QS_CRITICAL_BFS_DELTA = None    # crítico na quiescência)
E1_QUIESCENCE = None               # produção: ligada -- True/False liga/desliga a
E2_QUIESCENCE = None               # quiescência de muro
E1_LMR_PVS = None                  # produção: ligado -- True/False liga/desliga
E2_LMR_PVS = None                  # LMR+PVS

# Tabelas que ligam constante <-> flag do arena.exe. Um knob novo precisa de
# uma linha aqui e mais nada -- parse_args e a montagem do comando são
# geradas a partir delas.
#   (sufixo da flag, dest base, const E1, const E2, tipo)
MCAB_VALUE_KNOBS = [
    ("mcab-nodes",          "mcab_nodes",          E1_MCAB_NODES,          E2_MCAB_NODES,          int),
    ("mcab-leaf-depth",     "mcab_leaf_depth",     E1_MCAB_LEAF_DEPTH,     E2_MCAB_LEAF_DEPTH,     int),
    ("mcab-leaf-depth-max", "mcab_leaf_depth_max", E1_MCAB_LEAF_DEPTH_MAX, E2_MCAB_LEAF_DEPTH_MAX, int),
    ("mcab-cpuct",          "mcab_cpuct",          E1_MCAB_CPUCT,          E2_MCAB_CPUCT,          float),
    ("mcab-fpu",            "mcab_fpu",            E1_MCAB_FPU,            E2_MCAB_FPU,            float),
    ("mcab-score-scale",    "mcab_score_scale",    E1_MCAB_SCORE_SCALE,    E2_MCAB_SCORE_SCALE,    float),
    ("mcab-max-tree-depth", "mcab_max_tree_depth", E1_MCAB_MAX_TREE_DEPTH, E2_MCAB_MAX_TREE_DEPTH, int),
    ("mcab-root-select",    "mcab_root_select",    E1_MCAB_ROOT_SELECT,    E2_MCAB_ROOT_SELECT,    str),
    ("mcab-backup",         "mcab_backup",         E1_MCAB_BACKUP,         E2_MCAB_BACKUP,         str),
    ("mcab-widening-initial", "mcab_widening_initial", E1_MCAB_WIDENING_INITIAL, E2_MCAB_WIDENING_INITIAL, int),
    ("mcab-widening-coefficient", "mcab_widening_coefficient", E1_MCAB_WIDENING_COEFFICIENT, E2_MCAB_WIDENING_COEFFICIENT, float),
    ("mcab-widening-exponent", "mcab_widening_exponent", E1_MCAB_WIDENING_EXPONENT, E2_MCAB_WIDENING_EXPONENT, float),
    ("contempt",               "contempt",               E1_CONTEMPT,               E2_CONTEMPT,               int),
    ("policy-order-scale",     "policy_order_scale",     E1_POLICY_ORDER_SCALE,     E2_POLICY_ORDER_SCALE,     int),
    ("cat-score-scale",        "cat_score_scale",        E1_CAT_SCORE_SCALE,        E2_CAT_SCORE_SCALE,        int),
    ("lmr-min-depth",          "lmr_min_depth",          E1_LMR_MIN_DEPTH,          E2_LMR_MIN_DEPTH,          int),
    ("lmr-min-move-index",     "lmr_min_move_index",     E1_LMR_MIN_MOVE_INDEX,     E2_LMR_MIN_MOVE_INDEX,     int),
    ("lmr-divisor",            "lmr_divisor",            E1_LMR_DIVISOR,            E2_LMR_DIVISOR,            float),
    ("cat-hot-cm",             "cat_hot_cm",             E1_CAT_HOT_CM,             E2_CAT_HOT_CM,             int),
    ("cat-cold-cm",            "cat_cold_cm",            E1_CAT_COLD_CM,            E2_CAT_COLD_CM,            int),
    ("wall-bfs-order-max-ply", "wall_bfs_order_max_ply", E1_WALL_BFS_ORDER_MAX_PLY, E2_WALL_BFS_ORDER_MAX_PLY, int),
    ("qs-critical-bfs-delta",  "qs_critical_bfs_delta",  E1_QS_CRITICAL_BFS_DELTA,  E2_QS_CRITICAL_BFS_DELTA,  int),
    # inv/ab-policy (2026-08-23), direcao E: pre-filtro AB da raiz do MCTS.
    # 0/ausente = off (producao). Constantes por lado ficam None no topo;
    # passe --eX-ab-prefilter-{depth,topk,timefrac} explicitamente.
    ("ab-prefilter-depth",    "ab_prefilter_depth",    None, None, int),
    ("ab-prefilter-topk",     "ab_prefilter_topk",     None, None, int),
    ("ab-prefilter-timefrac", "ab_prefilter_timefrac", None, None, float),
]
# Knobs liga/desliga. `flag_on`/`flag_off` = o que o arena.exe aceita; None
# nesse campo significa "aquele lado é o default do binário, não há flag".
#   (dest base, const E1, const E2, flag_on, flag_off)
MCAB_FLAG_KNOBS = [
    ("mcab_adaptive_leaf_depth", E1_MCAB_ADAPTIVE_LEAF_DEPTH, E2_MCAB_ADAPTIVE_LEAF_DEPTH, "mcab-adaptive-leaf-depth", None),
    ("mcab_tree_reuse",          E1_MCAB_TREE_REUSE,          E2_MCAB_TREE_REUSE,          None, "mcab-no-tree-reuse"),
    ("mcab_clear_tt_per_move",   E1_MCAB_CLEAR_TT_PER_MOVE,   E2_MCAB_CLEAR_TT_PER_MOVE,   "mcab-clear-tt-per-move", None),
    ("mcab_equiv_mode",          E1_MCAB_EQUIV_MODE,          E2_MCAB_EQUIV_MODE,          "mcab-equiv-mode", None),
    ("mcab_progressive_widening", E1_MCAB_PROGRESSIVE_WIDENING, E2_MCAB_PROGRESSIVE_WIDENING, "mcab-progressive-widening", "mcab-no-progressive-widening"),
    ("quiescence",               E1_QUIESCENCE,               E2_QUIESCENCE,               "quiescence", "no-quiescence"),
    ("lmr_pvs",                  E1_LMR_PVS,                  E2_LMR_PVS,                  "lmr-pvs", "no-lmr-pvs"),
]
# --- Pasta de pesos NNUE por engine (resolve o bug de main vs local
# carregando o MESMO arquivo, ver defaultNnueWeightsPath em nnue.hpp) ---
# "default"/None/"" -> tenta <pasta_do_ref>/data/nnue/nnue_weights_int8.bin
# (os pesos DAQUELE ref/worktree, ex. o que esta comitado no main) e cai
# para PROJECT_ROOT/data/nnue/nnue_weights_int8.bin so se o ref nao tiver
# pesos proprios. Qualquer outro valor e tratado como pasta explicita
# (procura nnue_weights_int8.bin dentro dela, ou aceita caminho de arquivo
# .bin direto). --e1-nnue/--e2-nnue (arquivo exato) continuam tendo
# prioridade sobre isto quando passados.
E1_WEIGHTS_DIR = "default"
E2_WEIGHTS_DIR = "default"
# ==============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIN_DIR = os.path.join(PROJECT_ROOT, "bin")
ARENA_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "arena")

def calculate_elo_and_ci(wins_1, wins_2, draws):
    total = wins_1 + wins_2 + draws
    if total == 0:
        return 0.0, 0.0
    
    score = (wins_1 + 0.5 * draws) / total
    if score <= 0.0:
        elo_diff = -800.0
    elif score >= 1.0:
        elo_diff = 800.0
    else:
        elo_diff = -400.0 * math.log10(1.0 / score - 1.0)
    
    p_w = wins_1 / total
    p_l = wins_2 / total
    p_d = draws / total
    
    variance = (p_w * (1.0 - score)**2 + p_l * (0.0 - score)**2 + p_d * (0.5 - score)**2)
    std_error = math.sqrt(variance / total) if total > 1 else 0.0
    
    if 0.0 < score < 1.0:
        factor = 400.0 / (math.log(10) * score * (1.0 - score))
        margin = 1.96 * std_error * factor
    else:
        margin = 0.0
        
    return elo_diff, margin

def _bump_mtimes(src_dir, epoch_offset_seconds):
    # Contorna uma peculiaridade do GCC: quando os headers de duas árvores
    # de fonte são byte-idênticos (muito comum aqui -- normalmente só
    # search.hpp muda entre ref1/ref2, rules.hpp/cat.hpp/dsu.hpp/
    # endgame_race.hpp ficam iguais) E foram escritos no mesmo segundo, o
    # pragma once do cpplib os trata como "o mesmo arquivo já incluído"
    # mesmo estando em diretórios (e inodes) diferentes -- silenciosamente
    # descarta a inclusão do segundo, e o segundo motor acaba compilado
    # com os símbolos do primeiro. Forçar mtimes distintos entre as duas
    # árvores evita esse falso-positivo de forma determinística.
    base = os.path.join(src_dir, "src")
    if not os.path.isdir(base):
        return
    stamp = 1577836800 + epoch_offset_seconds  # 2020-01-01 UTC + offset
    for name in os.listdir(base):
        if name.endswith(".hpp"):
            p = os.path.join(base, name)
            os.utime(p, (stamp, stamp))

def compile_arena(dir1, dir2, output_exe):
    """
    Compila tools/arena/arena.cpp incluindo os headers REAIS de dir1 (ref1)
    e dir2 (ref2) no MESMO binário, sob namespaces separados (qr_e1 /
    qr_e2) -- ao contrário do compile_arena antigo, que só compilava dir1
    e fazia "Engine 2" ser, na prática, uma cópia do código de Engine 1
    (o worktree de --ref2 era criado e descartado sem nunca ser usado).
    """
    os.makedirs(os.path.dirname(output_exe), exist_ok=True)
    arena_src = os.path.join(PROJECT_ROOT, "tools", "arena", "arena.cpp")

    # CORREÇÃO: quando --ref1/--ref2 são omitidos (caso comum: comparar o
    # build local contra ele mesmo, ex. smoke-test), prepare_engine_source
    # devolve PROJECT_ROOT para os dois -- dir1 e dir2 acabam sendo o MESMO
    # diretório real. Nesse caso ENGINE1_SEARCH_HPP e ENGINE2_SEARCH_HPP
    # apontariam pro MESMO arquivo físico, e #pragma once descartaria a
    # segunda inclusão de verdade (não é uma falsa detecção de duplicata
    # por mtime -- é literalmente o mesmo arquivo), fazendo a compilação
    # falhar com "qr_e2 has not been declared" etc. _bump_mtimes sozinho
    # não resolve isso (dá timestamps diferentes ao MESMO arquivo, o que
    # não cria um segundo arquivo). Solução: se os diretórios coincidem,
    # cria uma cópia isolada de src/ só para a Engine 2 antes de seguir o
    # fluxo normal de mtimes.
    dir2_for_build = dir2
    tmp_dir2 = None
    if os.path.realpath(dir1) == os.path.realpath(dir2):
        tmp_dir2 = tempfile.mkdtemp(prefix="zquoridor_arena_e2_local_")
        shutil.copytree(os.path.join(dir2, "src"), os.path.join(tmp_dir2, "src"), dirs_exist_ok=True)
        dir2_for_build = tmp_dir2

    try:
        _bump_mtimes(dir1, 0)
        _bump_mtimes(dir2_for_build, 86400)

        search1 = os.path.join(dir1, "src", "search.hpp").replace("\\", "/")
        search2 = os.path.join(dir2_for_build, "src", "search.hpp").replace("\\", "/")
        cmd = (
            f"{COMPILER} {CXX_FLAGS} "
            f"-DENGINE1_SEARCH_HPP=\"\\\"{search1}\\\"\" "
            f"-DENGINE2_SEARCH_HPP=\"\\\"{search2}\\\"\" "
            f"\"{arena_src}\" -o \"{output_exe}\""
        )

        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[!] ERRO DE COMPILAÇÃO (arena dual, ref1={dir1} ref2={dir2}):\n{res.stderr}")
            sys.exit(1)
    finally:
        if tmp_dir2:
            shutil.rmtree(tmp_dir2, ignore_errors=True)

def _is_git_repo():
    res = subprocess.run("git rev-parse --is-inside-work-tree", shell=True, cwd=PROJECT_ROOT,
                          capture_output=True, text=True)
    return res.returncode == 0 and res.stdout.strip() == "true"

def prepare_engine_source(ref_name):
    """
    Se ref_name for None ou "", retorna o PROJECT_ROOT local com alterações atuais.
    Caso contrário, faz git worktree temporário da ref informada.

    CORRECAO: antes, um ref_name nao-vazio (ex. GIT_REF2="main" default)
    tentava sempre `git worktree add`, mesmo quando PROJECT_ROOT nao e um
    repositorio git de verdade (ex. rodando a partir de um .zip extraido
    sem .git) -- o comando falhava e o script abortava com sys.exit(1),
    mesmo que o usuario so quisesse comparar a versao local contra ela
    mesma. Agora detectamos isso antes e caimos para a versao local com um
    aviso, em vez de travar.
    """
    if not ref_name or ref_name.strip() == "":
        print(f"[*] Engine: Versao Local Atual (inclui alteracoes nao comitadas)")
        return PROJECT_ROOT, None

    if not _is_git_repo():
        print(f"[!] Aviso: '{ref_name}' foi pedido, mas {PROJECT_ROOT} nao e um repositorio git\n"
              f"    (rodando a partir de uma copia/zip sem .git?). Usando a versao local em vez\n"
              f"    de tentar 'git worktree add' -- para comparar duas refs de verdade, rode isto\n"
              f"    de dentro de um clone git normal do projeto.")
        return PROJECT_ROOT, None

    temp_dir = tempfile.mkdtemp(prefix=f"zquoridor_ref_{ref_name.replace('/', '_')}_")
    print(f"[*] Engine: Criando git worktree temporario para '{ref_name}' em {temp_dir}")
    cmd = f"git worktree add --detach \"{temp_dir}\" {ref_name}"
    res = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] ERRO ao criar worktree para git ref '{ref_name}':\n{res.stderr}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
    return temp_dir, temp_dir

def cleanup_worktree(temp_dir):
    if temp_dir and os.path.exists(temp_dir):
        print(f"[*] Removendo git worktree temporario {temp_dir}...")
        subprocess.run(f"git worktree remove --force \"{temp_dir}\"", shell=True, cwd=PROJECT_ROOT, capture_output=True)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

_NNUE_FILENAME = "nnue_weights_int8.bin"

def resolve_weights_path(weights_dir, engine_source_dir, engine_label):
    """Resolve o .bin de pesos NNUE de UM engine, respeitando qual ref/
    worktree ele veio -- corrige o bug em que Engine 1 (local) e Engine 2
    (--ref2, ex. 'main') acabavam carregando o MESMO arquivo default
    (data/nnue/nnue_weights_int8.bin resolvido contra o cwd do processo,
    nao contra a arvore de codigo de cada engine -- ver defaultNnueWeightsPath
    em src/nnue.hpp). weights_dir None/""/"default" (case-insensitive):
    tenta primeiro os pesos PROPRIOS daquele ref (engine_source_dir/data/
    nnue/nnue_weights_int8.bin -- ex. o .bin comitado no main); se nao
    existir, cai para o default global do projeto (PROJECT_ROOT/data/nnue/
    nnue_weights_int8.bin); se nenhum dos dois existir, devolve "" (arena.exe
    faz seu proprio fallback pra heuristica, com aviso). Qualquer outro
    valor de weights_dir e tratado como pasta explicita (ou arquivo .bin
    direto, se ja terminar em .bin)."""
    if weights_dir is None or weights_dir.strip() == "" or weights_dir.strip().lower() == "default":
        own_path = os.path.join(engine_source_dir, "data", "nnue", _NNUE_FILENAME)
        if os.path.isfile(own_path):
            print(f"[*] {engine_label}: pesos NNUE do proprio ref -> {own_path}")
            return own_path
        fallback_path = os.path.join(PROJECT_ROOT, "data", "nnue", _NNUE_FILENAME)
        if os.path.isfile(fallback_path):
            print(f"[*] {engine_label}: sem pesos proprios no ref, usando default global -> {fallback_path}")
            return fallback_path
        print(f"[!] {engine_label}: nenhum .bin encontrado (nem no ref, nem no default global) -- "
              f"cai para heuristica dentro do arena.exe")
        return ""
    explicit = weights_dir.strip()
    resolved = explicit if explicit.endswith(".bin") else os.path.join(explicit, _NNUE_FILENAME)
    if not os.path.isfile(resolved):
        print(f"[!] {engine_label}: pasta/arquivo de pesos informado nao existe -> {resolved}")
    else:
        print(f"[*] {engine_label}: pesos NNUE de pasta explicita -> {resolved}")
    return resolved

import multiprocessing

def worker_process(exe_path, worker_games, time_ms, random_plies, seed, report_games, bin_path, invert_colors, progress_queue, e1_nnue="", e2_nnue="", heuristic=False, e1_heuristic=False, e2_heuristic=False, policy_order=False, e1_policy_order=False, e2_policy_order=False, e1_policy_min_depth=3, e2_policy_min_depth=3, e1_mcab=None, e2_mcab=None, e1_engine_args=None, e2_engine_args=None):
    cmd = [
        exe_path,
        "--games", str(worker_games),
        "--time", str(time_ms),
        "--random-plies", str(random_plies),
        "--seed", str(seed)
    ]
    if not invert_colors:
        cmd.append("--no-invert")
    if report_games > 0:
        cmd.extend(["--report-games", str(report_games)])
    if bin_path:
        cmd.extend(["--bin-file", bin_path])
    if e1_nnue:
        cmd.extend(["--e1-nnue", e1_nnue])
    if e2_nnue:
        cmd.extend(["--e2-nnue", e2_nnue])
    if heuristic:
        cmd.append("--heuristic")
    if e1_heuristic:
        cmd.append("--e1-heuristic")
    if e2_heuristic:
        cmd.append("--e2-heuristic")
    if policy_order:
        cmd.append("--policy-order")
    # 2026-08: arena.exe agora nasce com policy ordering LIGADA por default
    # em cada engine (E1/E2_POLICY_ORDERING_DEFAULT=true) -- precisamos
    # mandar o "--eX-no-policy-order" explicito quando eX_policy_order for
    # False, ou o default do binario (ligado) prevalece silenciosamente.
    if e1_policy_order:
        cmd.extend(["--e1-policy-order", "--e1-policy-order-min-depth", str(e1_policy_min_depth)])
    else:
        cmd.append("--e1-no-policy-order")
    if e2_policy_order:
        cmd.extend(["--e2-policy-order", "--e2-policy-order-min-depth", str(e2_policy_min_depth)])
    else:
        cmd.append("--e2-no-policy-order")

    # MCTS hibrido com alpha-beta. Agora e a busca de PRODUCAO -- vem LIGADO
    # por default no arena.exe -- entao isto e tri-estado:
    #   None  = nao manda flag, o binario decide (o normal)
    #   True  = --eX-mcab      (forca ligado)
    #   False = --eX-no-mcab   (forca alpha-beta puro)
    # IMPORTANTE: um --refX anterior a esta feature nao tem searchLeaf; o
    # arena compila mesmo assim (dispatch SFINAE em src/mcab.hpp) e
    # imprime um aviso 1x, rodando AB puro daquele lado.
    if e1_mcab is True:
        cmd.append("--e1-mcab")
    elif e1_mcab is False:
        cmd.append("--e1-no-mcab")
    if e2_mcab is True:
        cmd.append("--e2-mcab")
    elif e2_mcab is False:
        cmd.append("--e2-no-mcab")
    # Demais knobs de engine (MCab + parametros de busca), ja resolvidos em
    # main() na forma "--e1-<flag> [valor]" -- lista vazia = tudo em producao.
    # Sao por engine porque o ponto e comparar duas configuracoes com UM
    # parametro diferente, com muito menos variancia do que medir cada uma
    # contra uma baseline separada.
    cmd.extend(e1_engine_args or [])
    cmd.extend(e2_engine_args or [])

    # cwd=PROJECT_ROOT: arena.exe agora tenta carregar o caminho default de
    # pesos NNUE (data/nnue/nnue_weights_int8.bin, RELATIVO) quando
    # --e1-nnue/--e2-nnue nao sao passados -- sem fixar o cwd aqui, esse
    # caminho relativo dependeria de onde run_arena.py foi chamado (ex.
    # rodar de dentro de teste/ faria o default "nao existir" e o arena
    # cair silenciosamente para heuristico, mesmo com os pesos presentes).
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=PROJECT_ROOT)
    
    for line in iter(proc.stdout.readline, ''):
        line = line.strip()
        if line.startswith("PROGRESS_JSON:"):
            try:
                data = json.loads(line.replace("PROGRESS_JSON:", "").strip())
                progress_queue.put(("PROGRESS", data))
            except Exception:
                pass
        elif line.startswith("RESULT_JSON:"):
            try:
                data = json.loads(line.replace("RESULT_JSON:", "").strip())
                progress_queue.put(("RESULT", data))
            except Exception:
                pass
        elif line.startswith("[arena]") or line.startswith("[nnue]"):
            # Confirmação de carregamento/fallback de NNUE (ou heurística
            # forçada) do binário -- com NNUE virando o default, é
            # importante o usuário conseguir ver se cada engine realmente
            # carregou os pesos ou caiu para heurístico, em vez desses
            # avisos ficarem mudos dentro do subprocess.
            # [nnue] (nnue.hpp) inclui: alem do "[arena]" (decisao de alto
            # nivel, NNUE vs heuristica), a razao especifica de uma falha
            # de load (tamanho de arquivo errado p/ NUM_FEATURES atual,
            # arquivo ausente, etc.) -- antes essas linhas nao batiam em
            # nenhum elif e eram lidas do pipe e descartadas sem aviso,
            # escondendo a causa raiz de qualquer fallback pra heuristica.
            progress_queue.put(("LOG", line))
            
    proc.wait()
    if proc.returncode != 0:
        progress_queue.put(("ERROR", "Worker finalizou com erro"))

def main():
    parser = argparse.ArgumentParser(description="Zquoridor Arena Selfplay: Confronto entre 2 Referencias Git ou Local")
    parser.add_argument("--ref1", default=GIT_REF1, help="Referencia Git do Engine 1 (deixe vazio/None para versao local)")
    parser.add_argument("--ref2", default=GIT_REF2, help="Referencia Git do Engine 2 (deixe vazio/None para versao local)")
    parser.add_argument("--games", type=int, default=GAMES, help=f"Total de jogos (padrao: {GAMES})")
    parser.add_argument("--report-games", type=int, default=REPORT_GAMES, help=f"Frequencia do relatorio de progresso (padrao: {REPORT_GAMES})")
    parser.add_argument("--create-bin", action="store_true", default=CREATE_BIN, help="Salva dataset .bin das partidas em data/arena/")
    parser.add_argument("--time", type=int, default=TIME_MS, help=f"Tempo por lance em ms (padrao: {TIME_MS})")
    parser.add_argument("--threads", type=int, default=THREADS, help=f"Numero de nucleos/threads (padrao: {THREADS})")
    parser.add_argument("--random-plies", type=int, default=RANDOM_OPENING_PLIES, help=f"Lances aleatorios na abertura (padrao: {RANDOM_OPENING_PLIES})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Semente RNG (padrao: {SEED})")
    # BUG CORRIGIDO (2026-08-13): as duas flags declaravam `default=` para o
    # MESMO dest. O argparse aplica o default da PRIMEIRA action registrada e
    # ignora o das seguintes, entao o default efetivo era o False do
    # --no-invert -- a constante INVERT_COLORS do topo do arquivo nunca valia,
    # e toda arena rodava sem inversao de cores, mesmo com INVERT_COLORS=True.
    # O default agora vem de set_defaults, uma vez so, a partir da constante.
    parser.add_argument("--no-invert", dest="invert_colors", action="store_false", help="Nao inverte as cores na mesma abertura")
    parser.add_argument("--invert-colors", dest="invert_colors", action="store_true", help="Inverte as cores jogando a mesma abertura 2x")
    parser.set_defaults(invert_colors=INVERT_COLORS)
    parser.add_argument("--e1-nnue", default="", help="Caminho para pesos NNUE quantizados (.qbin) do Engine 1 (default: data/nnue/nnue_weights_int8.bin, se existir)")
    parser.add_argument("--e2-nnue", default="", help="Caminho para pesos NNUE quantizados (.qbin) do Engine 2 (default: data/nnue/nnue_weights_int8.bin, se existir)")
    parser.add_argument("--e1-weights-dir", default=E1_WEIGHTS_DIR,
                         help=f"Pasta de pesos NNUE do Engine 1. 'default'/None: usa os pesos do PROPRIO --ref1 "
                              f"(worktree/data/nnue/nnue_weights_int8.bin), com fallback pro default global do "
                              f"projeto se o ref nao tiver pesos proprios. Ignorado se --e1-nnue for passado "
                              f"(padrao: {E1_WEIGHTS_DIR!r})")
    parser.add_argument("--e2-weights-dir", default=E2_WEIGHTS_DIR,
                         help=f"Pasta de pesos NNUE do Engine 2 (mesma logica de --e1-weights-dir, aplicada a "
                              f"--ref2 -- ex. pega o .bin comitado no 'main' em vez do .bin local). Ignorado se "
                              f"--e2-nnue for passado (padrao: {E2_WEIGHTS_DIR!r})")
    parser.add_argument("--heuristic", action="store_true", default=False, help="Forca avaliacao heuristica (evalSimple) nas duas engines, ignorando NNUE -- debug/historico/fallback")
    parser.add_argument("--e1-heuristic", dest="e1_heuristic", action="store_true", default=E1_HEURISTIC, help=f"Forca heuristica so no Engine 1 (padrao: {E1_HEURISTIC})")
    parser.add_argument("--e1-nnue-default", dest="e1_heuristic", action="store_false", help="Sobrepoe E1_HEURISTIC=True do arquivo, forcando NNUE default no Engine 1 sem editar/recompilar")
    parser.add_argument("--e2-heuristic", dest="e2_heuristic", action="store_true", default=E2_HEURISTIC, help=f"Forca heuristica so no Engine 2 (padrao: {E2_HEURISTIC})")
    parser.add_argument("--e2-nnue-default", dest="e2_heuristic", action="store_false", help="Sobrepoe E2_HEURISTIC=True do arquivo, forcando NNUE default no Engine 2 sem editar/recompilar")
    parser.add_argument("--policy-order", action="store_true", default=False, help="Liga a ordenacao por politica NNUE nas duas engines (sem efeito se a engine estiver em heuristica)")
    parser.add_argument("--e1-policy-order", dest="e1_policy_order", action="store_true", default=E1_POLICY_ORDER, help=f"Liga a ordenacao por politica NNUE so no Engine 1 (padrao: {E1_POLICY_ORDER})")
    parser.add_argument("--e1-no-policy-order", dest="e1_policy_order", action="store_false", help="Sobrepoe E1_POLICY_ORDER=True do arquivo, desligando no Engine 1 sem editar/recompilar")
    parser.add_argument("--e2-policy-order", dest="e2_policy_order", action="store_true", default=E2_POLICY_ORDER, help=f"Liga a ordenacao por politica NNUE so no Engine 2 (padrao: {E2_POLICY_ORDER})")
    parser.add_argument("--e2-no-policy-order", dest="e2_policy_order", action="store_false", help="Sobrepoe E2_POLICY_ORDER=True do arquivo, desligando no Engine 2 sem editar/recompilar")
    parser.add_argument("--policy-order-min-depth", type=int, default=None, help="Piso de profundidade da ordenacao por politica nas duas engines (padrao: 3 -- ver E1_POLICY_ORDER_MIN_DEPTH/E2_POLICY_ORDER_MIN_DEPTH)")
    parser.add_argument("--e1-policy-order-min-depth", type=int, default=E1_POLICY_ORDER_MIN_DEPTH, help=f"Piso de profundidade so no Engine 1 (padrao: {E1_POLICY_ORDER_MIN_DEPTH})")
    parser.add_argument("--e2-policy-order-min-depth", type=int, default=E2_POLICY_ORDER_MIN_DEPTH, help=f"Piso de profundidade so no Engine 2 (padrao: {E2_POLICY_ORDER_MIN_DEPTH})")
    # Tri-estado: o MCTS hibrido e a busca de PRODUCAO e vem ligado por
    # default no arena.exe. Sem nenhuma destas flags, nada e mandado e o
    # binario decide. --no-mcab e como se mede alpha-beta puro.
    parser.add_argument("--mcab", dest="mcab", action="store_true", default=None, help="Forca o MCTS hibrido LIGADO nas DUAS engines (ja e o default do binario)")
    parser.add_argument("--no-mcab", dest="mcab", action="store_false", help="Forca alpha-beta PURO nas DUAS engines")
    parser.add_argument("--e1-mcab", dest="e1_mcab", action="store_true", default=None, help="Forca o hibrido ligado so no Engine 1 (--ref1). Se o ref for anterior a feature, o arena avisa e roda AB puro nesse lado")
    parser.add_argument("--e1-no-mcab", dest="e1_mcab", action="store_false", help="Forca alpha-beta puro so no Engine 1")
    parser.add_argument("--e2-mcab", dest="e2_mcab", action="store_true", default=None, help="Forca o hibrido ligado so no Engine 2 (--ref2)")
    parser.add_argument("--e2-no-mcab", dest="e2_mcab", action="store_false", help="Forca alpha-beta puro so no Engine 2")
    # Knobs de engine (MCab + parametros de busca). Cada um tem a forma global
    # (as duas engines) e a forma --e1-/--e2- (so um lado); a per-engine vence
    # sobre a global, e as duas vencem sobre a constante do topo do arquivo.
    for lado, rotulo in (("", "nas DUAS engines"), ("e1-", "so no Engine 1"), ("e2-", "so no Engine 2")):
        for flag, dest, _c1, _c2, tipo in MCAB_VALUE_KNOBS:
            kwargs = {"default": None, "dest": lado.replace("-", "_") + dest,
                       "help": f"{flag} {rotulo} (vazio = valor de producao; ver a constante correspondente no topo do arquivo)"}
            if flag == "mcab-root-select":
                kwargs["choices"] = ["visits", "q", "visits-then-q"]
            elif flag == "mcab-backup":
                kwargs["choices"] = ["minimax", "avg"]
            else:
                kwargs["type"] = tipo
            parser.add_argument(f"--{lado}{flag}", **kwargs)
        for dest, _c1, _c2, flag_on, flag_off in MCAB_FLAG_KNOBS:
            d = lado.replace("-", "_") + dest
            if flag_on:
                parser.add_argument(f"--{lado}{flag_on}", dest=d, action="store_true", default=None,
                                    help=f"liga {dest} {rotulo}")
            if flag_off:
                parser.add_argument(f"--{lado}{flag_off}", dest=d, action="store_false", default=None,
                                    help=f"desliga {dest} {rotulo}")

    args = parser.parse_args()
    # --policy-order liga as duas engines, igual --heuristic; --e1-.../--e2-...
    # continuam disponiveis pra ligar so uma de cada vez (mesmo padrao que
    # --heuristic vs --e1-heuristic/--e2-heuristic acima).
    if args.policy_order:
        args.e1_policy_order = True
        args.e2_policy_order = True
    # --mcab/--no-mcab valem para as duas engines; o --eX- correspondente,
    # quando passado, vence para aquele lado.
    if args.mcab is not None:
        if args.e1_mcab is None:
            args.e1_mcab = args.mcab
        if args.e2_mcab is None:
            args.e2_mcab = args.mcab
    # Nada na CLI -> cai no bloco de config do topo do arquivo (que, se
    # tambem estiver vazio/None, deixa o binario decidir).
    if args.e1_mcab is None:
        args.e1_mcab = E1_MCAB
    if args.e2_mcab is None:
        args.e2_mcab = E2_MCAB
    # Monta a lista de flags extras de cada engine. Precedencia por lado:
    #   --eX-<knob>  >  --<knob> (global)  >  constante EX_ do topo  >  producao
    # Nada resolvido em nenhum nivel => nenhuma flag e mandada e o arena.exe
    # fica no valor de producao daquele knob.
    args.e1_engine_args, args.e2_engine_args = [], []
    args.engine_tuning_resumo = {"e1": [], "e2": []}
    for flag, dest, c1, c2, _tipo in MCAB_VALUE_KNOBS:
        globalv = getattr(args, dest)
        for lado, constante, destino, resumo in (("e1_", c1, args.e1_engine_args, args.engine_tuning_resumo["e1"]),
                                                  ("e2_", c2, args.e2_engine_args, args.engine_tuning_resumo["e2"])):
            v = getattr(args, lado + dest)
            if v is None:
                v = globalv
            if v is None:
                v = constante
            if v is not None:
                destino += [f"--{lado[:2]}-{flag}", str(v)]
                resumo.append(f"{flag}={v}")
    for dest, c1, c2, flag_on, flag_off in MCAB_FLAG_KNOBS:
        globalv = getattr(args, dest)
        for lado, constante, destino, resumo in (("e1_", c1, args.e1_engine_args, args.engine_tuning_resumo["e1"]),
                                                  ("e2_", c2, args.e2_engine_args, args.engine_tuning_resumo["e2"])):
            v = getattr(args, lado + dest)
            if v is None:
                v = globalv
            if v is None:
                v = constante
            # Só existe flag para o lado que difere do default do binário
            # (ex.: tree-reuse só tem "--...-no-tree-reuse"); pedir o outro
            # lado é justamente o default, então não há nada a mandar.
            if v is True and flag_on:
                destino.append(f"--{lado[:2]}-{flag_on}")
                resumo.append(f"{dest}=on")
            elif v is False and flag_off:
                destino.append(f"--{lado[:2]}-{flag_off}")
                resumo.append(f"{dest}=off")
    # --policy-order-min-depth (sem prefixo e1/e2) sobrepoe as duas de uma vez.
    if args.policy_order_min_depth is not None:
        args.e1_policy_order_min_depth = args.policy_order_min_depth
        args.e2_policy_order_min_depth = args.policy_order_min_depth
    
    ref1_name = args.ref1 if args.ref1 else "LOCAL"
    ref2_name = args.ref2 if args.ref2 else "LOCAL"
    
    ref1_label = args.ref1 if args.ref1 else "LOCAL (Nao Comitado)"
    ref2_label = args.ref2 if args.ref2 else "LOCAL (Nao Comitado)"
    
    print("=" * 65)
    print("        ZQUORIDOR ARENA SELFPLAY - BENCHMARK MULTI-CORE")
    print("=" * 65)
    print(f"  Engine 1 : {ref1_label}")
    print(f"  Engine 2 : {ref2_label}")
    print(f"  Config   : {args.games} jogos | {args.threads} threads | {args.time}ms/lance | Relatorio a cada {args.report_games} jogos")
    print(f"  Salvar .bin: {args.create_bin}")
    # So aparece o que foi tirado do valor de producao; silencio = as duas
    # engines em producao, que e o caso normal.
    for lado, rotulo in (("e1", "Engine 1"), ("e2", "Engine 2")):
        resumo = args.engine_tuning_resumo[lado]
        if resumo:
            print(f"  {rotulo} (fora do default): {', '.join(resumo)}")
    print(f"  Politica NNUE na ordenacao: Engine 1 {'LIGADA (min-depth=' + str(args.e1_policy_order_min_depth) + ')' if args.e1_policy_order else 'desligada'} | Engine 2 {'LIGADA (min-depth=' + str(args.e2_policy_order_min_depth) + ')' if args.e2_policy_order else 'desligada'}")
    print("=" * 65 + "\n")
    
    dir1, cleanup1 = prepare_engine_source(args.ref1)
    dir2, cleanup2 = prepare_engine_source(args.ref2)

    # Resolve os pesos NNUE de cada engine ANTES do build (so pra log; o
    # caminho e passado em runtime via --e1-nnue/--e2-nnue mesmo, nao afeta
    # a compilacao). --e1-nnue/--e2-nnue explicitos sempre vencem --e1/e2-
    # weights-dir -- mesma precedencia que ja existia entre eles e o
    # default do arena.exe.
    if not args.e1_nnue:
        args.e1_nnue = resolve_weights_path(args.e1_weights_dir, dir1, "Engine 1")
    if not args.e2_nnue:
        args.e2_nnue = resolve_weights_path(args.e2_weights_dir, dir2, "Engine 2")

    # BUG CORRIGIDO (2026-08): quando o path resolvido acima vem de um
    # worktree git temporario (ex. --ref2 "main"), ele apontava direto pra
    # dentro de <temp>/data/nnue/nnue_weights_int8.bin. Esse arquivo so e
    # lido em RUNTIME pelo arena.exe (fopen), nao e embutido na compilacao
    # -- mas o worktree era removido (cleanup_worktree, no finally logo
    # abaixo) IMEDIATAMENTE apos compile_arena(), ou seja, MUITO antes dos
    # worker_process/arena.exe sequer comecarem a rodar as partidas. O
    # resultado: --e2-nnue apontava pra um arquivo que ja nao existia mais
    # no disco, fopen() falhava, e o engine do ref2 caia silenciosamente
    # pra heuristica (o engine local/--ref1=None nunca sofria disso, pois
    # seu path aponta pro PROJECT_ROOT permanente, nao um worktree
    # descartavel). Fix: copiar os .bin resolvidos para um diretorio
    # temporario PROPRIO cuja vida dura ate o fim do arena inteiro (limpo
    # so no final de main(), depois de todas as partidas), e reapontar
    # args.e1_nnue/args.e2_nnue pra essas copias antes de derrubar os
    # worktrees.
    weights_cache_dir = tempfile.mkdtemp(prefix="zquoridor_arena_weights_")

    def _persist_weights(path, engine_label, filename):
        if not path or not os.path.isfile(path):
            return path
        dest = os.path.join(weights_cache_dir, filename)
        shutil.copy2(path, dest)
        print(f"[*] {engine_label}: pesos copiados para local persistente (sobrevive ao cleanup do worktree) -> {dest}")
        return dest

    args.e1_nnue = _persist_weights(args.e1_nnue, "Engine 1", "e1_nnue_weights_int8.bin")
    args.e2_nnue = _persist_weights(args.e2_nnue, "Engine 2", "e2_nnue_weights_int8.bin")

    cand_exe = os.path.join(BIN_DIR, "arena.exe")
    
    try:
        print(f"[*] Compilando executavel de arena (engine1={ref1_label} + engine2={ref2_label} no mesmo binario)...")
        compile_arena(dir1, dir2, cand_exe)
        print(f"[+] Compilacao concluida com sucesso! (as duas engines sao codigo REAL de cada ref, nao a mesma engine duplicada)")
    finally:
        cleanup_worktree(cleanup1)
        cleanup_worktree(cleanup2)
        
    threads_count = max(1, args.threads)
    total_pairs = (args.games + 1) // 2
    pairs_per_thread = total_pairs // threads_count
    extra_pairs = total_pairs % threads_count
    
    # Divide report_games entre threads para emissão frequente
    worker_report = max(2, (args.report_games // threads_count) // 2 * 2) if args.report_games > 0 else 0
    
    temp_bin_dir = tempfile.mkdtemp(prefix="zquoridor_arena_bins_") if args.create_bin else None
    
    tasks = []
    for w in range(threads_count):
        worker_pairs = pairs_per_thread + (1 if w < extra_pairs else 0)
        if worker_pairs <= 0:
            continue
        # SEMPRE par, mesmo sem inversao de cores: o arena.exe arredonda
        # `--games` impar para cima (totalGames++ em main()), entao um worker
        # com numero impar jogava uma partida a mais do que o Python contava --
        # o denominador do relatorio ficava menor que os jogos realmente
        # disputados (ex.: 100 pedidos, 112 jogados) e o Elo saia de uma
        # amostra diferente da anunciada.
        worker_games = worker_pairs * 2
        if worker_games <= 0: continue
        worker_seed = args.seed + w * 10007
        worker_bin = os.path.join(temp_bin_dir, f"part_{w}.bin") if temp_bin_dir else None
        tasks.append((worker_games, worker_seed, worker_bin))

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    processes = []
    for worker_games, worker_seed, worker_bin in tasks:
        p = multiprocessing.Process(
            target=worker_process,
            args=(cand_exe, worker_games, args.time, args.random_plies, worker_seed, worker_report, worker_bin, args.invert_colors, queue, args.e1_nnue, args.e2_nnue, args.heuristic, args.e1_heuristic, args.e2_heuristic, False, args.e1_policy_order, args.e2_policy_order, args.e1_policy_order_min_depth, args.e2_policy_order_min_depth, args.e1_mcab, args.e2_mcab, args.e1_engine_args, args.e2_engine_args)
        )
        p.start()
        processes.append(p)
    
    tot_eng1_wins = 0
    tot_eng2_wins = 0
    tot_draws = 0
    tot_eng1_nodes = 0
    tot_eng2_nodes = 0
    tot_eng1_time = 0.0
    tot_eng2_time = 0.0
    
    # Totais definitivos, montados so a partir dos RESULT_JSON (um por worker,
    # ja acumulado). Os tot_* acima seguem servindo a impressao de progresso.
    final_eng1_wins = 0
    final_eng2_wins = 0
    final_draws = 0
    final_eng1_nodes = 0
    final_eng2_nodes = 0
    final_eng1_time = 0.0
    final_eng2_time = 0.0
    got_final = False

    completed_games = 0
    next_report_target = args.report_games
    total_games = sum(t[0] for t in tasks)
    
    finished_workers = 0
    seen_log_lines = set()
    while finished_workers < len(tasks):
        msg_type, data = queue.get()
        if msg_type == "LOG":
            # Todo worker imprime a mesma confirmação de carregamento NNUE
            # (mesmos pesos, mesmo caminho default) -- deduplica para não
            # repetir a mesma linha uma vez por worker/thread.
            if data not in seen_log_lines:
                seen_log_lines.add(data)
                print(f"  {data}", flush=True)
        elif msg_type == "PROGRESS":
            # PROGRESS e so a visao AO VIVO: cada worker manda um delta a cada
            # `worker_report` partidas e as partidas restantes (worker_games %
            # worker_report) nunca viram um PROGRESS -- elas so aparecem no
            # RESULT_JSON final daquele worker. Os totais definitivos vem dos
            # RESULTs, abaixo; somar as duas fontes contaria os mesmos jogos
            # duas vezes.
            tot_eng1_wins += data["candWins"]
            tot_eng2_wins += data["baseWins"]
            tot_draws += data["draws"]
            tot_eng1_nodes += data["candNodes"]
            tot_eng2_nodes += data["baseNodes"]
            tot_eng1_time += data["candTimeSec"]
            tot_eng2_time += data["baseTimeSec"]

            completed_games += data["games"]
            if completed_games >= next_report_target or completed_games == total_games:
                elo_diff, elo_margin = calculate_elo_and_ci(tot_eng1_wins, tot_eng2_wins, tot_draws)
                sign = "+" if elo_diff >= 0 else ""
                eng1_nps = tot_eng1_nodes / tot_eng1_time if tot_eng1_time > 0 else 0.0
                eng2_nps = tot_eng2_nodes / tot_eng2_time if tot_eng2_time > 0 else 0.0
                print(f"  [Progresso: {completed_games:4d}/{total_games:4d} jogos] Eng1: {tot_eng1_wins:3d} ({eng1_nps:,.0f} nps) | Eng2: {tot_eng2_wins:3d} ({eng2_nps:,.0f} nps) | Empates: {tot_draws:2d} | Elo: {sign}{elo_diff:.1f} (±{elo_margin:.1f})", flush=True)
                while next_report_target <= completed_games and next_report_target < total_games:
                    next_report_target += args.report_games
        elif msg_type == "RESULT":
            finished_workers += 1
            # RESULT traz os totais ACUMULADOS do worker (nao um delta), entao
            # e a fonte definitiva: acumula aqui e, no fim, substitui o que os
            # PROGRESS montaram. Antes isto so era lido quando
            # --report-games 0, e com report ligado as partidas da sobra de
            # cada worker simplesmente sumiam do placar.
            if data:
                final_eng1_wins += data["candWins"]
                final_eng2_wins += data["baseWins"]
                final_draws += data["draws"]
                final_eng1_nodes += data["candNodes"]
                final_eng2_nodes += data["baseNodes"]
                final_eng1_time += data["candTimeSec"]
                final_eng2_time += data["baseTimeSec"]
                got_final = True
        elif msg_type == "ERROR":
            print(f"[!] {data}", flush=True)
            finished_workers += 1
            
    for p in processes:
        p.join()

    # Placar definitivo: os RESULTs de todos os workers. So cai de volta nos
    # contadores de progresso se nenhum RESULT tiver chegado (worker morto).
    if got_final:
        tot_eng1_wins, tot_eng2_wins, tot_draws = final_eng1_wins, final_eng2_wins, final_draws
        tot_eng1_nodes, tot_eng2_nodes = final_eng1_nodes, final_eng2_nodes
        tot_eng1_time, tot_eng2_time = final_eng1_time, final_eng2_time

    shutil.rmtree(weights_cache_dir, ignore_errors=True)

    # Salva dataset final concatenado em data/arena/ com numeração sequencial 000, 001...
    if args.create_bin and temp_bin_dir:
        os.makedirs(ARENA_DATA_DIR, exist_ok=True)
        r1_clean = ref1_name.replace("/", "_").replace("\\", "_")
        r2_clean = ref2_name.replace("/", "_").replace("\\", "_")
        
        idx = 0
        while True:
            candidate_name = f"arena_{r1_clean}_{r2_clean}_{total_games}_{idx:03d}.bin"
            candidate_path = os.path.join(ARENA_DATA_DIR, candidate_name)
            if not os.path.exists(candidate_path):
                final_bin_path = candidate_path
                break
            idx += 1
        
        total_bytes = 0
        with open(final_bin_path, "wb") as outfile:
            for _, _, b_file in tasks:
                if b_file and os.path.exists(b_file):
                    with open(b_file, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)
                    total_bytes = outfile.tell()
                    
        shutil.rmtree(temp_bin_dir, ignore_errors=True)
        total_samples = total_bytes // 32  # sizeof(TrainingSample) == 32
        print(f"\n[+] DATASET SALVO COM SUCESSO:")
        print(f"    Arquivo  : {final_bin_path}")
        print(f"    Amostras : {total_samples:,} posicoes ({total_bytes / (1024*1024):.2f} MB)")
    
    # Relatório estatístico
    if total_games > 0:
        elo_diff, elo_margin = calculate_elo_and_ci(tot_eng1_wins, tot_eng2_wins, tot_draws)
        eng1_nps = tot_eng1_nodes / tot_eng1_time if tot_eng1_time > 0 else 0.0
        eng2_nps = tot_eng2_nodes / tot_eng2_time if tot_eng2_time > 0 else 0.0
        
        print("\n" + "=" * 65)
        print("                  RELATORIO FINAL DE ELO & NODES/S")
        print("=" * 65)
        print(f"  Engine 1 ({ref1_label:^17}): {tot_eng1_wins:3d} vitorias ({100.0 * tot_eng1_wins / total_games:5.1f}%) | NPS: {eng1_nps:10,.0f} nps")
        print(f"  Engine 2 ({ref2_label:^17}): {tot_eng2_wins:3d} vitorias ({100.0 * tot_eng2_wins / total_games:5.1f}%) | NPS: {eng2_nps:10,.0f} nps")
        print(f"  Empates                     : {tot_draws:3d}          ({100.0 * tot_draws / total_games:5.1f}%)")
        print("-" * 65)
        sign = "+" if elo_diff >= 0 else ""
        print(f"  Diferenca Elo Engine 1 vs 2 : {sign}{elo_diff:.1f} (Margem ±{elo_margin:.1f})")
        if elo_diff > 0 and (elo_diff - elo_margin) > 0:
            print("  Status                      : [V] ENGINE 1 EH ESTATISTICAMENTE SUPERIOR!")
        elif elo_diff < 0 and (elo_diff + elo_margin) < 0:
            print("  Status                      : [X] ENGINE 2 EH ESTATISTICAMENTE SUPERIOR!")
        else:
            print("  Status                      : [~] DIFERENCA DENTRO DA MARGEM DE ERRO (Inconclusivo)")
        print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
