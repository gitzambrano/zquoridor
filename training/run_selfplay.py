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
TOTAL_GAMES   = 300000   # partidas totais a gerar nesta rodada
CHUNK_GAMES   = 2000    # partidas por arquivo .bin
                        # Cada chunk fica ~6.5 MB -- confortável pra 32 GB RAM

# --- Busca ---
MAX_DEPTH     = 50      # profundidade máxima do negamax (iterative deepening)
TIME_MS       = 50     # orçamento de tempo por lance em ms
                        # 200 ms = boa qualidade; reduza para 50-100 ms se quiser
                        # gerar muito volume rapidamente (em detrimento da força)
 
# --- Abertura aleatória ---
# Fase 1: lances iniciais (óbvios no Quoridor) com muito pouco ruído
OPENING_PLIES1   = 4       # lances 1 a N1 sujeitos a EPSILON_OPENING1
EPSILON_OPENING1 = 0.3     # baixo: não distorce os lances óbvios da abertura

# Fase 2: janela de exploração pesada para diversificar posições iniciais
OPENING_PLIES2   = 10     # lances N1+1 a N2 sujeitos a EPSILON_OPENING2
EPSILON_OPENING2 = 0.5    # alto: cria muita variedade de abertura (lance totalmente aleatório)
 
EPSILON_MIDGAME  = 0.01   # prob. de desvio no midgame: escolhe 2º ou 3º melhor lance (não totalmente aleatório)
 
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
THREADS       = 10      # 0 = auto (usa hardware_concurrency); ajuste se quiser
                        # reservar threads para outras tarefas

# --- Semente ---
SEED          = 983    # semente base do RNG; chunks subsequentes variam automaticamente

# --- Saída ---
# Use {shard:03d} para nomear os chunks automaticamente.
# Os arquivos ficam em data/selfplay/ relativo à raiz do projeto.
OUT_TEMPLATE  = "data/selfplay/selfplay_{shard:03d}.bin"

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
    teste/run_arena.py."""
    p = argparse.ArgumentParser(description="Orquestrador de self-play (config no topo do arquivo ou via flags)")
    p.add_argument("--games", type=int, default=TOTAL_GAMES, help=f"partidas totais (padrao: {TOTAL_GAMES})")
    p.add_argument("--chunk-games", type=int, default=CHUNK_GAMES, help=f"partidas por arquivo .bin (padrao: {CHUNK_GAMES})")
    p.add_argument("--depth", type=int, default=MAX_DEPTH, help=f"profundidade maxima (padrao: {MAX_DEPTH})")
    p.add_argument("--time-ms", type=int, default=TIME_MS, help=f"ms por lance (padrao: {TIME_MS})")
    p.add_argument("--opening-plies", type=int, default=OPENING_PLIES1)
    p.add_argument("--epsilon", type=float, default=EPSILON_OPENING1)
    p.add_argument("--opening-plies2", type=int, default=OPENING_PLIES2)
    p.add_argument("--epsilon-opening2", type=float, default=EPSILON_OPENING2)
    p.add_argument("--epsilon-midgame", type=float, default=EPSILON_MIDGAME)
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

    print("=" * 60)
    print(f"[run_selfplay] Iniciando geração de dados")
    print(f"  Executável  : {exe}")
    print(f"  Partidas    : {args.games} total / {args.chunk_games} por chunk")
    print(f"  Busca       : depth<={args.depth}, {args.time_ms} ms/lance")
    print(f"  Abertura    : fase1=[lances 1..{args.opening_plies}] eps={args.epsilon} (lance aleatório)")
    print(f"                fase2=[lances {args.opening_plies+1}..{args.opening_plies2}] eps={args.epsilon_opening2} (lance aleatório)")
    print(f"                midgame eps={args.epsilon_midgame} (2º/3º melhor lance)")
    print(f"  Avaliação   : {'heurística (evalSimple) -- forçada' if args.force_heuristic else 'NNUE (default do binário), com fallback automático para heurística se os pesos não existirem'}")
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
