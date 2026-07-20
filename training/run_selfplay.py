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
  - TrainingSample = 27 bytes; partida media ~120 posicoes -> ~3.2 KB/partida
  - CHUNK_GAMES = 2000 -> arquivo de ~6.5 MB (carrega inteiro em RAM sem stress)
  - TOTAL_GAMES = 20000 -> 10 chunks, ~65 MB total -> dataset sólido p/ inicio
  - Para treino com train_nnue.py: o default --ram-budget-gb 32 e
    --vram-budget-gb 6 já usam os 200k-posições de chunk size automaticamente.
"""

import os
import subprocess
import sys
import time

# =============================================================================
# CONFIG -- edite estas variaveis conforme necessario
# =============================================================================

# --- Geração de partidas ---
TOTAL_GAMES   = 200000   # partidas totais a gerar nesta rodada
CHUNK_GAMES   = 2000    # partidas por arquivo .bin
                        # Cada chunk fica ~6.5 MB -- confortável pra 32 GB RAM

# --- Busca ---
MAX_DEPTH     = 40      # profundidade máxima do negamax (iterative deepening)
TIME_MS       = 200     # orçamento de tempo por lance em ms
                        # 200 ms = boa qualidade; reduza para 50-100 ms se quiser
                        # gerar muito volume rapidamente (em detrimento da força)

# --- Abertura aleatória ---
OPENING_PLIES   = 8       # primeiros N lances sujeitos a epsilon-greedy
EPSILON         = 0.8    # probabilidade de lance aleatório na janela de abertura
EPSILON_MIDGAME = 0.03   # probabilidade de lance aleatório após a janela de abertura (0.02 = 2%)

# --- Segurança ---
MAX_PLIES     = 300     # corte: partidas que não terminam são descartadas

# --- Paralelismo ---
THREADS       = 14      # 0 = auto (usa hardware_concurrency); ajuste se quiser
                        # reservar threads para outras tarefas

# --- Semente ---
SEED          = 43      # semente base do RNG; chunks subsequentes variam automaticamente

# --- Saída ---
# Use {shard:03d} para nomear os chunks automaticamente.
# Os arquivos ficam em data/selfplay/ relativo à raiz do projeto.
OUT_TEMPLATE  = "data/selfplay/selfplay_{shard:03d}.bin"

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

def main():
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
    out_dir = os.path.dirname(os.path.join(root, OUT_TEMPLATE.split("{")[0]))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Monta o path de saída relativo à raiz do projeto.
    out_full = os.path.join(root, OUT_TEMPLATE).replace("\\", "/")

    # Monta os argumentos do executável.
    cmd = [
        exe,
        "--games",         str(TOTAL_GAMES),
        "--chunk-games",   str(CHUNK_GAMES),
        "--depth",         str(MAX_DEPTH),
        "--time-ms",       str(TIME_MS),
        "--opening-plies", str(OPENING_PLIES),
        "--epsilon",       str(EPSILON),
        "--epsilon-midgame", str(EPSILON_MIDGAME),
        "--max-plies",     str(MAX_PLIES),
        "--seed",          str(SEED),
        "--out",           out_full,
    ]
    if THREADS > 0:
        cmd += ["--threads", str(THREADS)]

    print("=" * 60)
    print(f"[run_selfplay] Iniciando geração de dados")
    print(f"  Executável : {exe}")
    print(f"  Partidas   : {TOTAL_GAMES} total / {CHUNK_GAMES} por chunk")
    print(f"  Busca      : depth<={MAX_DEPTH}, {TIME_MS} ms/lance")
    print(f"  Abertura   : plies={OPENING_PLIES}, epsilon={EPSILON} | midgame epsilon={EPSILON_MIDGAME}")
    print(f"  Threads    : {THREADS or 'auto'}")
    print(f"  Saída      : {out_full}")
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
