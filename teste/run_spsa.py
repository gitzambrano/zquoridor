#!/usr/bin/env python3
"""
run_spsa.py -- Orquestrador Python para o tuner SPSA (teste/tune_spsa.cpp).

Edite as variáveis da seção CONFIG abaixo e execute:
    python3 run_spsa.py

Mesmo padrão de training/run_selfplay.py e teste/run_arena.py: TODAS as
variáveis podem ser setadas aqui no topo do arquivo (sem precisar passar
nada por linha de comando nem recompilar C++); qualquer flag de CLI, se
passada, sobrepõe só aquela variável para esta execução.

O script localiza teste/tune_spsa.cpp relativo a este arquivo, compila
(g++) se o binário não existir ou se o .cpp for mais novo que o binário, e
chama o executável com os parâmetros configurados.

NNUE + ordenação assistida por política: LIGADAS por default (mesmo
default já usado por selfplay/arena/wasm) -- ver USE_NNUE/POLICY_ORDERING
abaixo. Só tuna parâmetros que afetam avaliação/ordenação em modo NNUE
(contempt, policyOrderScale, catScoreScale) -- NÃO tuna os pesos da
heurística evalSimple (EvalWeights), que ficaram fora de escopo desta
rodada.
"""

import argparse
import os
import subprocess
import sys

# =============================================================================
# CONFIG -- edite estas variaveis conforme necessario
# =============================================================================

# --- Modo ---
# "spsa": ajuste contínuo (Spall 1998) de contempt/policyOrderScale/
#   catScoreScale via partidas antitéticas.
# "sweep-mindepth": mini-torneio round-robin sobre valores discretos de
#   policyOrderingMinDepth (não se presta a SPSA contínuo -- ver
#   tune_spsa.cpp para a justificativa completa).
MODE = "spsa"

# --- SPSA (modo "spsa") ---
ITERATIONS       = 40        # iteracoes totais de SPSA
SEED             = 20260719
TIME_BUDGET_SEC  = None      # None = sem limite; ex: 3600.0 para 1h
CHECKPOINT       = "spsa_checkpoint.txt"   # relativo a PROJECT_ROOT

# Parametros continuos tunados e seus bounds ABSOLUTOS (nao relativos ao
# valor inicial -- contempt pode ser negativo, entao "0.1x/4x do inicial"
# nao funciona bem). None em CONTEMPT_INIT/POLICY_SCALE_INIT/CAT_SCALE_INIT
# usa o default hardcoded em search.hpp (-30 / 400 / 2).
TUNE_CONTEMPT       = True
CONTEMPT_INIT       = None
CONTEMPT_BOUNDS     = (-150.0, 0.0)

TUNE_POLICY_SCALE   = True
POLICY_SCALE_INIT   = None
POLICY_SCALE_BOUNDS = (0.0, 2000.0)

TUNE_CAT_SCALE      = True
CAT_SCALE_INIT       = None
CAT_SCALE_BOUNDS     = (0.0, 20.0)

# --- Sweep de policyOrderingMinDepth (modo "sweep-mindepth") ---
# Discreto e de faixa pequena -- ver tune_spsa.cpp: nao entra no SPSA
# continuo, roda como mini-torneio round-robin separado (mesma engine com
# os 3 parametros continuos acima fixos no valor de INIT/default).
MINDEPTH_CANDIDATES = [1, 2, 3, 4, 5]
MINDEPTH_GAMES       = 50     # partidas antiteticas por par de candidatos

# --- Config da partida de auto-jogo (rapida de proposito -- SPSA precisa
# de MUITAS partidas, nao de partidas profundas) ---
SEARCH_DEPTH          = 20
TIME_MS                = 120
MAX_PLIES              = 100
OPENING_RANDOM_PLIES   = 4

# --- Avaliacao de folha (NNUE + policy head ligados por default) ---
# NAO tunamos mais a heuristica (EvalWeights/evalSimple) nesta rodada --
# USE_NNUE=True e o default porque selfplay/arena/wasm ja rodam assim em
# producao; desligar aqui so serve para comparacao/depuracao (contempt
# ainda tem efeito em modo heuristico, mas policyOrderScale/catScoreScale
# nao tem efeito nenhum sem NNUE).
USE_NNUE               = True
NNUE_WEIGHTS_PATH       = None   # None = default do binario (data/nnue/nnue_weights_int8.bin)
POLICY_ORDERING         = True
POLICY_ORDER_MIN_DEPTH  = 3       # usado quando MODE="spsa"; ignorado em "sweep-mindepth"

# =============================================================================
# INTERNALS -- normalmente nao e necessario editar abaixo desta linha
# =============================================================================

def find_project_root():
    """Sobe na hierarquia de diretorios ate encontrar a pasta 'src'."""
    here = os.path.abspath(os.path.dirname(__file__))
    candidate = here
    for _ in range(5):
        if os.path.isdir(os.path.join(candidate, "src")):
            return candidate
        candidate = os.path.dirname(candidate)
    return here  # fallback

def find_tune_spsa_exe(root):
    for name in ("tune_spsa.exe", "tune_spsa"):
        path = os.path.join(root, "bin", name)
        if os.path.isfile(path):
            return path
    return None

def need_recompile(root, exe):
    src = os.path.join(root, "teste", "tune_spsa.cpp")
    if not os.path.isfile(exe):
        return True
    return os.path.getmtime(src) > os.path.getmtime(exe)

def compile_tune_spsa(root):
    """Tenta compilar tune_spsa usando o build script adequado."""
    import platform
    if platform.system() == "Windows":
        bat = os.path.join(root, "build", "build_tune_spsa.bat")
        if os.path.isfile(bat):
            print("[run_spsa] compilando tune_spsa...")
            ret = subprocess.call(["cmd", "/c", bat], cwd=root)
            return ret == 0
    else:
        sh = os.path.join(root, "build", "build_tune_spsa.sh")
        if os.path.isfile(sh):
            print("[run_spsa] compilando tune_spsa...")
            ret = subprocess.call(["bash", sh], cwd=root)
            return ret == 0
    return False


def parse_args():
    """CLI opcional -- toda flag aqui tem a constante correspondente na
    secao CONFIG acima como default. Rodar `python3 run_spsa.py` sem
    argumento nenhum usa 100% das constantes do arquivo; qualquer flag
    passada aqui sobrepoe so aquela constante para esta execucao, sem
    precisar editar/recompilar nada. Mesmo padrao de run_selfplay.py e
    run_arena.py."""
    p = argparse.ArgumentParser(description="Orquestrador do tuner SPSA (config no topo do arquivo ou via flags)")
    p.add_argument("--mode", choices=["spsa", "sweep-mindepth"], default=MODE)
    p.add_argument("--iterations", type=int, default=ITERATIONS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--time-budget-sec", type=float, default=TIME_BUDGET_SEC)
    p.add_argument("--checkpoint", default=CHECKPOINT)

    p.add_argument("--tune-contempt", dest="tune_contempt", action="store_true", default=TUNE_CONTEMPT)
    p.add_argument("--no-tune-contempt", dest="tune_contempt", action="store_false")
    p.add_argument("--contempt-init", type=float, default=CONTEMPT_INIT)
    p.add_argument("--contempt-bounds", nargs=2, type=float, default=list(CONTEMPT_BOUNDS))

    p.add_argument("--tune-policy-scale", dest="tune_policy_scale", action="store_true", default=TUNE_POLICY_SCALE)
    p.add_argument("--no-tune-policy-scale", dest="tune_policy_scale", action="store_false")
    p.add_argument("--policy-scale-init", type=float, default=POLICY_SCALE_INIT)
    p.add_argument("--policy-scale-bounds", nargs=2, type=float, default=list(POLICY_SCALE_BOUNDS))

    p.add_argument("--tune-cat-scale", dest="tune_cat_scale", action="store_true", default=TUNE_CAT_SCALE)
    p.add_argument("--no-tune-cat-scale", dest="tune_cat_scale", action="store_false")
    p.add_argument("--cat-scale-init", type=float, default=CAT_SCALE_INIT)
    p.add_argument("--cat-scale-bounds", nargs=2, type=float, default=list(CAT_SCALE_BOUNDS))

    p.add_argument("--mindepth-candidates", default=",".join(str(x) for x in MINDEPTH_CANDIDATES),
                    help=f"lista separada por virgula (padrao: {MINDEPTH_CANDIDATES})")
    p.add_argument("--mindepth-games", type=int, default=MINDEPTH_GAMES)

    p.add_argument("--depth", type=int, default=SEARCH_DEPTH)
    p.add_argument("--time-ms", type=int, default=TIME_MS)
    p.add_argument("--max-plies", type=int, default=MAX_PLIES)
    p.add_argument("--opening-plies", type=int, default=OPENING_RANDOM_PLIES)

    p.add_argument("--heuristic", dest="use_nnue", action="store_false", default=USE_NNUE,
                    help="forca avaliacao heuristica, ignorando NNUE (padrao: NNUE ligada)")
    p.add_argument("--nnue", dest="use_nnue", action="store_true",
                    help="sobrepoe USE_NNUE=False do arquivo, forcando NNUE sem editar/recompilar")
    p.add_argument("--nnue-weights", default=NNUE_WEIGHTS_PATH)
    p.add_argument("--policy-order", dest="policy_order", action="store_true", default=POLICY_ORDERING)
    p.add_argument("--no-policy-order", dest="policy_order", action="store_false")
    p.add_argument("--policy-order-min-depth", type=int, default=POLICY_ORDER_MIN_DEPTH)

    return p.parse_args()


def main():
    args = parse_args()
    root = find_project_root()
    exe = find_tune_spsa_exe(root)

    if exe is None or need_recompile(root, exe):
        ok = compile_tune_spsa(root)
        exe = find_tune_spsa_exe(root)
        if exe is None:
            print("ERRO: nao foi possivel encontrar ou compilar tune_spsa.", file=sys.stderr)
            print("  Execute manualmente: build/build_tune_spsa.sh (ou .bat)", file=sys.stderr)
            sys.exit(1)

    cmd = [
        exe,
        "--mode",             args.mode,
        "--iterations",       str(args.iterations),
        "--seed",             str(args.seed),
        "--checkpoint",       os.path.join(root, args.checkpoint).replace("\\", "/"),
        "--depth",            str(args.depth),
        "--time-ms",          str(args.time_ms),
        "--max-plies",        str(args.max_plies),
        "--opening-plies",    str(args.opening_plies),
    ]
    if args.time_budget_sec is not None:
        cmd += ["--time-budget-sec", str(args.time_budget_sec)]

    if not args.use_nnue:
        cmd += ["--heuristic"]
    elif args.nnue_weights:
        nnue_abs = os.path.join(root, args.nnue_weights)
        cmd += ["--nnue-weights", nnue_abs.replace("\\", "/")]

    if args.policy_order:
        cmd += ["--policy-order-min-depth", str(args.policy_order_min_depth)]
    else:
        cmd += ["--no-policy-order"]

    if args.mode == "spsa":
        if not args.tune_contempt:
            cmd += ["--no-tune-contempt"]
        if args.contempt_init is not None:
            cmd += ["--contempt", str(args.contempt_init)]
        cmd += ["--contempt-bounds", f"{args.contempt_bounds[0]},{args.contempt_bounds[1]}"]

        if not args.tune_policy_scale:
            cmd += ["--no-tune-policy-scale"]
        if args.policy_scale_init is not None:
            cmd += ["--policy-scale", str(args.policy_scale_init)]
        cmd += ["--policy-scale-bounds", f"{args.policy_scale_bounds[0]},{args.policy_scale_bounds[1]}"]

        if not args.tune_cat_scale:
            cmd += ["--no-tune-cat-scale"]
        if args.cat_scale_init is not None:
            cmd += ["--cat-scale", str(args.cat_scale_init)]
        cmd += ["--cat-scale-bounds", f"{args.cat_scale_bounds[0]},{args.cat_scale_bounds[1]}"]
    else:  # sweep-mindepth
        cmd += ["--mindepth-candidates", args.mindepth_candidates]
        cmd += ["--mindepth-games", str(args.mindepth_games)]

    print("=" * 70)
    print("[run_spsa] Iniciando tuner SPSA")
    print(f"  Executavel   : {exe}")
    print(f"  Modo         : {args.mode}")
    if args.mode == "spsa":
        print(f"  Iteracoes    : {args.iterations}")
        tuned = []
        if args.tune_contempt: tuned.append("contempt")
        if args.tune_policy_scale: tuned.append("policyOrderScale")
        if args.tune_cat_scale: tuned.append("catScoreScale")
        print(f"  Parametros   : {', '.join(tuned) if tuned else '(nenhum -- nada a fazer)'}")
    else:
        print(f"  Candidatos minDepth: {args.mindepth_candidates} ({args.mindepth_games} partidas/par)")
    print(f"  Busca        : depth<={args.depth}, {args.time_ms} ms/lance, max {args.max_plies} plies")
    print(f"  Avaliacao    : {'NNUE (default) + policy ordering' if args.use_nnue and args.policy_order else 'NNUE (default), policy ordering desligada' if args.use_nnue else 'heuristica (evalSimple) -- forcada'}")
    print(f"  Checkpoint   : {args.checkpoint}")
    print("=" * 70)
    print()

    ret = subprocess.call(cmd, cwd=root)
    if ret != 0:
        print(f"[run_spsa] ERRO: tune_spsa terminou com codigo {ret}", file=sys.stderr)
        sys.exit(ret)


if __name__ == "__main__":
    main()
