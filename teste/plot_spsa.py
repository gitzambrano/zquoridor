#!/usr/bin/env python3
"""
plot_spsa.py -- plota a convergencia de uma (ou varias) rodada(s) de
teste/tune_spsa.cpp a partir do CSV de historico que o proprio tune_spsa
grava a cada iteracao (--history, default spsa_history.csv; modo hybrid
grava um arquivo por candidato, com sufixo "_depth{D}").

Colunas esperadas no CSV (cabecalho escrito pelo tune_spsa):
    iter,elapsed_s,score,<param1>,<param2>,...

Uso basico (1 rodada, modo "spsa"):
    python3 teste/plot_spsa.py
    python3 teste/plot_spsa.py --history spsa_history.csv --out spsa_plot.png

Comparando os candidatos de uma rodada em modo "hybrid" (--glob casa
"spsa_history_depth*.csv" a partir do --history base):
    python3 teste/tune_spsa.cpp --mode hybrid ...
    python3 teste/plot_spsa.py --history spsa_history.csv --glob

Ou aponte pra um conjunto explicito de arquivos (rotulo = nome do arquivo,
sem extensao, a menos que --labels seja passado):
    python3 teste/plot_spsa.py spsa_history_depth2.csv spsa_history_depth3.csv

Gera 1 figura com 2 linhas de subplots:
  - topo: score bruto por iteracao (cinza, ruidoso por natureza) + media
    movel (janela --smooth, default 10) por cima, uma cor por rodada.
  - baixo: 1 subplot por parametro tunado (contempt/policyOrderScale/
    catScoreScale, o que estiver no CSV), valor de theta por iteracao.

Requer matplotlib (e opcionalmente pandas, mas usa so csv+listas puras se
pandas nao estiver instalado -- sem dependencia obrigatoria alem de
matplotlib).
"""
import argparse
import csv
import glob
import os
import sys


def read_history(path):
    """Le um CSV de historico do tune_spsa. Retorna dict de listas:
    {"iter": [...], "elapsed_s": [...], "score": [...], "<param>": [...], ...}
    Silenciosamente ignora linhas malformadas (ex.: escritas concorrentemente
    por um processo que ainda esta rodando -- corta a ultima linha se
    incompleta em vez de falhar)."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return None
    header = rows[0]
    cols = {name: [] for name in header}
    for row in rows[1:]:
        if len(row) != len(header):
            continue  # linha cortada (processo pode ainda estar escrevendo)
        for name, val in zip(header, row):
            try:
                cols[name].append(float(val))
            except ValueError:
                cols[name].append(None)
    return cols


def moving_average(values, window):
    if window <= 1:
        return list(values)
    out = []
    acc = 0.0
    q = []
    for v in values:
        q.append(v)
        acc += v
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out


def find_param_columns(cols):
    return [k for k in cols.keys() if k not in ("iter", "elapsed_s", "score")]


def label_for(path):
    base = os.path.basename(path)
    if base.endswith(".csv"):
        base = base[:-4]
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="CSV(s) de historico explicitos (sobrepoe --history/--glob)")
    p.add_argument("--history", default="spsa_history.csv",
                    help="CSV de historico base (default: spsa_history.csv); ignorado se 'files' for passado")
    p.add_argument("--glob", action="store_true",
                    help="em vez de --history sozinho, casa '<history sem .csv>_depth*.csv' "
                         "(saida de --mode hybrid) e plota todos juntos, um por cor")
    p.add_argument("--labels", nargs="*", default=None,
                    help="rotulos custom para cada arquivo (mesma ordem/quantidade de 'files' ou dos arquivos casados por --glob)")
    p.add_argument("--smooth", type=int, default=10,
                    help="janela da media movel sobre o score (default: 10 iteracoes; 1 = sem suavizar)")
    p.add_argument("--out", default="spsa_plot.png", help="arquivo de imagem de saida (default: spsa_plot.png)")
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args()

    if args.files:
        paths = args.files
    elif args.glob:
        base = args.history[:-4] if args.history.endswith(".csv") else args.history
        paths = sorted(glob.glob(base + "_depth*.csv"))
        if not paths:
            print(f"[plot_spsa] nenhum arquivo casou com '{base}_depth*.csv' -- rode --mode hybrid antes, "
                  f"ou passe --history/arquivos explicitos.", file=sys.stderr)
            sys.exit(1)
    else:
        paths = [args.history]

    labels = args.labels if args.labels else [label_for(pth) for pth in paths]
    if len(labels) != len(paths):
        print(f"[plot_spsa] --labels tem {len(labels)} itens mas ha {len(paths)} arquivo(s) -- ignorando --labels", file=sys.stderr)
        labels = [label_for(pth) for pth in paths]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_spsa] ERRO: matplotlib nao esta instalado -- pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    runs = []  # lista de (label, cols)
    for pth, lbl in zip(paths, labels):
        cols = read_history(pth)
        if cols is None or not cols.get("iter"):
            print(f"[plot_spsa] aviso: '{pth}' nao existe ou esta vazio -- pulando", file=sys.stderr)
            continue
        runs.append((lbl, cols))

    if not runs:
        print("[plot_spsa] ERRO: nenhum historico valido encontrado -- nada para plotar.", file=sys.stderr)
        sys.exit(1)

    # parametros tunados = uniao das colunas de todas as rodadas (uma rodada
    # pode ter tunado um subconjunto diferente via --no-tune-*)
    param_names = []
    for _, cols in runs:
        for name in find_param_columns(cols):
            if name not in param_names:
                param_names.append(name)

    n_param_rows = len(param_names)
    fig, axes = plt.subplots(1 + n_param_rows, 1, figsize=(9, 3.2 * (1 + n_param_rows)), sharex=True)
    if n_param_rows == 0:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax_score = axes[0]
    for i, (lbl, cols) in enumerate(runs):
        color = colors[i % len(colors)]
        it = cols["iter"]
        score = cols["score"]
        ax_score.plot(it, score, color=color, alpha=0.25, linewidth=0.8)
        smoothed = moving_average(score, args.smooth)
        ax_score.plot(it, smoothed, color=color, linewidth=1.8, label=f"{lbl} (media movel={args.smooth})")
    ax_score.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_score.set_ylabel("score da iteracao\n(+1 = plus venceu as 2, -1 = minus venceu as 2)")
    ax_score.set_title("Convergencia do SPSA -- teste/tune_spsa.cpp")
    ax_score.legend(fontsize=8, loc="best")
    ax_score.grid(alpha=0.25)

    for row, pname in enumerate(param_names):
        ax = axes[1 + row]
        for i, (lbl, cols) in enumerate(runs):
            if pname not in cols:
                continue
            color = colors[i % len(colors)]
            ax.plot(cols["iter"], cols[pname], color=color, linewidth=1.5, label=lbl)
        ax.set_ylabel(pname)
        ax.grid(alpha=0.25)
        if len(runs) > 1:
            ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("iteracao SPSA")
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"[plot_spsa] salvo em {args.out} ({len(runs)} rodada(s), {n_param_rows} parametro(s): {', '.join(param_names)})")


if __name__ == "__main__":
    main()
