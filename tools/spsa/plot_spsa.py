#!/usr/bin/env python3
"""
plot_spsa.py -- plota a convergencia de uma (ou varias) rodada(s) de
tools/spsa/tune_spsa.cpp a partir do CSV de historico que o proprio tune_spsa
grava a cada iteracao (--history, default spsa_history.csv; modo hybrid
grava um arquivo por candidato, com sufixo "_depth{D}").

Colunas esperadas no CSV do GA:
    generation,bestFitness,meanFitness,diversity,<param1>,<param2>,...

O script aceita também o CSV antigo do SPSA, detectando automaticamente as
colunas. Para o GA, o gráfico mostra a melhor fitness, média da população,
diversidade e a evolução dos genes.

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
    excluded = {"iter", "elapsed_s", "score", "generation", "bestFitness", "meanFitness", "diversity"}
    return [k for k in cols.keys() if k not in excluded]


def label_for(path):
    base = os.path.basename(path)
    if base.endswith(".csv"):
        base = base[:-4]
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="CSV(s) de historico explicitos (sobrepoe --history/--glob)")
    p.add_argument("--history", default="ga_history.csv",
                    help="CSV de historico base (default: ga_history.csv); ignorado se 'files' for passado")
    p.add_argument("--glob", action="store_true",
                    help="em vez de --history sozinho, casa '<history sem .csv>_depth*.csv' "
                         "(saida de --mode hybrid) e plota todos juntos, um por cor")
    p.add_argument("--labels", nargs="*", default=None,
                    help="rotulos custom para cada arquivo (mesma ordem/quantidade de 'files' ou dos arquivos casados por --glob)")
    p.add_argument("--smooth", type=int, default=10,
                    help="janela da media movel sobre o score (default: 10 iteracoes; 1 = sem suavizar)")
    p.add_argument("--out", default="ga_plot.png", help="arquivo de imagem de saida (default: ga_plot.png)")
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
        # Formato GA usa "generation" como eixo x; formato SPSA legado usa "iter".
        has_data = cols is not None and (cols.get("generation") or cols.get("iter"))
        if not has_data:
            print(f"[plot_spsa] aviso: '{pth}' nao existe ou esta vazio -- pulando", file=sys.stderr)
            continue
        runs.append((lbl, cols))

    if not runs:
        print("[plot_spsa] ERRO: nenhum historico valido encontrado -- nada para plotar.", file=sys.stderr)
        sys.exit(1)

    # Detecta formato GA ou SPSA antigo.
    is_ga = all("generation" in cols and "bestFitness" in cols for _, cols in runs)

    param_names = []
    for _, cols in runs:
        for name in find_param_columns(cols):
            if name not in param_names:
                param_names.append(name)

    n_param_rows = len(param_names)
    fig, axes = plt.subplots(2 + n_param_rows, 1,
                             figsize=(9, 3.0 * (2 + n_param_rows)),
                             sharex=True)
    if not isinstance(axes, (list, tuple)):
        import numpy as np
        axes = np.atleast_1d(axes)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax_fit = axes[0]
    ax_div = axes[1]
    for i, (lbl, cols) in enumerate(runs):
        color = colors[i % len(colors)]
        if is_ga:
            x = cols["generation"]
            ax_fit.plot(x, cols["bestFitness"], color=color, linewidth=1.8, label=f"{lbl} melhor")
            ax_fit.plot(x, cols["meanFitness"], color=color, linewidth=1.0, alpha=0.45, linestyle="--")
            ax_div.plot(x, cols["diversity"], color=color, linewidth=1.5, label=lbl)
        else:
            x = cols["iter"]
            score = cols["score"]
            ax_fit.plot(x, score, color=color, alpha=0.25, linewidth=0.8)
            ax_fit.plot(x, moving_average(score, args.smooth), color=color, linewidth=1.8, label=lbl)
    ax_fit.axhline(0.5 if is_ga else 0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_fit.set_ylabel("fitness" if is_ga else "score")
    ax_fit.set_title("Convergência do GA" if is_ga else "Convergência do SPSA legado")
    ax_fit.legend(fontsize=8, loc="best")

    if is_ga:
        ax_div.set_ylabel("diversidade normalizada")
        ax_div.set_title("Diversidade da população — útil para detectar convergência prematura")
        ax_div.legend(fontsize=8, loc="best")
    else:
        ax_div.set_visible(False)

    offset = 2 if is_ga else 1
    for row, pname in enumerate(param_names):
        ax = axes[offset + row]
        for i, (lbl, cols) in enumerate(runs):
            if pname not in cols:
                continue
            color = colors[i % len(colors)]
            x = cols["generation"] if is_ga else cols["iter"]
            ax.plot(x, cols[pname], color=color, linewidth=1.5, label=lbl)
        ax.set_ylabel(pname)
        ax.grid(alpha=0.25)
        if len(runs) > 1:
            ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("geração" if is_ga else "iteração SPSA")
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"[plot_spsa] salvo em {args.out} ({len(runs)} rodada(s), {n_param_rows} parametro(s): {', '.join(param_names)})")


if __name__ == "__main__":
    main()
