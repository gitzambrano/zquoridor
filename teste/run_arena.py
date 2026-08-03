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
GIT_REF2 = None             # Ref Git base para o confronto (ex: 'main', 'v1.0', 'HEAD')

INVERT_COLORS = True          # Se True, joga cada abertura 2x invertendo as cores (par). Se False, joga apenas 1x por abertura.
CREATE_BIN = True            # Se True, salva os dados das partidas em data/arena/ no formato .bin de treino
GAMES = 5000                   # Quantidade total de jogos
REPORT_GAMES = 50             # Atualiza e imprime o relatório parcial a cada N jogos concluídos (default 50)
TIME_MS = 500                 # Tempo de pensamento por lance em milissegundos
THREADS = 14                  # Número de núcleos / processos em paralelo (default 14)
RANDOM_OPENING_PLIES = 4      # Quantidade de lances aleatórios na abertura
SEED = 53                     # Semente aleatória
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
E2_HEURISTIC = True
# ==============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN_DIR = os.path.join(PROJECT_ROOT, "teste", "bin")
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
    Compila teste/arena.cpp incluindo os headers REAIS de dir1 (ref1)
    e dir2 (ref2) no MESMO binário, sob namespaces separados (qr_e1 /
    qr_e2) -- ao contrário do compile_arena antigo, que só compilava dir1
    e fazia "Engine 2" ser, na prática, uma cópia do código de Engine 1
    (o worktree de --ref2 era criado e descartado sem nunca ser usado).
    """
    os.makedirs(os.path.dirname(output_exe), exist_ok=True)
    arena_src = os.path.join(PROJECT_ROOT, "teste", "arena.cpp")

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

import multiprocessing

def worker_process(exe_path, worker_games, time_ms, random_plies, seed, report_games, bin_path, invert_colors, progress_queue, e1_nnue="", e2_nnue="", heuristic=False, e1_heuristic=False, e2_heuristic=False):
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
        elif line.startswith("[arena]"):
            # Confirmação de carregamento/fallback de NNUE (ou heurística
            # forçada) do binário -- com NNUE virando o default, é
            # importante o usuário conseguir ver se cada engine realmente
            # carregou os pesos ou caiu para heurístico, em vez desses
            # avisos ficarem mudos dentro do subprocess.
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
    parser.add_argument("--no-invert", dest="invert_colors", action="store_false", default=False, help="Nao inverte as cores na mesma abertura")
    parser.add_argument("--invert-colors", dest="invert_colors", action="store_true", default=True, help="Inverte as cores jogando a mesma abertura 2x")
    parser.add_argument("--e1-nnue", default="", help="Caminho para pesos NNUE quantizados (.qbin) do Engine 1 (default: data/nnue/nnue_weights_int8.bin, se existir)")
    parser.add_argument("--e2-nnue", default="", help="Caminho para pesos NNUE quantizados (.qbin) do Engine 2 (default: data/nnue/nnue_weights_int8.bin, se existir)")
    parser.add_argument("--heuristic", action="store_true", default=False, help="Forca avaliacao heuristica (evalSimple) nas duas engines, ignorando NNUE -- debug/historico/fallback")
    parser.add_argument("--e1-heuristic", dest="e1_heuristic", action="store_true", default=E1_HEURISTIC, help=f"Forca heuristica so no Engine 1 (padrao: {E1_HEURISTIC})")
    parser.add_argument("--e1-nnue-default", dest="e1_heuristic", action="store_false", help="Sobrepoe E1_HEURISTIC=True do arquivo, forcando NNUE default no Engine 1 sem editar/recompilar")
    parser.add_argument("--e2-heuristic", dest="e2_heuristic", action="store_true", default=E2_HEURISTIC, help=f"Forca heuristica so no Engine 2 (padrao: {E2_HEURISTIC})")
    parser.add_argument("--e2-nnue-default", dest="e2_heuristic", action="store_false", help="Sobrepoe E2_HEURISTIC=True do arquivo, forcando NNUE default no Engine 2 sem editar/recompilar")
    
    args = parser.parse_args()
    
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
    print("=" * 65 + "\n")
    
    dir1, cleanup1 = prepare_engine_source(args.ref1)
    dir2, cleanup2 = prepare_engine_source(args.ref2)
    
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
        worker_games = worker_pairs * 2 if args.invert_colors else (args.games // threads_count + (1 if w < (args.games % threads_count) else 0))
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
            args=(cand_exe, worker_games, args.time, args.random_plies, worker_seed, worker_report, worker_bin, args.invert_colors, queue, args.e1_nnue, args.e2_nnue, args.heuristic, args.e1_heuristic, args.e2_heuristic)
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
            if data and worker_report == 0:
                tot_eng1_wins += data["candWins"]
                tot_eng2_wins += data["baseWins"]
                tot_draws += data["draws"]
                tot_eng1_nodes += data["candNodes"]
                tot_eng2_nodes += data["baseNodes"]
                tot_eng1_time += data["candTimeSec"]
                tot_eng2_time += data["baseTimeSec"]
                completed_games += (data["candWins"] + data["baseWins"] + data["draws"])
        elif msg_type == "ERROR":
            print(f"[!] {data}", flush=True)
            finished_workers += 1
            
    for p in processes:
        p.join()

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
        total_samples = total_bytes // 27  # sizeof(TrainingSample) == 27
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
