#!/usr/bin/env python3
# build_arena_common.py -- compila teste/arena.cpp como um binario
# standalone (fora do fluxo completo de teste/run_arena.py, que faz git
# worktree + paralelismo + relatorio de Elo). Chamado por build_arena.sh
# e build_arena.bat.
#
# POR QUE ISTO NAO EXISTIA: tentar compilar teste/arena.cpp direto (g++
# teste/arena.cpp) sempre falhava com o #error no topo do arquivo --
# arena.cpp exige -DENGINE1_SEARCH_HPP/-DENGINE2_SEARCH_HPP apontando pra
# dois search.hpp reais (compila os DOIS motores no mesmo binario, sob
# namespaces qr_e1/qr_e2). Só teste/run_arena.py montava esse comando, e
# só via git worktree -- sem repo git (ex. rodando a partir de um zip),
# run_arena.py também não ajudava. Este script cobre o caso "só quero
# compilar/testar o arena.cpp local", sem precisar de git nem de duas refs
# de verdade.
#
# Uso:
#   python3 build_arena_common.py                  # engine1 = engine2 = raiz do projeto local
#   python3 build_arena_common.py DIR1              # engine2 = DIR1 também
#   python3 build_arena_common.py DIR1 DIR2         # cada engine usa o src/ de um diretorio diferente
#
# Cada DIRn deve ser a RAIZ de um checkout do projeto (contém DIRn/src/search.hpp).
import os
import sys
import shutil
import subprocess
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARENA_SRC = os.path.join(PROJECT_ROOT, "teste", "arena.cpp")
BIN_DIR = os.path.join(PROJECT_ROOT, "teste", "bin")

COMPILER = os.environ.get("CXX", "g++")
CXX_FLAGS = ["-O3", "-std=c++17"]


def bump_mtimes(src_dir, epoch_offset_seconds):
    # Mesmo mecanismo de teste/run_arena.py (_bump_mtimes): GCC trata dois
    # headers BYTE-IDENTICOS escritos no MESMO SEGUNDO como "o mesmo
    # arquivo já incluído" para fins de #pragma once, mesmo estando em
    # caminhos/inodes diferentes -- silenciosamente descarta a segunda
    # inclusão, e o segundo motor compila com os símbolos do primeiro.
    stamp = 1577836800 + epoch_offset_seconds  # 2020-01-01 UTC + offset
    for name in os.listdir(src_dir):
        if name.endswith(".hpp"):
            p = os.path.join(src_dir, name)
            os.utime(p, (stamp, stamp))


def main():
    args = sys.argv[1:]
    dir1 = os.path.abspath(args[0]) if len(args) >= 1 and args[0] else PROJECT_ROOT
    dir2 = os.path.abspath(args[1]) if len(args) >= 2 and args[1] else dir1

    for label, d in (("Engine 1", dir1), ("Engine 2", dir2)):
        if not os.path.isfile(os.path.join(d, "src", "search.hpp")):
            print(f"[!] ERRO: {label}: '{d}/src/search.hpp' nao existe.")
            sys.exit(1)

    os.makedirs(BIN_DIR, exist_ok=True)

    # SEMPRE duas cópias físicas distintas de src/, mesmo quando dir1==dir2
    # (auto-teste): se os dois -D apontassem pro MESMO arquivo real, o
    # #pragma once descartaria a segunda inclusão de verdade (não é uma
    # falsa detecção de duplicata -- É o mesmo arquivo), e o binário
    # resultante não teria namespace qr_e2 nenhum.
    work1 = tempfile.mkdtemp(prefix="zquoridor_arena_e1_")
    work2 = tempfile.mkdtemp(prefix="zquoridor_arena_e2_")
    try:
        shutil.copytree(os.path.join(dir1, "src"), os.path.join(work1, "src"), dirs_exist_ok=True)
        shutil.copytree(os.path.join(dir2, "src"), os.path.join(work2, "src"), dirs_exist_ok=True)
        bump_mtimes(os.path.join(work1, "src"), 0)
        bump_mtimes(os.path.join(work2, "src"), 86400)

        search1 = os.path.join(work1, "src", "search.hpp").replace("\\", "/")
        search2 = os.path.join(work2, "src", "search.hpp").replace("\\", "/")
        out_exe = os.path.join(BIN_DIR, "arena.exe")

        print(f"[*] Engine 1 : {dir1}")
        print(f"[*] Engine 2 : {dir2}")
        print(f"[*] Compilando {out_exe} ...")

        cmd = [COMPILER, *CXX_FLAGS,
               f"-DENGINE1_SEARCH_HPP=\"{search1}\"",
               f"-DENGINE2_SEARCH_HPP=\"{search2}\"",
               ARENA_SRC, "-o", out_exe]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print("[!] ERRO DE COMPILACAO (ver saida do compilador acima).")
            sys.exit(1)

        print()
        print(f"OK -- {out_exe}")
        print("NNUE e o default de avaliacao (tenta carregar")
        print("data/nnue/nnue_weights_int8.bin automaticamente); use --heuristic")
        print("para desligar, ou --e1-nnue/--e2-nnue para apontar pesos especificos.")
        print("Rode a partir da RAIZ do projeto para o caminho default resolver:")
        print(f"  {os.path.relpath(out_exe, PROJECT_ROOT)} --games 40 --time 50")
        print()
        print("Para um confronto de verdade entre duas refs git (paralelismo,")
        print("relatorio de Elo, dataset .bin), use teste/run_arena.py em vez")
        print("deste script -- este aqui e so para compilar/testar rapido.")
    finally:
        shutil.rmtree(work1, ignore_errors=True)
        shutil.rmtree(work2, ignore_errors=True)


if __name__ == "__main__":
    main()
