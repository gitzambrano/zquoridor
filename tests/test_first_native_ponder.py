#!/usr/bin/env python3
"""Smoke test first opponent ponder / first go real-clock tree reuse under uniform Lazy.

No games or opponent dependencies; derive the played move from the
engine's own ponder suggestion so it necessarily has a searched child.
"""
import argparse
import re
import subprocess
from pathlib import Path

def run(exe, weights, clock_hint, native_go):
    proc=subprocess.Popen(
        [str(Path(exe).resolve()),"--nnue",str(Path(weights).resolve()),"--nodes","2000000"],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        text=True,bufsize=1)
    def send(s):
        proc.stdin.write(s+"\n")
        proc.stdin.flush()
    def until(key):
        lines=[]
        for _ in range(20000):
            line=proc.stdout.readline()
            if not line:
                raise RuntimeError("EOF waiting for "+key)
            lines.append(line.strip())
            if line.startswith(key):
                return line.strip(), lines
        raise RuntimeError("Too many lines waiting for "+key)
    try:
        send("uci");until("uciok")
        send("isready");until("readyok")
        send("position startpos")
        send(f"ponder {'clock ' if clock_hint else ''}movetime 1500")
        line,_=until("ponderok")
        found=re.search(r"\bsuggested=(\S+)",line)
        assert found, line
        move=found.group(1)
        send("position startpos moves "+move)
        if native_go:
            send("go wtime 180000 btime 180000 winc 2000 binc 2000")
        else:
            send("go movetime 200")
        best,lines=until("bestmove ")
        infos=[l for l in lines if l.startswith("info depth ")]
        assert len(infos)==1,(best,lines[-10:])
        stats=infos[-1]
        tree=re.search(r"\btreeHit=(\d+)\b",stats)
        lazy=re.search(r"\blazy=(\d+)\b",stats)
        assert tree and lazy,stats
        result={"clock_hint":clock_hint,"native_go":native_go,"treeHit":int(tree.group(1)),
                "lazy":int(lazy.group(1)),"suggested":move,"bestmove":best.split()[1]}
        return result
    finally:
        try:send("quit")
        except (BrokenPipeError,ValueError):pass
        try:proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:proc.kill()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--exe",required=True)
    p.add_argument("--weights",required=True)
    a=p.parse_args()
    a1=run(a.exe,a.weights,False,False)
    a2=run(a.exe,a.weights,True,True)
    a3=run(a.exe,a.weights,False,True)
    assert (a1["treeHit"],a1["lazy"])==(1,1),a1
    # 4.03 uses Lazy on all game-clock protocols. Ponder roots remain
    # compatible whether the optional native-clock hint is supplied or not.
    assert (a2["treeHit"],a2["lazy"])==(1,1),a2
    assert (a3["treeHit"],a3["lazy"])==(1,1),a3
    for case in (a1,a2,a3):print("VALID PONDER ROOT",case,flush=True)
if __name__=="__main__":main()
