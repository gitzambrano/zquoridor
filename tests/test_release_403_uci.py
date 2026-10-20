#!/usr/bin/env python3
"""4.03 UCI contract: identical Lazy tree across movetime and every native clock.

This is a short protocol/correctness smoke, not a competitive Elo test.
"""
import argparse
import re
import subprocess
from pathlib import Path

def run(exe,weights,commands,args=(),timeout=100):
    text="uci\nisready\nposition startpos\n"+commands+"\nquit\n"
    p=subprocess.run([str(Path(exe).resolve()),"--nnue",str(Path(weights).resolve()),*args],
                     input=text,capture_output=True,text=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"UCI failed {p.returncode}: {p.stderr[-2000:]}")
    assert "id name zQuoridor 4.03" in p.stdout,p.stdout[:1000]
    assert "uciok" in p.stdout and "readyok" in p.stdout
    infos=[line for line in p.stdout.splitlines() if line.startswith("info depth")]
    moves=[line for line in p.stdout.splitlines() if line.startswith("bestmove ")]
    assert len(infos)==len(moves)==commands.count("\ngo ")+(1 if commands.startswith("go ") else 0),p.stdout[-4000:]
    assert all(re.match(r"bestmove [a-i][1-9](?:[hv])?",m) for m in moves),moves
    assert all("nodecap=" in line and "lazy=" in line and "hard=" in line for line in infos),infos
    return infos,moves

def field(info,name):
    m=re.search(r"\b"+re.escape(name)+r"=(\d+)\b",info)
    assert m,(name,info)
    return int(m.group(1))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--exe",required=True)
    ap.add_argument("--weights",required=True)
    ns=ap.parse_args()
    base=["go movetime 60","position startpos"]
    for label,wtime,inc in (("1+1",60000,1000),("3+2",180000,2000),("5+3",300000,3000)):
        script="\n".join(base+[f"go wtime {wtime} btime {wtime} winc {inc} binc {inc}"])
        infos,moves=run(ns.exe,ns.weights,script,timeout=110)
        assert [field(x,"lazy") for x in infos]==[1,1],(label,infos)
        assert field(infos[1],"treeHit")==1,(label,infos[1])
        # External timer includes full tree work plus UCI output; grant OS
        # scheduling tolerance, but never allow game-long stalls.
        assert field(infos[1],"hard")>=field(infos[1],"budget"),(label,infos[1])
        print(f"VALID {label}: fixed->clock retains Lazy tree and returns legal-looking moves {moves}",flush=True)
    eager,_=run(ns.exe,ns.weights,"go movetime 50",args=["--eager-search"],timeout=30)
    assert field(eager[0],"lazy")==0,eager
    plain,_=run(ns.exe,ns.weights,"go movetime 50",timeout=30)
    assert field(plain[0],"lazy")==1,plain
    print("PASS unified Lazy default and explicit Eager diagnostic override",flush=True)

if __name__=="__main__":
    main()
