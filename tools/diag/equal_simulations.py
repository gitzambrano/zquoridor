#!/usr/bin/env python3
"""Fixed *simulation count* experiment, not a clock benchmark.

All engines start with precisely the same 9x9 states, same NNUE and exactly
N MCGS backup calls. Each state is searched twice by the *same* persistent
UCI engine to reveal tree-reuse effects. This is intentionally not 3+2.
"""
import argparse,json,random,re,subprocess
from pathlib import Path

def positions(root, count):
    selected=[]
    books=["tools/external/openings_normal_confirm_400.jsonl",
           "tools/external/openings_center_rush_sound_5k.jsonl"]
    for k,book in enumerate(books):
        rows=[json.loads(s)["moves"] for s in (root/book).read_text().splitlines() if s.strip()]
        ix=list(range(len(rows)))
        random.Random(20261129+k).shuffle(ix)
        for j in ix[:count]:
            selected.append({"family":("normal","center")[k],"opening_idx":j,"moves":rows[j]})
    return selected

def run(exe,root,nnue,n,items):
    query="uci\nisready\n"
    for case in items:
        line="position startpos"+(" moves "+" ".join(case["moves"]) if case["moves"] else "")
        query+=line+"\ngo movetime 30000\n"+line+"\ngo movetime 30000\n"
    query+="quit\n"
    cmd=[str(exe),"--nnue",str(nnue),"--dump-root","--nodes","80000",
         "--diag-fixed-sims",str(n)]
    p=subprocess.run(cmd,input=query,text=True,capture_output=True,timeout=210, cwd=root)
    if p.returncode:raise RuntimeError(f"{exe} exit {p.returncode}: {p.stderr[-2000:]}")
    ls=p.stdout.splitlines()
    infos=[dict(re.findall(r"(?<!\S)(diag_[a-z]+)=([^\s]+)",l)) for l in ls
           if l.startswith("info depth ") and "diag_sims=" in l]
    roots=[json.loads(l.split("root_json ",1)[1]) for l in ls if "root_json " in l]
    moves=[l.split()[1] for l in ls if l.startswith("bestmove ")]
    expected=len(items)*2
    if not(len(infos)==len(roots)==len(moves)==expected):
        raise RuntimeError(f"{exe}: expected {expected} got infos={len(infos)} roots={len(roots)} moves={len(moves)} tail={ls[-10:]}")
    out=[]
    for j,(info,rs,move) in enumerate(zip(infos,roots,moves)):
        if int(info["diag_sims"])!=n or int(info["diag_stop"])!=5 or int(info["diag_simcap"])!=n or int(info["diag_adaptive"])!=0:
            raise RuntimeError(f"Not equal simulations: {exe} item={j} info={info}")
        if int(rs["simulations"])!=n:raise RuntimeError(f"Wrong root simulations: {exe} item={j} root={rs['simulations']}")
        if rs["bestmove"]!=move:raise RuntimeError("UCI/root bestmove mismatch")
        out.append({"bestmove":move,"best_action":rs["best_action"],
             "root_total_visits":rs["root_total_visits"],
             "simulations":n,
             "expanded_nodes":int(info["diag_nodes"]),
             "elapsed_ms":float(rs["elapsed_ms"]),
             "root_value":rs["root_value"],
             "priors":rs["priors"],
             "visits":rs["visit_counts"],
             "q_values":rs["q_values"],
             "tree_reused": None})
    return out

def tv(a,b):
    sa=sum(a);sb=sum(b)
    if not sa or not sb:return None
    return sum(abs(x/sa-y/sb) for x,y in zip(a,b))/2

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--main",required=True);p.add_argument("--lazy",required=True)
    p.add_argument("--combo",required=True);p.add_argument("--repo",default=".")
    p.add_argument("--sims",type=int,required=True)
    p.add_argument("--pairs-per-family",type=int,default=10)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    root=Path(a.repo).resolve()
    nnue=root/"data/nnue/nnue_weights_int8.bin"
    items=positions(root,a.pairs_per_family)
    data={name:run(Path(getattr(a,name)).resolve(),root,nnue,a.sims,items)
          for name in ("main","lazy","combo")}
    comparisons={}
    for left,right in (("combo","main"),("combo","lazy"),("lazy","main")):
        ar=data[left];br=data[right]
        rows=[]
        for j,(x,y) in enumerate(zip(ar,br)):
            if x["priors"]!=y["priors"]:
                raise AssertionError(f"Non-identical policy prior on position {j}: {left} vs {right}")
            rows.append({"opening":j//2,"second_same_root":bool(j%2),
              "family":items[j//2]["family"],
              "same_move":x["bestmove"]==y["bestmove"],
              "visits_total_variation":tv(x["visits"],y["visits"]),
              "expanded_node_delta":x["expanded_nodes"]-y["expanded_nodes"],
              "root_value_delta":(x["root_value"]-y["root_value"]) if x["root_value"] is not None and y["root_value"] is not None else None})
        comparisons[left+"_vs_"+right]={
            "same_move_fraction":sum(z["same_move"] for z in rows)/len(rows),
            "mean_visit_tv":sum(z["visits_total_variation"] for z in rows if z["visits_total_variation"] is not None)/len(rows),
            "mean_node_delta":sum(z["expanded_node_delta"] for z in rows)/len(rows),
            "rows":rows}
    report={"schema":"zquoridor.diag.equal_sims.v1","target_simulations":a.sims,
            "root_positions":len(items),"searches_per_engine":2*len(items),
            "paired_priors_bit_exact":True,"comparisons":comparisons,
            "by_engine":data,"positions":items}
    dest=Path(a.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,separators=(",",":"))+"\n")
    print("VALID FIXED SIM COUNT",a.sims,"PER SEARCH",len(items),"positions; repeated same-root tree reuse")
    print("EQUAL SIMULATION RESULT",json.dumps({k:{x:y for x,y in v.items() if x!="rows"} for k,v in comparisons.items()}),flush=True)

if __name__=="__main__":main()
