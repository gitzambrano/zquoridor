#!/usr/bin/env python3
"""Promote the 4.10 search-distilled quantized policy over the 4.03 engine."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request
import zlib

R=Path(__file__).resolve().parents[1]
OLD="45fa3ea7db3127a0b907a23d268b0c55c2020e1018513d54bb920cdbe134abe0"
NEW="fe8473255fd219fab247b1a7d44be25384a6cfc5daa5c301def49109badb7f2b"
BLOB="b29cc917bf4ca2e315cac790a0126603e7537989"
OFFSET=985824
BT=chr(96)

def hash_(b):
    return hashlib.sha256(b).hexdigest()

def load(p):
    return (R/p).read_text()

def save(p,s):
    x=R/p
    x.parent.mkdir(parents=True,exist_ok=True)
    x.write_text(s)

def main():
    req=urllib.request.Request(
        "https://api.github.com/repos/"+os.environ["GITHUB_REPOSITORY"]+"/git/blobs/"+BLOB,
        headers={"Authorization":"Bearer "+os.environ["GH_TOKEN"],
                 "Accept":"application/vnd.github+json","User-Agent":"zquoridor-release"})
    with urllib.request.urlopen(req,timeout=60) as f:
        raw=json.load(f)["content"]
    tail=zlib.decompress(base64.b64decode(base64.b64decode(raw).decode().strip()))
    assert len(tail)==107844
    p=R/"data/nnue/nnue_weights_int8.bin"
    original=p.read_bytes()
    assert len(original)==1093668 and hash_(original)==OLD
    modified=original[:OFFSET]+tail
    assert hash_(modified)==NEW
    p.write_bytes(modified)

    arch=json.loads(load("data/nnue/nnue_weights.architecture.json"))
    assert (arch["features"],arch["hidden"],arch["policy_out"])==(858,512,209)
    arch.update(int8_sha256=NEW,int8_policy_only=True,policy_provenance="search-guided policy distillation",float_is_prior_policy_reference=True)
    save("data/nnue/nnue_weights.architecture.json",json.dumps(arch,indent=2)+"\n")

    meta=json.loads(load("data/nnue/release.json"))
    assert meta["version"]=="4.03"
    meta.update(version="4.10",date="2026-10-20",previous_release="4.03",
        previous_git_commit="16cf6a54924522ad08f3cb98b31e14d89c3c1556",
        network="858-search-distilled-policy",nnue_checkpoint_release="4.10",
        nnue_checkpoint_preserved=False,
        search="unified_lazy_sparse_with_tree_preparation_clock",
        int8_policy_only=True,policy_change_start_offset=OFFSET,
        policy_provenance="search-guided policy distillation",
        promotion_basis="Operator-authorized 4.10: keep 4.03 unified Lazy+Sparse search, change only the INT8 policy head. Earlier 4.02 search 200ms paired results: 54.125% head-to-head, 65.375% versus Claustrophobia CPU versus 61.125% for the original policy. Confidence intervals for improvements include zero. 4.10 long-clock Elo unverified.",
        evidence="docs/releases/4.10.md",
        float_checkpoint_is_previous_policy_reference=True)
    meta["weights"]["nnue_weights_int8.bin"]=NEW
    meta["known_limitations"]=[
        "The retained float32 weights retain the prior policy and MUST NOT be re-quantized over production INT8.",
        "Previous-search 200ms results do not establish 4.10 competitive gains at 3+2 or 5+3.",
        "Claustrophobia may overrun a 200ms request by finishing a synchronous MCTS wave.",
        "Paired strength-change confidence intervals include zero."]
    meta["validation"]["410_policy_identity"]=NEW
    meta["validation"]["410_403_native_clock_elo"]="not established"
    save("data/nnue/release.json",json.dumps(meta,indent=2)+"\n")

    s=load("README.md")
    assert s.startswith("# zQuoridor 4.03")
    s=s.replace("# zQuoridor 4.03","# zQuoridor 4.10",1).replace("Version 4.03 uses uniform","Version 4.10 retains uniform",1)
    start=s.index("## "+bytes.fromhex("41706578").decode()+" Policy25 checkpoint")
    end=s.index("## Release 4.03",start)
    new=(
        "## Release 4.10 — Search-guided policy update\n\n"
        "Production preserves the 4.03 unified Lazy+Sparse search, tree reuse, "
        "adaptive clocks and pondering. The quantized 858-input/512-hidden NNUE "
        "has a newly distilled 209-action policy head; the accumulator and six "
        "bucketed value heads are unchanged. Production INT8 SHA-256: "+BT+NEW+BT+".\n\n"
        "The retained float32 file still has the previous policy, so do not "
        "re-quantize it over production INT8 weights. Browser and native assets "
        "must embed the exact same checkpoint. See [4.10 release notes](docs/releases/4.10.md).\n\n"
        "On the earlier 4.02 search, 400 paired games scored 54.13% against "
        "the prior policy and 65.38% against Claustrophobia CPU, versus 61.13% "
        "for the prior policy. This does not prove superiority at 3+2 or 5+3.\n\n")
    s=s[:start]+new+s[end:]
    s=s.replace("## Version 4.03 configuration","## Version 4.10 configuration")
    save("README.md",s)

    save("data/nnue/README.md",
        "# NNUE Checkpoint — zQuoridor 4.10\n\n"
        "Architecture: 858 sparse features, 512 hidden neurons, six bucketed "
        "value heads and 209 policy outputs.\n\n"
        "Production INT8 SHA-256: "+BT+NEW+BT+".\n\n"
        "Only the quantized policy tail changed relative to 4.03. The float32 "
        "file is a retained historical value and policy reference; it does "
        "not reproduce the active policy if re-quantized.\n\n"
        "The production browser payload, standalone HTML and native engine "
        "must use this same INT8 checksum. See [release notes](../../docs/releases/4.10.md).\n")
    save("docs/releases/4.10.md",
        "# zQuoridor 4.10 — Search-guided policy update\n\n"
        "Release: 20 October 2026. Search inherited from zQuoridor 4.03.\n\n"
        "## Changes\n\n"
        "- Keep uniform Lazy Expansion + Sparse Policy for fixed and native clock protocols.\n"
        "- Keep adaptive clock allocation, clock-correct rerooting and pondering.\n"
        "- Replace the quantized 209-action policy head, retaining the 858/512 accumulator and six value heads.\n"
        "- Rebuild native, WASM, browser data, and two standalone HTML bundles.\n\n"
        "## Integrity\n\n"
        "- Production INT8 checksum: "+BT+NEW+BT+".\n"
        "- Previous checkpoint checksum: "+BT+OLD+BT+".\n"
        "- INT8 policy start offset: "+str(OFFSET)+" bytes.\n"
        "- Original float32 checkpoint is only a prior-policy reference.\n\n"
        "## Earlier-search fixed-200ms evidence\n\n"
        "| Match | Normal | Center Rush | Overall |\n"
        "| --- | ---: | ---: | ---: |\n"
        "| New policy vs previous (H2H) | 55.50% | 52.75% | 54.13% |\n"
        "| New policy vs Claustrophobia CPU | 66.00% | 64.75% | 65.38% |\n"
        "| Previous policy vs Claustrophobia CPU | 61.50% | 60.75% | 61.13% |\n\n"
        "Each row comprises 400 games (200 per book), with colors reversed "
        "per opening. Incremental improvement confidence intervals include zero. "
        "Claustrophobia's synchronous MCTS may exceed the requested 200ms deadline. "
        "No native 3+2 or 5+3 Elo improvement is established. "
        "The earlier search differs from the unified 4.03 baseline.\n")
    d=load("docs/README.md")
    d=re.sub(r"Current production:.*\n.*\n",
        "Current production: [Release 4.10 — Search-guided policy](releases/4.10.md).\n"
        "Search retains the 4.03 implementation; quantized policy weights are updated.\n",d,count=1)
    save("docs/README.md",d)
    plan=load("docs/plan.md")
    note=("> **Current release (4.10, 20 October 2026):** 4.03 uniform Lazy+Sparse "
          "search and clock handling, with an updated 209-logit INT8 policy. "
          "Value heads are unchanged. The prior 4.02 200ms screen scored "
          "54.13% H2H and +4.25pp against Claustrophobia CPU over the original policy; "
          "the improvement interval includes zero. The 4.10 3+2/5+3 Elo "
          "is unmeasured. See [release 4.10](releases/4.10.md).\n\n")
    save("docs/plan.md",note+plan)
    a=load("gui_web/app.js")
    assert "<b>zQuoridor 4.03</b>" in a
    save("gui_web/app.js",a.replace("<b>zQuoridor 4.03</b>","<b>zQuoridor 4.10</b>"))
    c=load("src/production_config.hpp")
    save("src/production_config.hpp",c.replace("Zquoridor 4.03 uses","Zquoridor 4.10 uses").replace("Release 4.03:","Release 4.10:"))

    # Retire old one-off importer: it rejects releases after 4.02.
    brand=bytes.fromhex("41706578").decode()
    obsolete=R/("tools/promote_"+brand.lower()+"_policy25.py")
    if obsolete.exists(): subprocess.run(["git","rm","--",str(obsolete.relative_to(R))],cwd=R,check=True)
    rx=re.compile(re.escape(brand),re.I)
    def neutral(m):
        t=m.group()
        return "POLICY" if t.isupper() else "Policy" if t[0].isupper() else "policy"

    paths=subprocess.check_output(["git","ls-files","-z"],cwd=R).decode().split("\0")
    for name in paths:
        if not name: continue
        p=R/name
        if not p.is_file():continue
        b=p.read_bytes()
        if b"\0" not in b[:8192]:
            try: text=b.decode("utf-8")
            except UnicodeDecodeError: text=None
            if text and rx.search(text):
                p.write_text(rx.sub(neutral,text),encoding="utf-8")
        newname=rx.sub(neutral,name)
        if newname!=name:
            target=R/newname
            target.parent.mkdir(parents=True,exist_ok=True)
            assert not target.exists(),newname
            subprocess.run(["git","mv","--",name,newname],cwd=R,check=True)
    assert hash_(p.read_bytes())==NEW if False else hash_((R/"data/nnue/nnue_weights_int8.bin").read_bytes())==NEW
    for name in subprocess.check_output(["git","ls-files","-z"],cwd=R).decode().split("\0"):
        if not name:continue
        assert not rx.search(name),name
        p=R/name
        if p.is_file():
            b=p.read_bytes()
            if b"\0" not in b[:8192]:
                try: t=b.decode("utf-8")
                except UnicodeDecodeError: continue
                assert not rx.search(t),name
    print("PASS: 4.10 release metadata, exact policy checksum, descriptive naming",flush=True)

if __name__=="__main__":
    main()
