#!/usr/bin/env python3
"""Safely promote the user-provided Apex Policy25 zip to native and web production.
Does not fetch files or run matches. Run on a complete local checkout.
"""
import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

OLD = "45fa3ea7db3127a0b907a23d268b0c55c2020e1018513d54bb920cdbe134abe0"
NEW = "fe8473255fd219fab247b1a7d44be25384a6cfc5daa5c301def49109badb7f2b"
POLICY_START = 985824
SIZE = 1093668
FOLDER = "zQuoridor401_apex_ordering/"

def digest(b):
    return hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip",type=Path,help="Original zQuoridor401_apex_ordering.zip")
    ap.add_argument("--repo",type=Path,default=Path.cwd())
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--skip-bundle",action="store_true",help="Diagnostic only: regenerate web HTML before publishing")
    a=ap.parse_args()
    root=a.repo.resolve()
    with zipfile.ZipFile(a.zip) as z:
        before=z.read(FOLDER+"zq401_original_int8.bin")
        after=z.read(FOLDER+"zq401_apex_policy25_int8.bin")
        arch=json.loads(z.read(FOLDER+"zq401_apex_policy25.architecture.json"))
    if not (len(before)==len(after)==SIZE and digest(before)==OLD and digest(after)==NEW):
        raise RuntimeError("Unrecognized or damaged Apex zip")
    if before[:POLICY_START]!=after[:POLICY_START] or before[POLICY_START:]==after[POLICY_START:]:
        raise RuntimeError("Value features were modified, or policy head is unchanged")
    if arch["int8_sha256"]!=NEW or (arch["features"],arch["hidden"],arch["policy_out"])!=(858,512,209):
        raise RuntimeError("Architecture mismatch")
    nnue=root/"data"/"nnue"
    weights=nnue/"nnue_weights_int8.bin"
    browser=root/"gui_web"/"zquoridor.data"
    archpath=nnue/"nnue_weights.architecture.json"
    manifest=nnue/"release.json"
    float_weights=nnue/"nnue_weights.bin"
    for file in (weights,browser,archpath,manifest,float_weights):
        if not file.is_file():raise RuntimeError(f"Missing {file}")
    if not a.skip_bundle and not (root/"gui_web"/"build_standalone.py").exists():
        raise RuntimeError("Standalone bundler is missing")
    for file in (weights,browser):
        if digest(file.read_bytes()) not in (OLD,NEW):
            raise RuntimeError(f"Pre-existing native/browser model is unexpected: {file}")
    info=json.loads(archpath.read_text())
    if (info["features"],info["hidden"],info["policy_out"])!=(858,512,209):
        raise RuntimeError("Original architecture does not match Apex")
    release=json.loads(manifest.read_text())
    if release["version"] not in ("4.01","4.02"): raise RuntimeError("Unrecognized release")
    if a.dry_run:
        print("PASS: archive integrity, unchanged value body, old/current weights, browser, architecture")
        return
    backup=nnue/"champions"/"4.01-original-858"
    backup.mkdir(parents=True,exist_ok=True)
    old_file=backup/"nnue_weights_int8.bin"
    if old_file.exists() and digest(old_file.read_bytes())!=OLD:raise RuntimeError("Old backup differs")
    old_file.write_bytes(before)
    archive=nnue/"champions"/"4.02-apex-policy25"
    archive.mkdir(parents=True,exist_ok=True)
    (archive/"nnue_weights_int8.bin").write_bytes(after)
    (archive/"nnue_weights.architecture.json").write_text(json.dumps(arch,indent=2)+"\n")
    weights.write_bytes(after)
    browser.write_bytes(after)
    info["int8_sha256"]=NEW
    info["policy_source"]="Apex Policy25, INT8 only; float network remains original 4.01"
    info["policy_int8_only"]=True
    archpath.write_text(json.dumps(info,indent=2)+"\n")
    release.update({
      "version":"4.02","network":"858-apex-policy25","previous_release":"4.01",
      "search":"clock-aware Lazy+Sparse fixed movetime; eager+Sparse native clocks",
      "policy_teacher":"Apex Policy25 (policy head only; value network unchanged)",
      "policy_change_start_offset":POLICY_START,"int8_policy_only":True,
      "promotion_basis":"Explicit operator authorization; 40-game 200ms H2H scored 57.5%, uncertainty includes 50%; 3+2 gain not yet established",
      "baseline_200ms_h2h":{"games":40,"score_pct":57.5,"paired_bootstrap_95_pct":[47.5,67.5]}
    })
    release["weights"]["nnue_weights_int8.bin"]=NEW
    release["weights"]["nnue_weights.bin"]=digest(float_weights.read_bytes())
    limits=release.setdefault("known_limitations",[])
    note="Apex checkpoint is INT8 policy-only; the float checkpoint is still old and re-quantizing it would overwrite Apex."
    if note not in limits:limits.append(note)
    manifest.write_text(json.dumps(release,indent=2)+"\n")
    if not a.skip_bundle:
        subprocess.run([sys.executable,str(root/"gui_web"/"build_standalone.py")],cwd=root,check=True)
        for file in (root/"gui_web"/"zquoridor.html",root/"docs"/"index.html"):
            text=file.read_text()
            match=re.search(r'__QR_DATA_BYTES__\s*=\s*__qr_b64ToBytes\("([A-Za-z0-9+/=]+)"\)',text)
            if not match or digest(base64.b64decode(match.group(1)))!=NEW:
                raise RuntimeError(f"Standalone still carries old weights: {file}")
            print("PASS embedded Apex in",file)
    if digest(weights.read_bytes())!=NEW or digest(browser.read_bytes())!=NEW:
        raise RuntimeError("Native/browser checkpoint parity failed")
    print("Apex Policy25 promoted locally:",NEW)
    print("Commit weights, metadata, gui_web/zquoridor.data, gui_web/zquoridor.html and docs/index.html")

if __name__=="__main__":
    main()
