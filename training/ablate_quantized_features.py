#!/usr/bin/env python3
"""Zero selected first-layer feature rows in a quantized NNUE file."""
from __future__ import annotations
import argparse, hashlib, json, struct
from pathlib import Path

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--features", type=int, required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--range", dest="ranges", action="append", required=True,
                   help="half-open feature range START:END")
    a=p.parse_args()
    src=Path(a.input); dst=Path(a.output)
    raw=bytearray(src.read_bytes())
    if len(raw) < 8:
        raise SystemExit("file too small")
    qa,qb=struct.unpack_from("<ii", raw, 0)
    row_bytes=a.hidden*2
    w1_bytes=a.features*row_bytes
    if len(raw) < 8+w1_bytes:
        raise SystemExit("file is smaller than declared first layer")
    touched=[]
    for item in a.ranges:
        lo_s,hi_s=item.split(":",1)
        lo,hi=int(lo_s),int(hi_s)
        if not (0 <= lo < hi <= a.features):
            raise SystemExit(f"invalid range {item}")
        for feat in range(lo,hi):
            off=8+feat*row_bytes
            raw[off:off+row_bytes]=b"\x00"*row_bytes
        touched.append([lo,hi])
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_bytes(raw)
    meta={
        "schema":"zquoridor.nnue_feature_ablation.v1",
        "input":str(src),
        "output":str(dst),
        "features":a.features,
        "hidden":a.hidden,
        "qa":qa,"qb":qb,
        "zeroed_ranges":touched,
        "input_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
        "output_sha256":hashlib.sha256(raw).hexdigest(),
    }
    dst.with_suffix(dst.suffix+".json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
