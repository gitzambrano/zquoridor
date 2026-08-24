#!/usr/bin/env python3
"""extract_wasm_from_bundle.py -- recovers zquoridor.wasm and zquoridor.data
from a built standalone bundle.

build_wasm.sh needs the Emscripten toolchain to produce zquoridor.wasm. When
that toolchain is not installed, this script recovers the same bytes from the
base64 blobs that build_standalone.py embedded in index.html. The recovered
files let build_standalone.py rebuild the bundle from the JavaScript and CSS
sources alone, so front-end work does not need Emscripten.

The recovered wasm is the one the bundle already runs. It does NOT pick up
edits to engine_wasm.cpp. Any engine change still needs a real build_wasm.sh
run.

Usage (from the repository root):

    python3 gui_web/extract_wasm_from_bundle.py [bundle.html]

Default bundle is index.html at the repository root.
"""
import base64
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
BLOBS = (("zquoridor.wasm", "__QR_WASM_BYTES__"), ("zquoridor.data", "__QR_DATA_BYTES__"))


def main():
    bundle = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "index.html"
    if not bundle.exists():
        sys.exit(f"bundle not found: {bundle}")
    html = bundle.read_text(encoding="utf-8")

    wrote = 0
    for name, var in BLOBS:
        m = re.search(re.escape(f'const {var} = __qr_b64ToBytes("') + r'([A-Za-z0-9+/=]+)"', html)
        if not m:
            print(f"{name}: not embedded in {bundle.name}, skipped")
            continue
        raw = base64.b64decode(m.group(1))
        if name.endswith(".wasm") and raw[:4] != b"\0asm":
            sys.exit(f"{name}: recovered bytes are not a wasm module")
        (HERE / name).write_bytes(raw)
        print(f"{name}: {len(raw)} bytes")
        wrote += 1
    if not wrote:
        sys.exit("nothing recovered -- is this a standalone bundle?")


if __name__ == "__main__":
    main()
