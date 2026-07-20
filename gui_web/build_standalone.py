#!/usr/bin/env python3
"""build_standalone.py -- monta um único zquoridor.html autocontido, sem
precisar de servidor HTTP: embute o WASM como base64 dentro do próprio
arquivo (mesma técnica de sempre pra distribuir side-by-side).

Uso (depois de rodar build_wasm.sh nesta pasta, o que gera zquoridor.js e
zquoridor.wasm):

    python3 build_standalone.py

Gera gui_web/zquoridor.html.
"""
import base64
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent

def main():
    wasm_path = HERE / "zquoridor.wasm"
    loader_path = HERE / "zquoridor.js"
    html_path = HERE / "style.html"
    app_path = HERE / "app.js"
    out_path = HERE / "zquoridor.html"
    root_out_path = HERE.parent / "index.html"

    for p in (wasm_path, loader_path, html_path, app_path):
        if not p.exists():
            sys.exit(f"faltando {p} -- rode ./build_wasm.sh primeiro")

    wasm_b64 = base64.b64encode(wasm_path.read_bytes()).decode("ascii")
    loader_js = loader_path.read_text(encoding="utf-8")
    app_js = app_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")

    # standalone precisa passar os bytes do wasm direto pro módulo em vez
    # de deixar o Emscripten buscar zquoridor.wasm via fetch/XHR (que exige
    # servidor HTTP -- não funciona em file://).
    app_js_standalone = app_js.replace(
        "ZquoridorModule().then((Module) => {",
        "ZquoridorModule({ wasmBinary: __QR_WASM_BYTES__ }).then((Module) => {",
    )
    if app_js_standalone == app_js:
        sys.exit("não encontrei o padrão ZquoridorModule().then(...) em app.js pra adaptar")

    # remove as duas tags <script src="..."> e injeta loader+app inline
    html_no_scripts = re.sub(
        r'\s*<script src="zquoridor\.js"></script>\s*<script src="app\.js"></script>\s*',
        "\n<!--INLINE_SCRIPTS-->\n",
        html,
    )
    if "<!--INLINE_SCRIPTS-->" not in html_no_scripts:
        sys.exit("não encontrei as tags <script src=zquoridor.js/app.js> em style.html")

    b2b = (
        "function __qr_b64ToBytes(b64) {\n"
        "    const bin = atob(b64);\n"
        "    const bytes = new Uint8Array(bin.length);\n"
        "    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);\n"
        "    return bytes;\n"
        "}\n"
        f'const __QR_WASM_BYTES__ = __qr_b64ToBytes("{wasm_b64}");\n\n'
    )

    inline = (
        "<script>\n"
        "// --- WASM embutido em base64 (build standalone, sem servidor HTTP) ---\n"
        f"{b2b}"
        f"{loader_js}\n"
        "</script>\n"
        "<script>\n"
        f"{app_js_standalone}\n"
        "</script>\n"
    )

    out = html_no_scripts.replace("<!--INLINE_SCRIPTS-->", inline)
    out_path.write_text(out, encoding="utf-8")
    root_out_path.write_text(out, encoding="utf-8")
    print(f"OK: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    print(f"OK: {root_out_path} ({root_out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
