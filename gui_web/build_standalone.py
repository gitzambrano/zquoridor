#!/usr/bin/env python3
"""build_standalone.py -- monta um único zquoridor.html autocontido, sem
precisar de servidor HTTP: embute o WASM como base64 dentro do próprio
arquivo (mesma técnica de sempre pra distribuir side-by-side). Também:

  - roda o contrast gate (tools/gui/contrast_check.py) antes de empacotar;
  - embute as fontes Google (Cinzel / JetBrains Mono) como WOFF2 base64,
    com cache em fonts_cache/ -- o bundle standalone nunca depende de rede.

Uso (depois de rodar build_wasm.sh nesta pasta, o que gera zquoridor.js e
zquoridor.wasm):

    python3 build_standalone.py

Gera gui_web/zquoridor.html.
"""
import base64
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
FONT_CACHE = HERE / "fonts_cache"

GOOGLE_FONTS_CSS = ("https://fonts.googleapis.com/css2"
                    "?family=Cinzel:wght@600;700"
                    "&family=JetBrains+Mono:wght@300;400;500;600;700"
                    "&display=swap")
# A modern Chrome UA makes the css2 endpoint answer with woff2 sources.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def inline_fonts(html):
    """Replaces the Google Fonts <link> with inline @font-face rules whose
    url()s are base64 data URIs. Uses fonts_cache/ as an offline-safe store;
    on total failure keeps the <link> and warns (dev builds still work)."""
    link_re = re.compile(r'<link[^>]*fonts\.googleapis\.com[^>]*>\s*')
    if not link_re.search(html):
        return html, False
    try:
        FONT_CACHE.mkdir(exist_ok=True)
        css = FONT_CACHE / "fonts.css"
        if css.exists():
            css_text = css.read_text(encoding="utf-8")
        else:
            req = urllib.request.Request(GOOGLE_FONTS_CSS, headers={"User-Agent": UA})
            css_text = urllib.request.urlopen(req, timeout=20).read().decode("utf-8")
            css.write_text(css_text, encoding="utf-8")
        font_urls = set(re.findall(r"url\((https://[^)]+\.woff2)\)", css_text))
        for u in sorted(font_urls):
            name = hashlib.sha1(u.encode()).hexdigest()[:16] + ".woff2"
            p = FONT_CACHE / name
            if not p.exists():
                req = urllib.request.Request(u, headers={"User-Agent": UA})
                p.write_bytes(urllib.request.urlopen(req, timeout=30).read())
                print(f"    cached {name} ({p.stat().st_size // 1024} KB)")
            data = base64.b64encode(p.read_bytes()).decode("ascii")
            css_text = css_text.replace(u, f"data:font/woff2;base64,{data}")
        # drop unicode-range subsetting? keep it -- browsers honour it and the
        # data URIs stay valid. Inline everything into a single <style>.
        html = link_re.sub("", html)
        style = "<style>\n/* inlined fonts (standalone: no network needed) */\n" + css_text + "\n</style>\n"
        return html.replace("</head>", style + "</head>", 1), True
    except Exception as e:
        print(f"[AVISO] fontes não embutidas ({e}) -- mantendo <link> de rede")
        return html, False


def main():
    wasm_path = HERE / "zquoridor.wasm"
    data_path = HERE / "zquoridor.data"
    loader_path = HERE / "zquoridor.js"
    html_path = HERE / "style.html"
    app_path = HERE / "app.js"
    board_path = HERE / "board.js"
    out_path = HERE / "zquoridor.html"
    root_out_path = HERE.parent / "index.html"

    for p in (wasm_path, loader_path, html_path, app_path, board_path):
        if not p.exists():
            sys.exit(f"faltando {p} -- rode ./build_wasm.sh primeiro")

    # P9 gate: token contrast for every board x UI combination. A regression
    # fails the bundle build.
    import subprocess
    r = subprocess.run([sys.executable, str(HERE.parent / "tools" / "gui" / "contrast_check.py")])
    if r.returncode != 0:
        sys.exit("contrast check failed -- corrige os tokens antes de empacotar")

    wasm_b64 = base64.b64encode(wasm_path.read_bytes()).decode("ascii")
    data_b64 = base64.b64encode(data_path.read_bytes()).decode("ascii") if data_path.exists() else None
    loader_js = loader_path.read_text(encoding="utf-8")
    app_js = app_path.read_text(encoding="utf-8")
    board_js = board_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")

    html, fonts_inlined = inline_fonts(html)
    if fonts_inlined:
        print("    fontes embutidas (WOFF2 base64)")

    # standalone precisa passar os bytes do wasm direto pro módulo em vez
    # de deixar o Emscripten buscar zquoridor.wasm via fetch/XHR (que exige
    # servidor HTTP -- não funciona em file://).
    module_args = ["wasmBinary: __QR_WASM_BYTES__"]
    if data_b64:
        module_args.append("getPreloadedPackage: (remoteName, remotePackageSize) => __QR_DATA_BYTES__")
    
    app_js_standalone = app_js.replace(
        "ZquoridorModule().then((Module) => {",
        f"ZquoridorModule({{ {', '.join(module_args)} }}).then((Module) => {{",
    )
    if app_js_standalone == app_js:
        sys.exit("não encontrei o padrão ZquoridorModule().then(...) em app.js pra adaptar")

    # remove as duas tags <script src="..."> e injeta loader+app inline
    html_no_scripts = re.sub(
        r'\s*(<script src="board\.js"></script>\s*)?<script src="zquoridor\.js"></script>\s*<script src="app\.js"></script>\s*',
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
        f'const __QR_WASM_BYTES__ = __qr_b64ToBytes("{wasm_b64}");\n'
    )
    if data_b64:
        b2b += f'const __QR_DATA_BYTES__ = __qr_b64ToBytes("{data_b64}").buffer;\n'
    b2b += "\n"

    inline = (
        "<script>\n"
        "// --- WASM/DATA embutidos em base64 (build standalone, sem servidor HTTP) ---\n"
        f"{b2b}"
        f"{loader_js}\n{board_js}\n"
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
