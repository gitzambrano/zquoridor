#!/usr/bin/env python3
"""gen_weights_header.py -- converte data/nnue/nnue_weights_int8.bin num
array C++ estatico (nnue_weights_data.h), pra embutir os pesos DENTRO do
binario WASM em tempo de compilacao.

Por que: a Classe A do quoridor-arena exige um unico arquivo fonte (JS),
rodando num container read-only sem rede -- nao da pra abrir o .bin do
disco em runtime (nao ha filesystem real, so o MEMFS do Emscripten, que
precisaria de --preload-file/fetch assincrono). Embutindo os bytes como
array C++ em tempo de compilacao, o loadNnueFromMemory() em
engine_bridge.cpp le direto da memoria, sem tocar em FS nenhum.

Uso:
    python3 gen_weights_header.py <entrada.bin> <saida.h>
"""
import sys


def main():
    if len(sys.argv) != 3:
        print(f"uso: {sys.argv[0]} <entrada.bin> <saida.h>", file=sys.stderr)
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "rb") as f:
        data = f.read()

    with open(out_path, "w") as f:
        f.write("// GERADO AUTOMATICAMENTE por tools/gen_weights_header.py -- nao editar a mao.\n")
        f.write(f"// Fonte: {in_path} ({len(data)} bytes)\n")
        f.write("#pragma once\n")
        f.write("#include <cstddef>\n\n")
        f.write("static const unsigned char kNnueWeightsData[] = {\n")
        line = []
        for i, b in enumerate(data):
            line.append(f"0x{b:02x}")
            if len(line) == 20:
                f.write(",".join(line) + ",\n")
                line = []
        if line:
            f.write(",".join(line) + "\n")
        f.write("};\n")
        f.write(f"static const size_t kNnueWeightsLen = {len(data)};\n")

    print(f"OK: {out_path} ({len(data)} bytes embutidos)")


if __name__ == "__main__":
    main()
