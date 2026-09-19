"""Binary inference transport for the original Claustrophobia Rust MCTS.

This internal worker receives exact upstream encode20 tensors and legal masks.
The parent owns search. This worker owns only the TorchScript forward pass.
"""
import os
import struct
import sys
import numpy as np
import torch


def read_exact(stream, length):
    chunks = bytearray()
    while len(chunks) < length:
        part = stream.read(length - len(chunks))
        if not part:
            raise EOFError("inference transport closed")
        chunks.extend(part)
    return chunks


def infer(model, planes, masks, device):
    x = torch.from_numpy(planes).to(device)
    legal = torch.from_numpy(masks).to(device)
    with torch.inference_mode():
        result = model(x)
        logits, value = result[0].float(), result[1].float().reshape(-1)
        if logits.shape != (len(planes), 209) or not legal.any(1).all():
            raise ValueError("invalid inference shape or empty legal action set")
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            raise ValueError("non-finite teacher output")
        p = logits.masked_fill(~legal, -torch.inf).softmax(1).cpu().numpy()
    return p.astype("<f4"), value.clamp(-1, 1).cpu().numpy().astype("<f4")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: inference_worker checkpoint.pt cpu|gpu")
    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    device = "cuda" if sys.argv[2] in ("gpu", "cuda") else "cpu"
    torch.set_num_threads(1)
    model = torch.jit.load(sys.argv[1], map_location=device).eval()
    # Initialize the selected backend before the parent's search clock starts.
    infer(model, np.zeros((1, 20, 9, 9), np.float32), np.ones((1, 209), bool), device)
    source, output = sys.stdin.buffer, sys.stdout.buffer
    output.write(b"ZQIP")
    output.flush()
    while True:
        header = source.read(4)
        if not header:
            return
        if len(header) != 4:
            raise EOFError("incomplete batch header")
        n = struct.unpack("<I", header)[0]
        if not 0 < n <= 65536:
            raise ValueError("invalid inference batch size")
        planes = np.frombuffer(read_exact(source, n * 1620 * 4), dtype="<f4").copy().reshape(n, 20, 9, 9)
        masks = np.frombuffer(read_exact(source, n * 209), dtype=np.uint8).astype(bool).reshape(n, 209)
        probabilities, values = infer(model, planes, masks, device)
        output.write(probabilities.tobytes())
        output.write(values.tobytes())
        output.flush()


if __name__ == "__main__":
    main()
