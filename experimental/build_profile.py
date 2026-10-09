#!/usr/bin/env python3
"""Build an archived network and search profile without changing production."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "profile": "858_delta_dense_only",
    "registry": ROOT / "experimental" / "profiles.json",
    "output": ROOT / "experimental" / "artifacts" / "bin",
    "compiler": "g++",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=CONFIG["profile"])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    profiles = json.loads(CONFIG["registry"].read_text(encoding="utf-8"))["profiles"]
    if args.list:
        print("\n".join(sorted(profiles)))
        return
    profile = profiles[args.profile]
    weights = ROOT / profile["weights"]
    if hashlib.sha256(weights.read_bytes()).hexdigest() != profile["weights_sha256"]:
        raise RuntimeError("The archived profile weights do not match the registry.")
    source = ROOT / profile["source"]
    output = CONFIG["output"] / f"{args.profile}.exe"
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = [f"-D{key}={value}" for key, value in profile["defines"].items()]
    command = [CONFIG["compiler"], "-O3", "-DNDEBUG", "-std=c++17", "-pthread",
               *flags, "-I", str(source / "src"),
               str(source / "tools/external/zquoridor_uci.cpp"), "-o", str(output)]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Executable: {output}")
    print(f"Weights: {weights}")


if __name__ == "__main__":
    main()
