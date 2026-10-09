#!/usr/bin/env python3
"""Compare fixed-node EdgeAcc trees after a valid BFS-key alias history."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmarks" / "edge-parity-local"
CONFIG = {
    "compiler": "g++",
    "output": OUT / "edge_acc_alias_tree_parity.json",
    "source": ROOT / "tests" / "test_edge_acc_cache_path_alias_tree.cpp",
    "variants": ["off", "v1", "v3", "v3_node", "hybrid_sparse", "hybrid_dense",
                 "delta_dense_only", "v3_node_dense", "hybrid_dense_bfs",
                 "v3_node_dense_bfs"],
    "networks": ["504", "858"],
    "node_budgets": [1000, 4000],
    "base_flags": ["-O3", "-DNDEBUG", "-std=c++17", "-pthread", "-Isrc"],
    "weights": {
        "504": Path(r"C:\Projetos\Zquoridor\data\nnue\nnue_weights_int8.bin"),
        "858": Path(r"C:\Projetos\Zquoridor\results\experiments\contact_soup_tri_512\soup_tri_champion\student_int8.bin"),
    },
    "variant_macros": {
        "off": {},
        "v1": {"ZQ_EXP_EDGE_ACC_CACHE": 1},
        "v3": {"ZQ_EXP_EDGE_ACC_PATH_CONTEXT": 1},
        "v3_node": {"ZQ_EXP_EDGE_ACC_NODE_CONTEXT": 1},
        "hybrid_sparse": {"ZQ_EXP_EDGE_ACC_CACHE": 1, "ZQ_EXP_EDGE_FEATURE_DELTA": 1},
        "hybrid_dense": {"ZQ_EXP_EDGE_ACC_CACHE": 1, "ZQ_EXP_EDGE_FEATURE_DELTA": 1,
                          "ZQ_EXP_EDGE_DENSE_DELTA": 1},
        "delta_dense_only": {"ZQ_EXP_EDGE_FEATURE_DELTA": 1,
                              "ZQ_EXP_EDGE_DENSE_DELTA": 1,
                              "ZQ_EXP_EDGE_BFS_REPLAY": 1},
        "v3_node_dense": {"ZQ_EXP_EDGE_ACC_NODE_CONTEXT": 1,
                          "ZQ_EXP_EDGE_FEATURE_DELTA": 1,
                          "ZQ_EXP_EDGE_DENSE_DELTA": 1},
        "hybrid_dense_bfs": {"ZQ_EXP_EDGE_ACC_CACHE": 1,
                             "ZQ_EXP_EDGE_FEATURE_DELTA": 1,
                             "ZQ_EXP_EDGE_DENSE_DELTA": 1,
                             "ZQ_EXP_EDGE_BFS_REPLAY": 1},
        "v3_node_dense_bfs": {"ZQ_EXP_EDGE_ACC_NODE_CONTEXT": 1,
                              "ZQ_EXP_EDGE_FEATURE_DELTA": 1,
                              "ZQ_EXP_EDGE_DENSE_DELTA": 1,
                              "ZQ_EXP_EDGE_BFS_REPLAY": 1},
    },
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_edge_acc_campaign as campaign

ROW_RE = re.compile(
    r"EDGE_ALIAS_TREE variant=(?P<variant>\S+) phase=(?P<phase>\S+) "
    r"nodes=(?P<nodes>\d+) best=(?P<best>\d+) pool=(?P<pool>\d+) "
    r"simulations=(?P<simulations>\d+) expanded=(?P<expanded>\d+) "
    r"rootN=(?P<root_n>-?\d+) fp=(?P<fingerprint>[0-9a-f]+) "
    r"reused=(?P<reused>\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_binary(network: str, variant: str) -> Path:
    binary = OUT / "bin" / "alias_tree" / f"{network}_{variant}.exe"
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [CONFIG["compiler"], *CONFIG["base_flags"],
               *campaign.network_flags(network)]
    command.extend(f"-D{name}={value}" for name, value in CONFIG["variant_macros"][variant].items())
    command.append(f'-DEDGE_ALIAS_VARIANT="{variant}"')
    command.extend([str(CONFIG["source"]), "-o", str(binary)])
    subprocess.run(command, cwd=ROOT, check=True)
    return binary


def parse_output(stdout: str, network: str, expected_variant: str) -> list[dict]:
    rows = []
    for line in stdout.splitlines():
        match = ROW_RE.fullmatch(line.strip())
        if not match:
            continue
        row = match.groupdict()
        if row["variant"] != expected_variant:
            raise RuntimeError(f"variant label mismatch in {network}: {row['variant']}")
        for key in ("nodes", "best", "pool", "simulations", "expanded", "root_n", "reused"):
            row[key] = int(row[key])
        row["network"] = network
        rows.append(row)
    expected = len(CONFIG["node_budgets"]) * 2
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} result rows for {network}/{expected_variant}, got {len(rows)}")
    return rows


def main() -> int:
    source_hashes = campaign.source_hash_map()
    manifest = {"source_sha256": source_hashes,
                "test_source_sha256": sha256(CONFIG["source"]),
                "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "weights": {},
                "variants": CONFIG["variants"], "node_budgets": CONFIG["node_budgets"],
                "results": [], "comparisons": [], "passed": False}
    by_identity = {}
    for network in CONFIG["networks"]:
        weights = CONFIG["weights"][network]
        if not weights.is_file():
            raise FileNotFoundError(f"missing {network} weights: {weights}")
        manifest["weights"][network] = {"path": str(weights), "sha256": sha256(weights)}
        for variant in CONFIG["variants"]:
            binary = build_binary(network, variant)
            run = subprocess.run([str(binary), str(weights)], cwd=ROOT, check=True,
                                 text=True, capture_output=True)
            rows = parse_output(run.stdout, network, variant)
            manifest["results"].extend(rows)
            for row in rows:
                key = (network, row["nodes"], row["phase"])
                by_identity.setdefault(key, {})[variant] = row
            print(f"RAN {network}/{variant} rows={len(rows)}", flush=True)

    checked = ("best", "pool", "simulations", "expanded", "root_n", "fingerprint", "reused")
    for (network, nodes, phase), variants in sorted(by_identity.items()):
        baseline = variants.get("off")
        if not baseline:
            raise RuntimeError(f"missing OFF baseline for {network}/{nodes}/{phase}")
        for variant, candidate in variants.items():
            mismatches = {field: {"off": baseline[field], "candidate": candidate[field]}
                          for field in checked if candidate[field] != baseline[field]}
            result = {"network": network, "nodes": nodes, "phase": phase,
                      "variant": variant, "status": "PASS" if not mismatches else "FAIL",
                      "mismatches": mismatches}
            manifest["comparisons"].append(result)
            if mismatches:
                print(f"FAIL {network}/{variant} nodes={nodes} phase={phase} {mismatches}",
                      file=sys.stderr, flush=True)
    manifest["passed"] = all(row["status"] == "PASS" for row in manifest["comparisons"])
    if campaign.source_hash_map() != source_hashes:
        raise RuntimeError("source headers changed while the alias tree gate was running")
    manifest["finished_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    CONFIG["output"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["output"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {CONFIG['output']}")
    print(f"EDGE_ALIAS_TREE_PARITY {'PASS' if manifest['passed'] else 'FAIL'} "
          f"comparisons={len(manifest['comparisons'])}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
