#!/usr/bin/env python3
"""Compare fixed-node trees for natural sibling wall moves."""
from __future__ import annotations

import concurrent.futures
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
    "output": OUT / "edge_acc_natural_tree_parity.json",
    "source": ROOT / "tests" / "test_edge_acc_natural_alias_tree.cpp",
    "variants": ["off", "v3", "v3_node", "v3_node_dense",
                 "v3_node_dense_bfs", "delta_dense_only"],
    "networks": ["504", "858"],
    "node_budgets": [100000, 300000],
    "timeout_seconds": 180,
    "workers": 4,
    "base_flags": ["-O3", "-DNDEBUG", "-std=c++17", "-pthread", "-Isrc"],
    "weights": {
        "504": Path(r"C:\Projetos\Zquoridor\data\nnue\nnue_weights_int8.bin"),
        "858": Path(r"C:\Projetos\Zquoridor\results\experiments\contact_soup_tri_512\soup_tri_champion\student_int8.bin"),
    },
    "variant_macros": {
        "off": {},
        "v3": {"ZQ_EXP_EDGE_ACC_PATH_CONTEXT": 1},
        "v3_node": {"ZQ_EXP_EDGE_ACC_NODE_CONTEXT": 1},
        "v3_node_dense": {"ZQ_EXP_EDGE_ACC_NODE_CONTEXT": 1,
                          "ZQ_EXP_EDGE_FEATURE_DELTA": 1,
                          "ZQ_EXP_EDGE_DENSE_DELTA": 1},
        "v3_node_dense_bfs": {"ZQ_EXP_EDGE_ACC_NODE_CONTEXT": 1,
                              "ZQ_EXP_EDGE_FEATURE_DELTA": 1,
                              "ZQ_EXP_EDGE_DENSE_DELTA": 1,
                              "ZQ_EXP_EDGE_BFS_REPLAY": 1},
        "delta_dense_only": {"ZQ_EXP_EDGE_FEATURE_DELTA": 1,
                              "ZQ_EXP_EDGE_DENSE_DELTA": 1,
                              "ZQ_EXP_EDGE_BFS_REPLAY": 1},
    },
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_edge_acc_campaign as campaign

ROW_RE = re.compile(
    r"EDGE_NATURAL_TREE variant=(?P<variant>\S+) nodes=(?P<nodes>\d+) "
    r"best=(?P<best>\d+) pool=(?P<pool>\d+) simulations=(?P<simulations>\d+) "
    r"expanded=(?P<expanded>\d+) rootN=(?P<root_n>-?\d+) "
    r"fp=(?P<fingerprint>[0-9a-f]+) h63=(?P<h63>\d+) v63=(?P<v63>\d+) "
    r"h63N=(?P<h63_n>\d+) v63N=(?P<v63_n>\d+) reused=(?P<reused>\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one_run(network: str, variant: str, nodes: int) -> dict:
    binary = OUT / "bin" / "natural_tree" / f"{network}_{variant}_{nodes}.exe"
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [CONFIG["compiler"], *CONFIG["base_flags"], *campaign.network_flags(network)]
    command.extend(f"-D{name}={value}" for name, value in CONFIG["variant_macros"][variant].items())
    command.append(f'-DEDGE_NATURAL_VARIANT="{variant}"')
    command.extend([str(CONFIG["source"]), "-o", str(binary)])
    build = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if build.returncode:
        return {"network": network, "variant": variant, "nodes": nodes,
                "status": "BUILD_FAIL", "stderr": build.stderr[-8000:], "command": command}
    try:
        run = subprocess.run([str(binary), str(CONFIG["weights"][network]), str(nodes)],
                             cwd=ROOT, text=True, capture_output=True,
                             timeout=CONFIG["timeout_seconds"])
    except subprocess.TimeoutExpired as error:
        return {"network": network, "variant": variant, "nodes": nodes,
                "status": "TIMEOUT", "timeout_seconds": CONFIG["timeout_seconds"],
                "stdout": error.stdout or "", "stderr": error.stderr or ""}
    if run.returncode:
        return {"network": network, "variant": variant, "nodes": nodes,
                "status": "RUN_FAIL", "returncode": run.returncode,
                "stdout": run.stdout, "stderr": run.stderr}
    match = next((ROW_RE.fullmatch(line.strip()) for line in run.stdout.splitlines()
                  if line.startswith("EDGE_NATURAL_TREE ")), None)
    if not match:
        return {"network": network, "variant": variant, "nodes": nodes,
                "status": "PARSE_FAIL", "stdout": run.stdout, "stderr": run.stderr}
    result = match.groupdict()
    result.update({"network": network, "status": "RAN", "binary_sha256": sha256(binary)})
    for key in ("nodes", "best", "pool", "simulations", "expanded", "root_n", "h63", "v63",
                "h63_n", "v63_n", "reused"):
        result[key] = int(result[key])
    return result


def main() -> int:
    source_hashes = campaign.source_hash_map()
    manifest = {
        "schema_version": 1,
        "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_sha256": source_hashes,
        "test_source_sha256": sha256(CONFIG["source"]),
        "weights": {network: {"path": str(CONFIG["weights"][network]),
                              "sha256": sha256(CONFIG["weights"][network])}
                    for network in CONFIG["networks"]},
        "variants": CONFIG["variants"],
        "node_budgets": CONFIG["node_budgets"],
        "timeout_seconds": CONFIG["timeout_seconds"],
        "workers": CONFIG["workers"],
        "results": [],
        "comparisons": [],
        "passed": False,
    }
    jobs = [(network, variant, nodes) for network in CONFIG["networks"]
            for variant in CONFIG["variants"] for nodes in CONFIG["node_budgets"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["workers"]) as pool:
        futures = {pool.submit(one_run, *job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            manifest["results"].append(row)
            print(f"{row['status']} {row['network']}/{row['variant']} nodes={row['nodes']}", flush=True)

    checked = ("best", "pool", "simulations", "expanded", "root_n", "fingerprint", "h63_n", "v63_n")
    for network in CONFIG["networks"]:
        for nodes in CONFIG["node_budgets"]:
            rows = {row["variant"]: row for row in manifest["results"]
                    if row["network"] == network and row["nodes"] == nodes}
            baseline = rows.get("off", {})
            for variant in CONFIG["variants"]:
                candidate = rows.get(variant, {})
                mismatches = {}
                if candidate.get("status") != "RAN" or baseline.get("status") != "RAN":
                    status = "INCOMPLETE"
                elif (baseline.get("h63_n", 0) <= 0 or baseline.get("v63_n", 0) <= 0
                      or candidate.get("h63_n", 0) <= 0 or candidate.get("v63_n", 0) <= 0):
                    status = "INCOMPLETE"
                    mismatches["branch_coverage"] = {
                        "off": {"h63N": baseline.get("h63_n"), "v63N": baseline.get("v63_n")},
                        "candidate": {"h63N": candidate.get("h63_n"), "v63N": candidate.get("v63_n")}}
                else:
                    mismatches = {field: {"off": baseline.get(field), "candidate": candidate.get(field)}
                                  for field in checked if baseline.get(field) != candidate.get(field)}
                    if candidate.get("h63") != 1 or candidate.get("v63") != 1 or candidate.get("reused") != 0:
                        mismatches["setup"] = {"h63": candidate.get("h63"), "v63": candidate.get("v63"),
                                                "reused": candidate.get("reused")}
                    status = "PASS" if not mismatches else "FAIL"
                manifest["comparisons"].append({"network": network, "variant": variant,
                    "nodes": nodes, "status": status, "mismatches": mismatches})
    if campaign.source_hash_map() != source_hashes:
        raise RuntimeError("source headers changed while the natural tree gate was running")
    manifest["finished_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["passed"] = all(row["status"] == "PASS" for row in manifest["comparisons"])
    CONFIG["output"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["output"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {CONFIG['output']}")
    print(f"EDGE_NATURAL_TREE_PARITY {'PASS' if manifest['passed'] else 'FAIL/INCOMPLETE'} "
          f"comparisons={len(manifest['comparisons'])}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
