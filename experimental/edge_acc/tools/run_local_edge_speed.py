"""Measure isolated deterministic large-tree throughput for EdgeAcc variants."""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "out_dir": ROOT / "results" / "benchmarks" / "edge-parity-local",
    "parity_manifest": ROOT / "results" / "benchmarks" / "edge-parity-local" / "parity_manifest.json",
    "alias_gate": ROOT / "results" / "benchmarks" / "edge-parity-local" / "edge_acc_alias_tree_parity.json",
    "natural_gate": ROOT / "results" / "benchmarks" / "edge-parity-local" / "edge_acc_natural_tree_parity.json",
    "variants": ["off", "v1", "v2", "v3", "hybrid_sparse", "hybrid_dense",
                 "v3_node", "v3_node_dense", "hybrid_dense_bfs",
                 "v3_node_dense_bfs", "delta_dense_only"],
    "networks": ["n504", "contact858"],
    "finalist_extra_repeats": 2,
    "order_seed": 20261006,
    "weight_keys": {"n504": "n504", "contact858": "contact858"},
}
ROW = re.compile(r"(\w+)=([^ ]+)")


def timing_role(variant: str, large_status: str,
               alias_status: str, natural_status: str) -> str:
    """Classify performance samples without allowing failed variants into games."""
    if variant == "v1":
        return "speed_reference"
    if large_status in ("PASS", "BASELINE") and alias_status == "PASS" and natural_status == "PASS":
        return "candidate"
    return "reference_only"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_expanded(text: str) -> tuple[int, int]:
    total = cases = 0
    for line in text.splitlines():
        if line.startswith("CASE ") or line.startswith("REUSE "):
            row = dict(ROW.findall(line))
            if "expanded" not in row:
                raise RuntimeError(f"expanded field missing: {line}")
            total += int(row["expanded"])
            cases += 1
    if cases != 10:
        raise RuntimeError(f"expected 10 deterministic checkpoints, received {cases}")
    return total, cases


def normalized_hash_map(value: dict) -> dict[str, str]:
    return {str(k).replace("\\", "/").lower(): str(v).lower() for k, v in value.items()}


def alias_status_for(alias: dict, net: str, variant: str) -> str:
    network = "504" if net == "n504" else "858"
    rows = [r for r in alias.get("comparisons", [])
            if r.get("network") == network and r.get("variant") == variant]
    expected = {(phase, budget) for phase in ("warm", "alias") for budget in (1000, 4000)}
    actual = {(r.get("phase"), int(r.get("nodes", -1))) for r in rows}
    if actual != expected:
        return "NOT_TESTED"
    return "PASS" if all(r.get("status") == "PASS" and not r.get("mismatches") for r in rows) else "FAIL"


def natural_status_for(natural: dict, net: str, variant: str) -> str:
    network = "504" if net == "n504" else "858"
    rows = [r for r in natural.get("comparisons", [])
            if r.get("network") in (network, net) and r.get("variant") == variant]
    expected = {100000, 300000}
    actual = {int(r.get("nodes", -1)) for r in rows}
    if actual != expected or len(rows) != 2:
        return "NOT_TESTED"
    if any(row.get("status") != "PASS" or row.get("mismatches") for row in rows):
        return "FAIL" if all(row.get("status") != "INCOMPLETE" for row in rows) else "INCOMPLETE"
    return "PASS"


def sample(net: str, variant: str, repeat: int, order_index: int,
           parity: dict, alias: dict, natural: dict, result: dict) -> dict:
    exe = CONFIG["out_dir"] / "bin" / net / f"{variant}_large.exe"
    weights = Path(parity["weights"][net]["path"])
    raw = CONFIG["out_dir"] / "raw" / "speed" / net / f"r{repeat}_{variant}.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    proc = subprocess.run([str(exe), str(weights)], cwd=ROOT, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    raw.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"speed sample failed {net}/{variant}: {proc.stderr[-8000:]}")
    expanded, cases = parse_expanded(proc.stdout)
    alias_status = alias_status_for(alias, net, variant)
    natural_status = natural_status_for(natural, net, variant)
    return {
        "network": net,
        "variant": variant,
        "repeat": repeat,
        "phase": "pilot" if repeat == 0 else "finalist_repeat",
        "order_index": order_index,
        "seconds": seconds,
        "expanded_nodes": expanded,
        "nps": expanded / seconds,
        "checkpoints": cases,
        "raw": str(raw),
        "raw_sha256": sha256(raw),
        "binary": str(exe),
        "binary_sha256": sha256(exe),
        "parity_status": result.get("comparison_to_off", {}).get("status", "BASELINE"),
        "required_large_gate": result.get("comparison_to_off", {}).get("status", "BASELINE"),
        "additional_alias_gate": alias_status,
        "natural_gate": natural_status,
        "role": timing_role(variant, result.get("comparison_to_off", {}).get("status", "BASELINE"),
                            alias_status, natural_status),
    }


def main() -> int:
    parity = json.loads(CONFIG["parity_manifest"].read_text(encoding="utf-8"))
    alias = json.loads(CONFIG["alias_gate"].read_text(encoding="utf-8"))
    natural = json.loads(CONFIG["natural_gate"].read_text(encoding="utf-8"))
    alias_sources = alias.get("source_sha256")
    if not isinstance(alias_sources, dict) or normalized_hash_map(alias_sources) != normalized_hash_map(parity.get("source_sha256", {})):
        raise RuntimeError("alias gate headers do not match large-tree parity manifest")
    parity_weights = {"504": parity["weights"]["n504"]["sha256"].lower(),
                      "858": parity["weights"]["contact858"]["sha256"].lower()}
    alias_weights = {k: v["sha256"].lower() for k, v in alias.get("weights", {}).items()}
    if alias_weights != parity_weights:
        raise RuntimeError("alias gate weights do not match large-tree parity manifest")
    natural_sources = natural.get("source_sha256")
    if not isinstance(natural_sources, dict) or normalized_hash_map(natural_sources) != normalized_hash_map(parity.get("source_sha256", {})):
        raise RuntimeError("natural gate headers do not match large-tree parity manifest")
    natural_weights = {k: v["sha256"].lower() for k, v in natural.get("weights", {}).items()}
    if natural_weights != parity_weights:
        raise RuntimeError("natural gate weights do not match large-tree parity manifest")
    output_path = CONFIG["out_dir"] / "isolated_speed.json"
    variants = CONFIG["variants"]
    for net in CONFIG["networks"]:
        for variant in variants:
            result = parity["large_tree"].get(f"{net}/{variant}")
            if result is None:
                raise RuntimeError(f"missing parity result {net}/{variant}")
            status = result.get("comparison_to_off", {}).get("status", "BASELINE")
            if variant != "off" and status != "PASS" and variant != "v2":
                raise RuntimeError(f"refusing speed sample without completed large-tree comparison: {net}/{variant}={status}")
            binary = CONFIG["out_dir"] / "bin" / net / f"{variant}_large.exe"
            if not binary.is_file():
                raise FileNotFoundError(binary)

    rng = random.Random(CONFIG["order_seed"])
    results = {
        "schema_version": 1,
        "started_utc": utc_now(),
        "execution": {"workers": 1, "parallelism": "sequential single-process samples",
                      "cpu_count": __import__("os").cpu_count(),
                      "order_seed": CONFIG["order_seed"],
                      "source_sha256": parity["source_sha256"],
                      "main_source_sha256": parity.get("main_source_sha256"),
                      "alias_gate_path": str(CONFIG["alias_gate"]),
                      "alias_gate_source_sha256": alias_sources,
                      "alias_gate_test_source_sha256": alias.get("test_source_sha256"),
                      "alias_gate_finished_utc": alias.get("finished_utc"),
                      "natural_gate_path": str(CONFIG["natural_gate"]),
                      "natural_gate_source_sha256": natural_sources,
                      "natural_gate_finished_utc": natural.get("finished_utc"),
                      "weights": parity["weights"],
                      "flags_by_build": {},
                      "definition": "sum of expanded across all ten fixed-node checkpoints divided by wall seconds"},
        "pilot_variants": variants,
        "finalists": {},
        "samples": [],
        "summary": {},
    }
    for build in parity.get("builds", []):
        if build.get("kind") == "large":
            results["execution"]["flags_by_build"][f"{build['network']}/{build['variant']}"] = build["command"]

    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("execution", {}).get("source_sha256") != parity.get("source_sha256"):
            raise RuntimeError("existing speed samples use different source hashes")
        if previous.get("execution", {}).get("weights") != parity.get("weights"):
            raise RuntimeError("existing speed samples use different weight hashes")
        results = previous
        results["execution"].update({k: v for k, v in {
            "alias_gate_path": str(CONFIG["alias_gate"]),
            "alias_gate_source_sha256": alias_sources,
            "alias_gate_test_source_sha256": alias.get("test_source_sha256"),
            "alias_gate_finished_utc": alias.get("finished_utc"),
            "natural_gate_path": str(CONFIG["natural_gate"]),
            "natural_gate_source_sha256": natural_sources,
            "natural_gate_finished_utc": natural.get("finished_utc"),
        }.items()})
        results.setdefault("samples", [])
        results["resumed_utc"] = utc_now()
        for row in results["samples"]:
            p = parity["large_tree"][f"{row['network']}/{row['variant']}"]
            row["required_large_gate"] = p.get("comparison_to_off", {}).get("status", "BASELINE")
            row["additional_alias_gate"] = alias_status_for(alias, row["network"], row["variant"])
            row["natural_gate"] = natural_status_for(natural, row["network"], row["variant"])
            row["role"] = timing_role(row["variant"], row["required_large_gate"],
                                       row["additional_alias_gate"], row["natural_gate"])

    order_index = max((r["order_index"] for r in results["samples"]), default=-1) + 1
    # Interleave the networks for each seeded pilot variant order.
    pilot_order = variants[:]
    rng.shuffle(pilot_order)
    pilot_tasks = []
    for variant in pilot_order:
        nets = CONFIG["networks"][:]
        if rng.randrange(2):
            nets.reverse()
        for net in nets:
            pilot_tasks.append((net, variant))
    existing_keys = {(r["network"], r["variant"], r["repeat"]) for r in results["samples"]}
    for net, variant in pilot_tasks:
        if (net, variant, 0) in existing_keys:
            continue
        row = sample(net, variant, 0, order_index, parity, alias,
                     natural,
                     parity["large_tree"][f"{net}/{variant}"])
        results["samples"].append(row)
        existing_keys.add((net, variant, 0))
        order_index += 1
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"SPEED pilot net={net} variant={variant} expanded={row['expanded_nodes']} "
              f"seconds={row['seconds']:.2f} nps={row['nps']:.1f} "
              f"parity={row['parity_status']}", flush=True)

    pilot_map = {(r["network"], r["variant"]): r["nps"] for r in results["samples"] if r["repeat"] == 0}
    finalists = {}
    for net in CONFIG["networks"]:
        correct = [v for v in variants if v not in ("off", "v2") and all(
            parity["large_tree"][f"{other}/{v}"].get("comparison_to_off", {}).get("status") == "PASS" and
            alias_status_for(alias, other, v) == "PASS" and
            natural_status_for(natural, other, v) == "PASS" for other in CONFIG["networks"])]
        top = sorted(correct, key=lambda v: pilot_map[(net, v)], reverse=True)[:2]
        finalists[net] = list(dict.fromkeys(["off", "v1", *top]))
    results["finalists"] = finalists
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("SPEED_FINALISTS " + json.dumps(finalists, sort_keys=True), flush=True)

    for repeat in range(1, CONFIG["finalist_extra_repeats"] + 1):
        per_net = {net: finalists[net][:] for net in CONFIG["networks"]}
        for net in per_net:
            rng.shuffle(per_net[net])
        finalist_tasks = []
        for slot in range(max(len(v) for v in per_net.values())):
            nets = CONFIG["networks"][:]
            if rng.randrange(2):
                nets.reverse()
            for net in nets:
                finalist_tasks.append((net, per_net[net][slot]))
        for net, variant in finalist_tasks:
            if (net, variant, repeat) in existing_keys:
                continue
            row = sample(net, variant, repeat, order_index, parity, alias,
                         natural,
                         parity["large_tree"][f"{net}/{variant}"])
            results["samples"].append(row)
            existing_keys.add((net, variant, repeat))
            order_index += 1
            output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"SPEED repeat={repeat} net={net} variant={variant} "
                  f"expanded={row['expanded_nodes']} seconds={row['seconds']:.2f} "
                  f"nps={row['nps']:.1f}", flush=True)

    for net in CONFIG["networks"]:
        net_summary = {}
        medians = {}
        for variant in variants:
            values = [r["nps"] for r in results["samples"]
                      if r["network"] == net and r["variant"] == variant]
            if not values:
                continue
            medians[variant] = sorted(values)[len(values) // 2] if len(values) % 2 else (
                sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2
            sample_times = sorted(r["seconds"] for r in results["samples"]
                                  if r["network"] == net and r["variant"] == variant)
            median_seconds = sample_times[len(values) // 2] if len(values) % 2 else (
                sample_times[len(values) // 2 - 1] + sample_times[len(values) // 2]) / 2
            net_summary[variant] = {"samples": len(values), "median_nps": medians[variant],
                                    "median_seconds": median_seconds}
        for variant in net_summary:
            net_summary[variant]["speedup_vs_off"] = medians[variant] / medians["off"]
            if "v1" in medians:
                net_summary[variant]["speedup_vs_v1"] = medians[variant] / medians["v1"]
        results["summary"][net] = net_summary
    results["finished_utc"] = utc_now()
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for net, values in results["summary"].items():
        ordered = sorted(((v, row["median_nps"]) for v, row in values.items()),
                         key=lambda item: item[1], reverse=True)
        print(f"SPEED_SUMMARY {net} " + json.dumps(ordered), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
