"""Build and run local EdgeAcc parity gates for 504 and Contact858.

Keep all generated files under the ignored results/benchmarks tree. Edit CONFIG
for local paths or concurrency; command-line arguments are invocation overrides.
"""

from __future__ import annotations

import concurrent.futures
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "compiler": "g++",
    "out_dir": ROOT / "results" / "benchmarks" / "edge-parity-local",
    "main_worktree": Path(r"C:\Projetos\Zquoridor"),
    "contact_source": Path(
        r"C:\Projetos\Zquoridor\results\experiments\contact_soup_tri_512"
        r"\soup_tri_champion\student_int8.bin"
    ),
    "contact_local": ROOT / "results" / "benchmarks" / "edge-parity-local"
    / "weights" / "contact858_student_int8.bin",
    "networks": {
        "n504": {
            "weights": ROOT / "data" / "nnue" / "nnue_weights_int8.bin",
            "flags": [],
        },
        "contact858": {
            "weights": ROOT / "results" / "benchmarks" / "edge-parity-local"
            / "weights" / "contact858_student_int8.bin",
            "flags": [
                "-DZQ_NNUE_RACE_FEATURES=1",
                "-DZQ_NNUE_MULTIPATH_FEATURES=1",
                "-DZQ_NNUE_MARGIN_REGIME_FEATURES=0",
                "-DZQ_NNUE_PHASE_FEATURES=1",
                "-DZQ_NNUE_CONTACT_FEATURES=1",
                "-DZQ_NNUE_HIDDEN=512",
                "-DZQ_NNUE_VALUE_BUCKETS=6",
                "-DZQ_NNUE_VALUE_DEPTH=2",
            ],
        },
    },
    "variants": {
        "off": [],
        "v1": ["-DZQ_EXP_EDGE_ACC_CACHE=1"],
        "v2": ["-DZQ_EXP_EDGE_ACC_FULL_CANONICAL=1"],
        "v3": ["-DZQ_EXP_EDGE_ACC_PATH_CONTEXT=1"],
        "hybrid_sparse": ["-DZQ_EXP_EDGE_ACC_CACHE=1", "-DZQ_EXP_EDGE_FEATURE_DELTA=1"],
        "hybrid_dense": ["-DZQ_EXP_EDGE_ACC_CACHE=1", "-DZQ_EXP_EDGE_FEATURE_DELTA=1",
                         "-DZQ_EXP_EDGE_DENSE_DELTA=1"],
        "v3_node": ["-DZQ_EXP_EDGE_ACC_NODE_CONTEXT=1"],
        "v3_node_dense": ["-DZQ_EXP_EDGE_ACC_NODE_CONTEXT=1", "-DZQ_EXP_EDGE_FEATURE_DELTA=1",
                          "-DZQ_EXP_EDGE_DENSE_DELTA=1"],
        "v3_parent": ["-DZQ_EXP_EDGE_ACC_NODE_CONTEXT=1", "-DZQ_EXP_EDGE_ACC_PARENT_CONTEXT=1"],
        "hybrid_dense_bfs": ["-DZQ_EXP_EDGE_ACC_CACHE=1", "-DZQ_EXP_EDGE_FEATURE_DELTA=1",
                             "-DZQ_EXP_EDGE_DENSE_DELTA=1", "-DZQ_EXP_EDGE_BFS_REPLAY=1"],
        "v3_node_dense_bfs": ["-DZQ_EXP_EDGE_ACC_NODE_CONTEXT=1", "-DZQ_EXP_EDGE_FEATURE_DELTA=1",
                              "-DZQ_EXP_EDGE_DENSE_DELTA=1", "-DZQ_EXP_EDGE_BFS_REPLAY=1"],
        "delta_dense_only": ["-DZQ_EXP_EDGE_FEATURE_DELTA=1", "-DZQ_EXP_EDGE_DENSE_DELTA=1",
                             "-DZQ_EXP_EDGE_BFS_REPLAY=1"],
    },
    "logical_cpus": os.cpu_count() or 1,
    # Large trees allocate sizable node pools. Four processes give 16 logical
    # CPUs useful work without multiplying peak memory across all eight jobs.
    "parallel_runs": min(4, os.cpu_count() or 1),
    "parallel_aux": min(8, os.cpu_count() or 1),
    "fixed_node_count": 4000,
    "fixed_node_positions": 80,
    "require_large_cases": 10,
    "minimum_v1_v3_hits": 1000,
}

BASE = ["-O3", "-DNDEBUG", "-std=c++17", "-pthread", "-Isrc"]
SEMANTIC_FIELDS = ["nodes", "best", "pool", "sims", "expanded", "visits", "fp", "rootN"]
REUSE_FIELDS = SEMANTIC_FIELDS + ["treeHit", "reused"]
ROW_RE = re.compile(r"(\w+)=([^ ]+)")


def run(command: list[str], *, stdout=None, stderr=None, cwd=ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True, stdout=stdout, stderr=stderr)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_weights() -> None:
    CONFIG["out_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["contact_local"].parent.mkdir(parents=True, exist_ok=True)
    source = CONFIG["contact_source"]
    dest = CONFIG["contact_local"]
    if not dest.exists() or dest.stat().st_size != source.stat().st_size:
        shutil.copy2(source, dest)
    for net, spec in CONFIG["networks"].items():
        weights = spec["weights"]
        if not weights.is_file() or weights.stat().st_size == 0:
            raise FileNotFoundError(f"missing {net} weights: {weights}")


def compile_binary(net: str, variant: str, kind: str) -> dict:
    source = {
        "large": ROOT / "tests" / "test_edge_acc_large_tree.cpp",
        "accumulator": ROOT / "tests" / "test_edge_acc_transposition.cpp",
        "uci": ROOT / "tools" / "external" / "zquoridor_uci.cpp",
    }[kind]
    binary = CONFIG["out_dir"] / "bin" / net / f"{variant}_{kind}.exe"
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [CONFIG["compiler"], *BASE, *CONFIG["networks"][net]["flags"],
               *CONFIG["variants"][variant], str(source), "-o", str(binary)]
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    record = {"network": net, "variant": variant, "kind": kind,
              "command": command, "seconds": time.perf_counter() - start,
              "returncode": proc.returncode, "binary": str(binary)}
    if proc.returncode:
        record["stderr"] = proc.stderr[-12000:]
        raise RuntimeError(json.dumps(record, indent=2))
    return record


def parse_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not (line.startswith("CASE ") or line.startswith("REUSE ")):
            continue
        kind = line.split()[0]
        fields = dict(ROW_RE.findall(line))
        # The existing stress binary emits simulations and rootN; the complete
        # per-node and per-edge visit vectors are included in its tree hash.
        # Keep total simulation visits explicit in the structured manifest.
        fields["visits"] = fields.get("sims", "")
        label = line.split()[1] if kind == "CASE" else "step=" + fields["step"]
        rows.append({"kind": kind, "label": label, "fields": fields, "line": line})
    return rows


def run_large(net: str, variant: str) -> dict:
    exe = CONFIG["out_dir"] / "bin" / net / f"{variant}_large.exe"
    raw = CONFIG["out_dir"] / "raw" / net / f"{variant}_large.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with raw.open("w", encoding="utf-8") as output:
        proc = subprocess.run([str(exe), str(CONFIG["networks"][net]["weights"])],
                              cwd=ROOT, stdout=output, stderr=subprocess.PIPE, text=True)
    elapsed = time.perf_counter() - start
    if proc.returncode:
        return {"network": net, "variant": variant, "status": "ERROR",
                "seconds": elapsed, "stderr": proc.stderr[-12000:], "raw": str(raw)}
    return {"network": net, "variant": variant, "status": "RAN",
            "seconds": elapsed, "raw": str(raw), "rows": parse_rows(raw),
            "stdout_sha256": sha256(raw)}


def compare_large(off: dict, candidate: dict) -> dict:
    a, b = off["rows"], candidate["rows"]
    result = {"status": "PASS", "expected_checkpoints": CONFIG["require_large_cases"],
              "actual_off": len(a), "actual_candidate": len(b), "mismatches": []}
    if len(a) != CONFIG["require_large_cases"] or len(b) != CONFIG["require_large_cases"]:
        result["status"] = "FAIL"
        result["mismatches"].append({"reason": "checkpoint_count"})
    for i, (ra, rb) in enumerate(zip(a, b)):
        fields = REUSE_FIELDS if ra["kind"] == "REUSE" else SEMANTIC_FIELDS
        diff = {key: [ra["fields"].get(key), rb["fields"].get(key)]
                for key in fields if ra["fields"].get(key) != rb["fields"].get(key)}
        if ra["kind"] != rb["kind"] or ra["label"] != rb["label"] or diff:
            result["status"] = "FAIL"
            result["mismatches"].append({"index": i, "label_off": ra["label"],
                                         "label_candidate": rb["label"], "diff": diff,
                                         "off": ra["line"], "candidate": rb["line"]})
    result["candidate_cache_hits"] = sum(int(r["fields"].get("hits", 0)) for r in b)
    if candidate["variant"] in ("v1", "v3") and result["candidate_cache_hits"] < CONFIG["minimum_v1_v3_hits"]:
        result["status"] = "FAIL"
        result["mismatches"].append({"reason": "insufficient_cache_hits",
                                     "hits": result["candidate_cache_hits"]})
    return result


def run_accumulator(net: str, variant: str) -> dict:
    exe = CONFIG["out_dir"] / "bin" / net / f"{variant}_accumulator.exe"
    raw = CONFIG["out_dir"] / "raw" / net / f"{variant}_accumulator.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    proc = subprocess.run([str(exe), str(CONFIG["networks"][net]["weights"])],
                          cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    raw.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    passed = proc.returncode == 0 and "EDGE_ACC_PARITY_OK" in proc.stdout
    return {"network": net, "variant": variant, "status": "PASS" if passed else "FAIL",
            "returncode": proc.returncode, "seconds": elapsed, "raw": str(raw),
            "stdout": proc.stdout.strip(), "stderr": proc.stderr[-12000:]}


def fixed_node_histories() -> list[list[str]]:
    books = [ROOT / "tools/external/openings_normal_confirm_400.jsonl",
             ROOT / "tools/external/openings_center_rush_sound_5k.jsonl"]
    histories: list[list[str]] = []
    for book in books:
        for line in book.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            moves = json.loads(line).get("moves", [])
            histories.append(moves)
            if len(histories) >= CONFIG["fixed_node_positions"]:
                return histories
    return histories


def run_uci_positions(net: str, variant: str, histories: list[list[str]], exe_override: Path | None = None) -> dict:
    exe = exe_override or (CONFIG["out_dir"] / "bin" / net / f"{variant}_uci.exe")
    raw = CONFIG["out_dir"] / "raw" / net / f"{variant}_fixed_nodes.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen([str(exe), "--nnue", str(CONFIG["networks"][net]["weights"]),
                          "--nodes", str(CONFIG["fixed_node_count"]), "--no-tree-reuse",
                          "--clear-tt-per-move"], cwd=ROOT, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    def send(line: str) -> None:
        assert p.stdin
        p.stdin.write(line + "\n")
        p.stdin.flush()
    assert p.stdout
    send("uci")
    while p.stdout.readline().strip() != "uciok":
        if p.poll() is not None:
            raise RuntimeError("engine exited during uci handshake")
    send("isready")
    while p.stdout.readline().strip() != "readyok":
        if p.poll() is not None:
            raise RuntimeError("engine exited during ready handshake")
    decisions = []
    try:
        for moves in histories:
            send("ucinewgame")
            send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
            send("go movetime 60000")
            info, best = None, None
            while True:
                line = p.stdout.readline()
                if not line:
                    raise RuntimeError("engine exited while searching: " + p.stderr.read()[-4000:])
                line = line.strip()
                if line.startswith("info "):
                    info = line
                if line.startswith("bestmove "):
                    best = line.split()[1]
                    break
            decisions.append({"bestmove": best, "info": info, "moves": moves})
    finally:
        try:
            send("quit")
        except (BrokenPipeError, OSError):
            pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    raw.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    return {"network": net, "variant": variant, "status": "RAN", "raw": str(raw),
            "positions": len(decisions), "decisions": decisions}


def compile_main_off(net: str) -> tuple[Path, dict]:
    main_root = CONFIG["main_worktree"]
    binary = CONFIG["out_dir"] / "bin" / net / "main_off_uci.exe"
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [CONFIG["compiler"], *BASE, *CONFIG["networks"][net]["flags"],
               str(main_root / "tools" / "external" / "zquoridor_uci.cpp"), "-o", str(binary)]
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=main_root, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"main OFF build failed for {net}: {proc.stderr[-12000:]}")
    return binary, {"network": net, "variant": "main_off", "command": command,
                    "returncode": proc.returncode, "seconds": time.perf_counter() - start,
                    "binary": str(binary)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=CONFIG["variants"].keys(),
                        help="run OFF plus only the listed variants")
    parser.add_argument("--merge-existing", action="store_true",
                        help="merge results into the existing manifest after source/weight hash checks")
    args = parser.parse_args()
    selected_variants = list(CONFIG["variants"])
    if args.variants:
        selected_variants = ["off"] + [v for v in args.variants if v != "off"]
    selected_variants = list(dict.fromkeys(selected_variants))
    ensure_weights()
    out = CONFIG["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                     text=True).strip(),
                "config": {k: str(v) for k, v in CONFIG.items() if k != "networks"},
                "visit_evidence": {
                    "total": "sims emitted by test_edge_acc_large_tree.cpp",
                    "per_node_and_edge": "fp; treeFingerprintForInspection hashes every totalN and every N vector element",
                    "root": "rootN emitted at every checkpoint",
                },
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p)
                                   for p in sorted((ROOT / "src").glob("*.hpp"))},
                "main_source_sha256": {str(p.relative_to(CONFIG["main_worktree"])): sha256(p)
                                        for p in sorted((CONFIG["main_worktree"] / "src").glob("*.hpp"))},
                "networks": {}, "builds": [], "large_tree": {}, "accumulator": {},
                "fixed_node": {}, "main_fixed_node": {}}
    manifest["weights"] = {net: {"path": str(spec["weights"]),
                                "sha256": sha256(spec["weights"]),
                                "bytes": spec["weights"].stat().st_size}
                           for net, spec in CONFIG["networks"].items()}
    manifest_path = out / "parity_manifest.json"
    if args.merge_existing and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("source_sha256") != manifest.get("source_sha256"):
            raise RuntimeError("refusing manifest merge: source header hashes differ")
        if previous.get("main_source_sha256") != manifest.get("main_source_sha256"):
            raise RuntimeError("refusing manifest merge: main source header hashes differ")
        if previous.get("weights") != manifest.get("weights"):
            raise RuntimeError("refusing manifest merge: network weight hashes differ")
        for section in ("builds", "large_tree", "accumulator", "fixed_node", "main_fixed_node"):
            if section == "builds":
                previous.setdefault(section, []).extend(manifest[section])
            else:
                previous.setdefault(section, {}).update(manifest[section])
        previous.setdefault("campaigns", []).append({
            "started_local": manifest["started_local"],
            "variants": selected_variants,
        })
        manifest = previous
    if not args.skip_build:
        tasks = [(net, variant, kind) for net in CONFIG["networks"]
                 for variant in selected_variants
                 for kind in ("large", "accumulator", "uci")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, CONFIG["logical_cpus"])) as ex:
            futures = [ex.submit(compile_binary, *task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                build = future.result()
                manifest["builds"].append(build)
                print(f"BUILT {build['network']} {build['variant']} {build['kind']} "
                      f"{build['seconds']:.1f}s", flush=True)
                (out / "parity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    jobs = [(net, variant) for net in CONFIG["networks"] for variant in selected_variants]
    large_rows = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["parallel_runs"]) as ex:
        futures = {ex.submit(run_large, net, variant): (net, variant) for net, variant in jobs}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            row = future.result()
            large_rows[key] = row
            manifest["large_tree"][f"{key[0]}/{key[1]}"] = {k: v for k, v in row.items() if k != "rows"}
            (out / "parity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for net in CONFIG["networks"]:
        baseline = large_rows[(net, "off")]
        if baseline["status"] != "RAN":
            raise RuntimeError(f"baseline tree test failed: {net}")
        for variant in (v for v in selected_variants if v != "off"):
            candidate = large_rows[(net, variant)]
            result = compare_large(baseline, candidate) if candidate["status"] == "RAN" else {
                "status": "FAIL", "mismatches": [{"reason": candidate.get("status")}]
            }
            manifest["large_tree"][f"{net}/{variant}"]["comparison_to_off"] = result
            (out / "parity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"LARGE_PARITY net={net} variant={variant} status={result['status']} "
                  f"mismatches={len(result['mismatches'])} seconds={candidate.get('seconds')}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["parallel_aux"]) as ex:
        futures = {ex.submit(run_accumulator, net, variant): (net, variant) for net, variant in jobs}
        for future in concurrent.futures.as_completed(futures):
            net, variant = futures[future]
            result = future.result()
            manifest["accumulator"][f"{net}/{variant}"] = {k: v for k, v in result.items() if k != "stderr"}
            (out / "parity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"ACCUMULATOR_PARITY net={net} variant={variant} status={result['status']} "
                  f"{result.get('stdout','')}", flush=True)
    if not args.skip_fixed:
        histories = fixed_node_histories()[:CONFIG["fixed_node_positions"]]
        fixed_jobs = []
        for net in CONFIG["networks"]:
            for variant in selected_variants:
                fixed_jobs.append((net, variant, None))
            main_exe, build = compile_main_off(net)
            manifest["builds"].append(build)
            fixed_jobs.append((net, "main_off", main_exe))
        fixed_rows = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["parallel_aux"]) as ex:
            futures = {ex.submit(run_uci_positions, net, variant, histories, exe): (net, variant)
                       for net, variant, exe in fixed_jobs}
            for future in concurrent.futures.as_completed(futures):
                net, variant = futures[future]
                row = future.result()
                fixed_rows[(net, variant)] = row
                if variant == "main_off":
                    manifest["main_fixed_node"][net] = {k: v for k, v in row.items() if k != "decisions"}
                else:
                    manifest["fixed_node"][f"{net}/{variant}"] = {
                        k: v for k, v in row.items() if k != "decisions"}
                (out / "parity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for net in CONFIG["networks"]:
            main_row = fixed_rows[(net, "main_off")]
            experimental_off = manifest["fixed_node"][f"{net}/off"]
            main_decisions = json.loads(Path(main_row["raw"]).read_text(encoding="utf-8"))
            exp_decisions = json.loads(Path(experimental_off["raw"]).read_text(encoding="utf-8"))
            diffs = [{"index": i, "main": a["bestmove"], "experimental_off": b["bestmove"]}
                     for i, (a, b) in enumerate(zip(main_decisions, exp_decisions))
                     if a["bestmove"] != b["bestmove"]]
            manifest["main_fixed_node"][net]["comparison_to_experimental_off"] = {
                "status": "PASS" if not diffs and len(main_decisions) == len(exp_decisions) == len(histories) else "FAIL",
                "positions": len(histories), "mismatches": diffs}
            print(f"MAIN_OFF_FIXED_NODE net={net} status="
                  f"{manifest['main_fixed_node'][net]['comparison_to_experimental_off']['status']} "
                  f"positions={len(histories)} mismatches={len(diffs)}", flush=True)
            (out / "parity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            baseline = manifest["fixed_node"][f"{net}/off"]
            for variant in (v for v in selected_variants if v != "off"):
                candidate = manifest["fixed_node"][f"{net}/{variant}"]
                diffs = []
                a = json.loads(Path(baseline["raw"]).read_text(encoding="utf-8"))
                b = json.loads(Path(candidate["raw"]).read_text(encoding="utf-8"))
                for i, (x, y) in enumerate(zip(a, b)):
                    if x["bestmove"] != y["bestmove"]:
                        diffs.append({"index": i, "best_off": x["bestmove"], "best_candidate": y["bestmove"]})
                candidate["comparison_to_off"] = {"status": "PASS" if not diffs and len(a) == len(b) == len(histories) else "FAIL",
                                                   "positions": len(histories), "mismatches": diffs}
                print(f"FIXED_NODE_PARITY net={net} variant={variant} "
                      f"status={candidate['comparison_to_off']['status']} positions={len(histories)} "
                      f"mismatches={len(diffs)}", flush=True)
                (out / "parity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["finished_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "parity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    failed = any(data.get("comparison_to_off", {}).get("status") == "FAIL"
                 for data in manifest["large_tree"].values())
    failed |= any(data.get("status") == "FAIL" for data in manifest["accumulator"].values())
    failed |= any(data.get("comparison_to_off", {}).get("status") == "FAIL"
                  for data in manifest["fixed_node"].values())
    failed |= any(data.get("comparison_to_experimental_off", {}).get("status") == "FAIL"
                  for data in manifest["main_fixed_node"].values())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
