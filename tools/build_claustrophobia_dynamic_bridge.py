#!/usr/bin/env python3
"""Build an isolated deadline-aware bridge and pinned MCTS library."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.external import bot_setup
from tools.external.build_lock import build_lock

CONFIG = {
    "shared_root": ROOT,
    "output": ROOT / "results/benchmarks/claustrophobia_dynamic_bridge",
    "source": ROOT / "tools/external/claustrophobia_benchmark_bridge.rs",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deadline_mcts(source: str) -> str:
    """Add a clock entry point while preserving all fixed-count callers."""
    marker = "pub(crate) fn run_mcts_batched_impl_restricted<S, F>("
    start = source.index(marker)
    body = source.index("{\n    assert!(", start)
    header = source[start:body]
    timed_header = header.replace("run_mcts_batched_impl_restricted", "run_mcts_batched_impl_restricted_timed", 1)
    timed_header = timed_header.replace("    root_allowed: Option<&[bool]>,\n)",
        "    root_allowed: Option<&[bool]>,\n    deadline: Option<std::time::Instant>,\n)")
    if timed_header == header or "deadline:" not in timed_header:
        raise RuntimeError("Pinned MCTS signature does not match the deadline patch")
    wrapper = header + """{
    run_mcts_batched_impl_restricted_timed(root_state, batch_eval, n_simulations,
        c_puct, cfg, root_noise, rng, reuse, root_allowed, None)
}

#[allow(clippy::too_many_arguments)]
"""
    end = source.index("#[cfg(test)]\nmod tests", body)
    core = source[body:end]
    reserve = ".reserve(n_simulations as usize * AVG_BRANCH + 1);"
    if core.count(reserve) != 2:
        raise RuntimeError("Pinned MCTS reserve sites do not match")
    core = core.replace(reserve,
        ".reserve(if deadline.is_some() { 1024 * AVG_BRANCH + 1 } else { n_simulations as usize * AVG_BRANCH + 1 });")
    wave = "    while done < n_simulations {\n"
    if core.count(wave) != 1:
        raise RuntimeError("Pinned MCTS wave loop does not match")
    core = core.replace(wave, wave + """        // Stop only between complete waves: no pending virtual loss escapes.
        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
            break;
        }
""")
    result = "let mut result = assemble_result(tree, root_id, n_simulations);"
    if core.count(result) != 1:
        raise RuntimeError("Pinned MCTS result assembly does not match")
    core = core.replace(result,
        "let mut result = assemble_result(tree, root_id, if deadline.is_some() { done } else { n_simulations });")
    public = """
/// Search one persistent MCTS tree until the monotonic deadline.
/// Root evaluation counts against the caller's deadline. Synchronous evaluation
/// can overrun by one complete wave; total_simulations records actual backups.
pub fn run_mcts_batched_until<B: BatchEvaluator>(
    root_state: GameState,
    batch_eval: &B,
    deadline: std::time::Instant,
    c_puct: f64,
    cfg: BatchedConfig,
) -> SearchResult {
    run_mcts_batched_impl_restricted_timed(root_state,
        |s: &[GameState]| batch_eval.evaluate_batch(s), u32::MAX,
        c_puct, cfg, None, None, None, None, Some(deadline))
}

impl SearchResult {
    /// Actual backed-up root visits, including solver exits.
    pub fn completed_root_visits(&self) -> u32 {
        self.tree.node(self.root_id).visit_count
    }
    pub fn root_is_proven(&self) -> bool {
        self.tree.node(self.root_id).proven != 0
    }
}

"""
    return source[:start] + public + wrapper + timed_header + core + source[end:]


def _ensure_bridge(shared_root: Path = CONFIG["shared_root"], output: Path = CONFIG["output"],
                  *, build: bool = True) -> Path:
    shared_root = Path(shared_root).resolve()
    output = Path(output).resolve()
    checkout = shared_root / "external_bots/claustrophobia/repo"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if revision != bot_setup.CLAUSTROPHOBIA_SHA:
        raise RuntimeError("Claustrophobia checkout does not match the pinned revision")
    sources = {
        "bridge.rs": Path(CONFIG["source"]),
        "zq_ipc_eval.rs": ROOT / "training/teachers/claustrophobia_ipc_eval.rs",
        "zq_inference_worker.py": ROOT / "training/teachers/claustrophobia_inference_worker.py",
    }
    upstream_sources = {str(path.relative_to(checkout)): digest(path)
                        for path in sorted((checkout / "src").rglob("*.rs"))}
    patch_identity = {"builder_sha256": digest(Path(__file__)),
                      "sources": {name: digest(path) for name, path in sources.items()},
                      "upstream_sources": upstream_sources, "revision": revision}
    version = hashlib.sha256(json.dumps(patch_identity, sort_keys=True).encode()).hexdigest()[:16]
    output = output / version
    rustc = Path(bot_setup._find_cargo(shared_root)).with_name("rustc.exe" if os.name == "nt" else "rustc")
    executable = output / ("zq_benchmark_bridge.exe" if os.name == "nt" else "zq_benchmark_bridge")
    library = output / "libquoridor_clock.rlib"
    library_command = [str(rustc), "--edition=2021", "--crate-type=rlib", "--crate-name", "quoridor",
                       "-C", "opt-level=3", "-C", "panic=abort", "-C", "lto=fat",
                       "-C", "codegen-units=1", str(output / "src/lib.rs"), "-o", str(library)]
    command = [str(rustc), "--edition=2021", "-C", "opt-level=3", "-C", "panic=abort",
               "-C", "lto=fat", "-C", "codegen-units=1", "--crate-name", "zq_benchmark_bridge",
               str(output / "bridge.rs"), "--extern", f"quoridor={library}", "-o", str(executable)]
    identity = {**patch_identity, "command": command, "library_command": library_command,
                "protocol": "deadline-and-engine-clock-v2"}
    stamp = output / "build.json"
    if executable.is_file() and library.is_file() and stamp.is_file():
        previous = json.loads(stamp.read_text())
        if (previous.get("identity") == identity and previous.get("executable_sha256") == digest(executable)
                and previous.get("library_sha256") == digest(library)):
            return executable
    if not build:
        return executable
    output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        shutil.copy2(source, output / name)
    shutil.copytree(checkout / "src", output / "src", dirs_exist_ok=True)
    mcts = output / "src/mcts.rs"
    mcts.write_text(deadline_mcts(mcts.read_text(encoding="utf-8")), encoding="utf-8")
    environment = bot_setup._cargo_environment(shared_root)
    if os.name == "nt" and Path("C:/mingw64/bin").is_dir():
        environment["PATH"] = "C:/mingw64/bin" + os.pathsep + environment.get("PATH", "")
    environment["QUORIDOR_BUILD_SHA"] = revision
    subprocess.run(library_command, env=environment, check=True, cwd=output)
    subprocess.run(command, env=environment, check=True, cwd=output)
    stamp.write_text(json.dumps({"identity": identity, "executable_sha256": digest(executable),
                                "library_sha256": digest(library), "patched_mcts_sha256": digest(mcts)},
                               indent=2), encoding="utf-8")
    return executable


def ensure_bridge(shared_root: Path = CONFIG["shared_root"], output: Path = CONFIG["output"],
                  *, build: bool = True) -> Path:
    """Build or reuse one isolated deadline bridge under a cache lock."""
    if not build:
        return _ensure_bridge(shared_root, output, build=False)
    with build_lock(Path(output).resolve()):
        return _ensure_bridge(shared_root, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=CONFIG["shared_root"])
    parser.add_argument("--output", type=Path, default=CONFIG["output"])
    args = parser.parse_args()
    print(ensure_bridge(args.shared_root, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
