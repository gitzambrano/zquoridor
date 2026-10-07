"""Build an isolated, increment-aware Titanium UCI protocol bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.external import bot_setup
from tools.external.build_lock import build_lock


PINNED_REVISION = bot_setup.TITANIUM_SHA
CONFIG = {"shared_root": ROOT, "output": ROOT / "results/benchmarks/titanium_clock_bridge", "build_jobs": 1}
DEFAULT_OUTPUT = CONFIG["output"]
SESSION_RELATIVE = Path("src/titanium/uci/session.rs")
PATCH_MARKER = "zquoridor increment-aware clock bridge"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patched_session(source: str) -> str:
    if PATCH_MARKER in source:
        return source
    old = '''                    let mut rem_arg: Option<f64> = None;
                    let mut node_limit: Option<u64> = None;'''.replace("+", "")
    new = '''                    // zquoridor increment-aware clock bridge
                    let mut rem_arg: Option<f64> = None;
                    let mut inc_arg: Option<f64> = None;
                    let mut opp_arg: Option<f64> = None;
                    let mut node_limit: Option<u64> = None;'''
    if old not in source:
        raise RuntimeError("Pinned session.rs clock parser anchor changed")
    source = source.replace(old, new, 1)
    old = '''                            "nodes" | "rem" | "time" => {
                                let key = toks[i];'''
    new = '''                            "nodes" | "rem" | "time" | "inc" | "opp" => {
                                let key = toks[i];'''
    if old not in source:
        raise RuntimeError("Pinned session.rs go-keyword parser anchor changed")
    source = source.replace(old, new, 1)
    old = '''                                    "rem" => val.parse::<f64>().map(|v| rem_arg = Some(v)).is_ok(),
                                    _ => val.parse::<f64>().map(|v| time_sec = Some(v)).is_ok(),'''
    new = '''                                    "rem" => val.parse::<f64>().map(|v| rem_arg = Some(v)).is_ok(),
                                    "inc" => val.parse::<f64>().map(|v| inc_arg = Some(v)).is_ok(),
                                    "opp" => val.parse::<f64>().map(|v| opp_arg = Some(v)).is_ok(),
                                    _ => val.parse::<f64>().map(|v| time_sec = Some(v)).is_ok(),'''
    if old not in source:
        raise RuntimeError("Pinned session.rs numeric parser anchor changed")
    source = source.replace(old, new, 1)
    old = '''                    if let Some(msg) = bad {
                        err!(msg);
                        continue;
                    }
                    let use_rem = rem_arg.is_some();'''
    new = '''                    if let Some(msg) = bad {
                        err!(msg);
                        continue;
                    }
                    for (name, value) in [("rem", rem_arg), ("inc", inc_arg), ("opp", opp_arg)] {
                        if let Some(value) = value {
                            if !value.is_finite() || value < 0.0 {
                                bad = Some(format!("{name} must be a finite non-negative number"));
                                break;
                            }
                        }
                    }
                    if let Some(msg) = bad {
                        err!(msg);
                        continue;
                    }
                    let use_rem = rem_arg.is_some();'''
    if old not in source:
        raise RuntimeError("Pinned session.rs validation anchor changed")
    source = source.replace(old, new, 1)
    old = '''                    let time_ms = if let Some(rem_sec) = rem_arg {
                        let remaining_ms = (rem_sec * 1000.0).max(0.0) as u64;
                        let ja = crate::titanium::race::jump_aware_goal_distances(&mut current_g);'''
    new = '''                    let time_ms = if let Some(rem_sec) = rem_arg {
                        let remaining_ms = (rem_sec * 1000.0).max(0.0) as u64;
                        let ja = crate::titanium::race::jump_aware_goal_distances(&mut current_g);'''
    if old not in source:
        raise RuntimeError("Pinned session.rs clock allocation anchor changed")
    # Retain the original allocator call and capture its complete result.
    start = source.index(old)
    end = source.index('''                    } else if let Some(secs) = time_sec {''', start)
    original = source[start:end]
    original = original.replace('''                        crate::titanium::time_alloc::allocate_move_budget_with_dists_and_walls(''','''                        let budget = crate::titanium::time_alloc::allocate_move_budget_with_dists_and_walls(''', 1)
    original = original.replace('''                        .move_ms
                        .max(1)''','''                        ;
                        let increment_ms = (inc_arg.unwrap_or(0.0) * 1000.0).round() as u64;
                        let hard_cap = remaining_ms.saturating_sub(budget.safety_ms);
                        let allocated_ms = if inc_arg.unwrap_or(0.0) > 0.0 {
                            budget.move_ms.saturating_add(
                                (increment_ms as f64 * crate::titanium::timeman::time_alloc::MAX_RATIO)
                                    .round() as u64,
                            ).min(hard_cap).max(1)
                        } else {
                            budget.move_ms.max(1)
                        };
                        println!("info string clock remaining_ms={} inc_ms={} opp_ms={} allocated_ms={}",
                            remaining_ms, increment_ms,
                            (opp_arg.unwrap_or(0.0) * 1000.0).round() as u64,
                            allocated_ms);
                        allocated_ms'''.replace("+", ""), 1)
    if original == source[start:end] or ".move_ms" not in source[start:end]:
        raise RuntimeError("Could not adapt native allocation result")
    source = source[:start] + original + source[end:]
    return source


def _tree_hash(root: Path, overrides: dict[str, bytes] | None = None) -> str:
    overrides = overrides or {}
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts and "target" not in p.parts):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(overrides.get(relative, path.read_bytes()) + b"\0")
    return digest.hexdigest()


def _ensure_bridge(
    shared_root: str | Path = CONFIG["shared_root"],
    output: str | Path | None = None,
    *, build: bool = True,
) -> Path:
    """Build or reuse an isolated patched copy; return its executable path."""
    project_root = Path(shared_root).resolve()
    source_root = project_root
    upstream = source_root / "external_bots/titanium/repo"
    if not upstream.is_dir():
        bot_setup.ensure_bot("titanium", root=source_root, build=False)
    revision = bot_setup._run(["git", "-C", str(upstream), "rev-parse", "HEAD"])
    if revision != PINNED_REVISION:
        raise RuntimeError(f"Titanium checkout revision {revision} differs from pin {PINNED_REVISION}")
    upstream_session = upstream / SESSION_RELATIVE
    patched_bytes = _patched_session(upstream_session.read_text(encoding="utf-8")).encode("utf-8")
    upstream_build = upstream / "build.rs"
    build_source = upstream_build.read_text(encoding="utf-8")
    stamp_anchor = "    emit_git_commit();"
    if build_source.count(stamp_anchor) != 1:
        raise RuntimeError("Pinned build.rs provenance anchor changed")
    patched_build = build_source.replace(stamp_anchor,
        f'    println!("cargo:rustc-env=GIT_COMMIT_HASH={revision}-clock");', 1).encode("utf-8")
    overrides = {SESSION_RELATIVE.as_posix(): patched_bytes, "build.rs": patched_build}
    upstream_tree_sha256 = _tree_hash(upstream)
    patched_session_sha256 = _sha256_bytes(patched_bytes)
    expected_source_tree_sha256 = _tree_hash(
        upstream, overrides)
    builder_sha256 = _sha256_file(Path(__file__))
    identity = {
        "revision": revision,
        "pinned_revision": PINNED_REVISION,
        "builder_sha256": builder_sha256,
        "upstream_tree_sha256": upstream_tree_sha256,
        "patched_session_sha256": patched_session_sha256,
        "patched_build_sha256": _sha256_bytes(patched_build),
        "source_tree_sha256": expected_source_tree_sha256,
    }
    identity_sha256 = _sha256_bytes(json.dumps(identity, sort_keys=True).encode("utf-8"))
    output_root = Path(output) if output is not None else DEFAULT_OUTPUT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    destination = output_root.resolve() / identity_sha256[:20]
    source_dir = destination / "source"
    executable = bot_setup._executable(source_dir / "target/release/titanium")
    manifest = destination / "build.json"
    if manifest.is_file() and executable.is_file():
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
            if (record.get("identity") == identity
                    and record.get("source_tree_sha256") == _tree_hash(source_dir)
                    and record.get("binary_sha256") == _sha256_file(executable)):
                return executable.resolve()
        except (OSError, ValueError):
            pass
    if not build:
        return executable.resolve()
    if source_dir.exists():
        source_tree_sha256 = _tree_hash(source_dir)
        if source_tree_sha256 != expected_source_tree_sha256:
            raise RuntimeError(f"The content-addressed Titanium source copy changed: {source_dir}")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(upstream, source_dir, ignore=shutil.ignore_patterns(".git", "target"))
        (source_dir / SESSION_RELATIVE).write_bytes(patched_bytes)
        (source_dir / "build.rs").write_bytes(patched_build)
    cargo = bot_setup._find_cargo(source_root)
    environment = bot_setup._cargo_environment(source_root)
    environment["RUSTFLAGS"] = "-C target-cpu=native"
    environment["CARGO_BUILD_JOBS"] = str(CONFIG["build_jobs"])
    if os.name == "nt":
        environment["PATH"] = "C:/mingw64/bin" + os.pathsep + environment.get("PATH", "")
    bot_setup._run([cargo, "build", "--release", "--manifest-path", str(source_dir / "Cargo.toml"), "--bin", "titanium"], env=environment, cwd=source_dir)
    if not executable.is_file():
        raise RuntimeError(f"Cargo did not create Titanium executable at {executable}")
    record = {
        "upstream_repository": bot_setup.TITANIUM_REPOSITORY,
        "upstream_revision": revision,
        "identity": identity,
        "identity_sha256": identity_sha256,
        "upstream_tree_sha256": upstream_tree_sha256,
        "upstream_session_sha256": _sha256_file(upstream_session),
        "patched_session_sha256": _sha256_file(source_dir / SESSION_RELATIVE),
        "source_tree_sha256": _tree_hash(source_dir),
        "binary_sha256": _sha256_file(executable),
        "protocol": "go rem <seconds> inc <seconds> [opp <seconds>]",
        "patch": "native allocation plus min(remaining minus native safety, allocation plus round(increment * MAX_RATIO)); increment zero preserves native behavior",
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return executable.resolve()


def ensure_bridge(shared_root: str | Path = CONFIG["shared_root"],
                  output: str | Path | None = None, *, build: bool = True) -> Path:
    """Build or reuse one isolated Titanium bridge under a cache lock."""
    if not build:
        return _ensure_bridge(shared_root, output, build=False)
    directory = Path(output) if output is not None else DEFAULT_OUTPUT
    if not directory.is_absolute():
        directory = ROOT / directory
    with build_lock(directory.resolve()):
        return _ensure_bridge(shared_root, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=CONFIG["shared_root"])
    parser.add_argument("--output", type=Path, default=CONFIG["output"])
    parser.add_argument("--build-jobs", type=int, default=CONFIG["build_jobs"])
    args = parser.parse_args()
    if args.build_jobs <= 0:
        parser.error("build-jobs must be positive")
    CONFIG["build_jobs"] = args.build_jobs
    try:
        print(ensure_bridge(args.shared_root, args.output))
    except Exception as error:
        print(f"Titanium bridge build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
