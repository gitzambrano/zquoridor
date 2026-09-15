"""Provision pinned external bots for local teacher and benchmark jobs."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TITANIUM_REPOSITORY = "https://github.com/titaniummachine1/titanium-engine.git"
TITANIUM_SHA = "1ac94755f69799f01b2e06869403e701bbf0bd51"
CLAUSTROPHOBIA_REPOSITORY = "https://github.com/Plaaasma/Claustrophobia.git"
CLAUSTROPHOBIA_SHA = "ae093653e62ad700e201706fa5ed767093d0d68e"
CLAUSTROPHOBIA_CHECKPOINT_URL = (
    "https://github.com/Plaaasma/Claustrophobia/releases/download/v1.3.1/champion.pt"
)
CLAUSTROPHOBIA_CHECKPOINT_SHA256 = (
    "821bd10f2736fc433c82de4dfc090ab95fc81b524f04fa09e68c4a26e30b496c"
)


def _run(command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"The required command is not available: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "no command output").strip()
        raise RuntimeError(f"The command failed: {' '.join(command)}\n{detail}") from error
    return result.stdout.rstrip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    """Download an artifact atomically, unless a verified copy exists."""
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = _sha256(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"The SHA-256 for {url} is {actual}. Expected {expected_sha256}."
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


_MANAGED_MARKER = ".zquoridor-managed.sha256"
_MANAGED_PATHS = {
    "Cargo.toml",
    "src/bin/zq_encode_bridge.rs",
    "src/bin/zq_search_bridge.rs",
    "src/bin/zq_benchmark_bridge.rs",
    "src/bin/zq_ipc_eval.rs",
}


def _managed_checkout_digest(checkout: Path) -> str | None:
    status = _run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"]
    )
    paths = []
    for line in status.splitlines():
        path = line[3:].replace("\\", "/")
        if " -> " in path:
            return None
        if path == _MANAGED_MARKER:
            continue
        if path not in _MANAGED_PATHS:
            return None
        paths.append(path)
    digest = hashlib.sha256()
    for path in sorted(paths):
        file_path = checkout / path
        digest.update(path.encode("utf-8") + b"\0")
        if file_path.is_file():
            digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _record_managed_checkout(checkout: Path) -> None:
    digest = _managed_checkout_digest(checkout)
    if digest is None:
        raise RuntimeError(f"The checkout contains an unmanaged change: {checkout}")
    (checkout / _MANAGED_MARKER).write_text(digest + "\n", encoding="ascii")


def _managed_checkout_matches(checkout: Path) -> bool:
    marker = checkout / _MANAGED_MARKER
    if not marker.is_file():
        return False
    expected = marker.read_text(encoding="ascii").strip()
    return bool(expected) and expected == _managed_checkout_digest(checkout)


def _ensure_checkout(
    repository: str,
    revision: str,
    checkout: Path,
    *,
    allow_managed: bool = False,
) -> None:
    """Create a pinned checkout or verify a safe existing checkout."""
    if checkout.exists():
        git_dir = checkout / ".git"
        if not checkout.is_dir() or not git_dir.exists():
            raise RuntimeError(
                f"The provisioner refuses to replace the unknown checkout at {checkout}."
            )
        actual = _run(["git", "-C", str(checkout), "rev-parse", "HEAD"])
        dirty = _run(["git", "-C", str(checkout), "status", "--porcelain"])
        managed = allow_managed and dirty and _managed_checkout_matches(checkout)
        if actual != revision or (dirty and not managed):
            state = f"revision {actual}" if not dirty else "local changes"
            raise RuntimeError(
                f"The provisioner refuses to replace the checkout at {checkout}: {state}."
            )
        return

    checkout.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{checkout.name}.", dir=checkout.parent))
    try:
        _run(["git", "clone", "--quiet", repository, str(temporary)])
        _run(["git", "-C", str(temporary), "checkout", "--quiet", "--detach", revision])
        actual = _run(["git", "-C", str(temporary), "rev-parse", "HEAD"])
        if actual != revision:
            raise RuntimeError(f"The checkout is at {actual}. Expected {revision}.")
        os.replace(temporary, checkout)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _same_file_content(left: Path, right: Path) -> bool:
    return left.is_file() and right.is_file() and _sha256(left) == _sha256(right)


def _add_cargo_binary(cargo_toml: Path, name: str, *, neural: bool) -> bool:
    text = cargo_toml.read_text(encoding="utf-8")
    if f'name = "{name}"' in text:
        return False
    entry = f'\n[[bin]]\nname = "{name}"\npath = "src/bin/{name}.rs"\n'
    if neural:
        entry += 'required-features = ["nn"]\n'
    cargo_toml.write_text(text.rstrip() + "\n" + entry, encoding="utf-8")
    return True


def _prepare_claustrophobia_sources(project_root: Path, checkout: Path) -> bool:
    """Refresh each available bridge and add its Cargo target once."""
    helper = project_root / "training/teachers/claustrophobia_ipc_eval.rs"
    destination = checkout / "src/bin/zq_ipc_eval.rs"
    helper_changed = helper.is_file() and not _same_file_content(helper, destination)
    if helper_changed:
        shutil.copy2(helper, destination)
    # These managed bridge targets can use the std-only IPC evaluator.
    cargo_file = checkout / "Cargo.toml"
    text = cargo_file.read_text(encoding="utf-8")
    blocks = text.split("[[bin]]")
    for index, block in enumerate(blocks):
        if any(f'name = "{name}"' in block for name in ("zq_search_bridge", "zq_benchmark_bridge")):
            blocks[index] = block.replace('required-features = ["nn"]\n', '')
    updated = "[[bin]]".join(blocks)
    if updated != text:
        cargo_file.write_text(updated, encoding="utf-8")
        helper_changed = True
    sources = [
        (
            project_root / "training" / "teachers" / "claustrophobia_encode_bridge.rs",
            "zq_encode_bridge",
            False,
        ),
        (
            project_root / "training" / "teachers" / "claustrophobia_search_bridge.rs",
            "zq_search_bridge",
            False,
        ),
        (
            project_root / "tools" / "external" / "claustrophobia_benchmark_bridge.rs",
            "zq_benchmark_bridge",
            False,
        ),
    ]
    changed = bool(helper_changed)
    for source, name, neural in sources:
        if not source.is_file():
            continue
        destination = checkout / "src" / "bin" / f"{name}.rs"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not _same_file_content(source, destination):
            shutil.copy2(source, destination)
            changed = True
        changed = _add_cargo_binary(checkout / "Cargo.toml", name, neural=neural) or changed
    return changed


def _executable(path: Path) -> Path:
    return path.with_suffix(".exe") if os.name == "nt" else path


def _find_cargo(project_root: Path) -> str:
    candidates = [
        shutil.which("cargo"),
        str(project_root / "external_bots" / ".toolchain" / "cargo" / "bin" / "cargo.exe"),
        str(Path.home() / ".cargo" / "bin" / "cargo.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError(
        "Cargo is required to build the external bots. Install Rust from "
        "https://rustup.rs, or place cargo.exe in "
        f"{project_root / 'external_bots' / '.toolchain' / 'cargo' / 'bin'}."
    )


def _cargo_environment(project_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    toolchain = project_root / "external_bots" / ".toolchain"
    cargo_home = toolchain / "cargo"
    rustup_home = toolchain / "rustup"
    if cargo_home.is_dir() and rustup_home.is_dir():
        environment["CARGO_HOME"] = str(cargo_home)
        environment["RUSTUP_HOME"] = str(rustup_home)
        environment["PATH"] = str(cargo_home / "bin") + os.pathsep + environment.get("PATH", "")
    return environment


def _torch_environment(project_root: Path) -> tuple[dict[str, str], Path]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to build the Claustrophobia search bridges. "
            f"Install PyTorch for {sys.executable}."
        ) from error
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if not torch_lib.is_dir():
        raise RuntimeError(f"The PyTorch library directory does not exist: {torch_lib}")
    environment = _cargo_environment(project_root)
    environment["LIBTORCH_USE_PYTORCH"] = "1"
    variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    environment[variable] = str(torch_lib) + os.pathsep + environment.get(variable, "")
    return environment, torch_lib


def _build_titanium(project_root: Path, checkout: Path, executable: Path) -> None:
    if executable.is_file():
        return
    cargo = _find_cargo(project_root)
    environment = _cargo_environment(project_root)
    environment["RUSTFLAGS"] = "-C target-cpu=native"
    _run(
        [cargo, "build", "--release", "--manifest-path", str(checkout / "Cargo.toml"), "--bin", "titanium"],
        env=environment,
    )
    if not executable.is_file():
        raise RuntimeError(f"Cargo did not create the Titanium executable: {executable}")


def _build_claustrophobia(project_root: Path, checkout: Path, changed: bool) -> None:
    cargo = _find_cargo(project_root)
    release = checkout / "target" / "release"
    targets = ["zq_encode_bridge", "zq_search_bridge"]
    if (checkout / "src/bin/zq_benchmark_bridge.rs").is_file():
        targets.append("zq_benchmark_bridge")
    environment = _cargo_environment(project_root)
    if os.name == "nt" and Path("C:/mingw64/bin").is_dir():
        environment["PATH"] = "C:/mingw64/bin" + os.pathsep + environment.get("PATH", "")
    command = [cargo, "build", "--release", "--manifest-path", str(checkout / "Cargo.toml")]
    for name in targets:
        command.extend(["--bin", name])
    _run(command, env=environment)
    worker = project_root / "training/teachers/claustrophobia_inference_worker.py"
    destination = release / "zq_inference_worker.py"
    if not _same_file_content(worker, destination):
        shutil.copy2(worker, destination)


def ensure_bot(name: str, root: Path = ROOT, build: bool = True) -> dict[str, Path]:
    """Provision one bot and return its absolute artifact paths.

    Titanium returns ``executable``. Claustrophobia returns ``checkout``,
    ``checkpoint``, ``encode_bridge``, ``search_bridge``, and
    ``benchmark_bridge``. If ``build`` is false, a returned binary path can be
    absent. The source checkout and downloaded checkpoint are still verified.
    """
    project_root = Path(root).resolve()
    bot = name.strip().lower()
    if bot == "titanium":
        checkout = project_root / "external_bots" / "titanium" / "repo"
        _ensure_checkout(TITANIUM_REPOSITORY, TITANIUM_SHA, checkout)
        executable = _executable(checkout / "target" / "release" / "titanium")
        if build:
            _build_titanium(project_root, checkout, executable)
        return {"executable": executable.resolve()}
    if bot == "claustrophobia":
        base = project_root / "external_bots" / "claustrophobia"
        checkout = base / "repo"
        checkpoint = base / "champion.pt"
        _ensure_checkout(
            CLAUSTROPHOBIA_REPOSITORY,
            CLAUSTROPHOBIA_SHA,
            checkout,
            allow_managed=True,
        )
        _download_verified(
            CLAUSTROPHOBIA_CHECKPOINT_URL,
            checkpoint,
            CLAUSTROPHOBIA_CHECKPOINT_SHA256,
        )
        changed = _prepare_claustrophobia_sources(project_root, checkout)
        _record_managed_checkout(checkout)
        if build:
            _build_claustrophobia(project_root, checkout, changed)
        release = checkout / "target" / "release"
        return {
            "checkout": checkout.resolve(),
            "checkpoint": checkpoint.resolve(),
            "encode_bridge": _executable(release / "zq_encode_bridge").resolve(),
            "search_bridge": _executable(release / "zq_search_bridge").resolve(),
            "benchmark_bridge": _executable(release / "zq_benchmark_bridge").resolve(),
        }
    raise ValueError(f"Unknown bot: {name}. Select titanium or claustrophobia.")
