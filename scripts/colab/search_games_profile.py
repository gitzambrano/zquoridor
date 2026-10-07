"""Define isolated Colab profiles for Claustrophobia search games."""

from __future__ import annotations

from copy import deepcopy
import re
import shlex
from typing import Any, Mapping

try:
    from .config import WORKERS
except ImportError:
    from config import WORKERS


CONFIG = {
    "worker_ids": [1, 2, 3, 4, 5],
    "revision": "dcfc7f74c07c02e96521fd77d60f50779b84cea7",
    "repository_url": "https://github.com/gitzambrano/zquoridor.git",
    "checkout_root": "/content/zquoridor_search_games",
    "drive_root": "/content/drive/MyDrive/zquoridor_data/claustro_search_games_v1",
    "pairs": 5000,
    "workers": 1,
    "seed_stride": 1000001,
    "mode": "match",
    "claustrophobia_device": "gpu",
    "start_move_time_ms": 400,
    "end_move_time_ms": 50,
    "decay_start_ply": 14,
    "decay_end_ply": 80,
    "schedule_origin": "opening",
    "opening_weights": {"center_rush": 7, "normal": 2, "weakness": 1},
    "opening_temperature": 1.0,
    "temperature_plies": 14,
    "record_both_searches": True,
    "unique_openings_first": True,
}


BOOTLOADER_TEMPLATE = """from google.colab import drive
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import torch

if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')

assert torch.cuda.is_available(), 'Select a GPU runtime before collection.'
assert shutil.which('git') and shutil.which('g++'), 'Install Git and the C++ compiler.'
checkout = Path({checkout_dir_literal})
revision = {revision_literal}
repository_url = {repository_url_literal}
if not checkout.exists():
    checkout.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', repository_url, str(checkout)], check=True)
    subprocess.run(['git', '-C', str(checkout), 'fetch', 'origin', revision], check=True)
    subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', revision], check=True)
else:
    actual = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise RuntimeError('The existing checkout does not match the pinned revision.')
    status = subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain'], text=True).strip()
    if status:
        raise RuntimeError('The existing checkout contains local changes.')

os.environ['PATH'] = '/root/.cargo/bin:' + os.environ.get('PATH', '')
if not shutil.which('cargo'):
    subprocess.run(['bash', '-c', 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'], check=True)
os.environ['CARGO_BUILD_JOBS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
subprocess.run(['cargo', '--version'], check=True)
os.chdir(checkout)
runtime_dir = Path({runtime_dir_literal})
runtime_cache = Path({runtime_cache_literal})
cache_stamp = runtime_cache / 'artifacts.json'
output_dir = Path({drive_dir_literal})
if not cache_stamp.is_file():
    if (output_dir / 'manifest.json').is_file():
        raise RuntimeError('The run manifest exists, but its frozen runtime cache is absent.')
    if runtime_cache.exists():
        raise RuntimeError('The runtime cache is incomplete. Select a new campaign directory.')
    from tools.external.bot_setup import ensure_bot
    runtime_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(['g++', '-O3', '-std=c++17', '-mavx2', '-mfma', '-I', str(checkout / 'src'),
                    str(checkout / 'tools/external/zquoridor_uci.cpp'),
                    '-o', str(runtime_dir / 'zquoridor_uci')], check=True)
    info = ensure_bot('claustrophobia', root=checkout)
    runtime_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = Path(tempfile.mkdtemp(prefix='.runtime_', dir=runtime_cache.parent))
    artifacts = {{
        'zquoridor_uci': runtime_dir / 'zquoridor_uci',
        'zq_benchmark_bridge': info['benchmark_bridge'],
        'zq_inference_worker.py': info['benchmark_bridge'].parent / 'zq_inference_worker.py',
        'champion.pt': info['checkpoint'],
    }}
    hashes = {{}}
    for name, source in artifacts.items():
        destination = temporary_cache / name
        shutil.copy2(source, destination)
        hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (temporary_cache / 'artifacts.json').write_text(
        json.dumps({{'revision': revision, 'sha256': hashes}}, sort_keys=True), encoding='utf-8')
    temporary_cache.rename(runtime_cache)

stamp = json.loads(cache_stamp.read_text(encoding='utf-8'))
if stamp['revision'] != revision:
    raise RuntimeError('The runtime cache does not match the pinned revision.')
expected_files = {{'zquoridor_uci', 'zq_benchmark_bridge', 'zq_inference_worker.py', 'champion.pt'}}
if set(stamp['sha256']) != expected_files:
    raise RuntimeError('The runtime cache contains an unexpected artifact set.')
runtime_dir.mkdir(parents=True, exist_ok=True)
for name, expected in stamp['sha256'].items():
    source = runtime_cache / name
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError('The runtime cache contains a modified artifact: ' + name)
    destination = runtime_dir / name
    shutil.copy2(source, destination)
    if name in ('zquoridor_uci', 'zq_benchmark_bridge'):
        destination.chmod(0o755)
cpu_flags = Path('/proc/cpuinfo').read_text(encoding='utf-8').lower().split()
if 'avx2' not in cpu_flags or 'fma' not in cpu_flags:
    raise RuntimeError('The frozen Main executable requires an AVX2 and FMA CPU.')
print('Collect Claustrophobia search games for worker {worker_id}.', flush=True)
%cd {checkout_dir}
!python -u tools/run_search_games.py {cmd_args}
"""


def build_worker_profile(
    worker_id: int,
    workers: Mapping[int, Mapping[str, Any]] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy one account profile and replace its collection parameters."""
    settings = {**CONFIG, **(config or {})}
    registry = WORKERS if workers is None else workers
    if worker_id not in registry:
        raise ValueError(f"Unknown worker identifier: {worker_id}.")
    revision = str(settings["revision"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The revision must be a full lowercase Git commit hash.")
    if int(settings["pairs"]) <= 0 or int(settings["workers"]) <= 0:
        raise ValueError("The pair count and worker count must be positive.")
    worker = deepcopy(dict(registry[worker_id]))
    drive_dir = f"{str(settings['drive_root']).rstrip('/')}/worker_{worker_id}"
    checkout_dir = f"{str(settings['checkout_root']).rstrip('/')}/{revision}"
    runtime_dir = (
        f"{str(settings['checkout_root']).rstrip('/')}_runtime/{revision}/worker_{worker_id}"
    )
    runtime_cache = (
        f"{str(settings['drive_root']).rstrip('/')}/_runtime_worker_{worker_id}/{revision}"
    )
    seed = int(settings["seed_stride"]) * worker_id
    arguments = [
        "--mode", str(settings["mode"]),
        "--pairs", str(settings["pairs"]),
        "--workers", str(settings["workers"]),
        "--seed", str(seed),
        "--claustrophobia-device", str(settings["claustrophobia_device"]),
        "--start-move-time-ms", str(settings["start_move_time_ms"]),
        "--end-move-time-ms", str(settings["end_move_time_ms"]),
        "--decay-start-ply", str(settings["decay_start_ply"]),
        "--decay-end-ply", str(settings["decay_end_ply"]),
        "--schedule-origin", str(settings["schedule_origin"]),
        "--opening-temperature", str(settings["opening_temperature"]),
        "--temperature-plies", str(settings["temperature_plies"]),
        "--output", drive_dir,
        "--zq-executable", f"{runtime_dir}/zquoridor_uci",
        "--claustrophobia-bridge", f"{runtime_dir}/zq_benchmark_bridge",
        "--claustrophobia-checkpoint", f"{runtime_dir}/champion.pt",
        "--no-auto-setup",
        "--resume",
        "--export-targets",
    ]
    for book, weight in settings["opening_weights"].items():
        arguments.extend(["--opening-weight", f"{book}={weight}"])
    if settings["record_both_searches"]:
        arguments.append("--record-both-searches")
    if settings["unique_openings_first"]:
        arguments.append("--unique-openings-first")
    worker.update(
        worker_id=worker_id,
        drive_dir=drive_dir,
        total_games=int(settings["pairs"]) * 2,
        pairs=int(settings["pairs"]),
        seed=seed,
        positions="",
        extra_args=shlex.join(arguments),
        target_keywords=["run_search_games.py", "claustro_search_games_v1"],
        target_shard_prefix=f"worker_{worker_id}",
        checkout_dir=checkout_dir,
        checkout_dir_literal=repr(checkout_dir),
        drive_dir_literal=repr(drive_dir),
        runtime_dir_literal=repr(runtime_dir),
        runtime_cache_literal=repr(runtime_cache),
        revision_literal=repr(revision),
        repository_url_literal=repr(str(settings["repository_url"])),
    )
    return worker


def build_worker_profiles(
    workers: Mapping[int, Mapping[str, Any]] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    """Return independent profiles for each configured account."""
    settings = {**CONFIG, **(config or {})}
    return {
        worker_id: build_worker_profile(worker_id, workers, settings)
        for worker_id in settings["worker_ids"]
    }


def render_bootloader(worker: Mapping[str, Any]) -> str:
    """Format the notebook cell through the generic launcher contract."""
    return BOOTLOADER_TEMPLATE.format(**worker, cmd_args=worker["extra_args"])
