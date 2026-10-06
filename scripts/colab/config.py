"""Central configuration and worker registry for Colab automation in Zquoridor.

This module defines worker profiles, notebook URLs, CDP ports, and the resilient
bootloader template executed on Google Colab virtual machines.
"""

from pathlib import Path
from typing import Dict, Any, List

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

# Worker configurations for distributed Google Colab self-play generation.
WORKERS: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Colab 1",
        "worker_id": 1,
        "account": "gustavozambrano@gmail.com",
        "profile_dir": str(PROFILES_DIR / "gustavozambrano"),
        "notebook_url": "https://colab.research.google.com/drive/13O4yYpM8ElgOFzo774Jps488DAPzIbDT",
        "cdp_port": 9001,
        "target_keywords": ["run_colab_worker.py", "selfplay_contact_soup_858", "contact_soup", "zquoridor"],
        "target_shard_prefix": "c1_",
        "drive_dir": "/content/drive/MyDrive/zquoridor_data/selfplay_contact_soup_858",
        "total_games": 80000,
        "positions": "tools/external/openings_center_rush_sound_5k.jsonl",
        "extra_args": "--weights results/experiments/contact_soup_tri_512/soup_tri_champion/student_int8.bin --seed 1000001 --playout-cap --time-ms 400 --cheap-time-ms 50 --full-search-opening-plies 14 --full-search-prob 0.35",
    },
    2: {
        "name": "Colab 2",
        "worker_id": 2,
        "account": "flightdyn@gmail.com",
        "profile_dir": str(PROFILES_DIR / "flightdyn"),
        "notebook_url": "https://colab.research.google.com/drive/1rPSnvqg7stxwU5V8ITgpGVv02j7qVsD1",
        "cdp_port": 9002,
        "target_keywords": ["run_colab_worker.py", "selfplay_contact_soup_858", "contact_soup", "zquoridor"],
        "target_shard_prefix": "c2_",
        "drive_dir": "/content/drive/MyDrive/zquoridor_data/selfplay_contact_soup_858",
        "total_games": 80000,
        "positions": "tools/external/openings_irregular_bank.jsonl",
        "extra_args": "--weights results/experiments/contact_soup_tri_512/soup_tri_champion/student_int8.bin --seed 2000002 --playout-cap --time-ms 400 --cheap-time-ms 50 --full-search-opening-plies 14 --full-search-prob 0.35",
    },
    3: {
        "name": "Colab 3",
        "worker_id": 3,
        "account": "zambraprojects@gmail.com",
        "profile_dir": str(PROFILES_DIR / "zambraprojects"),
        "notebook_url": "https://colab.research.google.com/drive/1tTPVhIs4Jq0Qr1yHNPRfBDtohE3EjX9E",
        "cdp_port": 9003,
        "target_keywords": ["run_colab_worker.py", "selfplay_contact_soup_858", "contact_soup", "zquoridor"],
        "target_shard_prefix": "c3_",
        "drive_dir": "/content/drive/MyDrive/zquoridor_data/selfplay_contact_soup_858",
        "total_games": 80000,
        "positions": "tools/external/openings_weakness_variations.jsonl",
        "extra_args": "--weights results/experiments/contact_soup_tri_512/soup_tri_champion/student_int8.bin --seed 3000003 --playout-cap --time-ms 400 --cheap-time-ms 50 --full-search-opening-plies 14 --full-search-prob 0.35",
    },
    4: {
        "name": "Colab 4",
        "worker_id": 4,
        "account": "zquoridor@gmail.com",
        "profile_dir": str(PROFILES_DIR / "zquoridor"),
        "notebook_url": "https://colab.research.google.com/drive/1nC1LOjwFm1LyeJtx4kxg6T7wQXym9FnA",
        "cdp_port": 9004,
        "target_keywords": ["run_colab_worker.py", "selfplay_contact_soup_858", "contact_soup", "zquoridor"],
        "target_shard_prefix": "c4_",
        "drive_dir": "/content/drive/MyDrive/zquoridor_data/selfplay_contact_soup_858",
        "total_games": 80000,
        "positions": "tools/external/openings_weakness_variations.jsonl",
        "extra_args": "--weights results/experiments/contact_soup_tri_512/soup_tri_champion/student_int8.bin --seed 4000004 --playout-cap --time-ms 400 --cheap-time-ms 50 --full-search-opening-plies 14 --full-search-prob 0.35",
    },
    5: {
        "name": "Colab 5",
        "worker_id": 5,
        "account": "gustati2201@gmail.com",
        "profile_dir": str(PROFILES_DIR / "gustati2201"),
        "notebook_url": "https://colab.research.google.com/drive/1cPMl8_zEi-el5GAE8sv2Bw3T6uKDwK1i",
        "cdp_port": 9005,
        "target_keywords": ["run_colab_worker.py", "selfplay_contact_soup_858", "contact_soup", "zquoridor"],
        "target_shard_prefix": "c5_",
        "drive_dir": "/content/drive/MyDrive/zquoridor_data/selfplay_contact_soup_858",
        "total_games": 80000,
        "extra_args": "--weights results/experiments/contact_soup_tri_512/soup_tri_champion/student_int8.bin --seed 5000005 --mc-obvious-plies 10 --mc-temp-obvious 2.5 --mc-temp-opening 1.2 --mc-temp-decay-plies 30 --mc-temp-end 0.12 --playout-cap --time-ms 400 --cheap-time-ms 200 --cheap-time-end-ms 50 --cheap-time-decay-plies 30 --full-search-opening-plies 14 --full-search-prob 0.35",
    },
}

# Aliases mapping worker strings/names to canonical integer IDs
WORKER_ALIASES: Dict[Any, int] = {
    1: 1,
    "1": 1,
    "c1": 1,
    "colab1": 1,
    2: 2,
    "2": 2,
    "c2": 2,
    "colab2": 2,
    3: 3,
    "3": 3,
    "c3": 3,
    "colab3": 3,
    4: 4,
    "4": 4,
    "c4": 4,
    "colab4": 4,
    5: 5,
    "5": 5,
    "c5": 5,
    "colab5": 5,
}


def resolve_worker_key(key: Any) -> int:
    """Normalize input worker identifier to canonical integer worker ID."""
    if key in WORKER_ALIASES:
        return WORKER_ALIASES[key]
    key_str = str(key).strip().lower()
    if key_str in WORKER_ALIASES:
        return WORKER_ALIASES[key_str]
    try:
        val = int(key_str)
        if val in WORKERS:
            return val
    except ValueError:
        pass
    raise ValueError(f"Unknown worker identifier: {key}. Supported: {list(WORKERS.keys())}")


def get_all_worker_keys() -> List[int]:
    """Return all canonical worker IDs."""
    return list(WORKERS.keys())


# Resilient Python/bash bootloader executed inside the Colab cell.
# Automatically recovers repository state when the free VM recycles after 12 hours.
BOOTLOADER_TEMPLATE = """from google.colab import drive
import os

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

if not os.path.exists('/content/zquoridor'):
    !git clone https://github.com/gitzambrano/zquoridor.git /content/zquoridor

%cd /content/zquoridor
!git fetch origin main
!git reset --hard origin/main
!bash build/build_selfplay.sh -DZQ_NNUE_CONTACT_FEATURES=1 -DZQ_NNUE_VALUE_BUCKETS=6 -DZQ_NNUE_VALUE_DEPTH=2 -DZQ_NNUE_HIDDEN=512
!python tools/selfplay/run_colab_worker.py --worker-id {worker_id} --drive-dir {drive_dir} {cmd_args}"""
