"""Central configuration and worker registry for Colab automation in Zquoridor.

This module defines worker profiles, notebook URLs, CDP ports, and the resilient
bootloader template executed on Google Colab virtual machines.
"""

from typing import Dict, Any, List

# Worker configurations for distributed Google Colab self-play generation.
WORKERS: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Colab 1",
        "worker_id": 1,
        "account": "gustavozambrano@gmail.com",
        "profile_dir": r"C:\Projetos\TikTok\profiles\gustavozambrano",
        "notebook_url": "https://colab.research.google.com/github/gitzambrano/zquoridor/blob/main/notebooks/colab_central_worker.ipynb",
        "cdp_port": 9001,
        "target_keywords": ["run_colab_worker.py", "selfplay_central", "zquoridor"],
        "target_shard_prefix": "c1_",
        "drive_dir": "/content/drive/MyDrive/zquoridor_data/selfplay_central",
        "positions": "tools/external/openings_center_rush_sound_5k.jsonl",
    },
    2: {
        "name": "Colab 2",
        "worker_id": 2,
        "account": "flightdyn@gmail.com",
        "profile_dir": r"C:\Projetos\TikTok\profiles\flightdyn",
        "notebook_url": "https://colab.research.google.com/github/gitzambrano/zquoridor/blob/main/notebooks/colab_irregular_worker.ipynb",
        "cdp_port": 9002,
        "target_keywords": ["run_colab_worker.py", "selfplay_irregular", "zquoridor"],
        "target_shard_prefix": "c2_",
        "drive_dir": "/content/drive/MyDrive/zquoridor_data/selfplay_irregular",
        "positions": "tools/external/openings_irregular_bank.jsonl",
    },
    3: {
        "name": "Colab 3",
        "worker_id": 3,
        "account": "zambraprojects@gmail.com",
        "profile_dir": r"C:\Projetos\TikTok\profiles\zambraprojects",
        "notebook_url": "https://colab.research.google.com/drive/1tTPVhIs4Jq0Qr1yHNPRfBDtohE3EjX9E",
        "cdp_port": 9003,
        "target_keywords": ["run_colab_worker.py", "selfplay_targeted_weakness", "selfplay_15m", "zquoridor"],
        "target_shard_prefix": "c3_",
        "drive_dir": "/content/drive/MyDrive/zquoridor_data/selfplay_targeted_weakness",
        "positions": "tools/external/openings_targeted_weakness_bank.jsonl",
    },
    4: {
        "name": "Colab 4",
        "worker_id": 4,
        "account": "zquoridor@gmail.com",
        "profile_dir": r"C:\Projetos\TikTok\profiles\zquoridor",
        "notebook_url": "https://colab.research.google.com/github/gitzambrano/zquoridor/blob/main/notebooks/colab_weakness_worker.ipynb",
        "cdp_port": 9004,
        "target_keywords": ["run_colab_worker.py", "selfplay_targeted_weakness", "selfplay_15m", "zquoridor"],
        "target_shard_prefix": "c4_",
        "drive_dir": "/content/drive/MyDrive/zquoridor_data/selfplay_targeted_weakness",
        "positions": "tools/external/openings_targeted_weakness_bank.jsonl",
    },
    5: {
        "name": "Colab 5",
        "worker_id": 5,
        "account": "gustati2201@gmail.com",
        "profile_dir": r"C:\Projetos\TikTok\profiles\gustati2201",
        "notebook_url": "https://colab.research.google.com/drive/1cPMl8_zEi-el5GAE8sv2Bw3T6uKDwK1i",
        "cdp_port": 9005,
        "target_keywords": ["run_colab_worker.py", "selfplay_targeted_weakness", "selfplay_15m", "zquoridor"],
        "target_shard_prefix": "c5_",
        "drive_dir": "/content/drive/MyDrive/zquoridor_data/selfplay_targeted_weakness",
        "positions": "tools/external/openings_targeted_weakness_bank.jsonl",
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
!git pull origin main
!bash build/build_selfplay.sh
!python tools/selfplay/run_colab_worker.py --worker-id {worker_id} --drive-dir {drive_dir} --positions {positions}"""
