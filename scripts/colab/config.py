"""Central configuration and worker registry for Colab automation in Zquoridor.

This module defines worker profiles, notebook URLs, CDP ports, and the resilient
bootloader template executed on Google Colab virtual machines.
"""

from typing import Dict, Any, List

# Worker configurations for distributed Google Colab self-play generation.
WORKERS: Dict[int, Dict[str, Any]] = {
    3: {
        "name": "Colab 3",
        "worker_id": 3,
        "account": "zambraprojects@gmail.com",
        "profile_dir": r"C:\Projetos\TikTok\profiles\zambraprojects",
        "notebook_url": "https://colab.research.google.com/drive/1tTPVhIs4Jq0Qr1yHNPRfBDtohE3EjX9E",
        "cdp_port": 9003,
        "target_keywords": ["run_colab_worker.py", "selfplay_15m", "zquoridor"],
        "target_shard_prefix": "c3_",
    },
    4: {
        "name": "Colab 4",
        "worker_id": 4,
        "account": "zquoridor@gmail.com",
        "profile_dir": r"C:\Projetos\TikTok\profiles\zquoridor",
        "notebook_url": "https://colab.research.google.com/drive/1nC1LOjwFm1LyeJtx4kxg6T7wQXym9FnA",
        "cdp_port": 9004,
        "target_keywords": ["run_colab_worker.py", "selfplay_15m", "zquoridor"],
        "target_shard_prefix": "c4_",
    },
    5: {
        "name": "Colab 5",
        "worker_id": 5,
        "account": "gustati2201@gmail.com",
        "profile_dir": r"C:\Projetos\TikTok\profiles\gustati2201",
        "notebook_url": "https://colab.research.google.com/drive/1cPMl8_zEi-el5GAE8sv2Bw3T6uKDwK1i",
        "cdp_port": 9005,
        "target_keywords": ["run_colab_worker.py", "selfplay_15m", "zquoridor"],
        "target_shard_prefix": "c5_",
    },
}

# Aliases mapping worker strings/names to canonical integer IDs
WORKER_ALIASES: Dict[Any, int] = {
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
!python tools/selfplay/run_colab_worker.py --worker-id {worker_id}"""
