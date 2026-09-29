"""Central configuration and worker registry for Colab automation.

This module defines worker profiles, notebook URLs, and the resilient
bootloader template executed on Google Colab virtual machines.
"""

from typing import Dict, Any

# Worker configurations for distributed Google Colab self-play generation.
WORKERS: Dict[int, Dict[str, Any]] = {
    3: {
        "name": "Colab 3",
        "worker_id": 3,
        "account": "worker-3@example.invalid",
        "profile_dir": r"COLAB_PRIVATE_PROFILE_3",
        "notebook_url": "https://colab.research.google.com/drive/NOTEBOOK_ID_3",
        "cdp_port": 9003,
        "target_shard_prefix": "c3_",
    },
    4: {
        "name": "Colab 4",
        "worker_id": 4,
        "account": "worker-4@example.invalid",
        "profile_dir": r"COLAB_PRIVATE_PROFILE_4",
        "notebook_url": "https://colab.research.google.com/drive/NOTEBOOK_ID_4",
        "cdp_port": 9004,
        "target_shard_prefix": "c4_",
    },
    5: {
        "name": "Colab 5",
        "worker_id": 5,
        "account": "worker-5@example.invalid",
        "profile_dir": r"COLAB_PRIVATE_PROFILE_5",
        "notebook_url": "https://colab.research.google.com/drive/NOTEBOOK_ID_5",
        "cdp_port": 9005,
        "target_shard_prefix": "c5_",
    },
}

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
