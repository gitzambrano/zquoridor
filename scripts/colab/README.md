# Colab Worker Orchestration Suite

This directory contains automated, headless Playwright tooling to inspect,
bootstrap, and manage remote Google Colab self-play workers for Zquoridor.

## Worker Registry

| Worker ID | Name | Google Account | Notebook URL | Shard Prefix |
| :---: | :---: | :---: | :---: | :---: |
| 3 | Colab 3 | `zambraprojects@gmail.com` | [Notebook Colab 3](https://colab.research.google.com/drive/1tTPVhIs4Jq0Qr1yHNPRfBDtohE3EjX9E) | `c3_shard_*.bin` |
| 4 | Colab 4 | `zquoridor@gmail.com` | [Notebook Colab 4](https://colab.research.google.com/drive/1nC1LOjwFm1LyeJtx4kxg6T7wQXym9FnA) | `c4_shard_*.bin` |
| 5 | Colab 5 | `gustati2201@gmail.com` | [Notebook Colab 5](https://colab.research.google.com/drive/1cPMl8_zEi-el5GAE8sv2Bw3T6uKDwK1i) | `c5_shard_*.bin` |

## Scripts

### 1. `inspect_workers.py`
Inspects the current state of all workers headlessly without interrupting ongoing runs:
```bash
python scripts/colab/inspect_workers.py
```
- Checks runtime connection state (`CONNECTED`, `CONNECTING`, `DISCONNECTED`).
- Checks execution status (`ACTIVE RUNNING`, `PENDING VM`, `IDLE`).
- Detects GPU quota exhaustion warnings.
- Displays recent self-play progress (game count and positions).

### 2. `launch_workers.py`
Bootstraps workers with the resilient self-cloning cell code and starts generation:
```bash
python scripts/colab/launch_workers.py
```
- Skips workers that are already actively running (to prevent interrupting progress).
- Connects runtime if disconnected.
- Updates Monaco and notebook cell models with the self-cloning bootloader.
- Clicks the run button and dismisses confirmation dialogs.
- Override flags:
  - `--worker-ids 4 5`: Run specific workers.
  - `--force-restart`: Force re-execution even if the worker is currently running.

### 3. `manage_runtime.py`
Performs administrative operations on Colab notebook environments:
```bash
# Switch accelerator from GPU to standard CPU (resolves quota blocks)
python scripts/colab/manage_runtime.py --action switch-cpu

# Terminate stuck ghost sessions in "Gerenciar sessões"
python scripts/colab/manage_runtime.py --action terminate-active

# Disconnect and delete VM environment
python scripts/colab/manage_runtime.py --action reset
```

## Resilient Bootloader Logic

Google Colab free-tier VMs recycle every 12 hours. Upon recycling, `/content/zquoridor`
is erased. The resilient bootloader checks for local existence and clones the
repository automatically:

```python
from google.colab import drive
import os

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

if not os.path.exists('/content/zquoridor'):
    !git clone https://github.com/gitzambrano/zquoridor.git /content/zquoridor

%cd /content/zquoridor
!git pull origin main
!bash build/build_selfplay.sh
!python tools/selfplay/run_colab_worker.py --worker-id {worker_id}
```

> [!IMPORTANT]
> **CPU vs GPU Configuration**:
> Zquoridor self-play runs 100% in multi-threaded C++ using AVX2 SIMD and int8
> quantized NNUE on the CPU. It does not use the GPU. Always ensure the notebook
> hardware accelerator is set to **CPU** (Standard). Configuring GPU will exhaust
> daily free-tier compute units and block VM provisioning.
