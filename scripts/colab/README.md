# Colab Worker Orchestration Suite

This directory contains automated, headless Playwright tooling to inspect,
bootstrap, monitor, and manage remote Google Colab self-play workers for Zquoridor.

## Worker Registry

| Worker ID | Name | Google Account | Notebook URL | Shard Prefix |
| :---: | :---: | :---: | :---: | :---: |
| 1 | Colab 1 | `gustavozambrano@gmail.com` | [Notebook Colab 1](https://colab.research.google.com/drive/13O4yYpM8ElgOFzo774Jps488DAPzIbDT) | `c1_shard_*.bin` |
| 2 | Colab 2 | `flightdyn@gmail.com` | [Notebook Colab 2](https://colab.research.google.com/drive/1rPSnvqg7stxwU5V8ITgpGVv02j7qVsD1) | `c2_shard_*.bin` |
| 3 | Colab 3 | `zambraprojects@gmail.com` | [Notebook Colab 3](https://colab.research.google.com/drive/1tTPVhIs4Jq0Qr1yHNPRfBDtohE3EjX9E) | `c3_shard_*.bin` |
| 4 | Colab 4 | `zquoridor@gmail.com` | [Notebook Colab 4](https://colab.research.google.com/drive/1nC1LOjwFm1LyeJtx4kxg6T7wQXym9FnA) | `c4_shard_*.bin` |
| 5 | Colab 5 | `gustati2201@gmail.com` | [Notebook Colab 5](https://colab.research.google.com/drive/1cPMl8_zEi-el5GAE8sv2Bw3T6uKDwK1i) | `c5_shard_*.bin` |
| 6 | Colab 6 | `zchezzproject@gmail.com` | [Notebook Colab 6](https://colab.research.google.com/drive/1j8-gG7qApv--t2DBM8tf1gmgKjuUXiLC) | `c6_shard_*.bin` |
| 7 | Colab 7 | `zbrainproject@gmail.com` | [Notebook Colab 7](https://colab.research.google.com/drive/1WaoYFjPIl70cECrs9CGZEwEoMxVzBEp8) | `c7_shard_*.bin` |

## The 3 Core Tools

### 1. Disparar: `launch_workers.py`
Bootstraps workers with the resilient self-cloning cell code and starts generation:
```bash
python scripts/colab/launch_workers.py
```
- Skips workers that are already actively running (to prevent interrupting progress).
- Connects runtime if disconnected.
- Updates Monaco and notebook cell models with the self-cloning bootloader.
- Clicks the run button and dismisses confirmation dialogs.
- Override flags:
  - `--worker-ids 3 4 5`: Select specific workers.
  - `--force-restart`: Force re-execution even if the worker is currently running.

### 2. Monitorar: `watchdog_workers.py`
Continuous active watchdog and keep-alive monitor:
```bash
python scripts/colab/watchdog_workers.py
```
- Keeps persistent browser sessions open with periodic micro-interactions (mouse moves) to prevent Google Colab idle timeout disconnects.
- Continuous loop reporting real-time game and position counts.
- Detects VM disconnects and automatically reconnects and re-triggers execution.
- Captures periodic health screenshots in `artifacts/colab/`.

### 3. Reportar: `report_workers.py` (ou `inspect_workers.py`)
Snapshot inspection and structured audit:
```bash
python scripts/colab/report_workers.py
```
- Audits runtime status (`CONNECTED`, `CONNECTING`, `DISCONNECTED`).
- Audits execution status (`ACTIVE RUNNING`, `PENDING VM`, `IDLE`).
- Parses live progress: current shard ID, games completed, positions generated, and speed metrics.
- Saves clean visual screenshots for all workers to `artifacts/colab/`.
- Generates a consolidated Markdown audit report at `artifacts/colab/report.md`.

### Auxiliary Tool: `manage_runtime.py`
Administrative operations on Colab notebook environments:
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
