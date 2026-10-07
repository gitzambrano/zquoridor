# Search Game Collection on Google Colab

Use a Colab GPU runtime for Claustrophobia inference. Mount Google Drive before the run so the game ledger and exports persist after the runtime ends.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Clone the current `main` branch. Then enter the repository.

```python
!git clone --branch main https://github.com/gitzambrano/zquoridor.git /content/zquoridor
%cd /content/zquoridor
```

Check the GPU and compiler dependencies. The setup requires PyTorch, NumPy,
Git, a C++ compiler, and a Rust toolchain. Colab supplies PyTorch and NumPy.
Install Rust if the runtime does not provide Cargo.

```python
import os
import shutil
import subprocess
import torch

assert torch.cuda.is_available(), 'Select a GPU runtime before collection.'
assert shutil.which('g++') and shutil.which('git'), 'Install the native build dependencies.'
if not shutil.which('cargo') and not os.path.isfile('/root/.cargo/bin/cargo'):
    subprocess.run(
        ['bash', '-c', 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'],
        check=True,
    )
os.environ['PATH'] = '/root/.cargo/bin:' + os.environ['PATH']
os.environ['CARGO_BUILD_JOBS'] = '1'
subprocess.run(['cargo', '--version'], check=True)
```

Check the opening selection before the long run. The default books are committed in the repository. The weakness book has 63 rows, so a large pair count repeats weakness openings. Read `unique_openings` and `repeated_opening_pairs` in the dry-run output before you start.

```python
!python tools/run_search_games.py \
  --mode match \
  --pairs 50000 \
  --batch-games 250 \
  --compress-game-ledger \
  --no-export-final-run \
  --workers 1 \
  --start-move-time-ms 400 \
  --end-move-time-ms 50 \
  --decay-start-ply 14 \
  --decay-end-ply 80 \
  --schedule-origin opening \
  --claustrophobia-device gpu \
  --opening-weight center_rush=7 \
  --opening-weight normal=2 \
  --opening-weight weakness=1 \
  --opening-temperature 1.0 \
  --temperature-plies 14 \
  --record-both-searches \
  --unique-openings-first \
  --opening-book normal=tools/external/openings_irregular_bank.jsonl \
  --opening-book center_rush=tools/external/openings_center_rush_sound_5k.jsonl \
  --opening-book weakness=tools/external/weak_openings_mined.jsonl \
  --output /content/drive/MyDrive/Zquoridor/search_games \
  --dry-run
```

Run the same command without `--dry-run` to collect games. The schedule uses the opening as its origin. It keeps 400 ms through searched ply 14 after each supplied opening, then tapers linearly to 50 ms at searched ply 80. The pair count schedules each opening for both Zquoridor colors, so 50000 pairs means 100000 games. This count does not mean 100000 unique openings.

The output directory stores one JSON record per game under `games/`. It also stores `games.jsonl`, `positions.jsonl`, `labels.jsonl`, `teacher_targets.npz`, `summary.json`, and manifests. Positions include replayable move history. Labels contain normalized 209-action root visit policies and side-to-move root values from the actual Claustrophobia searches. Only complete goal or repetition games contribute training labels.

The runner resumes by default when the manifest matches. Use the same command and output directory after a runtime interruption. A game interrupted during a search is recorded as interrupted and is not retried. The run does not generate missing opening books. Automatic bot setup fetches the pinned Claustrophobia checkpoint; it does not train or create a model.

This command allocates 35000 pairs to Center Rush, 10000 to Normal, and 5000 to weakness openings. Temperature 1.0 samples actual visit distributions for the first 14 searched plies after each opening. The record preserves the best move and the sampled move separately. Both engines search each identical position with the same budget. The raw game ledger retains both roots for later comparisons and dataset weighting. Paired searches increase collection time.

To increase weakness-opening diversity, create or provide a larger weakness book before the run. The collector reports repeated openings but does not generate opening variations.

The native runner writes an atomic compressed batch every 250 games. Each batch includes both raw searches, replayable histories, labels, targets, hashes, and a manifest. The compressed per-game ledger preserves progress between batches. The `--no-export-final-run` option avoids a complete ledger rebuild after this large run. Resume uses saved batch identities and game records.

Run `python scripts/colab/watch_search_games.py` locally for automatic notebook reconnection and workflow resume. Use one watcher for the five account profiles. The watcher does not manage individual game batches.
