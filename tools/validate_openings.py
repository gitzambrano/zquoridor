import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.external.local_arena import Referee

def check_file(path_str):
    p = Path(path_str)
    print(f"Checking {p}...")
    errors = 0
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            row = json.loads(line)
            ref = Referee()
            for ply, m in enumerate(row["moves"]):
                try:
                    ref.apply(m)
                except Exception as e:
                    print(f"  Line {i} ply {ply} ({row.get('description')}): move '{m}' failed: {e}")
                    print(f"    History: {row['moves'][:ply+1]}")
                    errors += 1
                    break
    if errors == 0:
        print("  All openings valid!")
    else:
        print(f"  Found {errors} errors.")

if __name__ == "__main__":
    check_file("tools/external/openings_center_rush_50pairs.jsonl")
