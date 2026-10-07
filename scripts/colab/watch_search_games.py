"""Watch Colab search collection and resume the frozen workflow after disconnects."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.colab.switch_search_games import CONFIG as SWITCH_CONFIG, main as switch_main
from tools.external.build_lock import build_lock

CONFIG = {**SWITCH_CONFIG, "driver_recovery_attempts": 3, "driver_retry_seconds": 60}


def driver_failed(states: dict) -> bool:
    """Identify a closed browser driver without retrying account authentication."""
    errors = [str(state.get("error", "")).lower() for state in states.values()
              if state.get("status") == "blocked"]
    return bool(errors) and any("driver" in error and
                               ("closed" in error or "connection" in error) for error in errors)


def main() -> None:
    """Recreate the browser driver after a bounded recovery delay."""
    output = Path(CONFIG["output"])
    with build_lock(output / "watcher_lock"):
        for attempt in range(CONFIG["driver_recovery_attempts"] + 1):
            try:
                switch_main()
                status_path = output / "status.json"
                states = json.loads(status_path.read_text()) if status_path.is_file() else {}
                recover = driver_failed(states)
            except Exception as error:
                message = str(error).lower()
                recover = "driver" in message and ("closed" in message or "connection" in message)
                if not recover:
                    raise
            if not recover or attempt == CONFIG["driver_recovery_attempts"]:
                return
            print("The browser driver closed. Retry the frozen workflow after 60 seconds.", flush=True)
            time.sleep(CONFIG["driver_retry_seconds"])


if __name__ == "__main__":
    main()
