"""Compatibility entry point. Use colab.remote.finish_current_shard for the implementation."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from colab.compat import redirect
redirect(__name__, 'colab.remote.finish_current_shard')
