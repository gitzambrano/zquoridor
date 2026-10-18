"""Compatibility entry point. Use colab.remote.run_colab_arena for the implementation."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from colab.compat import redirect
redirect(__name__, 'colab.remote.run_colab_arena')
