from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis_outputs import run_all_analyses


if __name__ == "__main__":
    run_all_analyses()
