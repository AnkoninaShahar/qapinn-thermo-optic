from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.steady_experiment import run_steady_experiments


if __name__ == "__main__":
    run_steady_experiments()
