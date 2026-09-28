from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quantum_feasibility import run_quantum_derivative_gate


def main() -> None:
    result = run_quantum_derivative_gate()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passes"]:
        raise SystemExit("quantum derivative feasibility gate failed")


if __name__ == "__main__":
    main()
