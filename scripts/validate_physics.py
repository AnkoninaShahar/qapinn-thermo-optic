from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, validate_config
from src.reference_transient import (
    reference_convergence_check,
    solve_transient_crank_nicolson,
    validate_reference_solution,
)


def main() -> None:
    config = load_config("configs/physics.yaml")
    validate_config(config)
    xi, tau, theta = solve_transient_crank_nicolson(config)
    validation = validate_reference_solution(xi, tau, theta)
    convergence = reference_convergence_check(config)
    output = {
        "reference_validation": validation,
        "convergence": convergence,
        "peak_theta": float(np.max(theta)),
    }
    Path("results/reference").mkdir(parents=True, exist_ok=True)
    np.savez_compressed("results/reference/transient_reference.npz", xi=xi, tau=tau, theta=theta)
    with Path("results/metrics/reference_validation.json").open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
    print(json.dumps(output, indent=2, sort_keys=True))
    if not validation["passes"]:
        raise SystemExit("reference validation failed")


if __name__ == "__main__":
    main()
