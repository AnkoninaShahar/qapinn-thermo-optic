from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config, validate_config
from src.train import run_mvp, write_metrics_csv


def main() -> None:
    config = load_config("configs/physics.yaml")
    validate_config(config)
    records = run_mvp(
        models=(
            "classical_matched",
            "classical_standard",
            "q0_separable",
            "q1_entangled",
        )
    )
    metrics_path = Path("results/metrics/final_transient_metrics.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(records, metrics_path)
    df = pd.read_csv(metrics_path)
    summary = df.groupby("model_id").agg(["mean", "std"])
    summary.to_csv("results/metrics/final_transient_summary.csv")


if __name__ == "__main__":
    main()
