from __future__ import annotations

import csv
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .classical_pinn import TransientClassicalPINN
    from .config import ProjectConfig, load_config, validate_config
    from .evaluate import evaluate_model, write_run_record
    from .explainability import save_explainability_outputs
    from .losses import transient_pde_loss
    from .qapinn import QAPINN
except ImportError:  # pragma: no cover
    from classical_pinn import TransientClassicalPINN
    from config import ProjectConfig, load_config, validate_config
    from evaluate import evaluate_model, write_run_record
    from explainability import save_explainability_outputs
    from losses import transient_pde_loss
    from qapinn import QAPINN


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dtype_from_config(config: ProjectConfig) -> torch.dtype:
    if config.training.dtype != "float64":
        raise ValueError("MVP requires float64 for second derivatives")
    return torch.float64


def make_collocation_points(config: ProjectConfig, seed: int, dtype: torch.dtype) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    xi = rng.uniform(config.domain.xi_min, config.domain.xi_max, config.training.collocation_points)
    tau = rng.uniform(config.domain.tau_min, config.domain.tau_max, config.training.collocation_points)
    points = np.column_stack([xi, tau])
    return torch.tensor(points, dtype=dtype)


def make_model(model_id: str, config: ProjectConfig) -> torch.nn.Module:
    if model_id == "classical_matched":
        model = TransientClassicalPINN(hidden_layers=config.training.hidden_layers)
    elif model_id == "classical_standard":
        model = TransientClassicalPINN(hidden_layers=config.training.standard_hidden_layers)
    elif model_id == "q0_separable":
        model = QAPINN("q0_separable", tail_layers=config.training.quantum_tail_layers)
    elif model_id == "q1_entangled":
        model = QAPINN("q1_entangled", tail_layers=config.training.quantum_tail_layers)
    else:
        raise ValueError(f"unknown MVP model_id: {model_id}")
    return model.double()


def _grad_norm(parameters) -> float:
    squared = 0.0
    for param in parameters:
        if param.grad is not None:
            squared += float(torch.sum(param.grad.detach() ** 2))
    return float(squared**0.5)


def gradient_summary(model: torch.nn.Module) -> dict[str, float]:
    if hasattr(model, "quantum"):
        quantum_norm = _grad_norm(model.quantum.parameters())
        first_classical_norm = _grad_norm(model.classical_tail[0].parameters())
    else:
        quantum_norm = 0.0
        first_classical_norm = _grad_norm(model.net[0].parameters())
    total_norm = _grad_norm(model.parameters())
    return {
        "grad_norm_quantum": quantum_norm,
        "grad_norm_first_classical": first_classical_norm,
        "grad_norm_total": total_norm,
    }


def train_one(
    model_id: str,
    seed: int,
    config: ProjectConfig,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = dtype_from_config(config)
    set_all_seeds(seed)
    model = make_model(model_id, config)
    points = make_collocation_points(config, seed + 10_000, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    for epoch in range(1, config.training.epochs + 1):
        optimizer.zero_grad()
        loss = transient_pde_loss(model, points, config.heater)
        loss.backward()
        grads = gradient_summary(model)
        optimizer.step()
        if epoch == 1 or epoch % config.training.log_interval == 0 or epoch == config.training.epochs:
            history.append({"epoch": epoch, "loss": float(loss.detach()), **grads})

    runtime_seconds = time.perf_counter() - start
    metrics = evaluate_model(model, config, runtime_seconds, seed, model_id, output_dir, dtype)
    explainability = save_explainability_outputs(model, config, model_id, seed, output_dir, dtype)
    record: dict[str, Any] = {
        "model_id": model_id,
        "seed": seed,
        "git_commit": git_commit_hash(),
        "training_history": history,
        "final_training_loss": history[-1]["loss"],
        **metrics,
        **explainability,
    }
    write_run_record(output_dir / f"{model_id}_seed{seed}.json", record, config)
    return record


def write_metrics_csv(records: list[dict[str, Any]], path: Path) -> None:
    keys = [
        "model_id",
        "seed",
        "temperature_relative_l2",
        "pde_residual_rms",
        "active_phase_rmse",
        "victim_phase_rmse",
        "crosstalk_abs_error",
        "trainable_parameters",
        "quantum_parameters",
        "gate_count",
        "circuit_depth",
        "runtime_seconds",
        "final_training_loss",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in keys})


def run_mvp(
    config_path: str = "configs/physics.yaml",
    models: tuple[str, ...] = ("classical_matched", "q0_separable", "q1_entangled"),
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    validate_config(config)
    output_dir = Path("results/runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for model_id in models:
        for seed in config.training.seeds:
            print(f"training {model_id} seed={seed}")
            records.append(train_one(model_id, seed, config, output_dir))
    write_metrics_csv(records, Path("results/metrics/mvp_metrics.csv"))
    return records


if __name__ == "__main__":
    run_mvp()
