from __future__ import annotations

import csv
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    from .classical_pinn import build_mlp
    from .config import ProjectConfig, config_to_dict, load_config
    from .phase_metrics import crosstalk_ratio, phase_history, relative_l2, rmse
    from .physics import source_from_config_torch
    from .qapinn import QuantumFeatureLayer
    from .reference_static import solve_static_from_config
except ImportError:  # pragma: no cover
    from classical_pinn import build_mlp
    from config import ProjectConfig, config_to_dict, load_config
    from phase_metrics import crosstalk_ratio, phase_history, relative_l2, rmse
    from physics import source_from_config_torch
    from qapinn import QuantumFeatureLayer
    from reference_static import solve_static_from_config


def _count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SteadyClassicalPINN(nn.Module):
    def __init__(self, hidden_layers: tuple[int, ...]):
        super().__init__()
        self.net = build_mlp(1, hidden_layers, 1)

    def first_layer_features(self, xi: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net[0](xi))

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        return (1.0 - xi**2) * self.net(xi)


class SteadyQAPINN(nn.Module):
    def __init__(self, architecture: str, tail_layers: tuple[int, ...]):
        super().__init__()
        if architecture == "q0_separable":
            entangle = False
        elif architecture == "q1_entangled":
            entangle = True
        else:
            raise ValueError("steady MVP supports q0_separable and q1_entangled")
        self.architecture = architecture
        self.quantum = QuantumFeatureLayer(entangle=entangle)
        self.classical_tail = build_mlp(2, tail_layers, 1)

    def _quantum_input(self, xi: torch.Tensor) -> torch.Tensor:
        return torch.cat([xi, torch.zeros_like(xi)], dim=1)

    def quantum_features(self, xi: torch.Tensor) -> torch.Tensor:
        return self.quantum(self._quantum_input(xi))

    def first_layer_features(self, xi: torch.Tensor) -> torch.Tensor:
        return self.quantum_features(xi)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        return (1.0 - xi**2) * self.classical_tail(self.quantum_features(xi))

    def quantum_parameter_count(self) -> int:
        return sum(param.numel() for param in self.quantum.parameters())

    def gate_count(self) -> int:
        return self.quantum.gate_count()

    def circuit_depth(self) -> int:
        return self.quantum.circuit_depth()


def steady_residual(model: torch.nn.Module, xi_points: torch.Tensor, config: ProjectConfig) -> torch.Tensor:
    xi = xi_points.clone().detach().requires_grad_(True)
    theta = model(xi)
    theta_x = torch.autograd.grad(theta, xi, torch.ones_like(theta), create_graph=True, retain_graph=True)[0]
    theta_xx = torch.autograd.grad(theta_x, xi, torch.ones_like(theta_x), create_graph=True, retain_graph=True)[0]
    source = source_from_config_torch(xi, config.heater)
    return -theta_xx - source


def make_steady_model(model_id: str, config: ProjectConfig) -> torch.nn.Module:
    if model_id == "classical_matched":
        model = SteadyClassicalPINN(config.training.hidden_layers)
    elif model_id == "classical_standard":
        model = SteadyClassicalPINN(config.training.standard_hidden_layers)
    elif model_id == "q0_separable":
        model = SteadyQAPINN("q0_separable", config.training.quantum_tail_layers)
    elif model_id == "q1_entangled":
        model = SteadyQAPINN("q1_entangled", config.training.quantum_tail_layers)
    else:
        raise ValueError(f"unknown steady model: {model_id}")
    return model.double()


def train_steady_one(model_id: str, seed: int, config: ProjectConfig, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed)
    model = make_steady_model(model_id, config)
    rng = np.random.default_rng(seed + 20_000)
    xi_collocation = rng.uniform(config.domain.xi_min, config.domain.xi_max, config.training.collocation_points)
    points = torch.tensor(xi_collocation.reshape(-1, 1), dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    for epoch in range(1, config.training.epochs + 1):
        optimizer.zero_grad()
        residual = steady_residual(model, points, config)
        loss = torch.mean(residual**2)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % config.training.log_interval == 0 or epoch == config.training.epochs:
            history.append({"epoch": epoch, "loss": float(loss.detach())})
    runtime_seconds = time.perf_counter() - start

    xi_ref, theta_ref = solve_static_from_config(config, config.training.eval_n_xi)
    xi_tensor = torch.tensor(xi_ref.reshape(-1, 1), dtype=torch.float64)
    with torch.no_grad():
        theta_pred = model(xi_tensor).detach().cpu().numpy().reshape(-1)
    residual_eval = steady_residual(model, xi_tensor, config).detach().cpu().numpy().reshape(-1)

    active_ref = np.asarray([phase_history(theta_ref, xi_ref, config.optical, "active")])
    victim_ref = np.asarray([phase_history(theta_ref, xi_ref, config.optical, "victim")])
    active_pred = np.asarray([phase_history(theta_pred, xi_ref, config.optical, "active")])
    victim_pred = np.asarray([phase_history(theta_pred, xi_ref, config.optical, "victim")])
    xtalk_ref = crosstalk_ratio(active_ref, victim_ref)
    xtalk_pred = crosstalk_ratio(active_pred, victim_pred)

    features = model.first_layer_features(xi_tensor).detach().cpu().numpy()
    singular_values = np.linalg.svd(features - features.mean(axis=0, keepdims=True), compute_uv=False)
    if features.shape[1] > 1:
        corr = np.corrcoef(features, rowvar=False)
        corr_abs = float(np.mean(np.abs(corr[np.triu_indices_from(corr, k=1)])))
    else:
        corr_abs = 1.0

    from .analysis_outputs import effective_rank

    record: dict[str, Any] = {
        "pde": "steady_poisson",
        "model_id": model_id,
        "seed": seed,
        "git_commit": git_commit_hash(),
        "temperature_relative_l2": relative_l2(theta_pred, theta_ref),
        "pde_residual_rms": float(np.sqrt(np.mean(residual_eval**2))),
        "active_phase_rmse": rmse(active_pred, active_ref),
        "victim_phase_rmse": rmse(victim_pred, victim_ref),
        "crosstalk_abs_error": float(abs(xtalk_pred - xtalk_ref)),
        "crosstalk_pred": xtalk_pred,
        "crosstalk_ref": xtalk_ref,
        "trainable_parameters": _count_trainable_parameters(model),
        "quantum_parameters": int(model.quantum_parameter_count()) if hasattr(model, "quantum_parameter_count") else 0,
        "gate_count": int(model.gate_count()) if hasattr(model, "gate_count") else 0,
        "circuit_depth": int(model.circuit_depth()) if hasattr(model, "circuit_depth") else 0,
        "runtime_seconds": float(runtime_seconds),
        "final_training_loss": history[-1]["loss"],
        "training_history": history,
        "mean_abs_pairwise_correlation": corr_abs,
        "singular_values": [float(v) for v in singular_values],
        "effective_rank": effective_rank(singular_values),
        "config": config_to_dict(config),
    }
    arrays_path = output_dir / f"steady_{model_id}_seed{seed}_arrays.npz"
    np.savez_compressed(
        arrays_path,
        xi=xi_ref,
        theta_ref=theta_ref,
        theta_pred=theta_pred,
        residual=residual_eval,
        features=features,
    )
    record["array_artifact"] = str(arrays_path)
    record_path = output_dir / f"steady_{model_id}_seed{seed}.json"
    with record_path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
    return record


def write_steady_outputs(records: list[dict[str, Any]]) -> None:
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
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
        "mean_abs_pairwise_correlation",
        "effective_rank",
    ]
    with (metrics_dir / "steady_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in keys})

    import pandas as pd

    df = pd.DataFrame([{key: record.get(key) for key in keys} for record in records])
    df.groupby("model_id").agg(["mean", "std"]).to_csv(metrics_dir / "steady_summary.csv")

    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    best = {model_id: min([r for r in records if r["model_id"] == model_id], key=lambda r: r["temperature_relative_l2"]) for model_id in sorted({r["model_id"] for r in records})}
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_id, record in best.items():
        arrays = np.load(record["array_artifact"])
        ax.plot(arrays["xi"], arrays["theta_pred"], label=model_id)
    ref = np.load(next(iter(best.values()))["array_artifact"])
    ax.plot(ref["xi"], ref["theta_ref"], "k--", label="reference")
    ax.set_xlabel("xi")
    ax.set_ylabel("theta")
    ax.set_title("Steady Poisson best-seed comparison")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "steady_comparison.png", dpi=180)
    plt.close(fig)


def run_steady_experiments(
    models: tuple[str, ...] = ("classical_matched", "classical_standard", "q0_separable", "q1_entangled"),
) -> list[dict[str, Any]]:
    config = load_config("configs/physics.yaml")
    output_dir = Path("results/steady_runs")
    records: list[dict[str, Any]] = []
    for model_id in models:
        for seed in config.training.seeds:
            print(f"training steady {model_id} seed={seed}")
            records.append(train_steady_one(model_id, seed, config, output_dir))
    write_steady_outputs(records)
    return records


if __name__ == "__main__":
    run_steady_experiments()
