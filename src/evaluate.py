from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .config import ProjectConfig, config_to_dict
    from .losses import transient_pde_residual
    from .phase_metrics import crosstalk_ratio, phase_history, relative_l2, rmse
    from .physics import make_space_time_grid
    from .reference_transient import solve_transient_crank_nicolson
except ImportError:  # pragma: no cover
    from config import ProjectConfig, config_to_dict
    from losses import transient_pde_residual
    from phase_metrics import crosstalk_ratio, phase_history, relative_l2, rmse
    from physics import make_space_time_grid
    from reference_transient import solve_transient_crank_nicolson


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def predict_on_grid(model: torch.nn.Module, xi_grid: np.ndarray, tau_grid: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    points = np.column_stack([xi_grid.ravel(), tau_grid.ravel()])
    tensor = torch.tensor(points, dtype=dtype)
    model.eval()
    with torch.no_grad():
        pred = model(tensor).detach().cpu().numpy().reshape(xi_grid.shape)
    return pred


def residual_on_grid(model: torch.nn.Module, xi_grid: np.ndarray, tau_grid: np.ndarray, config: ProjectConfig, dtype: torch.dtype) -> np.ndarray:
    points = np.column_stack([xi_grid.ravel(), tau_grid.ravel()])
    tensor = torch.tensor(points, dtype=dtype)
    model.eval()
    residual = transient_pde_residual(model, tensor, config.heater)
    return residual.detach().cpu().numpy().reshape(xi_grid.shape)


def evaluate_model(
    model: torch.nn.Module,
    config: ProjectConfig,
    runtime_seconds: float,
    seed: int,
    model_id: str,
    output_dir: Path,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    xi, tau, xi_grid, tau_grid = make_space_time_grid(config, config.training.eval_n_xi, config.training.eval_n_tau)
    ref_xi, ref_tau, theta_ref = solve_transient_crank_nicolson(
        config,
        n_xi=config.training.eval_n_xi,
        n_tau=config.training.eval_n_tau,
    )
    theta_pred = predict_on_grid(model, xi_grid, tau_grid, dtype)
    residual = residual_on_grid(model, xi_grid, tau_grid, config, dtype)

    active_ref = phase_history(theta_ref, ref_xi, config.optical, "active")
    victim_ref = phase_history(theta_ref, ref_xi, config.optical, "victim")
    active_pred = phase_history(theta_pred, xi, config.optical, "active")
    victim_pred = phase_history(theta_pred, xi, config.optical, "victim")
    xtalk_ref = crosstalk_ratio(active_ref, victim_ref)
    xtalk_pred = crosstalk_ratio(active_pred, victim_pred)

    metrics = {
        "model_id": model_id,
        "seed": seed,
        "temperature_relative_l2": relative_l2(theta_pred, theta_ref),
        "pde_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "active_phase_rmse": rmse(active_pred, active_ref),
        "victim_phase_rmse": rmse(victim_pred, victim_ref),
        "crosstalk_abs_error": float(abs(xtalk_pred - xtalk_ref)),
        "crosstalk_pred": xtalk_pred,
        "crosstalk_ref": xtalk_ref,
        "trainable_parameters": count_trainable_parameters(model),
        "runtime_seconds": float(runtime_seconds),
    }
    if hasattr(model, "quantum_parameter_count"):
        metrics["quantum_parameters"] = int(model.quantum_parameter_count())
        metrics["gate_count"] = int(model.gate_count())
        metrics["circuit_depth"] = int(model.circuit_depth())
    else:
        metrics["quantum_parameters"] = 0
        metrics["gate_count"] = 0
        metrics["circuit_depth"] = 0

    artifact_path = output_dir / f"{model_id}_seed{seed}_arrays.npz"
    np.savez_compressed(
        artifact_path,
        xi=xi,
        tau=tau,
        theta_ref=theta_ref,
        theta_pred=theta_pred,
        residual=residual,
        active_ref=active_ref,
        active_pred=active_pred,
        victim_ref=victim_ref,
        victim_pred=victim_pred,
    )
    metrics["array_artifact"] = str(artifact_path)
    return metrics


def write_run_record(path: Path, record: dict[str, Any], config: ProjectConfig) -> None:
    serializable = dict(record)
    serializable["config"] = config_to_dict(config)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(serializable, stream, indent=2, sort_keys=True)
