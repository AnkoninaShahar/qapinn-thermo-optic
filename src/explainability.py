from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .config import ProjectConfig
    from .losses import transient_pde_residual
    from .physics import make_space_time_grid
except ImportError:  # pragma: no cover
    from config import ProjectConfig
    from losses import transient_pde_residual
    from physics import make_space_time_grid


def feature_values(model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "first_layer_features"):
        return model.first_layer_features(points)
    raise TypeError("model does not expose first_layer_features")


def mixed_interaction_scores(model: torch.nn.Module, points: torch.Tensor) -> list[float]:
    x = points.clone().detach().requires_grad_(True)
    feats = feature_values(model, x)
    scores: list[float] = []
    for idx in range(feats.shape[1]):
        feature = feats[:, idx : idx + 1]
        grad = torch.autograd.grad(
            feature,
            x,
            grad_outputs=torch.ones_like(feature),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        d_feature_d_xi = grad[:, 0:1]
        mixed = torch.autograd.grad(
            d_feature_d_xi,
            x,
            grad_outputs=torch.ones_like(d_feature_d_xi),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0][:, 1:2]
        scores.append(float(torch.mean(torch.abs(mixed)).detach().cpu()))
    return scores


def save_explainability_outputs(
    model: torch.nn.Module,
    config: ProjectConfig,
    model_id: str,
    seed: int,
    output_dir: Path,
    dtype: torch.dtype,
) -> dict[str, Any]:
    xi, tau, xi_grid, tau_grid = make_space_time_grid(config, config.training.eval_n_xi, config.training.eval_n_tau)
    points_np = np.column_stack([xi_grid.ravel(), tau_grid.ravel()])
    points = torch.tensor(points_np, dtype=dtype)
    model.eval()

    feats = feature_values(model, points).detach().cpu().numpy()
    residual = transient_pde_residual(model, points, config.heater).detach().cpu().numpy()
    interaction = mixed_interaction_scores(model, points)

    feature_maps = feats.reshape((len(tau), len(xi), feats.shape[1]))
    residual_map = np.abs(residual.reshape((len(tau), len(xi))))

    npz_path = output_dir / f"{model_id}_seed{seed}_explainability.npz"
    np.savez_compressed(
        npz_path,
        xi=xi,
        tau=tau,
        feature_maps=feature_maps,
        residual_abs=residual_map,
        interaction_scores=np.asarray(interaction),
    )

    figure_paths: list[str] = []
    for idx in range(feature_maps.shape[2]):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        image = ax.imshow(
            feature_maps[:, :, idx],
            extent=[xi.min(), xi.max(), tau.min(), tau.max()],
            origin="lower",
            aspect="auto",
        )
        ax.set_xlabel("xi")
        ax.set_ylabel("tau")
        ax.set_title(f"{model_id} seed {seed} feature {idx}")
        fig.colorbar(image, ax=ax)
        path = output_dir / f"{model_id}_seed{seed}_feature{idx}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths.append(str(path))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    image = ax.imshow(
        residual_map,
        extent=[xi.min(), xi.max(), tau.min(), tau.max()],
        origin="lower",
        aspect="auto",
    )
    ax.set_xlabel("xi")
    ax.set_ylabel("tau")
    ax.set_title(f"{model_id} seed {seed} |PDE residual|")
    fig.colorbar(image, ax=ax)
    residual_path = output_dir / f"{model_id}_seed{seed}_residual.png"
    fig.tight_layout()
    fig.savefig(residual_path, dpi=160)
    plt.close(fig)
    figure_paths.append(str(residual_path))

    return {
        "explainability_artifact": str(npz_path),
        "interaction_scores": interaction,
        "explainability_figures": figure_paths,
    }
