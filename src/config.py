from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DomainConfig:
    xi_min: float = -1.0
    xi_max: float = 1.0
    tau_min: float = 0.0
    tau_max: float = 1.0


@dataclass(frozen=True)
class HeaterConfig:
    amplitude: float = 1.0
    center: float = 0.0
    width: float = 0.20
    waveform: str = "step"


@dataclass(frozen=True)
class OpticalConfig:
    active_xi: float = 0.0
    victim_xi: float = 0.55
    mode_width: float = 0.12
    phase_scale: float = 1.0


@dataclass(frozen=True)
class ReferenceConfig:
    n_xi: int = 81
    n_tau: int = 81
    convergence_grid_sizes: tuple[int, ...] = (41, 81)


@dataclass(frozen=True)
class TrainingConfig:
    dtype: str = "float64"
    collocation_points: int = 96
    eval_n_xi: int = 41
    eval_n_tau: int = 41
    epochs: int = 60
    learning_rate: float = 0.01
    seeds: tuple[int, ...] = (11, 23, 37)
    log_interval: int = 10
    hidden_layers: tuple[int, ...] = (2, 16, 16)
    standard_hidden_layers: tuple[int, ...] = (32, 32, 32)
    quantum_tail_layers: tuple[int, ...] = (16, 16)


@dataclass(frozen=True)
class ProjectConfig:
    domain: DomainConfig = field(default_factory=DomainConfig)
    heater: HeaterConfig = field(default_factory=HeaterConfig)
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_to_dict(config: ProjectConfig) -> dict[str, Any]:
    return asdict(config)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    data = config_to_dict(ProjectConfig())
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        data = _merge_dict(data, loaded)

    return ProjectConfig(
        domain=DomainConfig(**data["domain"]),
        heater=HeaterConfig(**data["heater"]),
        optical=OpticalConfig(**data["optical"]),
        reference=ReferenceConfig(
            n_xi=int(data["reference"]["n_xi"]),
            n_tau=int(data["reference"]["n_tau"]),
            convergence_grid_sizes=tuple(data["reference"]["convergence_grid_sizes"]),
        ),
        training=TrainingConfig(
            dtype=data["training"]["dtype"],
            collocation_points=int(data["training"]["collocation_points"]),
            eval_n_xi=int(data["training"]["eval_n_xi"]),
            eval_n_tau=int(data["training"]["eval_n_tau"]),
            epochs=int(data["training"]["epochs"]),
            learning_rate=float(data["training"]["learning_rate"]),
            seeds=tuple(int(seed) for seed in data["training"]["seeds"]),
            log_interval=int(data["training"]["log_interval"]),
            hidden_layers=tuple(int(width) for width in data["training"]["hidden_layers"]),
            standard_hidden_layers=tuple(int(width) for width in data["training"]["standard_hidden_layers"]),
            quantum_tail_layers=tuple(int(width) for width in data["training"]["quantum_tail_layers"]),
        ),
    )


def validate_config(config: ProjectConfig) -> None:
    if not config.domain.xi_min < config.domain.xi_max:
        raise ValueError("xi_min must be less than xi_max")
    if not config.domain.tau_min == 0.0 or not config.domain.tau_min < config.domain.tau_max:
        raise ValueError("transient domain must start at tau=0 and have positive duration")
    if config.heater.width <= 0:
        raise ValueError("heater width must be positive")
    if config.optical.mode_width <= 0:
        raise ValueError("optical mode width must be positive")
    if config.reference.n_xi < 5 or config.reference.n_tau < 3:
        raise ValueError("reference grids are too small")
    if config.training.collocation_points < 8:
        raise ValueError("too few collocation points")
