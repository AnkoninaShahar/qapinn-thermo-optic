from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _amplitude_spectrum(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    spectrum = np.abs(np.fft.rfft(centered))
    total = np.sum(spectrum)
    if total > 0:
        spectrum = spectrum / total
    return spectrum


def spectral_error(pred: np.ndarray, ref: np.ndarray) -> float:
    pred_spec = _amplitude_spectrum(pred)
    ref_spec = _amplitude_spectrum(ref)
    n = min(pred_spec.size, ref_spec.size)
    denom = np.sum(ref_spec[:n] ** 2)
    if denom <= 1e-15:
        return float("nan")
    return float(np.sum((pred_spec[:n] - ref_spec[:n]) ** 2) / denom)


def effective_rank(singular_values: np.ndarray) -> float:
    values = np.asarray(singular_values, dtype=float)
    total = np.sum(values)
    if total <= 1e-15:
        return 0.0
    probs = values / total
    entropy = -np.sum(probs * np.log(probs + 1e-15))
    return float(np.exp(entropy))


def _latest_record_paths(results_dir: Path) -> list[Path]:
    return sorted(path for path in results_dir.glob("*_seed*.json") if "smoke" not in str(path))


def _load_records(results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _latest_record_paths(results_dir):
        with path.open("r", encoding="utf-8") as stream:
            records.append(json.load(stream))
    return records


def generate_fourier_analysis(results_dir: Path = Path("results/runs"), output_dir: Path = Path("results/metrics")) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in _load_records(results_dir):
        arrays = np.load(record["array_artifact"])
        xi = arrays["xi"]
        tau = arrays["tau"]
        theta_ref = arrays["theta_ref"]
        theta_pred = arrays["theta_pred"]
        active_ref = arrays["active_ref"]
        active_pred = arrays["active_pred"]

        for tau_target in (0.25, 0.5, 1.0):
            idx = int(np.argmin(np.abs(tau - tau_target)))
            rows.append(
                {
                    "model_id": record["model_id"],
                    "seed": record["seed"],
                    "kind": "temperature_spatial",
                    "location": f"tau={tau[idx]:.3f}",
                    "spectral_error": spectral_error(theta_pred[idx, :], theta_ref[idx, :]),
                }
            )
        rows.append(
            {
                "model_id": record["model_id"],
                "seed": record["seed"],
                "kind": "active_phase_temporal",
                "location": "active_waveguide",
                "spectral_error": spectral_error(active_pred, active_ref),
            }
        )

        explain = np.load(record["explainability_artifact"])
        feature_maps = explain["feature_maps"]
        if record["model_id"].startswith("q"):
            for feature_idx in range(feature_maps.shape[2]):
                for tau_target in (0.25, 0.5, 1.0):
                    idx = int(np.argmin(np.abs(tau - tau_target)))
                    rows.append(
                        {
                            "model_id": record["model_id"],
                            "seed": record["seed"],
                            "kind": f"feature{feature_idx}_spatial_spectrum",
                            "location": f"tau={tau[idx]:.3f}",
                            "spectral_error": float("nan"),
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "fourier_spectral_errors.csv", index=False)
    summary = df.dropna(subset=["spectral_error"]).groupby(["model_id", "kind"])["spectral_error"].agg(["mean", "std"]).reset_index()
    summary.to_csv(output_dir / "fourier_spectral_summary.csv", index=False)
    return df


def generate_feature_diversity(results_dir: Path = Path("results/runs"), output_dir: Path = Path("results/metrics")) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in _load_records(results_dir):
        explain = np.load(record["explainability_artifact"])
        features = explain["feature_maps"].reshape(-1, explain["feature_maps"].shape[2])
        if features.shape[1] == 1:
            corr_abs = 1.0
        else:
            corr = np.corrcoef(features, rowvar=False)
            upper = corr[np.triu_indices_from(corr, k=1)]
            corr_abs = float(np.mean(np.abs(upper)))
        singular_values = np.linalg.svd(features - features.mean(axis=0, keepdims=True), compute_uv=False)
        rows.append(
            {
                "model_id": record["model_id"],
                "seed": record["seed"],
                "mean_abs_pairwise_correlation": corr_abs,
                "singular_values": json.dumps([float(v) for v in singular_values]),
                "effective_rank": effective_rank(singular_values),
                "interaction_scores": json.dumps(record["interaction_scores"]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "feature_diversity.csv", index=False)
    summary = df.groupby("model_id")[["mean_abs_pairwise_correlation", "effective_rank"]].agg(["mean", "std"])
    summary.to_csv(output_dir / "feature_diversity_summary.csv")
    return df


def generate_final_figures(results_dir: Path = Path("results/runs"), output_dir: Path = Path("results/figures")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv("results/metrics/final_transient_metrics.csv")

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axhline(0, color="black", linewidth=1)
    ax.fill_between([-0.18, 0.18], -0.18, 0.18, color="#ef4444", alpha=0.35, label="heater")
    ax.fill_between([-0.09, 0.09], -0.5, -0.28, color="#2563eb", alpha=0.5, label="active waveguide")
    ax.fill_between([0.49, 0.61], -0.5, -0.28, color="#7c3aed", alpha=0.45, label="victim waveguide")
    ax.annotate("effective thermal coordinate xi", xy=(-0.9, 0.1), xytext=(-0.9, 0.45), arrowprops={"arrowstyle": "->"})
    ax.text(0, 0.25, "localized Gaussian heat source", ha="center", fontsize=10)
    ax.text(0, -0.62, "active", ha="center", fontsize=10)
    ax.text(0.55, -0.62, "victim", ha="center", fontsize=10)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.75, 0.65)
    ax.set_yticks([])
    ax.set_xlabel("normalized lateral position xi")
    ax.set_title("Reduced thermo-optic phase-shifter model")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "reduced_device_schematic.png", dpi=180)
    plt.close(fig)

    summary = metrics.groupby("model_id")["temperature_relative_l2"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    x_pos = np.arange(len(summary))
    ax.bar(x_pos, summary["mean"], yerr=summary["std"], capsize=4)
    ax.set_ylabel("relative L2 temperature error")
    ax.set_xticks(x_pos, summary["model_id"], rotation=20, ha="right")
    ax.set_title("Transient accuracy by model")
    fig.tight_layout()
    fig.savefig(output_dir / "transient_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(metrics["trainable_parameters"], metrics["temperature_relative_l2"], s=60)
    for _, row in metrics.iterrows():
        ax.annotate(row["model_id"], (row["trainable_parameters"], row["temperature_relative_l2"]), fontsize=7, alpha=0.7)
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("relative L2 temperature error")
    ax.set_title("Accuracy versus parameter count")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_parameters.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(metrics["runtime_seconds"], metrics["temperature_relative_l2"], s=60)
    for _, row in metrics.iterrows():
        ax.annotate(row["model_id"], (row["runtime_seconds"], row["temperature_relative_l2"]), fontsize=7, alpha=0.7)
    ax.set_xlabel("runtime seconds")
    ax.set_ylabel("relative L2 temperature error")
    ax.set_title("Accuracy versus runtime")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_runtime.png", dpi=180)
    plt.close(fig)

    records = _load_records(results_dir)
    selected = {}
    for model_id in ("classical_standard", "q0_separable", "q1_entangled"):
        candidates = [r for r in records if r["model_id"] == model_id]
        selected[model_id] = sorted(candidates, key=lambda r: r["temperature_relative_l2"])[0]

    fig, axes = plt.subplots(3, 3, figsize=(10, 9), sharex=True, sharey=True)
    for row_idx, (model_id, record) in enumerate(selected.items()):
        arrays = np.load(record["array_artifact"])
        data = [arrays["theta_ref"], arrays["theta_pred"], arrays["theta_pred"] - arrays["theta_ref"]]
        titles = ["reference", "prediction", "error"]
        xi = arrays["xi"]
        tau = arrays["tau"]
        for col_idx, values in enumerate(data):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(values, extent=[xi.min(), xi.max(), tau.min(), tau.max()], origin="lower", aspect="auto")
            ax.set_title(f"{model_id} {titles[col_idx]}")
            fig.colorbar(image, ax=ax, fraction=0.046)
    for ax in axes[-1, :]:
        ax.set_xlabel("xi")
    for ax in axes[:, 0]:
        ax.set_ylabel("tau")
    fig.tight_layout()
    fig.savefig(output_dir / "transient_prediction_error_comparison.png", dpi=180)
    plt.close(fig)

    ref_arrays = np.load(selected["classical_standard"]["array_artifact"])
    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(
        ref_arrays["theta_ref"],
        extent=[ref_arrays["xi"].min(), ref_arrays["xi"].max(), ref_arrays["tau"].min(), ref_arrays["tau"].max()],
        origin="lower",
        aspect="auto",
    )
    ax.set_xlabel("xi")
    ax.set_ylabel("tau")
    ax.set_title("Reference transient temperature field")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "reference_transient_temperature.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for model_id, record in selected.items():
        arrays = np.load(record["array_artifact"])
        ax.plot(arrays["tau"], arrays["active_pred"], label=f"{model_id} active")
    ref = np.load(selected["classical_standard"]["array_artifact"])
    ax.plot(ref["tau"], ref["active_ref"], "k--", label="reference active")
    ax.set_xlabel("tau")
    ax.set_ylabel("normalized phase")
    ax.set_title("Active phase response")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "active_phase_response.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for model_id in ("classical_matched", "classical_standard", "q0_separable", "q1_entangled"):
        candidates = [r for r in records if r["model_id"] == model_id]
        best = sorted(candidates, key=lambda r: r["temperature_relative_l2"])[0]
        hist = pd.DataFrame(best["training_history"])
        grad_col = "grad_norm_quantum" if model_id.startswith("q") else "grad_norm_first_classical"
        ax.plot(hist["epoch"], hist[grad_col], marker="o", label=model_id)
    ax.set_xlabel("epoch")
    ax.set_ylabel("representational gradient norm")
    ax.set_title("Gradient histories for best seed per model")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_histories.png", dpi=180)
    plt.close(fig)

    residual_models = ("classical_standard", "q0_separable", "q1_entangled")
    residual_maps = {
        model_id: np.abs(np.load(selected[model_id]["array_artifact"])["residual"])
        for model_id in residual_models
    }
    residual_vmax = max(float(np.max(values)) for values in residual_maps.values())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharex=True, sharey=True, constrained_layout=True)
    for ax, model_id in zip(axes, residual_models):
        arrays = np.load(selected[model_id]["array_artifact"])
        image = ax.imshow(
            residual_maps[model_id],
            extent=[arrays["xi"].min(), arrays["xi"].max(), arrays["tau"].min(), arrays["tau"].max()],
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=residual_vmax,
        )
        ax.set_title(model_id)
        ax.set_xlabel("xi")
    axes[0].set_ylabel("tau")
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02, label="|PDE residual|")
    fig.suptitle("PDE residual localization")
    fig.savefig(output_dir / "residual_localization.png", dpi=180)
    plt.close(fig)

    for model_id in ("q0_separable", "q1_entangled"):
        record = selected[model_id]
        explain = np.load(record["explainability_artifact"])
        xi = explain["xi"]
        tau = explain["tau"]
        fmap = explain["feature_maps"]
        fig, axes = plt.subplots(1, fmap.shape[2], figsize=(8, 3.5), sharex=True, sharey=True)
        for idx, ax in enumerate(np.atleast_1d(axes)):
            image = ax.imshow(fmap[:, :, idx], extent=[xi.min(), xi.max(), tau.min(), tau.max()], origin="lower", aspect="auto")
            ax.set_title(f"{model_id} feature {idx}")
            ax.set_xlabel("xi")
            fig.colorbar(image, ax=ax, fraction=0.046)
        np.atleast_1d(axes)[0].set_ylabel("tau")
        fig.tight_layout()
        fig.savefig(output_dir / f"{model_id}_feature_maps.png", dpi=180)
        plt.close(fig)

    diversity = pd.read_csv("results/metrics/feature_diversity.csv")
    rows = []
    for _, row in diversity.iterrows():
        scores = json.loads(row["interaction_scores"])
        for idx, score in enumerate(scores):
            rows.append({"model_id": row["model_id"], "seed": row["seed"], "feature": idx, "interaction": score})
    inter = pd.DataFrame(rows)
    inter = inter[inter["model_id"].isin(["q0_separable", "q1_entangled"])]
    inter_summary = inter.groupby(["model_id", "feature"])["interaction"].agg(["mean", "std"]).reset_index()
    inter_summary["label"] = inter_summary.apply(
        lambda row: f"{'Q0' if row['model_id'] == 'q0_separable' else 'Q1'} feature {int(row['feature'])}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    labels = inter_summary["label"].tolist()
    x_pos = np.arange(len(labels))
    colors = ["#9ca3af", "#9ca3af", "#9ca3af", "#2563eb"]
    ax.bar(x_pos, inter_summary["mean"], yerr=inter_summary["std"].fillna(0), capsize=4, color=colors)
    ax.set_ylabel("mean |d2 feature / dxi dtau|", fontsize=9)
    ax.set_title("Mixed space-time interaction by quantum feature", fontsize=11)
    ax.set_xticks(x_pos, labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylim(0, max(4.2, float(inter_summary["mean"].max()) * 1.1))
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for x, mean in zip(x_pos, inter_summary["mean"]):
        y = mean + 0.08 if mean > 0.01 else 0.08
        ax.text(x, y, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "mixed_interaction_comparison.png", dpi=180)
    plt.close(fig)

    fourier = pd.read_csv("results/metrics/fourier_spectral_summary.csv")
    temp = fourier[fourier["kind"] == "temperature_spatial"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x_pos = np.arange(len(temp))
    ax.bar(x_pos, temp["mean"], yerr=temp["std"].fillna(0), capsize=4)
    ax.set_ylabel("spectral error")
    ax.set_title("Spatial temperature spectral error")
    ax.set_xticks(x_pos, temp["model_id"], rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "spatial_spectral_error.png", dpi=180)
    plt.close(fig)


def run_all_analyses() -> None:
    generate_fourier_analysis()
    generate_feature_diversity()
    generate_final_figures()


if __name__ == "__main__":
    run_all_analyses()
