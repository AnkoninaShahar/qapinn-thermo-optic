from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .qapinn import QAPINN
except ImportError:  # pragma: no cover
    from qapinn import QAPINN


def run_quantum_derivative_gate(output_path: str | Path = "results/metrics/quantum_derivative_gate.json") -> dict[str, Any]:
    torch.manual_seed(123)
    model = QAPINN("q1_entangled").double()
    x = torch.tensor([[0.2, 0.35], [-0.4, 0.7]], dtype=torch.float64, requires_grad=True)
    features = model.quantum_features(x)
    q0_sum = features[:, 0:1].sum()
    grad_q0 = torch.autograd.grad(q0_sum, x, create_graph=True, retain_graph=True)[0]
    d2_q0_xi = torch.autograd.grad(
        grad_q0[:, 0:1].sum(),
        x,
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    loss = (features**2).mean() + (d2_q0_xi**2).mean()
    loss.backward()

    eps = 1e-5
    x0 = x.detach().clone()
    xp = x0.clone()
    xm = x0.clone()
    xp[0, 0] += eps
    xm[0, 0] -= eps
    with torch.no_grad():
        numerical = (model.quantum_features(xp)[0, 0] - model.quantum_features(xm)[0, 0]) / (2 * eps)
    autograd_value = grad_q0[0, 0].detach()
    numerical_error = float(torch.abs(numerical - autograd_value))

    quantum_grads = [param.grad for param in model.quantum.parameters()]
    quantum_grad_finite = all(grad is not None and torch.isfinite(grad).all().item() for grad in quantum_grads)
    result = {
        "features_finite": bool(torch.isfinite(features).all().item()),
        "dq_dxi_finite": bool(torch.isfinite(grad_q0[:, 0]).all().item()),
        "dq_dtau_finite": bool(torch.isfinite(grad_q0[:, 1]).all().item()),
        "d2q_dxi2_finite": bool(torch.isfinite(d2_q0_xi).all().item()),
        "quantum_parameter_gradients_finite": bool(quantum_grad_finite),
        "numerical_dq_dxi_error": numerical_error,
        "passes": bool(
            torch.isfinite(features).all().item()
            and torch.isfinite(grad_q0).all().item()
            and torch.isfinite(d2_q0_xi).all().item()
            and quantum_grad_finite
            and numerical_error < 1e-4
        ),
        "feature_sample": features.detach().cpu().numpy().tolist(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    return result


if __name__ == "__main__":
    print(json.dumps(run_quantum_derivative_gate(), indent=2, sort_keys=True))
