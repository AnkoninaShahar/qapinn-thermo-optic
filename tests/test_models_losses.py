import torch

from src.classical_pinn import TransientClassicalPINN
from src.config import ProjectConfig
from src.losses import transient_pde_residual
from src.qapinn import QAPINN


def test_transient_hard_constraints_classical():
    model = TransientClassicalPINN().double()
    points = torch.tensor(
        [[-1.0, 0.2], [1.0, 0.7], [0.3, 0.0]],
        dtype=torch.float64,
    )
    out = model(points)
    assert torch.max(torch.abs(out)).item() < 1e-12


def test_transient_residual_shape_and_finite():
    model = TransientClassicalPINN().double()
    points = torch.rand((8, 2), dtype=torch.float64)
    points[:, 0] = points[:, 0] * 2.0 - 1.0
    residual = transient_pde_residual(model, points, ProjectConfig().heater)
    assert residual.shape == (8, 1)
    assert torch.isfinite(residual).all()


def test_q0_q1_parameter_counts_match():
    q0 = QAPINN("q0_separable").double()
    q1 = QAPINN("q1_entangled").double()
    assert sum(p.numel() for p in q0.parameters()) == sum(p.numel() for p in q1.parameters())
    assert q0.quantum_parameter_count() == q1.quantum_parameter_count()
    assert q1.gate_count() == q0.gate_count() + 1
