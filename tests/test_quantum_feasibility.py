from src.quantum_feasibility import run_quantum_derivative_gate


def test_quantum_derivative_gate_passes(tmp_path):
    result = run_quantum_derivative_gate(tmp_path / "gate.json")
    assert result["passes"]
