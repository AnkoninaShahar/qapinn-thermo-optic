from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
FIG_DIR = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"


def fmt(value: float) -> str:
    return f"{value:.4f}"


def summary_rows(path: Path) -> list[list[str]]:
    df = pd.read_csv(path)
    rows = [["Model", "Rel L2", "PDE RMS", "Active RMSE", "Crosstalk err", "Params", "Runtime s"]]
    for model_id, group in df.groupby("model_id"):
        rows.append(
            [
                model_id,
                f"{group['temperature_relative_l2'].mean():.4f} +/- {group['temperature_relative_l2'].std():.4f}",
                f"{group['pde_residual_rms'].mean():.4f} +/- {group['pde_residual_rms'].std():.4f}",
                f"{group['active_phase_rmse'].mean():.4f} +/- {group['active_phase_rmse'].std():.4f}",
                f"{group['crosstalk_abs_error'].mean():.4f} +/- {group['crosstalk_abs_error'].std():.4f}",
                str(int(group["trainable_parameters"].mean())),
                f"{group['runtime_seconds'].mean():.2f} +/- {group['runtime_seconds'].std():.2f}",
            ]
        )
    return rows


def protocol_rows() -> list[list[str]]:
    with (ROOT / "configs" / "final_experiment_manifest.yaml").open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    with (ROOT / "configs" / "physics.yaml").open("r", encoding="utf-8") as stream:
        physics = yaml.safe_load(stream)
    shared = manifest["shared_protocol"]
    reference = physics["reference"]
    quantum = manifest["controlled_quantum_comparison"]
    return [
        ["Item", "Frozen value"],
        ["Random seeds", ", ".join(str(seed) for seed in shared["seeds"])],
        ["Optimizer", shared["optimizer"]],
        ["Learning rate", str(shared["learning_rate"])],
        ["Training epochs", str(shared["epochs"])],
        ["Transient collocation count", str(shared["transient_collocation_points"])],
        ["Steady collocation count", str(shared["steady_collocation_points"])],
        ["Transient evaluation grid", shared["transient_eval_grid"]],
        ["Steady evaluation points", str(shared["steady_eval_points"])],
        ["Numerical dtype", shared["dtype"]],
        ["Reference grid", f"n_xi={reference['n_xi']}, n_tau={reference['n_tau']}"],
        ["Reference convergence grids", ", ".join(str(v) for v in reference["convergence_grid_sizes"])],
        ["PennyLane device/backend", "default.qubit"],
        ["Shots", "None"],
        ["Differentiation method", "backprop"],
        ["Q0/Q1 controlled difference", quantum["q1"]],
    ]


def add_table(story: list, rows: list[list[str]], widths: list[float]) -> None:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.08 * inch))


def figure_flowable(filename: str, width: float, max_height: float) -> RLImage | None:
    path = FIG_DIR / filename
    if not path.exists():
        return None
    with PILImage.open(path) as image:
        source_width, source_height = image.size
    height = width * source_height / source_width
    if height > max_height:
        height = max_height
        width = height * source_width / source_height
    return RLImage(str(path), width=width, height=height)


def add_figure(story: list, filename: str, caption: str, width: float = 5.7 * inch, max_height: float = 2.7 * inch) -> None:
    image = figure_flowable(filename, width, max_height)
    if image is not None:
        story.append(image)
        story.append(Paragraph(caption, CAPTION))
        story.append(Spacer(1, 0.07 * inch))


styles = getSampleStyleSheet()
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontSize=19, leading=22, spaceAfter=7)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, leading=16, spaceBefore=7, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11.2, leading=13, spaceBefore=5, spaceAfter=3)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.1, leading=11.7, spaceAfter=4)
CAPTION = ParagraphStyle("Caption", parent=styles["BodyText"], fontSize=8.1, leading=9.5, textColor=colors.HexColor("#4b5563"), spaceAfter=4)
REF = ParagraphStyle("Reference", parent=styles["BodyText"], fontSize=6.9, leading=7.8, spaceAfter=1.5)


def p(text: str) -> Paragraph:
    return Paragraph(text, BODY)


def h1(text: str) -> Paragraph:
    return Paragraph(text, H1)


def h2(text: str) -> Paragraph:
    return Paragraph(text, H2)


def build_pdf() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    story: list = []
    story.append(Paragraph("Explainable Quantum-Assisted PINNs for Reduced Thermo-Optic Phase-Shifter Modeling", TITLE))
    story.append(p("Authors: Amoggh Bellad <amogghb@gmail.com> and Shahar Ankonina <shahar.ankonina05@gmail.com>"))
    story.append(
        p(
            "This report studies a reduced one-dimensional thermo-optic phase-shifter model and compares a "
            "parameter-matched classical PINN, a larger standard classical PINN, a separable two-qubit QAPINN "
            "(Q0), and an entangled two-qubit QAPINN (Q1). The challenge asks for explanation of the quantum "
            "layer rather than a performance-only claim [6]."
        )
    )
    story.append(p("All experiments use self-generated synthetic reference solutions and simulator-only quantum circuits."))

    story.append(h1("1. Research Question and Scope"))
    story.append(
        p(
            "The central question is how entanglement and quantum angle encoding change physics-informed learning "
            "for static and transient thermal models of a silicon-photonic thermo-optic phase shifter. The model is "
            "a reduced-order lateral thermal coordinate with a localized Gaussian heater, an active waveguide, and "
            "a neighboring victim waveguide. It is not a fabricated-device simulator and is not calibrated to predict "
            "absolute insertion loss, P-pi, or multilayer heat leakage. The reduced photonics scope follows the "
            "thermo-optic and thermal-modeling motivation in prior phase-shifter work [4,5]."
        )
    )
    add_figure(story, "reduced_device_schematic.png", "Figure 1. Reduced thermo-optic model used for the experiments.", width=5.4 * inch, max_height=2.0 * inch)

    story.append(h1("2. Governing Equations"))
    story.append(
        p(
            "The transient heat equation is theta_tau = theta_xx + S(xi)u(tau), with xi in [-1, 1], tau in [0, 1], "
            "zero boundary conditions, and zero initial condition. The steady equation is -theta_xx = S(xi), with "
            "zero boundary conditions. The Gaussian source S(xi) is configured in configs/physics.yaml. Neural models "
            "enforce constraints with hard output transforms: tau*(1 - xi^2)*N(xi,tau) for transient and "
            "(1 - xi^2)*N(xi) for steady. This follows the PINN residual formulation introduced for physics-informed "
            "neural networks [1]."
        )
    )
    story.append(
        p(
            "Optical phase is computed after thermal prediction by mode-weighting temperature at the active and victim "
            "waveguide locations. The normalized phase scale is explicit and no physically calibrated phase-shifter "
            "claim is made."
        )
    )
    add_figure(story, "reference_transient_temperature.png", "Figure 2. Independent Crank-Nicolson transient reference solution.", width=4.4 * inch, max_height=2.5 * inch)

    story.append(h1("3. Models and Controlled Comparison"))
    arch_rows = [
        ["Model", "Purpose", "Architecture", "Params", "Q params", "Gates / depth"],
        ["classical_matched", "parameter budget control", "2 -> 2 -> 16 -> 16 -> 1 transient", "343", "0", "0/0"],
        ["classical_standard", "conventional classical baseline", "2 -> 32 -> 32 -> 32 -> 1 transient", "2241", "0", "0/0"],
        ["Q0", "separable quantum layer", "RY(xi), RY(tau), local Rot, <Z0>, <Z1>", "343", "6", "4/2"],
        ["Q1", "entanglement ablation", "identical to Q0 plus CNOT(0,1)", "343", "6", "5/3"],
    ]
    add_table(story, arch_rows, [1.1 * inch, 1.45 * inch, 2.15 * inch, 0.55 * inch, 0.75 * inch, 0.85 * inch])
    story.append(
        p(
            "Q0 and Q1 are controlled: they have the same trainable parameter count, measurements, classical tail, "
            "seeds, optimizer, collocation rule, dtype, and evaluation grid. Q1 differs by one CNOT. The larger "
            "classical baseline is intentionally not parameter matched. The QAPINN structure follows the challenge's "
            "hybrid quantum-classical architecture and prior QAPINN benchmark framing [2,6]."
        )
    )
    story.append(h2("Frozen protocol"))
    add_table(story, protocol_rows(), [2.2 * inch, 4.25 * inch])

    story.append(h1("4. Validation"))
    story.append(
        p(
            "The reference solver satisfies the initial and boundary conditions exactly to numerical tolerance. The "
            "41-to-81 grid convergence relative L2 check is 0.000641. The quantum derivative gate verifies finite "
            "dq/dxi, dq/dtau, d2q/dxi2, and quantum parameter gradients; the finite-difference derivative sanity "
            "error is 3.18e-10."
        )
    )

    story.append(h1("5. Transient Results"))
    add_table(story, summary_rows(METRICS_DIR / "final_transient_metrics.csv"), [1.35 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 0.55 * inch, 0.85 * inch])
    story.append(
        p(
            "Q0 improves over the parameter-matched classical baseline on mean transient relative L2 error and PDE "
            "residual. However, Q0 does not outperform the standard classical PINN, which has lower mean relative L2 "
            "error and much lower simulator runtime. Q1 creates a richer nonseparable feature but does not improve "
            "transient physical accuracy under this frozen protocol."
        )
    )
    add_figure(story, "transient_accuracy.png", "Figure 3. Transient temperature error across models.", width=4.8 * inch, max_height=2.5 * inch)
    add_figure(story, "transient_prediction_error_comparison.png", "Figure 4. Best-seed transient predictions and errors.", width=5.6 * inch, max_height=3.9 * inch)

    story.append(h1("6. Explainability"))
    story.append(
        p(
            "The mixed interaction score I_j = mean(abs(d2 q_j / dxi dtau)) measures whether a first-layer feature "
            "depends jointly on space and time. Q0 quantum features have interaction scores near zero, while Q1 has "
            "one feature with mean interaction around 3.80. This confirms that the CNOT changes representation, but "
            "the added interaction does not reduce transient error. Fourier-spectrum interpretation is motivated by "
            "the role of data encoding in variational quantum model expressivity [3]."
        )
    )
    add_figure(story, "q0_separable_feature_maps.png", "Figure 5. Q0 quantum-feature maps for the best Q0 seed.", width=4.8 * inch, max_height=2.2 * inch)
    add_figure(story, "q1_entangled_feature_maps.png", "Figure 6. Q1 quantum-feature maps for the best Q1 seed.", width=4.8 * inch, max_height=2.2 * inch)
    add_figure(story, "mixed_interaction_comparison.png", "Figure 7. Q0/Q1 mixed space-time interaction scores.", width=4.6 * inch, max_height=2.25 * inch)
    add_figure(story, "gradient_histories.png", "Figure 8. Representative gradient histories from saved training records.", width=4.6 * inch, max_height=2.25 * inch)

    story.append(h1("7. Fourier and Residual Analysis"))
    story.append(
        p(
            "Fourier analysis compares normalized spectra of predictions and reference fields at common sampled times, "
            "plus active-phase temporal spectra. It tests whether apparent accuracy corresponds to useful frequency "
            "content rather than arbitrary feature complexity."
        )
    )
    add_figure(story, "spatial_spectral_error.png", "Figure 9. Spatial temperature spectral error from saved run artifacts.", width=4.7 * inch, max_height=2.35 * inch)
    add_figure(story, "residual_localization.png", "Figure 10. PDE residual localization with a shared color scale.", width=5.7 * inch, max_height=2.0 * inch)
    add_figure(story, "active_phase_response.png", "Figure 11. Active-waveguide normalized phase response.", width=4.6 * inch, max_height=2.3 * inch)

    story.append(h1("8. Steady Poisson Results"))
    add_table(story, summary_rows(METRICS_DIR / "steady_metrics.csv"), [1.35 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 0.55 * inch, 0.85 * inch])
    story.append(
        p(
            "The steady experiment adds a second PDE class. Q1 has the lowest mean steady relative L2 error in this "
            "small-budget run, but its feature effective rank is 1.0 and it remains much slower than classical models. "
            "This result is useful but not strong enough to claim that entanglement is generally preferable for static "
            "thermal fields."
        )
    )
    add_figure(story, "steady_comparison.png", "Figure 12. Best-seed steady Poisson comparison.", width=4.8 * inch, max_height=2.5 * inch)

    story.append(h1("9. Interpretation"))
    story.append(
        p(
            "The clearest finding is negative and mechanistic: entanglement can create nonseparable space-time quantum "
            "features, but those features did not improve transient physical accuracy under a controlled training "
            "budget. Q0 is competitive with the parameter-matched classical model, yet a conventional larger classical "
            "PINN remains a stronger transient baseline. Parameter reduction is therefore separate from computational "
            "efficiency."
        )
    )
    story.append(
        p(
            "A justified circuit-design rule is: start with the smallest separable quantum feature layer when testing "
            "parameter efficiency; add entanglement only when interaction-score, spectral, and physical-error evidence "
            "show that nonseparable features help the PDE. Do not infer advantage from parameter count alone."
        )
    )

    story.append(h1("10. Limitations and Future Work"))
    story.append(
        p(
            "Limitations include a small optimizer budget, no Q2 data re-uploading in the main study, no hardware "
            "execution, no memory profiling in the final tables, no physically calibrated thermal constants, and no "
            "Maxwell optical-mode solving. Future work should add Q2 only after the reportable core remains stable, "
            "then test whether the extra Fourier capacity captures missing reference frequencies. Other extensions "
            "include a classical Fourier-feature control, heterogeneous thermal conductivity, pulse heating, and "
            "generalization over heater widths or amplitudes."
        )
    )
    story.append(h1("11. Conclusion"))
    story.append(
        p(
            "The final result is a controlled negative-and-positive finding. Q0 is a compact parameter-matched "
            "baseline that improves over the matched classical PINN on the transient task, but the larger standard "
            "classical PINN remains the strongest transient accuracy baseline. Q1 confirms that one CNOT creates a "
            "measurable mixed space-time feature, but that feature does not improve transient physical error under "
            "the fixed protocol. Future work should therefore test whether added frequency capacity improves physical "
            "metrics, not only whether it increases feature complexity."
        )
    )
    story.append(h1("AI Use and Contributions"))
    story.append(
        p(
            "ChatGPT and Codex assisted with code development, code inspection, numerical workflow development, "
            "validation infrastructure, analysis tooling, visualization, documentation, report preparation, slide "
            "preparation, and interpretation support. The submitted numerical results come from the repository's "
            "executed numerical workflows. The authors are responsible for the submitted work and conclusions."
        )
    )
    story.append(
        p(
            "Amoggh Bellad led silicon-photonics modeling; heater and waveguide physical interpretation; optical phase "
            "and crosstalk models; photonics literature review; reduced-device assumptions; quantum-circuit design; "
            "explainability, Fourier, gradient, residual, and feature analyses; limitations; and future research "
            "recommendations. Shahar Ankonina led numerical PDE implementation; classical PINN implementation; QAPINN "
            "software implementation; optimizer and training implementation; numerical reference computation; "
            "experiment infrastructure; reproducibility workflow; and software infrastructure. Both authors contributed "
            "to design decisions, interpretation, final review, the report, the presentation, validation, and packaging."
        )
    )
    story.append(h1("Recommendations for Future Research"))
    recommendation_rows = [
        ["Priority", "Recommendation", "Reason"],
        ["1", "Run optimizer-budget sensitivity.", "Separate architecture effects from short training-budget effects."],
        ["2", "Test Q2 data re-uploading.", "Measure whether added frequency capacity improves physical error."],
        ["3", "Add a classical Fourier-feature control.", "Separate quantum angle encoding from trigonometric inductive bias."],
        ["4", "Add pulse heating and generalization tests.", "Test cooling dynamics and robustness across heater settings."],
        ["5", "Defer 2D/3D photonic simulation.", "Keep high-fidelity modeling for after the reduced workflow is stable."],
    ]
    add_table(story, recommendation_rows, [0.55 * inch, 2.35 * inch, 3.6 * inch])
    story.append(h1("References"))
    references = [
        "[1] Maziar Raissi, Paris Perdikaris, and George Em Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 2019.",
        "[2] Jay Shah, Rut Lineswala, and Abhishek Chopra. Benchmarking Quantum-Assisted PINN (QA-PINN) for Computational Fluid Dynamics. IEEE International Conference on Quantum Computing and Engineering (QCE), 2024. DOI: 10.1109/QCE60285.2024.00199.",
        "[3] Schuld, Sweke, and Meyer. Effect of data encoding on the expressive power of variational quantum-machine-learning models. Physical Review A, 2021. DOI: 10.1103/PhysRevA.103.032430.",
        "[4] Parra, Navarro-Arenas, and Sanchis. Silicon thermo-optic phase shifters: a review of configurations and optimization strategies. Advanced Photonics Nexus, 2024. DOI: 10.1117/1.APN.3.4.044001.",
        "[5] Coenen et al. Static and Dynamic Thermal Modelling of Si Photonic Thermo-Optic Phase Shifter. IEEE ITherm, 2024. DOI: 10.1109/ITHERM55375.2024.10709411.",
        "[6] WISER and BQP. WISER <> BQP Quantum Assisted Physics-Informed Neural Networks for CFD / Summer Program 2026 BQP Challenge, 2026.",
    ]
    left_refs = [Paragraph(ref, REF) for ref in references[:3]]
    right_refs = [Paragraph(ref, REF) for ref in references[3:]]
    table = Table([[left_refs, right_refs]], colWidths=[3.25 * inch, 3.25 * inch])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(table)

    pdf_path = REPORT_DIR / "technical_report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.52 * inch,
        leftMargin=0.52 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.48 * inch,
        title="Explainable QAPINNs for Reduced Thermo-Optic Phase-Shifter Modeling",
    )
    doc.build(story)


if __name__ == "__main__":
    build_pdf()
