import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Users/amogghbellad/Documents/Codex/qapinn-thermo-optic";
const OUT = path.join(ROOT, "slides", "qapinn_submission_slides.pptx");
const RENDER_DIR = path.join(ROOT, "slides", "rendered");
const FIG = path.join(ROOT, "results", "figures");
const W = 1280;
const H = 720;
const C = {
  ink: "#000000",
  muted: "#4B5563",
  panel: "#F2F2F2",
  rule: "#B8BCC4",
  blue: "#3D8DFF",
  paleBlue: "#EAF5FB",
};

async function pngBytes(file) {
  return await fs.readFile(path.join(FIG, file));
}

async function writeBlob(file, blob) {
  await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer()));
}

function addText(slide, text, left, top, width, height, fontSize = 24, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize,
    typeface: "Helvetica Neue",
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
    alignment: opts.alignment ?? "left",
    verticalAlignment: opts.verticalAlignment ?? "top",
    autoFit: opts.autoFit ?? "shrinkText",
  };
  return shape;
}

function addTitle(slide, title, subtitle = "") {
  addText(slide, title, 42, 30, 1120, 82, 40, { bold: false });
  if (subtitle) addText(slide, subtitle, 42, 112, 1080, 44, 20, { color: C.muted });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 42, top: 162, width: 1196, height: 1 },
    fill: C.rule,
    line: { style: "solid", fill: C.rule, width: 0 },
  });
}

async function addImage(slide, file, left, top, width, height, alt) {
  slide.images.add({
    blob: await pngBytes(file),
    contentType: "image/png",
    alt,
    fit: "contain",
    position: { left, top, width, height },
  });
}

function addBullets(slide, items, left, top, width, height, fontSize = 22) {
  addText(slide, items.map((x) => `- ${x}`).join("\n"), left, top, width, height, fontSize, { color: C.ink });
}

function addNotes(slide, lines) {
  slide.speakerNotes.textFrame.setText(lines.join("\n"));
  slide.speakerNotes.setVisible(true);
}

function addFooter(slide, num) {
  addText(slide, String(num), 1184, 660, 54, 24, 13, { alignment: "right", verticalAlignment: "bottom" });
}

function addPanel(slide, left, top, width, height) {
  slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: C.panel,
    line: { style: "solid", fill: C.rule, width: 1 },
  });
}

async function main() {
  await fs.mkdir(path.dirname(OUT), { recursive: true });
  await fs.mkdir(RENDER_DIR, { recursive: true });
  const deck = Presentation.create({ slideSize: { width: W, height: H } });

  let s = deck.slides.add();
  s.background.fill = "#FFFFFF";
  addText(s, "Explainable Quantum-Assisted PINNs", 58, 86, 1040, 122, 56, { bold: false });
  addText(s, "Reduced thermo-optic phase-shifter modeling", 60, 232, 900, 48, 31, { color: C.muted });
  addText(s, "Static and transient thermal PDEs | Classical controls | Q0/Q1 entanglement ablation", 60, 415, 900, 75, 28);
  addText(s, "Amoggh Bellad <amogghb@gmail.com> | Shahar Ankonina <shahar.ankonina05@gmail.com>", 60, 525, 1060, 35, 20, { color: C.muted });
  addText(s, "WISER-BQP Quantum-Assisted PINN Challenge", 60, 600, 760, 28, 20, { color: C.muted });
  addFooter(s, 1);
  addNotes(s, [
    "[Sources] BQP - WISER Quantum Challenge [SHARED] (3).pdf; Summer Program 2026 BQP Challenge (1).pdf; configs/final_experiment_manifest.yaml.",
  ]);

  s = deck.slides.add();
  addTitle(s, "The study asks what the quantum layer changes", "The reduced device model keeps the physics defensible and the ablation controlled.");
  addBullets(s, [
    "Thermal PDE is learned; optical phase and crosstalk are computed afterward.",
    "Q0 and Q1 are parameter matched; Q1 differs by one CNOT.",
    "No hardware speedup, quantum advantage, or fabricated-device calibration is claimed.",
  ], 54, 205, 505, 330, 27);
  await addImage(s, "reduced_device_schematic.png", 595, 195, 610, 360, "Reduced thermo-optic phase-shifter schematic");
  addFooter(s, 2);
  addNotes(s, ["[Sources] report/technical_report.pdf; results/figures/reduced_device_schematic.png."]);

  s = deck.slides.add();
  addTitle(s, "The protocol was frozen around four core models", "Same physics, seeds, optimizer, dtype, collocation rule, and evaluation grid.");
  const rows = [
    ["classical_matched", "parameter budget control", "343 params"],
    ["classical_standard", "higher-capacity classical PINN", "2241 params"],
    ["Q0 separable", "two-qubit feature layer without CNOT", "343 params"],
    ["Q1 entangled", "Q0 plus one CNOT(0,1)", "343 params"],
  ];
  addText(s, "Model", 90, 202, 260, 32, 23, { bold: true });
  addText(s, "Role", 390, 202, 480, 32, 23, { bold: true });
  addText(s, "Budget", 930, 202, 220, 32, 23, { bold: true });
  rows.forEach((row, i) => {
    const y = 248 + i * 72;
    addPanel(s, 80, y - 8, 1080, 54);
    addText(s, row[0], 95, y, 260, 34, 22, { bold: true });
    addText(s, row[1], 390, y, 500, 34, 22);
    addText(s, row[2], 930, y, 200, 34, 22);
  });
  addText(s, "Derivative gate: finite dq/dxi, dq/dtau, d2q/dxi2, and quantum parameter gradients.", 70, 555, 1000, 52, 26);
  addFooter(s, 3);
  addNotes(s, ["[Sources] results/metrics/quantum_derivative_gate.json; results/metrics/reference_validation.json; configs/final_experiment_manifest.yaml."]);

  s = deck.slides.add();
  addTitle(s, "On the transient PDE, Q0 beat the matched baseline but not the standard PINN", "The standard classical control prevents overclaiming a parameter-efficiency result.");
  await addImage(s, "transient_accuracy.png", 54, 188, 600, 360, "Transient accuracy bar chart");
  await addImage(s, "accuracy_vs_runtime.png", 685, 188, 520, 360, "Accuracy versus runtime scatter plot");
  addText(s, "Mean relative L2: standard classical 0.0868, Q0 0.1235, matched classical 0.2074, Q1 0.2137.", 60, 590, 1100, 50, 25);
  addFooter(s, 4);
  addNotes(s, ["[Sources] results/metrics/final_transient_metrics.csv; results/metrics/final_transient_summary.csv; generated figures."]);

  s = deck.slides.add();
  addTitle(s, "Entanglement created interaction, not better transient accuracy", "The CNOT changed representation in the intended way, but the physical error did not improve.");
  await addImage(s, "q0_separable_feature_maps.png", 48, 178, 560, 245, "Q0 feature maps");
  await addImage(s, "q1_entangled_feature_maps.png", 48, 435, 560, 245, "Q1 feature maps");
  await addImage(s, "mixed_interaction_comparison.png", 650, 195, 555, 350, "Mixed interaction comparison");
  addText(s, "Q1 feature 1: mean interaction 3.80; Q0 features remain near zero.", 655, 570, 520, 45, 23);
  addFooter(s, 5);
  addNotes(s, ["[Sources] results/metrics/feature_diversity.csv; results/runs/*_explainability.npz; generated feature-map figures."]);

  s = deck.slides.add();
  addTitle(s, "Spectra and residuals show why accuracy matters more than feature complexity", "Extra representation structure is only useful if it reduces physical error.");
  await addImage(s, "spatial_spectral_error.png", 50, 188, 540, 335, "Spatial spectral error chart");
  await addImage(s, "residual_localization.png", 640, 185, 575, 335, "Residual localization heatmaps");
  addText(s, "Fourier and residual analyses are computed from saved arrays, not manually typed result values.", 60, 585, 1080, 45, 24);
  addFooter(s, 6);
  addNotes(s, ["[Sources] results/metrics/fourier_spectral_errors.csv; results/metrics/fourier_spectral_summary.csv; results/runs/*_arrays.npz."]);

  s = deck.slides.add();
  addTitle(s, "The steady Poisson task adds a second PDE class", "Only position is required, so entanglement should not be assumed useful.");
  await addImage(s, "steady_comparison.png", 55, 190, 625, 380, "Steady Poisson comparison");
  addBullets(s, [
    "Q1 had the lowest mean steady relative L2 in this small-budget protocol.",
    "The standard classical model was close and far faster.",
    "The steady result is not enough to claim a general entanglement advantage.",
  ], 705, 225, 455, 320, 26);
  addFooter(s, 7);
  addNotes(s, ["[Sources] results/metrics/steady_metrics.csv; results/metrics/steady_summary.csv; results/steady_runs/*.json."]);

  s = deck.slides.add();
  addTitle(s, "The defensible conclusion is conditional", "The project supports a circuit-selection rule, not a quantum-advantage claim.");
  addBullets(s, [
    "Use the shallow separable circuit as a compact parameter-efficiency control.",
    "Add entanglement only when interaction, spectrum, and physical metrics improve together.",
    "Report parameter count separately from simulator runtime.",
    "Preserve negative results: Q1 added interaction but did not help transient accuracy.",
  ], 90, 205, 980, 360, 30);
  addFooter(s, 8);
  addNotes(s, ["[Sources] docs/results_interpretation.md; report/technical_report.pdf."]);

  s = deck.slides.add();
  addTitle(s, "The submission is reproducible from scripts and saved records", "Every table and figure is backed by machine-readable outputs.");
  addBullets(s, [
    "Environment: Python 3.11, Torch 2.2.2, PennyLane 0.35.1, NumPy 1.26.4.",
    "Core commands: pytest, validate_physics.py, test_quantum_derivatives.py, run_final_transient.py, run_steady.py, analyze_results.py.",
    "Outputs: metrics CSVs, JSON run records, NPZ arrays, figures, report PDF, and this deck.",
  ], 70, 192, 1080, 210, 27);
  addPanel(s, 78, 432, 330, 126);
  addPanel(s, 475, 432, 330, 126);
  addPanel(s, 872, 432, 330, 126);
  addText(s, "8 passed", 105, 454, 260, 40, 32, { bold: true });
  addText(s, "focused tests", 105, 503, 240, 28, 22);
  addText(s, "2 gates", 502, 454, 260, 40, 32, { bold: true });
  addText(s, "reference and derivative", 502, 503, 245, 28, 22);
  addText(s, "1 package", 899, 454, 260, 40, 32, { bold: true });
  addText(s, "report, slides, source, evidence", 899, 503, 250, 28, 22);
  addFooter(s, 9);
  addNotes(s, ["[Sources] README.md; requirements-lock.txt; results/metrics/*.csv; results/runs/*.json; results/steady_runs/*.json."]);

  s = deck.slides.add();
  addTitle(s, "Next work should test mechanisms, not bigger circuits", "The current result supports a cautious research path.");
  addPanel(s, 70, 198, 1080, 82);
  addText(s, "1", 96, 214, 50, 45, 35, { bold: true, color: C.blue });
  addText(s, "Run optimizer-budget sensitivity", 160, 211, 430, 35, 27, { bold: true });
  addText(s, "Check whether Q0, Q1, and the classical baselines change rank order with more training.", 555, 212, 555, 44, 24);
  addPanel(s, 70, 320, 1080, 82);
  addText(s, "2", 96, 336, 50, 45, 35, { bold: true, color: C.blue });
  addText(s, "Add Q2 data re-uploading", 160, 333, 430, 35, 27, { bold: true });
  addText(s, "Report added parameters, gates, depth, runtime, spectra, gradients, and physical error together.", 555, 334, 555, 44, 24);
  addPanel(s, 70, 442, 1080, 82);
  addText(s, "3", 96, 458, 50, 45, 35, { bold: true, color: C.blue });
  addText(s, "Add classical Fourier features", 160, 455, 360, 35, 27, { bold: true });
  addText(s, "Separate quantum angle encoding from a general trigonometric inductive bias.", 555, 456, 555, 44, 24);
  addText(s, "Defer 2D/3D photonic simulation until the reduced-order workflow is stable under these tests.", 90, 585, 1040, 44, 25);
  addFooter(s, 10);
  addNotes(s, ["[Sources] configs/final_experiment_manifest.yaml; docs/results_interpretation.md."]);

  for (const [i, slide] of deck.slides.items.entries()) {
    const png = await deck.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(RENDER_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`), png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(RENDER_DIR, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text());
  }
  const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(RENDER_DIR, "montage.webp"), montage);
  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(OUT);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
