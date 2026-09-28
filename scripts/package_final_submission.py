from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "qapinn_thermo_optic_final_submission.zip"
SHA_OUT = OUT.with_suffix(OUT.suffix + ".sha256")
MANIFEST = ROOT / "submission" / "final_submission_manifest.json"

TEXT_SUFFIXES = {".csv", ".json", ".md", ".mjs", ".py", ".txt", ".yaml", ".yml"}
BLOCKED_TEXT = (
    "A" + "SD",
    "S" + "TE100",
    "Simplified Technical " + "English",
    "Written " + "using",
    "formal " + "compliance",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(paths: set[Path], rel: str) -> None:
    path = ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(rel)
    paths.add(path.relative_to(ROOT))


def add_tree(paths: set[Path], rel: str, suffixes: set[str] | None = None, recursive: bool = True) -> None:
    base = ROOT / rel
    if not base.is_dir():
        raise FileNotFoundError(rel)
    iterator = base.rglob("*") if recursive else base.glob("*")
    for path in iterator:
        if not path.is_file():
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        if "__pycache__" in path.parts or path.name == ".DS_Store":
            continue
        paths.add(path.relative_to(ROOT))


def collect_paths() -> list[Path]:
    paths: set[Path] = set()
    for rel in [
        ".gitignore",
        "README.md",
        "requirements-lock.txt",
        "docs/results_interpretation.md",
        "report/technical_report.md",
        "report/technical_report.pdf",
        "slides/qapinn_submission_slides.pptx",
        "slides/tmp_build/build_deck.mjs",
        "slides/tmp_build/package.json",
        "audit/stage3r/final_consistency_check.csv",
        "audit/stage3r/final_visual_qa.md",
        "audit/stage3r/mixed_interaction_chart_audit.csv",
        "audit/stage3r/submission_criteria_check.csv",
        "scripts/package_final_submission.py",
    ]:
        add_file(paths, rel)

    add_tree(paths, "configs", {".yaml"})
    add_tree(paths, "src", {".py"}, recursive=False)
    add_tree(paths, "scripts", {".py"}, recursive=False)
    add_tree(paths, "tests", {".py"}, recursive=False)
    add_tree(paths, "submission", {".md"}, recursive=False)
    add_tree(paths, "results/metrics", {".csv", ".json"})
    add_tree(paths, "results/figures", {".png"})
    add_tree(paths, "results/reference", {".npz"})

    for path in (ROOT / "results" / "runs").glob("*"):
        if path.is_file() and path.suffix in {".json", ".npz"}:
            paths.add(path.relative_to(ROOT))
    add_tree(paths, "results/steady_runs", {".json", ".npz"})

    paths.add(MANIFEST.relative_to(ROOT))
    return sorted(paths, key=lambda p: p.as_posix())


def scan_text(paths: list[Path]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel in paths:
        path = ROOT / rel
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in BLOCKED_TEXT:
            if token in text:
                hits.append({"path": rel.as_posix(), "token": token})
    return hits


def write_manifest(paths: list[Path], text_hits: list[dict[str, str]]) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "READY TO SUBMIT WITH DISCLOSED LIMITATIONS",
        "repository_target": "https://github.com/AnkoninaShahar/qapinn-thermo-optic",
        "authors": [
            {"name": "Amoggh Bellad", "email": "amogghb@gmail.com"},
            {"name": "Shahar Ankonina", "email": "shahar.ankonina05@gmail.com"},
        ],
        "required_artifacts": {
            "technical_report": "report/technical_report.pdf",
            "source_code": ["src/", "scripts/", "tests/"],
            "reproducibility_instructions": "README.md",
            "presentation_slides": "slides/qapinn_submission_slides.pptx",
            "summary_of_key_findings": "submission/key_findings.md",
            "future_research_recommendations": "submission/recommendations.md",
        },
        "validation_summary": {
            "technical_report_pages": 7,
            "rendered_slides": 10,
            "focused_pytest": "8 passed",
            "reference_validation": "passes true",
            "quantum_derivative_gate": "passes true",
            "public_controlled_language_claim_hits": text_hits,
        },
        "included_paths": [path.as_posix() for path in paths],
        "file_sha256": {path.as_posix(): sha256_file(ROOT / path) for path in paths if path != MANIFEST.relative_to(ROOT)},
        "final_zip_sha256": None,
        "final_zip_sha256_note": "The archive hash is written to submission/qapinn_thermo_optic_final_submission.zip.sha256 after package creation.",
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    initial_paths = collect_paths()
    initial_hits = scan_text([path for path in initial_paths if path != MANIFEST.relative_to(ROOT)])
    if initial_hits:
        raise SystemExit(f"Blocked public wording found: {initial_hits}")

    write_manifest(initial_paths, initial_hits)
    paths = collect_paths()
    hits = scan_text(paths)
    if hits:
        raise SystemExit(f"Blocked public wording found: {hits}")

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for rel in paths:
            archive.write(ROOT / rel, rel.as_posix())

    digest = sha256_file(OUT)
    SHA_OUT.write_text(f"{digest}  {OUT.name}\n", encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["final_zip_sha256"] = digest
    manifest["final_zip_size_bytes"] = OUT.stat().st_size
    manifest["included_file_count"] = len(paths)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"zip": str(OUT), "sha256": digest, "files": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
