from __future__ import annotations

from pathlib import Path


FORBIDDEN_TOKENS = (
    "/Users/andreborchert",
    "FXAI-main2/FXAI",
)

TEXT_EXTENSIONS = {
    ".golden",
    ".html",
    ".json",
    ".md",
    ".tsv",
    ".txt",
}


def test_repository_artifacts_do_not_embed_operator_paths():
    root = Path(__file__).resolve().parents[2]
    scan_roots = [
        root / "Tools/Benchmarks",
        root / "Tools/OfflineLab",
        root / "Tools/tests/golden",
    ]

    offenders: list[str] = []
    for scan_root in scan_roots:
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(token in text for token in FORBIDDEN_TOKENS):
                offenders.append(str(path.relative_to(root)))

    assert not offenders, "operator-specific paths remain in tracked artifacts: " + ", ".join(sorted(offenders))
