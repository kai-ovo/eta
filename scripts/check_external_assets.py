"""Check expected local data/checkpoint layout without downloading anything."""

from __future__ import annotations

from pathlib import Path


SAMPLE_DATA_DOI = "10.5281/zenodo.21635446"
MODEL_ARCHIVE_DOI = "10.5281/zenodo.21635468"


EXPECTED_PATHS = (
    ("data/toy.pth", SAMPLE_DATA_DOI),
    ("data/era5land_USA_SouthEast_1999-2023_dailymax.nc", SAMPLE_DATA_DOI),
    ("models/precip-srcnn", MODEL_ARCHIVE_DOI),
    ("models/fm", MODEL_ARCHIVE_DOI),
    ("samples", MODEL_ARCHIVE_DOI),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _work_dir() -> Path:
    import os

    return Path(os.environ.get("ETA_WORK_DIR", _repo_root())).expanduser()


def _check_paths(root: Path) -> list[str]:
    problems: list[str] = []
    if not root.exists():
        problems.append(f"ETA work root does not exist: {root}")
        return problems
    if not root.is_dir():
        problems.append(f"ETA work root is not a directory: {root}")
        return problems

    for rel, doi_placeholder in EXPECTED_PATHS:
        path = root / rel
        if not path.exists():
            problems.append(f"Missing expected path under {root}: {rel} (see {doi_placeholder})")
    return problems


def main() -> int:
    root = _work_dir()
    problems = _check_paths(root)

    if problems:
        print("Local asset layout check failed:")
        print(f"Checked root: {root}")
        for problem in problems:
            print(f"  - {problem}")
        print()
        print("No download was attempted.")
        print(f"Dataset archive: https://doi.org/{SAMPLE_DATA_DOI}")
        print(f"Model archive:   https://doi.org/{MODEL_ARCHIVE_DOI}")
        return 1

    print(f"Local asset layout is present under: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
