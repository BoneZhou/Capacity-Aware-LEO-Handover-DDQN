"""Validate repository-local evaluation CSVs and invoke the formal paper plotter."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def validate_schema(path: Path, expected: set[str]) -> None:
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        fields = set(csv.DictReader(stream).fieldnames or [])
    absent = expected - fields
    if absent:
        raise ValueError(f"Invalid plotting schema in {path}: missing {sorted(absent)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="render figures after schema validation")
    args = parser.parse_args()
    required = {
        ROOT / "outputs/ablation_vs_ue/ablation_vs_ue_raw.csv": {
            "method", "ue_count", "capacity", "average_throughput_mbps", "blocking_rate_percent",
            "average_handovers_per_ue"
        },
        ROOT / "outputs/capacity_sensitivity/capacity_sensitivity_raw.csv": {
            "method", "ue_count", "capacity", "average_throughput_mbps", "blocking_rate_percent",
            "average_handovers_per_ue"
        },
    }
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing evaluation output(s); run the corresponding evaluation first:\n" + "\n".join(map(str, missing)))
    for path, expected in required.items():
        try:
            validate_schema(path, expected)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    print("plotting input schema validated")
    if args.execute:
        command = [sys.executable, str(ROOT / "plotting/plot_icc2027_integrated_errorbars.py")]
        raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)
    print("validation only; pass --execute to render figures")


if __name__ == "__main__":
    main()
