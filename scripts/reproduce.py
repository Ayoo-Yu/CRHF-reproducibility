"""Rebuild observations, statistics, numeric tables, and figures from the release."""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "reproduction")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "outputs"):
        parser.error("Use a directory under outputs/ or outside the repository.")
    jobs = [
        ("prepare_observations.py", "--output", output / "observations_15min.csv.gz"),
        ("evaluate.py", "--check", "--output", output / "metrics"),
        ("tables.py", "--out", output / "tables"),
        ("figures.py", "--output", output / "figures"),
    ]
    for script, *arguments in jobs:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, ROOT / "scripts" / script, *arguments], check=True)
    print(f"Completed the observation, prediction, table and figure checks. Output: {output}")


if __name__ == "__main__":
    main()
