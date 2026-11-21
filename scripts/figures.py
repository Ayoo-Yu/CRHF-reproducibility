"""Recompute plot data, check against published values, and render all eighteen figures."""

import argparse
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "figures"))
from plot_statistics import compare_statistics, paper_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "results/reference/test_predictions.csv.gz"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figures")
    parser.add_argument(
        "--verify-only", action="store_true", help="Check plot values without rendering images."
    )
    args = parser.parse_args()
    data = pd.read_csv(args.predictions, float_precision="round_trip")
    if len(data) != 43721 or data.duplicated(["farm_id", "issue_slot"]).any():
        raise ValueError("Expected 43,721 unique test origins")
    summaries = compare_statistics(paper_statistics(data), ROOT / "figures/source_data")
    print(f"Verified {len(summaries)} plot-data tables covering 14 empirical figures.")
    if args.verify_only:
        return
    from draw import DRAW
    import matplotlib.pyplot as plt

    out = args.output.resolve()
    published = (ROOT / "figures/published").resolve()
    if published == out or published in out.parents:
        parser.error("Choose an output directory outside the archived publication artwork.")
    out.mkdir(parents=True, exist_ok=True)
    for k in ["01", "02", "03", "04"]:
        for ext in ["png", "svg"]:
            filename = f"figure_{k}.{ext}"
            shutil.copyfile(published / filename, out / filename)
    for number, draw in DRAW.items():
        fig = draw(summaries)
        if number in {"05", "06", "12"}:
            note = "May* denotes a partial month."
            if number == "12":
                note += "  Low / high: score < 0.5 / ≥ 0.5."
            fig.supxlabel(note, fontsize=8)
        fig.savefig(
            out / f"figure_{number}.png", dpi=180, metadata={"Software": "CRHF reproducibility"}
        )
        fig.savefig(
            out / f"figure_{number}.svg", metadata={"Date": None, "Creator": "CRHF reproducibility"}
        )
        plt.close(fig)
    print(f"Wrote 14 data-driven charts and the four method diagrams to {out}.")


if __name__ == "__main__":
    main()
