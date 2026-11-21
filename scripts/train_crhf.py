"""Train or replay CRHF on the public SCADA release."""

from argparse import ArgumentParser
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "prepare", help="Build and check the full feature grid and sequence dimensions."
    )
    train_parser = commands.add_parser(
        "train", help="Fit all stages, ablations, and four forward residual fits."
    )
    train_parser.add_argument("--output", type=Path, default=ROOT / "outputs/retrained")
    replay_parser = commands.add_parser(
        "replay", help="Run the saved model with sequentially arriving feedback."
    )
    replay_parser.add_argument("--output", type=Path, default=ROOT / "outputs/replay")
    replay_parser.add_argument("--bundle", type=Path, default=ROOT / "models/crhf.joblib")
    replay_parser.add_argument(
        "--quick",
        action="store_true",
        help="Evaluate the first two test days and the full model only.",
    )
    stage2 = commands.add_parser("stage2", help="Fit the nine available-power comparisons.")
    stage2.add_argument("--output", type=Path, default=ROOT / "outputs/stage2")
    stage2.add_argument(
        "--bridges",
        type=Path,
        help="Use stage1_bridges.joblib from a preceding public training run.",
    )
    diagnostics = commands.add_parser(
        "diagnostics", help="Replay risk scores, fixed candidates, the gate and bias tests."
    )
    diagnostics.add_argument("--output", type=Path, default=ROOT / "outputs/diagnostics")
    diagnostics.add_argument("--bundle", type=Path, default=ROOT / "models/crhf.joblib")
    diagnostics.add_argument("--quick", action="store_true", help="Run the first two test days.")
    args = parser.parse_args()
    from crhf.pipeline import ROLES, new_output, reconstruct, replay

    if args.command == "prepare":
        data = reconstruct()
        print(f"Forecast rows: {len(data['frame']):,}. Sequence tensor: {data['sequence'].shape}.")
        print(f"Public farms: {', '.join(sorted(data['frame'].farm.unique()))}.")
        return
    if args.command == "stage2":
        from crhf.experiments import available_experiment

        available_experiment(new_output(args.output), args.bridges)
    elif args.command == "diagnostics":
        from crhf.experiments import diagnostic_experiment

        if not args.bundle.is_file():
            parser.error(f"Cannot find the trained bundle at {args.bundle}.")
        diagnostic_experiment(new_output(args.output), args.bundle, args.quick)
    elif args.command == "train":
        from crhf.training import train

        train(new_output(args.output))
    else:
        if not args.bundle.is_file():
            parser.error(
                f"Cannot find {args.bundle}. Pass a model exported by the train command with --bundle."
            )
        replay(
            new_output(args.output),
            roles=("full",) if args.quick else ROLES,
            until="1998-02-03" if args.quick else None,
            bundle_path=args.bundle,
        )


if __name__ == "__main__":
    main()
