"""Command-line interface."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .io import inventory
from .pipeline import analyse_one, analyse_folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="microscopy-pipeline",
        description="Inventory and perform baseline QC morphometry on microscopy TIFFs.")
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="Write TIFF metadata inventory CSV")
    inv.add_argument("input", type=Path)
    inv.add_argument("--output", type=Path, default=Path("inventory.csv"))
    one = sub.add_parser("analyse", help="Analyse one TIFF")
    one.add_argument("input", type=Path)
    one.add_argument("--output", type=Path, default=Path("results"))
    one.add_argument("--pixel-size-um", type=float, default=None)
    batch = sub.add_parser("batch", help="Analyse all TIFFs recursively in a folder")
    batch.add_argument("input", type=Path)
    batch.add_argument("--output", type=Path, default=Path("results"))
    batch.add_argument("--pixel-size-um", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inventory":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(inventory(args.input)).to_csv(args.output, index=False)
        print(f"Wrote {args.output}")
    elif args.command == "analyse":
        rows = analyse_one(args.input, args.output, args.pixel_size_um)
        pd.DataFrame(rows).to_csv(args.output / "measurements.csv", index=False)
        print(f"Wrote results to {args.output}")
    elif args.command == "batch":
        result = analyse_folder(args.input, args.output, args.pixel_size_um)
        print(f"Processed {len(result)} measurement rows; results in {args.output}")


if __name__ == "__main__":
    main()
