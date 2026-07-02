from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_forecasting.metrics import prediction_interval_metrics, regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction CSV files.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "metrics_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for csv_path in sorted(args.results_dir.glob("*.csv")):
        if csv_path.name == args.output.name:
            continue
        frame = pd.read_csv(csv_path)
        required = {"y_true", "q0.1", "q0.5", "q0.9"}
        if not required.issubset(frame.columns):
            continue
        y_true = frame["y_true"].to_numpy()
        predictions = frame[["q0.1", "q0.5", "q0.9"]].to_numpy()
        row = {"model": csv_path.stem}
        row.update(regression_metrics(y_true, frame["q0.5"].to_numpy()))
        row.update(prediction_interval_metrics(y_true, predictions))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("model")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
