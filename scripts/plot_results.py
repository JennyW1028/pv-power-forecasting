from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prediction diagnostics.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "notebook_exports")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--max-points", type=int, default=600)
    return parser.parse_args()


def plot_prediction_interval(frame: pd.DataFrame, output_path: Path, max_points: int) -> None:
    work = frame.head(max_points).copy()
    x = range(len(work))
    plt.figure(figsize=(11, 4.5))
    plt.plot(x, work["y_true"], color="#1f2937", linewidth=1.4, label="Observed")
    plt.plot(x, work["q0.5"], color="#2563eb", linewidth=1.3, label="Median forecast")
    plt.fill_between(x, work["q0.1"], work["q0.9"], color="#93c5fd", alpha=0.35, label="0.1-0.9 interval")
    plt.xlabel("Sample")
    plt.ylabel("Aggregated active power")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_scatter(frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(frame["y_true"], frame["q0.5"], s=8, alpha=0.45, color="#2563eb")
    lower = min(frame["y_true"].min(), frame["q0.5"].min())
    upper = max(frame["y_true"].max(), frame["q0.5"].max())
    plt.plot([lower, upper], [lower, upper], color="#111827", linewidth=1, linestyle="--")
    plt.xlabel("Observed")
    plt.ylabel("Median forecast")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(args.results_dir.glob("*.csv")):
        if csv_path.stem == "metrics_summary":
            continue
        frame = pd.read_csv(csv_path)
        required = {"y_true", "q0.1", "q0.5", "q0.9"}
        if not required.issubset(frame.columns):
            continue
        interval_path = args.output_dir / f"{csv_path.stem}_interval.png"
        scatter_path = args.output_dir / f"{csv_path.stem}_scatter.png"
        plot_prediction_interval(frame, interval_path, args.max_points)
        plot_scatter(frame, scatter_path)
        print(f"Saved {interval_path}")
        print(f"Saved {scatter_path}")


if __name__ == "__main__":
    main()
