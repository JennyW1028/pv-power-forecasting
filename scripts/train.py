from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_forecasting.data import prepare_graph_data, prepare_station_data
from pv_forecasting.models import build_model
from pv_forecasting.training import save_prediction_frame, set_seed, train_graph_experiment, train_station_experiment


GRAPH_MODELS = {"gcn-informer", "gat-informer", "gcn-lstm"}
STATION_MODELS = {"informer", "lstm"}
ALL_MODELS = ["gcn-informer", "gat-informer", "gcn-lstm", "informer", "lstm"]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train photovoltaic forecasting models.")
    parser.add_argument("--model", choices=ALL_MODELS + ["all"], default="gcn-informer")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--points-per-day", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--graph-threshold", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def pick(args: argparse.Namespace, config: dict, name: str):
    value = getattr(args, name.replace("-", "_"))
    return config[name] if value is None else value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_dir = args.data_dir or ROOT / config["data_dir"]
    results_dir = args.results_dir or ROOT / config["results_dir"]
    seq_len = pick(args, config, "seq_len")
    points_per_day = pick(args, config, "points_per_day")
    train_ratio = pick(args, config, "train_ratio")
    graph_threshold = pick(args, config, "graph_threshold")
    hidden_dim = pick(args, config, "hidden_dim")
    batch_size = pick(args, config, "batch_size")
    epochs = pick(args, config, "epochs")
    lr = args.lr if args.lr is not None else config["learning_rate"]
    seed = pick(args, config, "seed")

    set_seed(seed)
    models = ALL_MODELS if args.model == "all" else [args.model]

    graph_data = None
    station_data = None

    for model_name in models:
        print(f"Running model: {model_name}")
        if model_name in GRAPH_MODELS:
            if graph_data is None:
                graph_data = prepare_graph_data(
                    data_dir=data_dir,
                    seq_len=seq_len,
                    points_per_day=points_per_day,
                    train_ratio=train_ratio,
                    graph_threshold=graph_threshold,
                )
            model = build_model(
                model_name,
                input_features=graph_data.x_train.shape[-1],
                seq_len=seq_len,
                num_nodes=graph_data.x_train.shape[2],
                hidden_dim=hidden_dim,
            )
            _, frame = train_graph_experiment(model, graph_data, batch_size=batch_size, epochs=epochs, lr=lr)
        else:
            if station_data is None:
                station_data = prepare_station_data(
                    data_dir=data_dir,
                    seq_len=seq_len,
                    points_per_day=points_per_day,
                    train_ratio=train_ratio,
                )

            def model_factory(name=model_name):
                return build_model(name, input_features=station_data.input_features, seq_len=seq_len, hidden_dim=hidden_dim)

            _, frame = train_station_experiment(model_factory, station_data, batch_size=batch_size, epochs=epochs, lr=lr)

        output_path = results_dir / f"{model_name}.csv"
        save_prediction_frame(frame, output_path)
        print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
