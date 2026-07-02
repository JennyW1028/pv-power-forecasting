from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import GraphData, StationData, as_numpy, inverse_transform_quantiles


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def quantile_loss(predictions: torch.Tensor, targets: torch.Tensor, quantiles: Sequence[float] = (0.1, 0.5, 0.9)) -> torch.Tensor:
    if targets.dim() > 1:
        targets = targets.squeeze(-1)
    expanded_targets = targets.unsqueeze(-1).expand_as(predictions)
    errors = expanded_targets - predictions
    quantile_tensor = torch.tensor(quantiles, device=predictions.device, dtype=predictions.dtype).view(1, -1)
    quantile_tensor = quantile_tensor.expand_as(predictions)
    return torch.max(quantile_tensor * errors, (quantile_tensor - 1) * errors).mean()


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    edge_index: torch.Tensor | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> tuple[list[float], list[float], torch.nn.Module]:
    device = device or resolve_device()
    model = model.to(device)
    edge_index = edge_index.to(device) if edge_index is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses: list[float] = []
    test_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_samples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = quantile_loss(model(batch_x, edge_index), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * batch_x.size(0)
            train_samples += batch_x.size(0)

        train_loss = total_train_loss / max(train_samples, 1)
        test_loss = evaluate_loss(model, test_loader, edge_index=edge_index, device=device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch + 1:03d}/{epochs:03d} | train_loss={train_loss:.6f} | test_loss={test_loss:.6f}")

    return train_losses, test_losses, model.cpu()


def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> float:
    device = device or resolve_device()
    model.eval()
    total_loss = 0.0
    samples = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = quantile_loss(model(batch_x, edge_index), batch_y)
            total_loss += loss.item() * batch_x.size(0)
            samples += batch_x.size(0)
    return total_loss / max(samples, 1)


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def train_graph_experiment(
    model: torch.nn.Module,
    data: GraphData,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    train_loader = make_loader(data.x_train, data.y_train, batch_size=batch_size, shuffle=True)
    test_loader = make_loader(data.x_test, data.y_test, batch_size=batch_size, shuffle=False)
    _, _, trained_model = train_model(model, train_loader, test_loader, edge_index=data.edge_index, epochs=epochs, lr=lr)
    predictions = predict(trained_model, test_loader, edge_index=data.edge_index)
    predictions = inverse_transform_quantiles(data.target_scaler, predictions)
    y_true = inverse_transform_quantiles(data.target_scaler, as_numpy(data.y_test).reshape(-1, 1)).reshape(-1)
    return trained_model, prediction_frame(y_true, predictions)


def predict(model: torch.nn.Module, loader: DataLoader, edge_index: torch.Tensor | None = None) -> np.ndarray:
    device = resolve_device()
    model = model.to(device)
    edge_index = edge_index.to(device) if edge_index is not None else None
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch_x, _ in loader:
            outputs.append(model(batch_x.to(device), edge_index).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def prediction_frame(y_true: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": np.asarray(y_true).reshape(-1)})
    frame[["q0.1", "q0.5", "q0.9"]] = predictions
    return frame


def save_prediction_frame(frame: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def train_station_experiment(
    model_factory,
    data: StationData,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
) -> tuple[list[torch.nn.Module], pd.DataFrame]:
    trained_models = []
    prediction_sum = None
    truth_sum = None

    for station_idx, (x_train, y_train, x_test, y_test, scaler) in enumerate(
        zip(data.x_train, data.y_train, data.x_test, data.y_test, data.target_scalers),
        start=1,
    ):
        print(f"Training station {station_idx:02d}")
        train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
        test_loader = make_loader(x_test, y_test, batch_size=batch_size, shuffle=False)
        model = model_factory()
        _, _, trained_model = train_model(model, train_loader, test_loader, epochs=epochs, lr=lr)
        trained_models.append(trained_model)

        station_predictions = inverse_transform_quantiles(scaler, predict(trained_model, test_loader))
        station_truth = inverse_transform_quantiles(scaler, as_numpy(y_test).reshape(-1, 1)).reshape(-1)

        prediction_sum = station_predictions if prediction_sum is None else prediction_sum + station_predictions
        truth_sum = station_truth if truth_sum is None else truth_sum + station_truth

    return trained_models, prediction_frame(truth_sum, prediction_sum)
