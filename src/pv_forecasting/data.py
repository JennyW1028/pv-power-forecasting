from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


DEFAULT_EDGE_INDEX = torch.tensor(
    [
        [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6],
        [2, 4, 2, 3, 4, 5, 6, 0, 1, 3, 4, 5, 6, 1, 2, 4, 5, 6, 0, 1, 2, 3, 5, 6, 1, 2, 3, 4, 6, 1, 2, 3, 4, 5],
    ],
    dtype=torch.long,
)


@dataclass
class GraphData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    target_scaler: MinMaxScaler
    feature_scalers: List[MinMaxScaler]
    adjacency: torch.Tensor
    feature_names: List[str]


@dataclass
class StationData:
    x_train: List[torch.Tensor]
    y_train: List[torch.Tensor]
    x_test: List[torch.Tensor]
    y_test: List[torch.Tensor]
    target_scalers: List[MinMaxScaler]
    feature_scalers: List[MinMaxScaler]
    input_features: int


def load_site_frames(data_dir: str | Path) -> List[pd.DataFrame]:
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_path.resolve()}")

    frames: List[pd.DataFrame] = []
    for file_path in files:
        frame = pd.read_csv(file_path, index_col=0)
        frame.index = pd.to_datetime(frame.index)
        frames.append(frame)
    return frames


def split_by_date(frames: Sequence[pd.DataFrame], train_ratio: float = 0.8) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    train_frames: List[pd.DataFrame] = []
    test_frames: List[pd.DataFrame] = []

    for frame in frames:
        work = frame.copy()
        work["_date"] = pd.to_datetime(work.index).date
        dates = sorted(work["_date"].unique())
        split_idx = int(len(dates) * train_ratio)
        train_dates = set(dates[:split_idx])

        train_frames.append(work[work["_date"].isin(train_dates)].drop(columns=["_date"]))
        test_frames.append(work[~work["_date"].isin(train_dates)].drop(columns=["_date"]))

    return train_frames, test_frames


def build_correlation_adjacency(
    frames: Sequence[pd.DataFrame],
    target_column: str = "Active_Power",
    threshold: float = 0.6,
    include_self: bool = True,
) -> torch.Tensor:
    power_matrix = pd.concat([frame[target_column].reset_index(drop=True) for frame in frames], axis=1)
    corr = power_matrix.corr(method="pearson").abs().fillna(0.0).to_numpy()
    adjacency = (corr >= threshold).astype(np.float32)
    if include_self:
        np.fill_diagonal(adjacency, 1.0)
    else:
        np.fill_diagonal(adjacency, 0.0)
    return torch.tensor(adjacency, dtype=torch.float32)


def select_feature_columns(frame: pd.DataFrame, feature_columns: Sequence[str] | None = None) -> List[str]:
    if feature_columns is None:
        return list(frame.columns)
    missing = [name for name in feature_columns if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return list(feature_columns)


def create_graph_windows(
    site_arrays: Sequence[np.ndarray],
    targets: np.ndarray,
    seq_len: int,
    points_per_day: int = 120,
    horizon: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    stacked = np.stack(site_arrays, axis=1)
    total_days = len(targets) // points_per_day
    x_windows = []
    y_windows = []

    for day in range(total_days):
        start_idx = day * points_per_day
        end_idx = start_idx + points_per_day
        last_start = end_idx - seq_len - horizon + 1
        for idx in range(start_idx, last_start):
            x_windows.append(stacked[idx : idx + seq_len])
            y_windows.append(targets[idx + seq_len + horizon - 1])

    return torch.tensor(np.asarray(x_windows), dtype=torch.float32), torch.tensor(np.asarray(y_windows), dtype=torch.float32)


def prepare_graph_data(
    data_dir: str | Path,
    seq_len: int = 24,
    points_per_day: int = 120,
    train_ratio: float = 0.8,
    graph_threshold: float = 0.6,
    feature_columns: Sequence[str] | None = None,
) -> GraphData:
    frames = load_site_frames(data_dir)
    train_frames, test_frames = split_by_date(frames, train_ratio=train_ratio)
    selected_columns = select_feature_columns(train_frames[0], feature_columns)
    adjacency = build_correlation_adjacency(train_frames, threshold=graph_threshold)

    feature_scalers: List[MinMaxScaler] = []
    x_train_arrays: List[np.ndarray] = []
    x_test_arrays: List[np.ndarray] = []

    for train_frame, test_frame in zip(train_frames, test_frames):
        scaler = MinMaxScaler()
        x_train_arrays.append(scaler.fit_transform(train_frame[selected_columns]))
        x_test_arrays.append(scaler.transform(test_frame[selected_columns]))
        feature_scalers.append(scaler)

    train_target = sum(frame["Active_Power"] for frame in train_frames)
    test_target = sum(frame["Active_Power"] for frame in test_frames)

    target_scaler = MinMaxScaler()
    y_train = target_scaler.fit_transform(np.asarray(train_target).reshape(-1, 1))
    y_test = target_scaler.transform(np.asarray(test_target).reshape(-1, 1))

    x_train, y_train_tensor = create_graph_windows(x_train_arrays, y_train, seq_len, points_per_day)
    x_test, y_test_tensor = create_graph_windows(x_test_arrays, y_test, seq_len, points_per_day)

    return GraphData(
        x_train=x_train,
        y_train=y_train_tensor,
        x_test=x_test,
        y_test=y_test_tensor,
        target_scaler=target_scaler,
        feature_scalers=feature_scalers,
        adjacency=adjacency,
        feature_names=selected_columns,
    )


def create_station_windows(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
    points_per_day: int = 120,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_days = len(targets) // points_per_day
    x_windows = []
    y_windows = []

    for day in range(total_days):
        start_idx = day * points_per_day
        end_idx = start_idx + points_per_day
        for idx in range(start_idx, end_idx - seq_len):
            x_windows.append(features[idx : idx + seq_len])
            y_windows.append(targets[idx + seq_len])

    return torch.tensor(np.asarray(x_windows), dtype=torch.float32), torch.tensor(np.asarray(y_windows), dtype=torch.float32)


def prepare_station_data(
    data_dir: str | Path,
    seq_len: int = 24,
    points_per_day: int = 120,
    train_ratio: float = 0.8,
    target_column: str = "Active_Power",
    feature_columns: Sequence[str] | None = None,
) -> StationData:
    frames = load_site_frames(data_dir)
    train_frames, test_frames = split_by_date(frames, train_ratio=train_ratio)
    selected_columns = select_feature_columns(train_frames[0], feature_columns)

    x_train: List[torch.Tensor] = []
    y_train: List[torch.Tensor] = []
    x_test: List[torch.Tensor] = []
    y_test: List[torch.Tensor] = []
    feature_scalers: List[MinMaxScaler] = []
    target_scalers: List[MinMaxScaler] = []

    for train_frame, test_frame in zip(train_frames, test_frames):
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        train_features = feature_scaler.fit_transform(train_frame[selected_columns])
        test_features = feature_scaler.transform(test_frame[selected_columns])
        train_targets = target_scaler.fit_transform(train_frame[[target_column]]).reshape(-1)
        test_targets = target_scaler.transform(test_frame[[target_column]]).reshape(-1)

        station_x_train, station_y_train = create_station_windows(train_features, train_targets, seq_len, points_per_day)
        station_x_test, station_y_test = create_station_windows(test_features, test_targets, seq_len, points_per_day)

        x_train.append(station_x_train)
        y_train.append(station_y_train)
        x_test.append(station_x_test)
        y_test.append(station_y_test)
        feature_scalers.append(feature_scaler)
        target_scalers.append(target_scaler)

    return StationData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        target_scalers=target_scalers,
        feature_scalers=feature_scalers,
        input_features=len(selected_columns),
    )


def inverse_transform_quantiles(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return scaler.inverse_transform(values)


def as_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)
