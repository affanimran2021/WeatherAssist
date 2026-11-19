"""Chronological sliding-window Dataset and leak-free FeatureScaler.

Key guarantees:
- Scaler statistics (mean, std) are fitted ONLY on the 2022-2023 training split.
- Sliding windows are constructed strictly within each split so no window crosses
  train/val/test boundaries.
- Supports both 2-feature baseline (['hour', 'demand_mw']) and 4-feature context
  (['hour', 'temperature', 'holiday', 'demand_mw']).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils import load_json, save_dict_to_json

BASELINE_FEATURES = ("hour", "demand_mw")
TARGET_COLUMN = "demand_mw"


@dataclass
class FeatureScaler:
    """JSON-serializable standard scaler fit exclusively on training data."""

    feature_cols: list[str]
    means: dict[str, float]
    stds: dict[str, float]
    target_col: str = TARGET_COLUMN

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        feature_cols: Sequence[str] = BASELINE_FEATURES,
        target_col: str = TARGET_COLUMN,
    ) -> "FeatureScaler":
        cols = list(feature_cols)
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for col in set(cols + [target_col]):
            arr = train_df[col].to_numpy(dtype=np.float64)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            if std_val == 0.0:
                std_val = 1.0
            means[col] = mean_val
            stds[col] = std_val
        return cls(feature_cols=cols, means=means, stds=stds, target_col=target_col)

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self.feature_cols:
            arr = df[col].to_numpy(dtype=np.float32)
            cols.append((arr - self.means[col]) / self.stds[col])
        return np.stack(cols, axis=-1).astype(np.float32)

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return ((arr - self.means[self.target_col]) / self.stds[self.target_col]).astype(np.float32)

    def inverse_transform_target(self, scaled_values: np.ndarray) -> np.ndarray:
        arr = np.asarray(scaled_values, dtype=np.float64)
        return arr * self.stds[self.target_col] + self.means[self.target_col]

    def save(self, path: str | Path) -> None:
        save_dict_to_json(
            {
                "feature_cols": self.feature_cols,
                "target_col": self.target_col,
                "means": self.means,
                "stds": self.stds,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        data = load_json(path)
        return cls(
            feature_cols=list(data["feature_cols"]),
            means={k: float(v) for k, v in data["means"].items()},
            stds={k: float(v) for k, v in data["stds"].items()},
            target_col=str(data.get("target_col", TARGET_COLUMN)),
        )


class DemandWindowDataset(Dataset):
    """24-hour sliding window -> next-hour electricity demand target.

    Window index `i` maps:
      - Input X[i]:  rows [i, i + 1, ..., i + window_size - 1]
      - Target y[i]: row  (i + window_size)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: FeatureScaler,
        window_size: int = 24,
        target_override_mw: np.ndarray | None = None,
    ) -> None:
        if len(df) <= window_size:
            raise ValueError(f"Split length ({len(df)}) must exceed window_size ({window_size}).")
        self.window_size = window_size
        self.features = scaler.transform_features(df)
        raw_targets = df[scaler.target_col].to_numpy(dtype=np.float64)
        if target_override_mw is not None:
            expected_len = len(df) - window_size
            if len(target_override_mw) != expected_len:
                raise ValueError(
                    f"target_override_mw length ({len(target_override_mw)}) != expected ({expected_len})"
                )
            self.targets_mw = np.asarray(target_override_mw, dtype=np.float64)
        else:
            self.targets_mw = raw_targets[window_size:].astype(np.float64)

        self.targets_scaled = scaler.transform_target(self.targets_mw)
        self.target_dates = df["date"].iloc[window_size:].reset_index(drop=True)
        self.target_hours = df["hour"].iloc[window_size:].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.targets_scaled)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.window_size]
        y = self.targets_scaled[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
