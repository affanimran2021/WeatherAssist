"""Unit and integrity tests for Group 6 baseline LSTM & anomaly injection pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.anomaly_injection import inject_anomalies
from src.dataset import DemandWindowDataset, FeatureScaler
from src.model import DemandLSTM


def _make_synthetic_df(n: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    hours = (dates.hour % 24) + 1
    demand = 5500.0 + 800.0 * np.sin(2 * np.pi * np.arange(n) / 24.0)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "hour": hours.astype(int),
            "demand_mw": demand,
            "year": dates.year,
        }
    )


def test_parameter_counts_match_proposal_appendix_a() -> None:
    baseline = DemandLSTM(input_size=2, hidden_size=32, num_layers=1, output_size=1)
    context = DemandLSTM(input_size=4, hidden_size=32, num_layers=1, output_size=1)
    assert baseline.count_parameters() == 4641
    assert context.count_parameters() == 4897


def test_forward_shape() -> None:
    model = DemandLSTM(input_size=2, hidden_size=32, num_layers=1, output_size=1)
    x = torch.randn(16, 24, 2)
    y = model(x)
    assert y.shape == (16,)


def test_scaler_and_window_alignment_no_leakage() -> None:
    df = _make_synthetic_df(120)
    train_df = df.iloc[:80].reset_index(drop=True)
    val_df = df.iloc[80:].reset_index(drop=True)

    scaler = FeatureScaler.fit(train_df)
    assert np.isclose(scaler.means["demand_mw"], train_df["demand_mw"].mean())
    assert np.isclose(scaler.stds["demand_mw"], train_df["demand_mw"].std(ddof=0))

    ds = DemandWindowDataset(val_df, scaler, window_size=24)
    assert len(ds) == len(val_df) - 24
    x0, y0 = ds[0]
    assert x0.shape == (24, 2)
    # Inverse-transformed target 0 must match row 24 of val_df
    recovered = scaler.inverse_transform_target(np.array([y0.item()]))[0]
    assert np.isclose(recovered, val_df["demand_mw"].iloc[24], atol=1e-3)


def test_anomaly_injection_preserves_control_and_is_deterministic() -> None:
    control_df = _make_synthetic_df(300)
    snapshot = control_df.copy(deep=True)
    cfg = {"label": "Mixed", "type": "mixed", "rate": 0.05, "min_spacing": 6, "mode": "target_only"}

    run1 = inject_anomalies(control_df, "mixed_test", cfg, window_size=24, seed=42)
    run2 = inject_anomalies(control_df, "mixed_test", cfg, window_size=24, seed=42)

    # Control DataFrame must remain untouched
    pd.testing.assert_frame_equal(control_df, snapshot)
    # Deterministic across identical seeds
    np.testing.assert_array_equal(run1.is_anomaly, run2.is_anomaly)
    np.testing.assert_allclose(run1.targets_mw, run2.targets_mw)
    pd.testing.assert_frame_equal(run1.manifest_df, run2.manifest_df)
    # Manifest rows match flagged anomalies count
    assert len(run1.manifest_df) == int(run1.is_anomaly.sum())
    assert int(run1.is_anomaly.sum()) > 0
