"""Controlled, deterministic anomaly injection for electricity demand time series.

Designed to be reused identically across both the demand-only baseline LSTM and
the context-aware LSTM (temperature + holidays) for apples-to-apples comparison.

Guarantees:
- Original control DataFrame is never mutated (always copied).
- Every injection event is deterministically reproducible via numpy Generator(seed).
- Full per-hour manifest is returned for traceability and qualitative error analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class InjectedCase:
    case_id: str
    label: str
    mode: str  # "target_only" (Mode A) or "raw_series" (Mode B)
    seed: int
    corrupted_df: pd.DataFrame
    targets_mw: np.ndarray
    is_anomaly: np.ndarray  # bool array aligned with window targets [window_size .. N-1]
    manifest_df: pd.DataFrame


def _sample_spaced_starts(
    rng: np.random.Generator,
    n_windows: int,
    block_lengths: list[int],
    min_spacing: int,
) -> list[tuple[int, int]]:
    """Sample non-overlapping (window_idx, length) intervals with minimum spacing."""
    occupied = np.zeros(n_windows, dtype=bool)
    placed: list[tuple[int, int]] = []
    # Leave 1-hour margin at index 0 so flatlines can copy the preceding hour cleanly
    candidates = np.arange(1, n_windows, dtype=int)

    for length in block_lengths:
        rng.shuffle(candidates)
        found = False
        for start in candidates:
            end = start + length
            if end > n_windows:
                continue
            lo = max(0, start - min_spacing)
            hi = min(n_windows, end + min_spacing)
            if not occupied[lo:hi].any():
                occupied[start:end] = True
                placed.append((int(start), int(length)))
                found = True
                break
        if not found:
            # Relax spacing to non-overlapping only if crowded
            for start in candidates:
                end = start + length
                if end <= n_windows and not occupied[start:end].any():
                    occupied[start:end] = True
                    placed.append((int(start), int(length)))
                    found = True
                    break
        if not found:
            raise RuntimeError("Could not place all anomaly blocks within validation series.")

    placed.sort(key=lambda item: item[0])
    return placed


def inject_anomalies(
    control_df: pd.DataFrame,
    case_id: str,
    case_cfg: dict[str, Any],
    window_size: int = 24,
    seed: int = 42,
) -> InjectedCase:
    """Create an isolated corrupted copy of control_df according to case_cfg."""
    rng = np.random.default_rng(seed)
    corrupted_df = control_df.copy(deep=True)
    n_rows = len(corrupted_df)
    n_windows = n_rows - window_size
    if n_windows <= 0:
        raise ValueError(f"Control dataframe too short ({n_rows}) for window_size={window_size}.")

    demand = corrupted_df["demand_mw"].to_numpy(dtype=np.float64).copy()
    original_demand = demand.copy()
    is_anomaly = np.zeros(n_windows, dtype=bool)
    manifest_rows: list[dict[str, Any]] = []

    ctype = str(case_cfg["type"])
    mode = str(case_cfg.get("mode", "target_only"))
    rate = float(case_cfg.get("rate", 0.02))
    min_spacing = int(case_cfg.get("min_spacing", 24))
    target_hours = max(1, int(round(rate * n_windows)))

    def record_corruption(w_idx: int, new_val: float, a_type: str, severity: str) -> None:
        row_idx = w_idx + window_size
        orig = float(original_demand[row_idx])
        new_clamped = float(max(0.0, new_val))
        demand[row_idx] = new_clamped
        is_anomaly[w_idx] = True
        manifest_rows.append(
            {
                "window_idx": w_idx,
                "row_idx": row_idx,
                "date": str(control_df["date"].iloc[row_idx]),
                "hour": int(control_df["hour"].iloc[row_idx]),
                "original_mw": round(orig, 3),
                "injected_mw": round(new_clamped, 3),
                "delta_mw": round(new_clamped - orig, 3),
                "delta_pct": round(((new_clamped - orig) / max(orig, 1e-6)) * 100.0, 2),
                "anomaly_type": a_type,
                "severity": severity,
            }
        )

    if ctype == "point_spike_drop":
        delta = float(case_cfg["delta"])
        severity = str(case_cfg.get("severity", f"{int(delta * 100)}pct"))
        events = _sample_spaced_starts(rng, n_windows, [1] * target_hours, min_spacing=min_spacing)
        for start, _ in events:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            a_type = "point_spike" if sign > 0 else "point_drop"
            row_idx = start + window_size
            record_corruption(start, original_demand[row_idx] * (1.0 + sign * delta), a_type, severity)

    elif ctype == "dropout_or_scale":
        d_lo, d_hi = case_cfg.get("duration_range", [1, 3])
        block_lengths: list[int] = []
        total = 0
        while total < target_hours:
            dur = int(rng.integers(d_lo, d_hi + 1))
            dur = min(dur, target_hours - total)
            block_lengths.append(dur)
            total += dur
        events = _sample_spaced_starts(rng, n_windows, block_lengths, min_spacing=min_spacing)
        for start, length in events:
            is_zero = bool(rng.random() < 0.5)
            for k in range(length):
                w_idx = start + k
                row_idx = w_idx + window_size
                if is_zero:
                    record_corruption(w_idx, 0.0, "sensor_dropout_zero", "severe")
                else:
                    record_corruption(w_idx, original_demand[row_idx] * 0.1, "decimal_shift_0.1x", "severe")

    elif ctype == "flatline":
        block_len = int(case_cfg.get("block_length", 5))
        n_blocks = max(1, target_hours // block_len)
        events = _sample_spaced_starts(rng, n_windows, [block_len] * n_blocks, min_spacing=min_spacing)
        for start, length in events:
            hold_val = float(original_demand[start + window_size - 1])
            for k in range(length):
                record_corruption(start + k, hold_val, "flatline_stuck_meter", "subtle")

    elif ctype == "mixed":
        # Stratified 25% moderate spikes/drops, 25% severe spikes/drops, 25% dropouts, 25% flatlines
        n_mod = max(1, target_hours // 4)
        n_sev = max(1, target_hours // 4)
        n_drop_hours = max(1, target_hours // 4)
        n_flat_hours = max(5, target_hours - n_mod - n_sev - n_drop_hours)

        specs: list[tuple[int, str]] = []
        specs.extend((1, "mod_spike_drop") for _ in range(n_mod))
        specs.extend((1, "sev_spike_drop") for _ in range(n_sev))

        rem = n_drop_hours
        while rem > 0:
            dur = min(int(rng.integers(1, 4)), rem)
            specs.append((dur, "dropout_or_scale"))
            rem -= dur

        n_flat_blocks = max(1, n_flat_hours // 5)
        specs.extend((5, "flatline") for _ in range(n_flat_blocks))

        rng.shuffle(specs)
        lengths = [s[0] for s in specs]
        events = _sample_spaced_starts(rng, n_windows, lengths, min_spacing=min_spacing)

        # Pair each placed event with its corresponding spec length
        # Since _sample_spaced_starts sorts by start, re-map specs by matching lengths cleanly
        by_len: dict[int, list[str]] = {}
        for length, kind in specs:
            by_len.setdefault(length, []).append(kind)

        for start, length in events:
            kind = by_len[length].pop(0)
            if kind == "mod_spike_drop":
                sign = 1.0 if rng.random() < 0.5 else -1.0
                a_type = "point_spike" if sign > 0 else "point_drop"
                record_corruption(start, original_demand[start + window_size] * (1.0 + sign * 0.30), a_type, "moderate")
            elif kind == "sev_spike_drop":
                sign = 1.0 if rng.random() < 0.5 else -1.0
                a_type = "point_spike" if sign > 0 else "point_drop"
                record_corruption(start, original_demand[start + window_size] * (1.0 + sign * 0.50), a_type, "severe")
            elif kind == "dropout_or_scale":
                is_zero = bool(rng.random() < 0.5)
                for k in range(length):
                    w_idx = start + k
                    row_idx = w_idx + window_size
                    if is_zero:
                        record_corruption(w_idx, 0.0, "sensor_dropout_zero", "severe")
                    else:
                        record_corruption(w_idx, original_demand[row_idx] * 0.1, "decimal_shift_0.1x", "severe")
            elif kind == "flatline":
                hold_val = float(original_demand[start + window_size - 1])
                for k in range(length):
                    record_corruption(start + k, hold_val, "flatline_stuck_meter", "subtle")

    else:
        raise ValueError(f"Unsupported anomaly case type: {ctype}")

    corrupted_df["demand_mw"] = demand
    manifest_df = pd.DataFrame(manifest_rows)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values("window_idx").reset_index(drop=True)

    return InjectedCase(
        case_id=case_id,
        label=str(case_cfg.get("label", case_id)),
        mode=mode,
        seed=seed,
        corrupted_df=corrupted_df,
        targets_mw=demand[window_size:].copy(),
        is_anomaly=is_anomaly,
        manifest_df=manifest_df,
    )
