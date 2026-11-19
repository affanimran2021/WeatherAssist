"""Download, clean, and chronologically split IESO Toronto hourly demand data (2022-2025).

Data source (IESO Terms of Use, attribution required):
    https://reports-public.ieso.ca/public/DemandZonal/PUB_DemandZonal_<YEAR>.csv

Chronological split (no leakage across splits):
    - Train:      2022-2023 (50%)
    - Validation: 2024      (25%, leap year -> 8,784 hours)
    - Test:       2025      (25%, held out until Milestone 3 final evaluation)
"""
from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import save_dict_to_json, set_logger, sha256_file

IESO_BASE_URL = "https://reports-public.ieso.ca/public/DemandZonal"
DEFAULT_YEARS = (2022, 2023, 2024, 2025)


def download_ieso_year(year: int, raw_dir: Path, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = f"PUB_DemandZonal_{year}.csv"
    dest = raw_dir / filename
    if dest.exists() and not force:
        logging.info("Using cached raw file: %s", dest)
        return dest

    url = f"{IESO_BASE_URL}/{filename}"
    logging.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


def load_and_clean_ieso_files(raw_paths: list[Path], zone: str = "Toronto") -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    for path in raw_paths:
        # IESO files have 3 metadata header lines beginning with '\\'
        df = pd.read_csv(path, skiprows=3)
        df.columns = [c.strip() for c in df.columns]
        if "Date" not in df.columns or "Hour" not in df.columns or zone not in df.columns:
            raise ValueError(f"Expected columns Date, Hour, {zone} in {path}; got {list(df.columns)}")
        sub = df[["Date", "Hour", zone]].copy()
        sub.columns = ["date", "hour", "demand_mw"]
        frames.append(sub)

    combined = pd.concat(frames, ignore_index=True)
    raw_rows = len(combined)

    # Parse and coerce types
    combined["date"] = pd.to_datetime(combined["date"], format="%Y-%m-%d", errors="coerce")
    combined["hour"] = pd.to_numeric(combined["hour"], errors="coerce")
    combined["demand_mw"] = pd.to_numeric(combined["demand_mw"], errors="coerce")

    invalid_date_hour = int(combined["date"].isna().sum() + combined["hour"].isna().sum())
    if invalid_date_hour > 0:
        combined = combined.dropna(subset=["date", "hour"])

    combined["hour"] = combined["hour"].astype(int)

    # Remove duplicate (date, hour) rows keeping first occurrence
    dup_count = int(combined.duplicated(subset=["date", "hour"]).sum())
    if dup_count > 0:
        combined = combined.drop_duplicates(subset=["date", "hour"], keep="first")

    # Sort strictly chronologically
    combined = combined.sort_values(["date", "hour"]).reset_index(drop=True)

    # Inspect missing or invalid (<= 0) demand readings and linearly interpolate if any exist
    invalid_demand_mask = combined["demand_mw"].isna() | (combined["demand_mw"] <= 0)
    invalid_demand_count = int(invalid_demand_mask.sum())
    if invalid_demand_count > 0:
        logging.warning(
            "Found %d missing or non-positive demand readings; interpolating linearly.",
            invalid_demand_count,
        )
        combined.loc[invalid_demand_mask, "demand_mw"] = np.nan
        combined["demand_mw"] = combined["demand_mw"].interpolate(method="linear").bfill().ffill()

    combined["year"] = combined["date"].dt.year
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    report = {
        "zone": zone,
        "raw_rows": raw_rows,
        "cleaned_rows": len(combined),
        "dropped_invalid_date_hour": invalid_date_hour,
        "dropped_duplicates": dup_count,
        "interpolated_invalid_demand": invalid_demand_count,
        "hour_min": int(combined["hour"].min()),
        "hour_max": int(combined["hour"].max()),
        "demand_min_mw": float(combined["demand_mw"].min()),
        "demand_max_mw": float(combined["demand_mw"].max()),
        "demand_mean_mw": float(combined["demand_mw"].mean()),
        "demand_std_mw": float(combined["demand_mw"].std()),
    }
    return combined, report


def build_splits(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": df[df["year"].isin([2022, 2023])].reset_index(drop=True),
        "val": df[df["year"] == 2024].reset_index(drop=True),
        "test": df[df["year"] == 2025].reset_index(drop=True),
    }

    # Verify strict chronological separation (zero overlap / zero leakage)
    assert splits["train"]["date"].max() < splits["val"]["date"].min(), "Train/Val date overlap!"
    assert splits["val"]["date"].max() < splits["test"]["date"].min(), "Val/Test date overlap!"

    all_path = out_dir / "toronto_demand_all.csv"
    df.to_csv(all_path, index=False)

    split_info: dict[str, Any] = {}
    for name, sdf in splits.items():
        path = out_dir / f"{name}.csv"
        sdf.to_csv(path, index=False)
        split_info[name] = {
            "path": str(path),
            "rows": len(sdf),
            "start_date": str(sdf["date"].iloc[0]),
            "end_date": str(sdf["date"].iloc[-1]),
            "demand_mean_mw": float(sdf["demand_mw"].mean()),
            "demand_std_mw": float(sdf["demand_mw"].std()),
            "demand_min_mw": float(sdf["demand_mw"].min()),
            "demand_max_mw": float(sdf["demand_mw"].max()),
            "sha256": sha256_file(path),
        }
        logging.info(
            "Saved %s split: %d rows (%s to %s), mean=%.1f MW, std=%.1f MW",
            name,
            len(sdf),
            split_info[name]["start_date"],
            split_info[name]["end_date"],
            split_info[name]["demand_mean_mw"],
            split_info[name]["demand_std_mw"],
        )
    return split_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IESO Toronto hourly demand dataset (2022-2025).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data directory.")
    parser.add_argument("--zone", type=str, default="Toronto", help="IESO demand zone column.")
    parser.add_argument("--force-download", action="store_true", help="Re-download raw IESO CSV files.")
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    processed_dir = args.data_dir / "processed"
    set_logger(processed_dir / "data_prep.log")

    raw_paths = [download_ieso_year(y, raw_dir, force=args.force_download) for y in DEFAULT_YEARS]
    cleaned_df, clean_report = load_and_clean_ieso_files(raw_paths, zone=args.zone)
    split_info = build_splits(cleaned_df, processed_dir)

    summary = {
        "source": IESO_BASE_URL,
        "years": list(DEFAULT_YEARS),
        "cleaning": clean_report,
        "splits": split_info,
    }
    save_dict_to_json(summary, processed_dir / "dataset_summary.json")
    logging.info("Data preparation complete. Total rows: %d", len(cleaned_df))


if __name__ == "__main__":
    main()
