"""Train the baseline demand-only LSTM on 2022-2023 IESO Toronto demand data.

Saves:
- best.pth.tar / last.pth.tar (gitignored checkpoints)
- scaler.json (train-only mean/std)
- history.csv (per-epoch train/val loss, val RMSE/MAE in MW)
- metrics_val.json (best epoch summary)
- params_resolved.json, environment.json, train.log
- learning_curve.png (train vs. val MSE + val RMSE in MW)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import DemandWindowDataset, FeatureScaler
from src.model import DemandLSTM
from src.utils import (
    Params,
    append_history,
    capture_environment,
    get_device,
    save_checkpoint,
    save_dict_to_json,
    seed_everything,
    seed_worker,
    set_logger,
    sha256_file,
)


def evaluate_forecast(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    scaler: FeatureScaler,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    count = 0
    preds_scaled: list[np.ndarray] = []
    targets_scaled: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            n = x.size(0)
            total_loss += float(loss.item()) * n
            count += n
            preds_scaled.append(out.cpu().numpy())
            targets_scaled.append(y.cpu().numpy())

    preds_sc = np.concatenate(preds_scaled)
    targets_sc = np.concatenate(targets_scaled)
    preds_mw = scaler.inverse_transform_target(preds_sc)
    targets_mw = scaler.inverse_transform_target(targets_sc)

    residuals_mw = targets_mw - preds_mw
    rmse_mw = float(np.sqrt(np.mean(residuals_mw**2)))
    mae_mw = float(np.mean(np.abs(residuals_mw)))
    mape_pct = float(np.mean(np.abs(residuals_mw) / np.maximum(np.abs(targets_mw), 1e-6)) * 100.0)

    metrics = {
        "mse_scaled": total_loss / max(count, 1),
        "rmse_mw": rmse_mw,
        "mae_mw": mae_mw,
        "mape_pct": mape_pct,
        "residual_mean_mw": float(np.mean(residuals_mw)),
        "residual_std_mw": float(np.std(residuals_mw)),
        "count": count,
    }
    return metrics, preds_mw, targets_mw


def plot_learning_curve(history_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(df["epoch"], df["train_mse"], label="Train MSE (scaled)", marker="o", markersize=3)
    axes[0].plot(df["epoch"], df["val_mse"], label="Val MSE (scaled)", marker="s", markersize=3)
    best_idx = int(df["val_mse"].idxmin())
    axes[0].axvline(df.loc[best_idx, "epoch"], color="gray", linestyle="--", alpha=0.7,
                    label=f"Best epoch ({int(df.loc[best_idx, 'epoch'])})")
    axes[0].set_title("Baseline LSTM Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (Scaled)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["epoch"], df["val_rmse_mw"], label="Val RMSE (MW)", color="tab:orange", marker="o", markersize=3)
    axes[1].plot(df["epoch"], df["val_mae_mw"], label="Val MAE (MW)", color="tab:green", marker="s", markersize=3)
    axes[1].axvline(df.loc[best_idx, "epoch"], color="gray", linestyle="--", alpha=0.7)
    axes[1].set_title("Validation Forecast Error on 2024 Control Split")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error (MW)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def train_model(model_dir: Path, data_dir: Path, overwrite: bool = False) -> dict[str, float]:
    params_path = model_dir / "params.json"
    if not params_path.is_file():
        raise FileNotFoundError(f"Missing params.json in {model_dir}")
    params = Params(params_path)

    if not overwrite and any((model_dir / f).exists() for f in ("history.csv", "best.pth.tar", "last.pth.tar")):
        raise FileExistsError(f"Run artifacts already exist in {model_dir}. Pass --overwrite to re-run.")
    if overwrite:
        for pattern in ("*.pth.tar", "history.csv", "metrics_val.json", "train.log",
                        "params_resolved.json", "environment.json", "scaler.json", "learning_curve.png"):
            for p in model_dir.glob(pattern):
                p.unlink()

    set_logger(model_dir / "train.log")
    seed_everything(int(params.seed))
    device = get_device(str(params.device))

    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Fit scaler strictly on 2022-2023 training split (zero leakage)
    scaler = FeatureScaler.fit(train_df, feature_cols=params.feature_cols, target_col=params.target_col)
    scaler.save(model_dir / "scaler.json")

    train_ds = DemandWindowDataset(train_df, scaler, window_size=int(params.window_size))
    val_ds = DemandWindowDataset(val_df, scaler, window_size=int(params.window_size))

    g = torch.Generator()
    g.manual_seed(int(params.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(params.batch_size),
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=int(params.batch_size), shuffle=False)

    model = DemandLSTM(
        input_size=int(params.input_size),
        hidden_size=int(params.hidden_size),
        num_layers=int(params.num_layers),
        output_size=int(params.output_size),
    ).to(device)

    param_count = model.count_parameters()
    logging.info("Initialized %s with %d trainable parameters on device=%s", params.model_name, param_count, device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params.learning_rate),
        weight_decay=float(getattr(params, "weight_decay", 0.0)),
    )

    resolved = dict(params.dict)
    resolved.update(
        {
            "param_count": param_count,
            "train_csv": str(train_csv.resolve()),
            "train_sha256": sha256_file(train_csv),
            "val_csv": str(val_csv.resolve()),
            "val_sha256": sha256_file(val_csv),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
        }
    )
    save_dict_to_json(resolved, model_dir / "params_resolved.json")
    save_dict_to_json(capture_environment(str(device)), model_dir / "environment.json")

    best_val_mse = float("inf")
    best_metrics: dict[str, float] = {}
    t0 = time.perf_counter()

    for epoch in range(1, int(params.max_epochs) + 1):
        model.train()
        total_train_loss = 0.0
        n_train = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            total_train_loss += float(loss.item()) * bs
            n_train += bs

        train_mse = total_train_loss / max(n_train, 1)
        val_metrics, _, _ = evaluate_forecast(model, val_loader, loss_fn, scaler, device)
        val_mse = val_metrics["mse_scaled"]
        is_best = val_mse < best_val_mse

        if is_best:
            best_val_mse = val_mse
            best_metrics = dict(val_metrics)
            best_metrics["best_epoch"] = epoch
            best_metrics["train_mse_at_best"] = train_mse

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
                "val_mse": val_mse,
                "param_count": param_count,
            },
            is_best=is_best,
            checkpoint_dir=model_dir,
        )

        row = {
            "epoch": epoch,
            "train_mse": round(train_mse, 6),
            "val_mse": round(val_mse, 6),
            "val_rmse_mw": round(val_metrics["rmse_mw"], 3),
            "val_mae_mw": round(val_metrics["mae_mw"], 3),
            "val_mape_pct": round(val_metrics["mape_pct"], 4),
            "is_best": int(is_best),
        }
        append_history(model_dir / "history.csv", row)
        logging.info(
            "Epoch %02d/%02d | train_mse=%.5f | val_mse=%.5f | val_rmse=%.2f MW | val_mae=%.2f MW%s",
            epoch,
            int(params.max_epochs),
            train_mse,
            val_mse,
            val_metrics["rmse_mw"],
            val_metrics["mae_mw"],
            " *" if is_best else "",
        )

    elapsed = time.perf_counter() - t0
    best_metrics["training_seconds"] = round(elapsed, 2)
    best_metrics["param_count"] = param_count
    save_dict_to_json(best_metrics, model_dir / "metrics_val.json")
    plot_learning_curve(model_dir / "history.csv", model_dir / "learning_curve.png")
    logging.info(
        "Training finished in %.2fs. Best epoch=%d | val_rmse=%.2f MW | val_mae=%.2f MW",
        elapsed,
        int(best_metrics["best_epoch"]),
        best_metrics["rmse_mw"],
        best_metrics["mae_mw"],
    )
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline DemandLSTM model.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("experiments/baseline_lstm"),
        help="Experiment directory containing params.json.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing train.csv and val.csv.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run artifacts.")
    args = parser.parse_args()
    train_model(args.model_dir, args.data_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
