"""Evaluate a trained DemandLSTM checkpoint on a clean split (val or test)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import DemandWindowDataset, FeatureScaler
from src.model import DemandLSTM
from src.train import evaluate_forecast
from src.utils import Params, get_device, load_checkpoint, save_dict_to_json, set_logger, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DemandLSTM checkpoint.")
    parser.add_argument("--model-dir", type=Path, default=Path("experiments/baseline_lstm"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--restore-file", type=str, default="best.pth.tar")
    args = parser.parse_args()

    set_logger(args.model_dir / f"evaluate_{args.split}.log")
    params = Params(args.model_dir / "params.json")
    scaler = FeatureScaler.load(args.model_dir / "scaler.json")
    device = get_device(str(params.device))

    split_csv = args.data_dir / f"{args.split}.csv"
    df = pd.read_csv(split_csv)
    ds = DemandWindowDataset(df, scaler, window_size=int(params.window_size))
    loader = DataLoader(ds, batch_size=int(params.batch_size), shuffle=False)

    model = DemandLSTM(
        input_size=int(params.input_size),
        hidden_size=int(params.hidden_size),
        num_layers=int(params.num_layers),
        output_size=int(params.output_size),
    ).to(device)
    load_checkpoint(args.model_dir / args.restore_file, model)

    metrics, preds_mw, targets_mw = evaluate_forecast(model, loader, nn.MSELoss(), scaler, device)
    metrics["split"] = args.split
    metrics["split_sha256"] = sha256_file(split_csv)
    save_dict_to_json(metrics, args.model_dir / f"metrics_{args.split}_eval.json")

    out_df = pd.DataFrame(
        {
            "date": ds.target_dates,
            "hour": ds.target_hours,
            "actual_mw": targets_mw,
            "predicted_mw": preds_mw,
            "residual_mw": targets_mw - preds_mw,
        }
    )
    out_df.to_csv(args.model_dir / f"predictions_{args.split}.csv", index=False)
    logging.info(
        "Split=%s | RMSE=%.2f MW | MAE=%.2f MW | MAPE=%.2f%%",
        args.split,
        metrics["rmse_mw"],
        metrics["mae_mw"],
        metrics["mape_pct"],
    )


if __name__ == "__main__":
    main()
