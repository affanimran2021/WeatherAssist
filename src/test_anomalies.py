"""Execute Test Plan: Baseline with Injected Anomalies.

Runs the frozen baseline DemandLSTM against:
1. Unchanged 2024 validation Control split (with pre/post SHA-256 verification).
2. Controlled injected anomaly cases (Cases 1a, 1b, 1c, 2, 3, 4 Mode A & Mode B).
3. Threshold sweep on Case 4 (Mode A) to select standardized residual threshold tau*
   subject to Control FPR <= max_control_fpr.
4. Repeatability check (exact re-run with seed=42) and multi-seed stability check
   (seeds 42, 123, 456).
5. Automated evaluation against pre-registered pass thresholds + visualization export.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.anomaly_injection import InjectedCase, inject_anomalies
from src.dataset import DemandWindowDataset, FeatureScaler
from src.model import DemandLSTM
from src.utils import (
    Params,
    capture_environment,
    get_device,
    load_checkpoint,
    load_json,
    save_dict_to_json,
    seed_everything,
    set_logger,
    sha256_file,
)


def run_inference(
    model: DemandLSTM,
    input_df: pd.DataFrame,
    targets_mw: np.ndarray,
    scaler: FeatureScaler,
    window_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Run deterministic inference and return (preds_mw, residuals_mw, dates, hours)."""
    ds = DemandWindowDataset(
        input_df,
        scaler,
        window_size=window_size,
        target_override_mw=targets_mw,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    preds_scaled: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            preds_scaled.append(out.cpu().numpy())

    preds_mw = scaler.inverse_transform_target(np.concatenate(preds_scaled))
    residuals_mw = np.abs(targets_mw - preds_mw)
    return preds_mw, residuals_mw, ds.target_dates, ds.target_hours


def compute_detection_metrics(
    z_scores: np.ndarray, is_anomaly: np.ndarray, tau: float
) -> dict[str, Any]:
    flagged = z_scores > tau
    tp = int(np.sum(flagged & is_anomaly))
    fp = int(np.sum(flagged & (~is_anomaly)))
    fn = int(np.sum((~flagged) & is_anomaly))
    tn = int(np.sum((~flagged) & (~is_anomaly)))

    n_pos = tp + fn
    n_neg = fp + tn
    recall = float(tp / n_pos) if n_pos > 0 else None
    fpr = float(fp / n_neg) if n_neg > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (0.0 if n_pos > 0 else None)
    if recall is not None and precision is not None and (recall + precision) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0 if n_pos > 0 else None

    return {
        "tau": round(float(tau), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_anomalies": n_pos,
        "n_normal": n_neg,
        "detection_rate_recall": recall,
        "false_positive_rate": fpr,
        "precision": precision,
        "f1": f1,
    }


def sweep_threshold(
    control_z: np.ndarray,
    case4_z: np.ndarray,
    case4_labels: np.ndarray,
    min_tau: float,
    max_tau: float,
    step: float,
    max_control_fpr: float,
    selection_metric: str = "f1",
) -> tuple[float, pd.DataFrame]:
    taus = np.round(np.arange(min_tau, max_tau + 1e-9, step), 4)
    rows: list[dict[str, Any]] = []
    for tau in taus:
        ctrl_fpr = float(np.mean(control_z > tau))
        m = compute_detection_metrics(case4_z, case4_labels, float(tau))
        rows.append(
            {
                "tau": float(tau),
                "control_fpr": ctrl_fpr,
                "case4_recall": m["detection_rate_recall"],
                "case4_precision": m["precision"],
                "case4_fpr": m["false_positive_rate"],
                "case4_f1": m["f1"],
                "meets_fpr_cap": ctrl_fpr <= max_control_fpr,
            }
        )
    sweep_df = pd.DataFrame(rows)
    eligible = sweep_df[sweep_df["meets_fpr_cap"]]
    if eligible.empty:
        best_row = sweep_df.sort_values(["control_fpr", "case4_f1"], ascending=[True, False]).iloc[0]
    elif selection_metric == "recall":
        best_row = eligible.sort_values(
            ["case4_recall", "case4_f1", "control_fpr"], ascending=[False, False, True]
        ).iloc[0]
    elif selection_metric == "f1":
        # Maximize F1 (balances detection rate & precision) under FPR cap; prefer higher recall on ties
        best_row = eligible.sort_values(
            ["case4_f1", "case4_recall", "control_fpr"], ascending=[False, False, True]
        ).iloc[0]
    else:
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")
    return float(best_row["tau"]), sweep_df


def plot_threshold_tradeoff(sweep_df: pd.DataFrame, tau_star: float, max_fpr: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(sweep_df["tau"], sweep_df["case4_recall"] * 100, label="Anomaly Detection Rate / Recall (%)", linewidth=2)
    ax.plot(sweep_df["tau"], sweep_df["case4_precision"] * 100, label="Precision (%)", linewidth=2)
    ax.plot(sweep_df["tau"], sweep_df["control_fpr"] * 100, label="Control False Positive Rate (%)", linewidth=2, linestyle="--")
    ax.axvline(tau_star, color="crimson", linestyle=":", linewidth=1.8, label=f"Selected tau* = {tau_star:.2f}")
    ax.axhline(max_fpr * 100, color="gray", linestyle="-.", linewidth=1.1, alpha=0.7, label=f"FPR Cap ({max_fpr*100:.1f}%)")
    ax.set_title("Baseline LSTM Threshold Sweep on 2024 Validation (Case 4 Mixed Suite)")
    ax.set_xlabel("Standardized Residual Threshold (tau = |y - y_hat| / sigma_ctrl)")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_case_comparison(case_results: dict[str, dict[str, Any]], out_path: Path) -> None:
    ordered_keys = [k for k in ("case_1a", "case_1b", "case_1c", "case_2", "case_3", "case_4_mode_a", "case_4_mode_b") if k in case_results]
    labels = [
        "1a: Mild\n(±15%)",
        "1b: Mod\n(±30%)",
        "1c: Sev\n(±50%)",
        "2: Dropout\n(0 / 0.1x)",
        "3: Flatline\n(5h hold)",
        "4A: Mixed\n(Target)",
        "4B: Mixed\n(Raw Ser.)",
    ][: len(ordered_keys)]

    recalls = [case_results[k]["metrics"]["detection_rate_recall"] * 100 for k in ordered_keys]
    precs = [case_results[k]["metrics"]["precision"] * 100 for k in ordered_keys]
    fprs = [case_results[k]["metrics"]["false_positive_rate"] * 100 for k in ordered_keys]

    x = np.arange(len(ordered_keys))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    b1 = ax.bar(x - w, recalls, width=w, label="Detection Rate / Recall (%)")
    b2 = ax.bar(x, precs, width=w, label="Precision (%)")
    b3 = ax.bar(x + w, fprs, width=w, label="False Positive Rate (%)")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=7.5, xytext=(0, 2), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Baseline Demand-Only LSTM — Performance Across Injected Anomaly Cases (2024 Val)", pad=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", ncol=3, frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_actual_vs_predicted(
    preds_df: pd.DataFrame,
    out_path: Path,
    window_hours: int = 240,
) -> None:
    """Plot a 10-day window from Case 4 Mode A containing multiple injected anomalies."""
    anom_indices = np.flatnonzero(preds_df["is_anomaly"].to_numpy())
    if len(anom_indices) == 0:
        return
    # Pick a 240-hour slice with high anomaly density for clear qualitative visualization
    best_start = 0
    best_count = -1
    max_start = max(0, len(preds_df) - window_hours)
    for s in range(0, max_start + 1, 24):
        c = int(preds_df["is_anomaly"].iloc[s : s + window_hours].sum())
        if c > best_count:
            best_count = c
            best_start = s
    sub = preds_df.iloc[best_start : best_start + window_hours].reset_index(drop=True)
    xs = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.plot(xs, sub["observed_mw"], label="Observed (with Injected Anomalies)", color="tab:blue", linewidth=1.3)
    ax.plot(xs, sub["predicted_mw"], label="Baseline LSTM Prediction", color="tab:orange", linewidth=1.3, alpha=0.85)

    tp_mask = sub["is_anomaly"] & sub["flagged"]
    fn_mask = sub["is_anomaly"] & (~sub["flagged"])
    fp_mask = (~sub["is_anomaly"]) & sub["flagged"]

    if tp_mask.any():
        ax.scatter(xs[tp_mask], sub.loc[tp_mask, "observed_mw"], color="crimson", s=55, zorder=5, label="True Positive (Detected)")
    if fn_mask.any():
        ax.scatter(xs[fn_mask], sub.loc[fn_mask, "observed_mw"], color="purple", marker="x", s=65, zorder=5, label="False Negative (Missed)")
    if fp_mask.any():
        ax.scatter(xs[fp_mask], sub.loc[fp_mask, "observed_mw"], color="goldenrod", marker="^", s=45, zorder=4, label="False Positive")

    start_date = sub["date"].iloc[0]
    end_date = sub["date"].iloc[-1]
    ax.set_title(f"Baseline LSTM: Actual vs. Predicted Toronto Demand with Highlighted Anomalies ({start_date} to {end_date})")
    ax.set_xlabel("Hour Offset in 10-Day Validation Window")
    ax.set_ylabel("Electricity Demand (MW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def evaluate_case(
    model: DemandLSTM,
    control_df: pd.DataFrame,
    case: InjectedCase,
    scaler: FeatureScaler,
    window_size: int,
    batch_size: int,
    device: torch.device,
    sigma_ctrl: float,
    tau: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    input_df = control_df if case.mode == "target_only" else case.corrupted_df
    preds_mw, abs_res_mw, dates, hours = run_inference(
        model, input_df, case.targets_mw, scaler, window_size, batch_size, device
    )
    z_scores = abs_res_mw / sigma_ctrl
    metrics = compute_detection_metrics(z_scores, case.is_anomaly, tau)

    # Also break down recall by sub-type if mixed
    sub_breakdown: dict[str, dict[str, Any]] = {}
    if not case.manifest_df.empty:
        flagged = z_scores > tau
        for a_type, grp in case.manifest_df.groupby("anomaly_type"):
            idxs = grp["window_idx"].to_numpy(dtype=int)
            tp_sub = int(flagged[idxs].sum())
            n_sub = len(idxs)
            sub_breakdown[str(a_type)] = {
                "count": n_sub,
                "detected": tp_sub,
                "recall": round(float(tp_sub / n_sub), 4) if n_sub > 0 else 0.0,
            }

    preds_df = pd.DataFrame(
        {
            "date": dates,
            "hour": hours,
            "control_mw": control_df["demand_mw"].iloc[window_size:].to_numpy(dtype=np.float64),
            "observed_mw": case.targets_mw,
            "predicted_mw": preds_mw,
            "abs_residual_mw": abs_res_mw,
            "z_score": z_scores,
            "is_anomaly": case.is_anomaly,
            "flagged": z_scores > tau,
        }
    )
    return {
        "case_id": case.case_id,
        "label": case.label,
        "mode": case.mode,
        "seed": case.seed,
        "metrics": metrics,
        "breakdown_by_type": sub_breakdown,
    }, preds_df


def run_test_plan(
    model_dir: Path,
    data_dir: Path,
    config_path: Path,
    out_dir: Path,
    restore_file: str = "best.pth.tar",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_logger(out_dir / "test_anomalies.log")

    params = Params(model_dir / "params.json")
    cfg = load_json(config_path)
    scaler = FeatureScaler.load(model_dir / "scaler.json")
    device = get_device(str(params.device))

    split_name = str(cfg.get("split", "val"))
    control_csv = data_dir / f"{split_name}.csv"
    sha_before = sha256_file(control_csv)
    control_df = pd.read_csv(control_csv)
    window_size = int(params.window_size)
    batch_size = int(params.batch_size)
    primary_seed = int(cfg.get("primary_seed", 42))
    seed_everything(primary_seed)

    model = DemandLSTM(
        input_size=int(params.input_size),
        hidden_size=int(params.hidden_size),
        num_layers=int(params.num_layers),
        output_size=int(params.output_size),
    ).to(device)
    ckpt_path = model_dir / restore_file
    ckpt_state = load_checkpoint(ckpt_path, model)
    ckpt_sha = sha256_file(ckpt_path)
    logging.info("Loaded frozen baseline checkpoint %s (epoch=%s, sha256=%s...)", ckpt_path, ckpt_state.get("epoch"), ckpt_sha[:12])

    # 1. Control run (unchanged 2024 validation split)
    control_targets = control_df["demand_mw"].iloc[window_size:].to_numpy(dtype=np.float64)
    ctrl_preds_mw, ctrl_abs_res, ctrl_dates, ctrl_hours = run_inference(
        model, control_df, control_targets, scaler, window_size, batch_size, device
    )
    signed_res = control_targets - ctrl_preds_mw
    sigma_ctrl = float(np.std(signed_res))
    ctrl_rmse = float(np.sqrt(np.mean(signed_res**2)))
    ctrl_mae = float(np.mean(ctrl_abs_res))
    ctrl_z = ctrl_abs_res / sigma_ctrl
    logging.info("Control (%s) | RMSE=%.2f MW | MAE=%.2f MW | sigma_ctrl=%.2f MW", split_name, ctrl_rmse, ctrl_mae, sigma_ctrl)

    # 2. Generate Case 4 (Mode A) for threshold sweep
    case4_cfg = cfg["cases"]["case_4_mode_a"]
    case4_primary = inject_anomalies(control_df, "case_4_mode_a", case4_cfg, window_size=window_size, seed=primary_seed)
    c4_preds_mw, c4_abs_res, _, _ = run_inference(
        model, control_df, case4_primary.targets_mw, scaler, window_size, batch_size, device
    )
    c4_z = c4_abs_res / sigma_ctrl

    sweep_cfg = cfg["threshold_sweep"]
    selection_metric = str(sweep_cfg.get("selection_metric", "f1"))
    tau_star, sweep_df = sweep_threshold(
        ctrl_z,
        c4_z,
        case4_primary.is_anomaly,
        min_tau=float(sweep_cfg["min_tau"]),
        max_tau=float(sweep_cfg["max_tau"]),
        step=float(sweep_cfg["step"]),
        max_control_fpr=float(sweep_cfg["max_control_fpr"]),
        selection_metric=selection_metric,
    )
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)
    plot_threshold_tradeoff(sweep_df, tau_star, float(sweep_cfg["max_control_fpr"]), out_dir / "threshold_tradeoff.png")
    logging.info("Selected operating threshold tau* = %.2f (residual threshold = %.2f MW)", tau_star, tau_star * sigma_ctrl)

    # Control metrics at tau*
    control_metrics = compute_detection_metrics(ctrl_z, np.zeros(len(ctrl_z), dtype=bool), tau_star)
    control_metrics.update(
        {
            "rmse_mw": round(ctrl_rmse, 3),
            "mae_mw": round(ctrl_mae, 3),
            "sigma_ctrl_mw": round(sigma_ctrl, 3),
            "threshold_mw": round(tau_star * sigma_ctrl, 3),
        }
    )

    # Save control predictions
    ctrl_dir = out_dir / "control"
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": ctrl_dates,
            "hour": ctrl_hours,
            "control_mw": control_targets,
            "predicted_mw": ctrl_preds_mw,
            "abs_residual_mw": ctrl_abs_res,
            "z_score": ctrl_z,
            "flagged": ctrl_z > tau_star,
        }
    ).to_csv(ctrl_dir / "predictions.csv", index=False)

    # 3. Evaluate all injected cases at tau*
    case_results: dict[str, dict[str, Any]] = {}
    case4_preds_df: pd.DataFrame | None = None
    for case_id, c_cfg in cfg["cases"].items():
        injected = inject_anomalies(control_df, case_id, c_cfg, window_size=window_size, seed=primary_seed)
        res, preds_df = evaluate_case(
            model, control_df, injected, scaler, window_size, batch_size, device, sigma_ctrl, tau_star
        )
        case_results[case_id] = res
        c_dir = out_dir / case_id
        c_dir.mkdir(parents=True, exist_ok=True)
        injected.manifest_df.to_csv(c_dir / "injection_manifest.csv", index=False)
        preds_df.to_csv(c_dir / "predictions.csv", index=False)
        save_dict_to_json(res, c_dir / "metrics.json")
        if case_id == "case_4_mode_a":
            case4_preds_df = preds_df
        m = res["metrics"]
        logging.info(
            "%-14s | n_anom=%3d | Recall=%6.2f%% | Precision=%6.2f%% | FPR=%5.2f%% | F1=%6.2f%%",
            case_id,
            m["n_anomalies"],
            m["detection_rate_recall"] * 100,
            m["precision"] * 100,
            m["false_positive_rate"] * 100,
            m["f1"] * 100,
        )

    # 4. Exact repeatability check (re-run Control and Case 4 Mode A with primary_seed)
    _, ctrl_abs_res_repeat, _, _ = run_inference(
        model, control_df, control_targets, scaler, window_size, batch_size, device
    )
    ctrl_repeat_delta = float(np.max(np.abs(ctrl_abs_res - ctrl_abs_res_repeat)))

    case4_repeat = inject_anomalies(control_df, "case_4_mode_a", case4_cfg, window_size=window_size, seed=primary_seed)
    res_repeat, _ = evaluate_case(
        model, control_df, case4_repeat, scaler, window_size, batch_size, device, sigma_ctrl, tau_star
    )
    case4_repeat_delta = max(
        abs(res_repeat["metrics"]["detection_rate_recall"] - case_results["case_4_mode_a"]["metrics"]["detection_rate_recall"]),
        abs(res_repeat["metrics"]["false_positive_rate"] - case_results["case_4_mode_a"]["metrics"]["false_positive_rate"]),
        abs(res_repeat["metrics"]["precision"] - case_results["case_4_mode_a"]["metrics"]["precision"]),
    )
    max_repeat_delta = max(ctrl_repeat_delta, case4_repeat_delta)

    # 5. Multi-seed stability check on Case 4 Mode A
    stability_seeds = [int(s) for s in cfg.get("stability_seeds", [42, 123, 456])]
    seed_runs: list[dict[str, Any]] = []
    for s in stability_seeds:
        inj_s = inject_anomalies(control_df, "case_4_mode_a", case4_cfg, window_size=window_size, seed=s)
        res_s, _ = evaluate_case(
            model, control_df, inj_s, scaler, window_size, batch_size, device, sigma_ctrl, tau_star
        )
        seed_runs.append(
            {
                "seed": s,
                "recall": res_s["metrics"]["detection_rate_recall"],
                "precision": res_s["metrics"]["precision"],
                "fpr": res_s["metrics"]["false_positive_rate"],
                "f1": res_s["metrics"]["f1"],
            }
        )
    recalls_s = [r["recall"] for r in seed_runs]
    fprs_s = [r["fpr"] for r in seed_runs]
    precs_s = [r["precision"] for r in seed_runs]
    stability_summary = {
        "seeds": stability_seeds,
        "per_seed": seed_runs,
        "recall_mean": float(np.mean(recalls_s)),
        "recall_std": float(np.std(recalls_s)),
        "precision_mean": float(np.mean(precs_s)),
        "precision_std": float(np.std(precs_s)),
        "fpr_mean": float(np.mean(fprs_s)),
        "fpr_std": float(np.std(fprs_s)),
    }

    # 6. Verify control CSV unchanged via SHA-256
    sha_after = sha256_file(control_csv)
    sha_match = sha_before == sha_after
    (out_dir / "control_sha256.txt").write_text(
        f"before: {sha_before}\nafter:  {sha_after}\nmatch:  {sha_match}\n",
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t0

    # 7. Evaluate pre-registered pass thresholds
    th = cfg["pass_thresholds"]
    r1a = case_results["case_1a"]["metrics"]["detection_rate_recall"]
    r1b = case_results["case_1b"]["metrics"]["detection_rate_recall"]
    r1c = case_results["case_1c"]["metrics"]["detection_rate_recall"]
    p1c = case_results["case_1c"]["metrics"]["precision"]
    r2 = case_results["case_2"]["metrics"]["detection_rate_recall"]
    p2 = case_results["case_2"]["metrics"]["precision"]

    checks = {
        "control_integrity": {
            "passed": bool(sha_match and max_repeat_delta < float(th["repeat_max_delta"])),
            "observed": f"SHA-256 match={sha_match}, max_repeat_delta={max_repeat_delta:.2e}",
            "threshold": f"Exact SHA-256 match; delta < {th['repeat_max_delta']}",
        },
        "control_fpr": {
            "passed": bool(control_metrics["false_positive_rate"] <= float(th["control_max_fpr"])),
            "observed": f"{control_metrics['false_positive_rate']*100:.2f}%",
            "threshold": f"<= {float(th['control_max_fpr'])*100:.1f}%",
        },
        "case_1b_moderate_recall": {
            "passed": bool(r1b >= float(th["case_1b_min_recall"])),
            "observed": f"{r1b*100:.2f}%",
            "threshold": f">= {float(th['case_1b_min_recall'])*100:.1f}%",
        },
        "case_1c_severe_recall_precision": {
            "passed": bool(r1c >= float(th["case_1c_min_recall"]) and p1c >= float(th["case_1c_min_precision"])),
            "observed": f"Recall={r1c*100:.2f}%, Precision={p1c*100:.2f}%",
            "threshold": f"Recall >= {float(th['case_1c_min_recall'])*100:.1f}%, Precision >= {float(th['case_1c_min_precision'])*100:.1f}%",
        },
        "case_2_dropout_recall_precision": {
            "passed": bool(r2 >= float(th["case_2_min_recall"]) and p2 >= float(th["case_2_min_precision"])),
            "observed": f"Recall={r2*100:.2f}%, Precision={p2*100:.2f}%",
            "threshold": f"Recall >= {float(th['case_2_min_recall'])*100:.1f}%, Precision >= {float(th['case_2_min_precision'])*100:.1f}%",
        },
        "severity_monotonicity": {
            "passed": bool(r1a <= r1b <= r1c),
            "observed": f"Mild {r1a*100:.2f}% <= Mod {r1b*100:.2f}% <= Sev {r1c*100:.2f}%",
            "threshold": "Recall(1a) <= Recall(1b) <= Recall(1c)",
        },
        "multi_seed_stability": {
            "passed": bool(
                stability_summary["recall_std"] <= float(th["multi_seed_max_std"])
                and stability_summary["fpr_std"] <= float(th["multi_seed_max_std"])
            ),
            "observed": f"Recall std={stability_summary['recall_std']*100:.2f}pp, FPR std={stability_summary['fpr_std']*100:.2f}pp",
            "threshold": f"<= {float(th['multi_seed_max_std'])*100:.1f} percentage points",
        },
        "runtime": {
            "passed": bool(elapsed < float(th["max_runtime_seconds"])),
            "observed": f"{elapsed:.2f}s",
            "threshold": f"< {th['max_runtime_seconds']:.0f}s",
        },
    }

    # 8. Plots
    plot_case_comparison(case_results, out_dir / "case_comparison.png")
    if case4_preds_df is not None:
        plot_actual_vs_predicted(case4_preds_df, out_dir / "actual_vs_predicted_anomalies.png")

    summary = {
        "model_dir": str(model_dir),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": ckpt_state.get("epoch"),
        "param_count": model.count_parameters(),
        "split": split_name,
        "control_sha256_before": sha_before,
        "control_sha256_after": sha_after,
        "selected_tau": tau_star,
        "threshold_mw": round(tau_star * sigma_ctrl, 3),
        "runtime_seconds": round(elapsed, 2),
        "control": control_metrics,
        "cases": case_results,
        "repeatability": {
            "ctrl_max_abs_delta_mw": ctrl_repeat_delta,
            "case4_max_metric_delta": case4_repeat_delta,
        },
        "multi_seed_stability": stability_summary,
        "pass_checks": checks,
        "all_required_passed": all(v["passed"] for v in checks.values()),
    }
    save_dict_to_json(summary, out_dir / "results_summary.json")
    save_dict_to_json(cfg, out_dir / "params.json")
    save_dict_to_json(capture_environment(str(device)), out_dir / "environment.json")

    logging.info("Test plan completed in %.2fs | all_required_passed=%s", elapsed, summary["all_required_passed"])
    for k, v in checks.items():
        logging.info("  [%s] %-32s | observed=%s | threshold=%s", "PASS" if v["passed"] else "FAIL", k, v["observed"], v["threshold"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Test Plan: Baseline with Injected Anomalies.")
    parser.add_argument("--model-dir", type=Path, default=Path("experiments/baseline_lstm"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("experiments/baseline_injected_anomalies/test_config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/baseline_injected_anomalies/run_01"))
    parser.add_argument("--restore-file", type=str, default="best.pth.tar")
    args = parser.parse_args()
    run_test_plan(args.model_dir, args.data_dir, args.config, args.out_dir, restore_file=args.restore_file)


if __name__ == "__main__":
    main()
