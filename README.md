# Detecting Anomalous Values in Electricity Demand Reporting for City of Toronto

**ARTE 4110U: Deep Learning Architecture and Algorithms (Fall 2026) — Group 6 (Anomalous Demand)**  
**Team:** Horace Zeng (`100924967`), Sayed Essa Sadat (`100870693`), Muhammad Affan (`100806394`)

This repository implements a reproducible PyTorch pipeline for detecting anomalous reporting values in hourly electricity demand for the City of Toronto (IESO Toronto zone, 2022–2025). The demand-only baseline LSTM (`input_size=2`, `hidden_size=32`, `1` layer, `4,641` parameters) predicts next-hour demand from a 24-hour sliding window and flags reporting anomalies via standardized forecast residuals.

---

## Project Structure

```text
├── src/
│   ├── data_prep.py          # Downloads & cleans IESO 2022–2025 Toronto data; saves chronological splits
│   ├── dataset.py            # Leak-free train-only FeatureScaler & 24h sliding-window Dataset
│   ├── model.py              # DemandLSTM (4,641 params baseline / 4,897 params context-aware)
│   ├── train.py              # Training loop, per-epoch validation, checkpointing & learning curve
│   ├── evaluate.py           # Clean split evaluation (2024 val or 2025 final test)
│   ├── anomaly_injection.py  # Deterministic anomaly injector (spikes/drops, dropouts, flatlines, mixed)
│   ├── test_anomalies.py     # Full Test Plan runner (control SHA-256, threshold sweep, metrics & plots)
│   └── utils.py              # Params, seeding, logging, environment & SHA-256 utilities
├── experiments/
│   ├── baseline_lstm/        # Baseline params.json, scaler.json, history.csv, logs & learning curve
│   └── baseline_injected_anomalies/  # Pre-registered test_config.json & run artifacts
├── tests/
│   └── test_pipeline.py      # Unit tests for param counts, split alignment & deterministic injection
├── CONTRIBUTIONS.md          # Team contribution log per milestone
└── requirements.txt          # Python dependencies
```

---

## Quickstart & Reproducibility

Requires **Python 3.10–3.12**. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Obtain & Prepare Data (IESO 2022–2025 Toronto Zone)
Datasets are intentionally excluded from Git (see `.gitignore`). Run the script below to fetch the public IESO Hourly Zonal Demand files (`2022`–`2025`) and generate strict chronological splits (`2022–2023` train, `2024` validation, `2025` held-out test):

```bash
python -m src.data_prep
```

### 2. Run Unit & Integrity Tests
Verifies parameter counts (`4,641` / `4,897`), leak-free train-only scaling, control immutability, and deterministic seeding:

```bash
pytest -v
```

### 3. Train Baseline Demand-Only LSTM
Trains for 30 epochs on `2022–2023` and validates on `2024` after every epoch, writing `history.csv`, `metrics_val.json`, `scaler.json`, `learning_curve.png`, and `best.pth.tar`:

```bash
python -m src.train --model-dir experiments/baseline_lstm --data-dir data/processed --overwrite
```

### 4. Run Test Plan: Baseline with Injected Anomalies (2024 Validation Split)
Runs the unchanged 2024 control check (with pre/post SHA-256 verification), sweeps standardized residual threshold `tau` on the mixed suite under a `<= 5%` control FPR cap, evaluates all anomaly cases (`1a`, `1b`, `1c`, `2`, `3`, `4A`, `4B`), verifies exact repeatability and multi-seed stability, and saves all logs, manifests, metrics, and figures:

```bash
python -m src.test_anomalies \
  --model-dir experiments/baseline_lstm \
  --data-dir data/processed \
  --config experiments/baseline_injected_anomalies/test_config.json \
  --out-dir experiments/baseline_injected_anomalies/run_01
```

### 5. Clean Split Evaluation (Demo / Milestone 3 Final Test)
Reserve `--split test` (2025) for the single final evaluation in Milestone 3:

```bash
python -m src.evaluate --model-dir experiments/baseline_lstm --split val
```

---

## Reproducibility Checklist
- [x] Random seeds set and logged (`seed = 42`; multi-seed stability checked on `42, 123, 456`).
- [x] Scaler fit strictly on `2022–2023` training data only; zero cross-split window leakage.
- [x] `2025` test split untouched during training and threshold tuning.
- [x] Every reported number traces to an experiment folder with `params.json`, `environment.json`, `history.csv` / `results_summary.json`, and log files.
- [x] Raw datasets and model weight checkpoints (`*.pth.tar`) kept out of Git.

---

## References & Attribution
- **Course starter code:** Utility patterns (`Params`, seeding, checkpointing, experiment folder layout) adapted from the ARTE 4110U Fall 2026 PyTorch starter examples (`pytorch/`).
- **Data:** Independent Electricity System Operator (IESO), "Hourly Zonal Demand Report," 2022–2025. Available: https://reports-public.ieso.ca/public/DemandZonal/ (IESO Terms of Use; attribution required).
- **Related Work:**
  - E. Altamimi, A. Al-Ali, A. K. Al-Ali, and Q. M. Malluhi, "Distribution-Based Residual Analysis of Load Forecasting for Enhanced Anomaly Detection," *International Journal of Electrical Power & Energy Systems*, vol. 173, p. 111435, 2025.
  - M. S. Piscitelli, G. Buscemi, C. Roselli, E. Marrasso, G. Pallotta, and A. Capozzoli, "A Novel Data-Analytics Based Process for Load Profiling and Meter-Level Anomaly Detection in Building Energy Consumption Time Series," *Energy and Buildings*, vol. 368, p. 117829, 2026.
