# Test Plan: Baseline LSTM with Injected Anomalies

**Course:** ARTE 4110U Deep Learning Architecture & Algorithms  
**Project:** Group 6 — Anomalous Demand (Toronto)  
**Team:** Horace Zeng (`100924967`), Sayed Essa Sadat (`100870693`), Muhammad Affan (`100806394`)  
**Repository:** `https://github.com/ARTE4110U-Fall2026/project-group-06`  
**Status:** **8 / 8 Pre-Registered Checks PASSED**  
**PDF Report:** [Test_Plan_Baseline_with_Injected_Anomalies.pdf](Test_Plan_Baseline_with_Injected_Anomalies.pdf)

---

## 1. Purpose & Experimental Scope

This test verifies how well our baseline demand-only LSTM detects artificial reporting errors in hourly electricity demand. All testing is carried out on the **2024 validation dataset**, keeping the uncorrupted 2024 data as our control benchmark. Per course guidelines, the **2025 dataset remains untouched** and is reserved for the final Milestone 3 evaluation.

---

## 2. System & Dataset Setup

- **Dataset:** IESO Hourly Zonal Demand (Toronto zone, 2022–2025). The series contains 35,063 clean hourly records with zero missing or duplicate values.
- **Splits:** Chronological partition — Train on 2022–2023 (17,520 hours); Validation on 2024 (8,784 hours, leap year); Test on 2025 (8,759 hours, held out).
- **Preprocessing:** Standard scaling fit exclusively on training data (mean = 5,710.6 MW, std = 945.9 MW). We use a 24-hour sliding window to forecast next-hour demand.
- **Architecture:** 1-layer LSTM (input size = 2: hour and demand; hidden units = 32; output size = 1) totaling **4,641 parameters** (matches Proposal Appendix A).
- **Training & Baseline Results:** MSE loss, Adam (`lr = 0.001`), 30 epochs, batch size 64 (17s CPU runtime). **2024 Control Validation:** RMSE = 63.11 MW, MAE = 46.26 MW, MAPE = 0.80%, residual std (`sigma_ctrl`) = 60.95 MW.
- **Anomaly Score & Threshold:** Standardized residual $z_t = |y_t - \hat{y}_t| / \sigma_{\text{ctrl}}$. Operating threshold **$\tau^* = 5.50$** (335.24 MW residual) was selected by sweeping $\tau \in [1.50, 6.00]$ to maximize F1 score while holding control false alarms to **0.07%** (6 hours all year).

---

## 3. Test Results Across Anomaly Cases ($\tau^* = 5.50$)

| Case | Description / Severity | Count | Recall | Precision | FPR | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Control** | Clean 2024 series (no injection) | 8,760 | — | — | **0.07%** (6 FPs) | — | **PASS** |
| **Case 1a** | Mild Spikes/Drops (±15%, 2% rate) | 175 | 100.0% | 96.7% | 0.07% | 98.3% | **PASS (diag)** |
| **Case 1b** | Moderate Spikes/Drops (±30%, 2% rate) | 175 | 100.0% | 96.7% | 0.07% | 98.3% | **PASS** |
| **Case 1c** | Severe Spikes/Drops (±50%, 2% rate) | 175 | 100.0% | 96.7% | 0.07% | 98.3% | **PASS** |
| **Case 2** | Sensor Dropout / 0.1x Scale (1% rate) | 88 | 100.0% | 93.6% | 0.07% | 96.7% | **PASS** |
| **Case 3** | Stuck Meter / Flatline (5h hold, 1% rate) | 85 | 7.1% | 16.7% | 0.35% | 9.9% | **DIAGNOSTIC** |
| **Case 4A** | Mixed Suite (Target-Only Injection) | 174 | 89.1% | 96.3% | 0.07% | 92.5% | **PASS** |
| **Case 4B** | Mixed Suite (Raw-Series Pre-Window) | 174 | 75.3% | 34.4% | 2.91% | 47.2% | **DIAGNOSTIC** |

---

## 4. Pre-Registered Pass Thresholds Verification

| Verification Criterion | Pre-Registered Target | Observed Outcome | Result |
| :--- | :--- | :--- | :--- |
| **Control Data Integrity** | Exact SHA-256 match; repeat delta < 1e-6 | SHA-256 matched; repeat delta = 0.00 | **PASS** |
| **Control False Positive Rate** | FPR ≤ 5.0% | 0.07% (6 FPs out of 8,760 hours) | **PASS** |
| **Moderate Spikes/Drops (Case 1b)** | Recall ≥ 75.0% | 100.0% (175 / 175 detected) | **PASS** |
| **Severe Spikes/Drops (Case 1c)** | Recall ≥ 90%, Precision ≥ 70% | Recall = 100.0%, Precision = 96.7% | **PASS** |
| **Sensor Dropouts (Case 2)** | Recall ≥ 95%, Precision ≥ 75% | Recall = 100.0%, Precision = 93.6% | **PASS** |
| **Severity Monotonicity** | Recall(1a) ≤ Recall(1b) ≤ Recall(1c) | 100.0% ≤ 100.0% ≤ 100.0% | **PASS** |
| **Multi-Seed Repeatability (42, 123, 456)** | Standard deviation ≤ 3.0 pp | Recall std = 0.27 pp, FPR std = 0.00 pp | **PASS** |
| **Evaluation Runtime** | Runtime < 300 seconds | 1.90 seconds | **PASS** |

---

## 5. Visual Evidence & Diagnostic Plots

- **Learning Curve:** `experiments/baseline_lstm/learning_curve.png`
- **Threshold Trade-Off:** `experiments/baseline_injected_anomalies/run_02_f1/threshold_tradeoff.png`
- **Case Comparison:** `experiments/baseline_injected_anomalies/run_02_f1/case_comparison.png`
- **Actual vs. Predicted Snapshot:** `experiments/baseline_injected_anomalies/run_02_f1/actual_vs_predicted_anomalies.png`

---

## 6. Key Findings & Discussion

1. **Threshold tuning is critical under class imbalance:** Because anomalies represent only 1% to 2% of total hours, tuning for raw recall under a 5% FPR cap resulted in 384 false alarms and reduced precision to 30%. Tuning for the F1 score selected $\tau^* = 5.50$, which kept recall at 100% for spikes and dropouts while cutting false alarms to just 6 across the entire year and lifting precision to 96.3%.
2. **Real-world storm outage identified in control data:** Of the 6 false alarms in the clean control data, the largest error ($z = 18.2$) occurred on **July 16, 2024 at 2:00 PM** (actual demand dropped to 6,429 MW vs 5,317 MW predicted). This coincides with the severe thunderstorm and flash flooding outage that hit Toronto that afternoon. Because the baseline model only sees hour and past demand, it cannot anticipate sudden weather shocks. This confirms our core hypothesis: weather and holiday context are essential to avoid false alarms during genuine grid events.
3. **Baseline failure modes:** Residual-based detection struggles with stuck sensors during low-variance periods (Case 3 recall was only 7.1%). In addition, when corrupted values feed back into subsequent 24-hour input windows (Case 4B), false alarms rise to 2.91% and precision falls to 34.4%, proving that online data cleaning/imputation is required before reusing past inputs.
4. **Milestone 3 refinement:** Because the baseline model is accurate (RMSE = 63.11 MW, ~1.1% of mean load), ±15% shifts are easily detected. For the final baseline vs. context-aware comparison, we will add finer-grained levels (±3%, ±5%, ±10%) to test which model best detects subtle reporting errors.
