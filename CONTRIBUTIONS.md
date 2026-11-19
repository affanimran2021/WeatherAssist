# Team Contributions — ARTE 4110U Group 6 (Anomalous Demand)

> **Note for Milestone 2 & 3:** Update each member's completed items and ensure commits in Git history come from each member's own GitHub account (per ARTE 4110U Final Project Guidelines §6.4).

## Milestone 1: Proposal & Project Plan (Nov 2, 2026)
- **Horace Zeng (`100924967`):** Problem framing, baseline architecture design (Appendix A), proposal drafting.
- **Sayed Essa Sadat (`100870693`):** Weather/holiday data integration plan, context-aware model specification, proposal drafting.
- **Muhammad Affan (`100806394`):** Evaluation plan, anomaly detection metrics & test plan design, Gantt chart & proposal drafting.

## Milestone 2: Technical Checkpoint (Due Nov 20, 2026)
- **Muhammad Affan (`100806394`):**
  - End-to-end IESO Toronto demand data preparation & chronological splitting (`src/data_prep.py`, `src/dataset.py`).
  - Baseline demand-only LSTM (`4,641` params) implementation, training pipeline, learning curve & validation evaluation (`src/model.py`, `src/train.py`, `src/evaluate.py`).
  - Controlled anomaly-injection suite & test plan harness (`src/anomaly_injection.py`, `src/test_anomalies.py`, `tests/test_pipeline.py`, `experiments/baseline_injected_anomalies/`).
- **Horace Zeng (`100924967`):** *(Add Milestone 2 commits / baseline review / checkpoint report sections)*
- **Sayed Essa Sadat (`100870693`):** *(Add Milestone 2 commits / Toronto weather + holiday preprocessing / context LSTM progress)*

## Milestone 3: Final Submission & Demo (Due Dec 4, 2026)
- **Horace Zeng (`100924967`):** *(Pending — Context model anomaly testing, final analysis, presentation)*
- **Sayed Essa Sadat (`100870693`):** *(Pending — Context-aware LSTM training & tuning, final analysis, presentation)*
- **Muhammad Affan (`100806394`):** *(Pending — Baseline vs. Context controlled comparison on 2025 held-out test set, demo workflow, presentation)*

## External Tools & Code Attribution (Guidelines §7.2 & §7.3)
- **Starter repository structure & utility patterns:** Adapted from ARTE 4110U Fall 2026 course starter code (`pytorch/nlp/utils.py`).
- **Generative AI disclosure:** Antigravity (Gemini) was used for test plan structuring, code scaffolding, and debugging assistance; all experiments were executed locally on real IESO 2022–2025 data and verified via automated tests and experiment logs.
