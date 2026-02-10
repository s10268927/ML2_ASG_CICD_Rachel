"""
test_model.py — Automated Quality Gate for Bike Sharing Demand Model
=====================================================================
This script validates the saved model against minimum performance
standards before it can be accepted into the deployment pipeline.

It loads the best model (model.joblib) from Task 1, runs predictions
on the 2011 evaluation data, and checks three acceptance criteria:
  1. RMSE must be at most 95% of the baseline (Linear Regression)
  2. MAE must be at most 95% of the baseline
  3. R-squared must exceed a minimum explanatory threshold

If any criterion fails, the script exits with code 1, which causes
the GitHub Actions workflow to report a red (failed) status.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Acceptance criteria ──────────────────────────────────────────────
# The baselines come from the Linear Regression model (Task 1) evaluated
# on the full 2011 dataset — the same data this quality gate uses.
# Per the assignment: assert rmse <= 0.95 * rmse_baseline.
RMSE_BASELINE      = 678.76        # LR baseline RMSE on full 2011 data
MAE_BASELINE       = 514.54        # LR baseline MAE  on full 2011 data
R2_BASELINE        = 0.7570        # LR baseline R2   on full 2011 data
IMPROVEMENT_FACTOR = 0.95          # model must achieve <= 95 % of baseline


def load_artefacts():
    """Return the saved model and the evaluation dataframe."""
    root = os.path.join(os.path.dirname(__file__), "..")
    model = joblib.load(os.path.join(root, "model.joblib"))
    df    = pd.read_csv(os.path.join(root, "data", "day_2011.csv"))
    return model, df


def evaluate(model, df):
    """Predict on the evaluation set and return RMSE, MAE, R2."""
    features = ["season", "mnth", "holiday", "weekday", "workingday",
                "weathersit", "temp", "atemp", "hum", "windspeed"]
    X = df[features]
    y = df["cnt"]
    y_hat = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae  = mean_absolute_error(y, y_hat)
    r2   = r2_score(y, y_hat)
    return rmse, mae, r2


def check_gate(rmse, mae, r2):
    """
    Run each acceptance check and return a list of (name, passed, detail)
    tuples so the log is easy to read.
    """
    rmse_limit = IMPROVEMENT_FACTOR * RMSE_BASELINE
    mae_limit  = IMPROVEMENT_FACTOR * MAE_BASELINE

    checks = [
        ("RMSE", rmse <= rmse_limit,
         f"{rmse:>10.2f}  (threshold: <= {rmse_limit:.2f})"),
        ("MAE",  mae  <= mae_limit,
         f"{mae:>10.2f}  (threshold: <= {mae_limit:.2f})"),
        ("R2",   r2   >= R2_BASELINE,
         f"{r2:>10.4f}  (threshold: >= {R2_BASELINE})"),
    ]
    return checks


def main():
    print("=" * 58)
    print("  AUTOMATED QUALITY GATE — Bike Sharing Demand Model")
    print("=" * 58)

    # Step 1 — load
    model, df = load_artefacts()
    print(f"\n  Model loaded   : model.joblib")
    print(f"  Eval dataset   : data/day_2011.csv  ({len(df)} rows)")
    print(f"  Baseline (LR on full 2011): RMSE={RMSE_BASELINE}, MAE={MAE_BASELINE}, R2={R2_BASELINE}")
    print(f"  RMSE threshold : {IMPROVEMENT_FACTOR} x {RMSE_BASELINE} = {IMPROVEMENT_FACTOR * RMSE_BASELINE:.2f}")
    print(f"  MAE  threshold : {IMPROVEMENT_FACTOR} x {MAE_BASELINE} = {IMPROVEMENT_FACTOR * MAE_BASELINE:.2f}")
    print(f"  R2   threshold : >= {R2_BASELINE}\n")

    # Step 2 — evaluate
    rmse, mae, r2 = evaluate(model, df)

    print("-" * 58)
    print("  Metric        Value          Acceptance Criterion")
    print("-" * 58)

    checks = check_gate(rmse, mae, r2)
    failures = 0
    for name, passed, detail in checks:
        status = "OK" if passed else "FAIL"
        print(f"  {name:<6}  {detail}   [{status}]")
        if not passed:
            failures += 1

    print("-" * 58)

    # Step 3 — verdict
    if failures == 0:
        print("\n  >> Quality Gate PASSED — model is ready for deployment.\n")
    else:
        print(f"\n  >> Quality Gate FAILED — {failures} check(s) did not meet threshold.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
