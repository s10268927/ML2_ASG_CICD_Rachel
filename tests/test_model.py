"""
test_model.py — Automated Quality Gate for Bike Sharing Demand Model
=====================================================================
This script is executed by the GitHub Actions CI/CD pipeline to validate
the saved model against minimum performance standards before it can be
accepted into the deployment pipeline.

Workflow:
  1. Load the best model (model.joblib) saved from Task 1.
  2. Load the evaluation dataset (data/day_2011.csv).
  3. Generate predictions and compute RMSE, MAE, and R².
  4. Compare each metric against acceptance thresholds derived from
     the Linear Regression baseline (Task 1).
  5. Exit with code 0 (pass) or code 1 (fail) to signal the result
     to GitHub Actions.

Acceptance Criteria:
  - RMSE must be <= 95% of the Linear Regression baseline RMSE.
  - MAE  must be <= 95% of the Linear Regression baseline MAE.
  - R²   must be >= the Linear Regression baseline R².
"""

# ── Imports ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")          # suppress scikit-learn version warnings

import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Acceptance Thresholds ────────────────────────────────────────────
# These baselines are the actual metrics of the Linear Regression model
# (Task 1) evaluated on the full 2011 dataset — the same dataset this
# quality gate evaluates on, ensuring a fair apples-to-apples comparison.
#
# Formula (per assignment): assert rmse <= 0.95 * rmse_baseline
#   RMSE threshold = 0.95 x 678.76 = 644.82
#   MAE  threshold = 0.95 x 514.54 = 488.81
#   R²   threshold = >= 0.757

RMSE_BASELINE      = 678.76        # Linear Regression RMSE on full 2011 data
MAE_BASELINE       = 514.54        # Linear Regression MAE  on full 2011 data
R2_BASELINE        = 0.7570        # Linear Regression R²   on full 2011 data
IMPROVEMENT_FACTOR = 0.95          # improved model must achieve <= 95% of baseline


# ── Helper Functions ─────────────────────────────────────────────────

def load_artefacts():
    """
    Load the saved model and evaluation data.
    - model.joblib: Gradient Boosting Regressor (best model from Task 1)
    - day_2011.csv: 365 daily records used for evaluation
    """
    root  = os.path.join(os.path.dirname(__file__), "..")
    model = joblib.load(os.path.join(root, "model.joblib"))
    df    = pd.read_csv(os.path.join(root, "data", "day_2011.csv"))
    return model, df


def evaluate(model, df):
    """
    Run predictions on the evaluation set and compute metrics.
    Uses the same 10 features that the model was trained on.
    Returns: (rmse, mae, r2)
    """
    features = [
        "season", "mnth", "holiday", "weekday", "workingday",
        "weathersit", "temp", "atemp", "hum", "windspeed"
    ]
    X     = df[features]
    y     = df["cnt"]
    y_hat = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae  = mean_absolute_error(y, y_hat)
    r2   = r2_score(y, y_hat)
    return rmse, mae, r2


def check_gate(rmse, mae, r2):
    """
    Compare each metric against its acceptance threshold.
    Returns a list of (metric_name, passed, detail_string) tuples.
    - RMSE and MAE must be BELOW 95% of the baseline (lower is better).
    - R² must be ABOVE the baseline value (higher is better).
    """
    rmse_limit = IMPROVEMENT_FACTOR * RMSE_BASELINE   # 644.82
    mae_limit  = IMPROVEMENT_FACTOR * MAE_BASELINE    # 488.81

    checks = [
        ("RMSE", rmse <= rmse_limit,
         f"{rmse:>10.2f}  (threshold: <= {rmse_limit:.2f})"),
        ("MAE",  mae  <= mae_limit,
         f"{mae:>10.2f}  (threshold: <= {mae_limit:.2f})"),
        ("R2",   r2   >= R2_BASELINE,
         f"{r2:>10.4f}  (threshold: >= {R2_BASELINE})"),
    ]
    return checks


# ── Main Execution ───────────────────────────────────────────────────

def main():
    # ---- Header ----
    print("=" * 58)
    print("  AUTOMATED QUALITY GATE — Bike Sharing Demand Model")
    print("=" * 58)

    # ---- Step 1: Load model and data ----
    model, df = load_artefacts()
    print(f"\n  Model loaded   : model.joblib")
    print(f"  Eval dataset   : data/day_2011.csv  ({len(df)} rows)")
    print(f"  Baseline (LR on full 2011): RMSE={RMSE_BASELINE}, MAE={MAE_BASELINE}, R2={R2_BASELINE}")
    print(f"  RMSE threshold : {IMPROVEMENT_FACTOR} x {RMSE_BASELINE} = {IMPROVEMENT_FACTOR * RMSE_BASELINE:.2f}")
    print(f"  MAE  threshold : {IMPROVEMENT_FACTOR} x {MAE_BASELINE} = {IMPROVEMENT_FACTOR * MAE_BASELINE:.2f}")
    print(f"  R2   threshold : >= {R2_BASELINE}\n")

    # ---- Step 2: Evaluate model performance ----
    rmse, mae, r2 = evaluate(model, df)

    # ---- Step 3: Display results table ----
    print("-" * 58)
    print("  Metric        Value          Acceptance Criterion")
    print("-" * 58)

    checks   = check_gate(rmse, mae, r2)
    failures = 0
    for name, passed, detail in checks:
        status = "OK" if passed else "FAIL"
        print(f"  {name:<6}  {detail}   [{status}]")
        if not passed:
            failures += 1

    print("-" * 58)

    # ---- Step 4: Final verdict ----
    # Exit code 0 = green (pass) on GitHub Actions
    # Exit code 1 = red  (fail) on GitHub Actions
    if failures == 0:
        print("\n  >> Quality Gate PASSED — model is ready for deployment.\n")
    else:
        print(f"\n  >> Quality Gate FAILED — {failures} check(s) did not meet threshold.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
