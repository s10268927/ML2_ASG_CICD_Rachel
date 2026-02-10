import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_quality_gate():
    """
    Quality Gate: Loads the saved best model and evaluates it on the 2011 dataset.
    The test passes only if the model meets minimum performance thresholds.
    """
    # --- Quality Gate Config ---
    print("--- Quality Gate Config ---")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "day_2011.csv")

    # Baseline metrics (from Linear Regression baseline model)
    BASELINE_RMSE = 690.51
    BASELINE_MAE = 501.26
    R2_MIN = 0.8
    QUALITY_FACTOR = 0.95

    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"DATA_PATH : {DATA_PATH}")
    print(f"BASELINE RMSE: {BASELINE_RMSE}")
    print(f"BASELINE MAE : {BASELINE_MAE}")
    print(f"R2_MIN: {R2_MIN}")
    print(f"QUALITY_FACTOR: {QUALITY_FACTOR}")

    # Compute thresholds
    RMSE_THRESHOLD = QUALITY_FACTOR * BASELINE_RMSE
    MAE_THRESHOLD = QUALITY_FACTOR * BASELINE_MAE

    print(f"RMSE THRESHOLD: {RMSE_THRESHOLD}")
    print(f"MAE THRESHOLD : {MAE_THRESHOLD}")
    print(f"R2 MIN: {R2_MIN}")
    print("=" * 40)

    # Load model and data
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    feature_cols = ["season", "mnth", "holiday", "weekday", "workingday",
                    "weathersit", "temp", "atemp", "hum", "windspeed"]
    X = df[feature_cols]
    y = df["cnt"]

    # Predict and evaluate
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print()
    print("--- Model Performance ---")
    print(f"RMSE: {rmse}")
    print(f"MAE : {mae}")
    print(f"R2  : {r2}")
    print("=" * 40)

    # Quality Gate Checks
    print()
    rmse_pass = rmse <= RMSE_THRESHOLD
    mae_pass = mae <= MAE_THRESHOLD
    r2_pass = r2 >= R2_MIN

    print(f"[{'PASS' if rmse_pass else 'FAIL'}] RMSE {rmse:.3f} <= {RMSE_THRESHOLD:.3f}")
    print(f"[{'PASS' if mae_pass else 'FAIL'}] MAE {mae:.3f} <= {MAE_THRESHOLD:.3f}")
    print(f"[{'PASS' if r2_pass else 'FAIL'}] R2 {r2:.3f} >= {R2_MIN}")
    print()

    # Final gate decision
    all_passed = rmse_pass and mae_pass and r2_pass

    if all_passed:
        print("Quality Gate: PASSED")
    else:
        print("Quality Gate: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    run_quality_gate()
