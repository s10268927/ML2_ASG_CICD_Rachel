import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def test_model_quality_gate():
    """
    Quality Gate: Loads the saved best model and evaluates it on the 2011 dataset.
    The test passes only if the model meets minimum performance thresholds.
    """
    # Load the saved model
    model_path = os.path.join(os.path.dirname(__file__), "..", "model.joblib")
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # Load evaluation data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "day_2011.csv")
    df = pd.read_csv(data_path)
    print(f"Data loaded from {data_path}, shape: {df.shape}")

    # Prepare features
    feature_cols = ["season", "mnth", "holiday", "weekday", "workingday",
                    "weathersit", "temp", "atemp", "hum", "windspeed"]
    X = df[feature_cols]
    y = df["cnt"]

    # Predict
    y_pred = model.predict(X)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")

    # --- Quality Gate Thresholds ---
    # The baseline Linear Regression RMSE is approximately 1200.
    # The improved model must achieve RMSE <= 0.95 * baseline_rmse.
    rmse_baseline = 1200
    rmse_threshold = 0.95 * rmse_baseline

    print(f"Quality Gate: RMSE must be <= {rmse_threshold:.2f} (95% of baseline RMSE {rmse_baseline})")

    # Assert quality gate
    assert rmse <= rmse_threshold, (
        f"QUALITY GATE FAILED: RMSE {rmse:.2f} exceeds threshold {rmse_threshold:.2f}"
    )
    print("QUALITY GATE PASSED: Model meets performance threshold.")


if __name__ == "__main__":
    test_model_quality_gate()
