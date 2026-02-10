# Bike Sharing Demand Prediction — MLOps Assignment

## Project Overview
This project applies MLOps principles to predict daily bike rental demand using the Capital Bikeshare dataset (Washington D.C.).

## Repository Structure
- `src/` — Jupyter notebook with model development, drift analysis, and MLflow tracking
- `tests/` — Automated quality gate test script
- `data/` — Bike sharing daily datasets (2011 and 2012)
- `model.joblib` — Exported best model (Gradient Boosting Regressor)
- `.github/workflows/python-app.yml` — GitHub Actions CI/CD pipeline

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Open `src/ML2_ASG_Notebook_ChuaSingYiRachel.ipynb` in Jupyter and run all cells.

### 3. Run Tests Locally
```bash
python -m pytest tests/test_model.py -v
```

### 4. MLflow Tracking
Start the MLflow server before running the notebook:
```bash
mlflow server --host 127.0.0.1 --port 5000
```

## Author
Chua Sing Yi Rachel
