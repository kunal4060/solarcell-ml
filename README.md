# solarcell-ml

Machine-learning benchmark for solar-cell device-parameter prediction from SCAPS simulation data.

Given five device inputs — ETL thickness, absorber-layer thickness, HTL thickness,
absorber defect density and absorber acceptor density — the models predict four
outputs: open-circuit voltage (**Voc**), short-circuit current density (**Jsc**),
fill factor (**FF**) and power-conversion efficiency (**eta**).

## Dataset

`ml_training/dataset.csv` — 2,944 SCAPS simulation samples (Tim's own simulation data).

| | |
|---|---|
| Features (5) | `etl_thickness_um`, `absorber_thickness_um`, `htl_thickness_um`, `defect_density_cm3`, `acceptor_density_cm3` |
| Targets (4) | `eta_pct`, `voc_V`, `jsc_mAcm2`, `ff_pct` |
| Split | 80/20 train/test, `random_state=42` (same split for ML and NN, so results are directly comparable) |

## Classical ML pipeline (`ml_training/training.py`)

1. Load `dataset.csv`, drop rows with missing values.
2. 80/20 train/test split.
3. Train six regressors (tree ensembles on raw features; linear/distance models behind a `StandardScaler` pipeline):
   - XGBoost (`MultiOutputRegressor` wrapper)
   - RandomForest
   - GradientBoosting (`MultiOutputRegressor` wrapper)
   - LinearRegression (+ StandardScaler)
   - Ridge, alpha=1.0 (+ StandardScaler)
   - KNeighbors, k=5 (+ StandardScaler)
4. Evaluate on the held-out test set: R², RMSE, MAE per target + overall (mean across targets).
5. Save metrics, predictions and plots to `ml_training/results/`.

## Neural network pipeline (`nural_network/neural_network_training.py`)

1. Same `dataset.csv`, same 80/20 split.
2. Standardize X and y (scalers fitted on the training fold).
3. Train a dense PyTorch MLP `5 → 128 → 64 → 32 → 4` with BatchNorm, Dropout(0.2),
   ReLU, Adam (lr=1e-3), MSE loss and early stopping (patience 30, max 500 epochs)
   on a 15% validation split.
4. Evaluate in original units: R², RMSE, MAE per target + overall.
5. Save metrics, predictions, weights, scalers and plots to `neural_network_results/`.

## Results (test set, 589 samples)

Overall scores (mean R² across the four targets):

| Model | R² (overall) | RMSE (overall) | MAE (overall) |
|---|---|---|---|
| RandomForest | **0.999988** | 0.000134 | 0.000081 |
| GradientBoosting | 0.999965 | 0.000286 | 0.000198 |
| KNeighbors | 0.999958 | 0.000543 | 0.000325 |
| XGBoost | 0.999851 | 0.000248 | 0.000178 |
| MLP (PyTorch) | 0.998624 | 0.003398 | 0.002685 |
| LinearRegression | 0.992356 | 0.009192 | 0.007826 |
| Ridge | 0.992333 | 0.009197 | 0.007834 |

**Best model: RandomForest (overall R² = 0.999988).** Tree-based ensembles clearly
outperform linear models on this data; the neural network lands between the
ensembles and the linear baselines.

Per-target R² of the best model (RandomForest):

| Target | R² | RMSE | MAE |
|---|---|---|---|
| eta (%) | 0.999998 | 0.000199 | 0.000118 |
| Voc (V) | 0.999961 | 0.000010 | 0.000006 |
| Jsc (mA/cm²) | 1.000000 | 0.000069 | 0.000019 |
| FF (%) | 0.999992 | 0.000259 | 0.000180 |

## Metrics

For true values y, predictions ŷ and n samples:

- RMSE = √( (1/n) Σ (yᵢ − ŷᵢ)² )
- MAE = (1/n) Σ |yᵢ − ŷᵢ|
- R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²

## Repository layout

```
├── README.md
├── combined_model_performance.csv   # all 7 models × (4 targets + overall)
├── combined_model_outputs.csv       # test-set actuals + every model's predictions
├── ml_training/
│   ├── dataset.csv                  # 2,944 samples, 5 features + 4 targets
│   ├── training.py                  # six-regressor benchmark
│   └── results/
│       ├── model_metrics.csv
│       ├── test_set_predictions.csv
│       ├── actual_vs_predicted_<model>.png
│       └── model_performance_*.png
├── nural_network/
│   └── neural_network_training.py   # PyTorch MLP
└── neural_network_results/
    ├── nn_model_metrics.csv
    ├── nn_test_predictions.csv
    ├── neural_network_model.pt      # trained weights
    ├── scalers.npz                  # StandardScaler parameters
    ├── training_history.png
    ├── actual_vs_predicted.png
    └── metrics_bar_charts.png
```

## How to run

```bash
pip install pandas numpy scikit-learn xgboost torch matplotlib
python ml_training/training.py
python nural_network/neural_network_training.py
```

## Notes

- All code in this repo is written from scratch for this project; the training
  methodology (80/20 split, six classical regressors + MLP, R²/RMSE/MAE reporting)
  follows the standard solar-cell ML benchmarking setup.
- The dataset is Tim's own SCAPS simulation data and is committed here as
  `ml_training/dataset.csv` for reproducibility.
