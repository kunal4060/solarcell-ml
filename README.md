# solarcell-ml

Machine-learning benchmark for solar-cell device-parameter prediction from SCAPS simulation data.

Given five device inputs — ETL thickness, absorber-layer thickness, HTL thickness,
absorber defect density and absorber acceptor density — the models predict four
outputs: open-circuit voltage (**Voc**), short-circuit current density (**Jsc**),
fill factor (**FF**) and power-conversion efficiency (**eta**).

## Dataset

`ml_training/dataset.csv` — 2,944 raw SCAPS simulation samples (Tim's own simulation data).
`ml_training/dataset_dedup.csv` — **1,328 unique design points** actually used for training.

> **Deduplication:** the raw data contained 1,616 near-duplicate rows — design points
> differing only in the 5th–6th decimal place of the features (physically meaningless
> for µm-scale thickness values). A random train/test split placed near-twins of the
> same point on both sides, letting models memorize instead of learn (data leakage,
> R² ≈ 1.0000). Deduplication (rounding the 5 features to 4 decimals and dropping
> duplicates) fixes this: the test set now contains genuinely unseen design points,
> so the reported scores are honest generalization performance.

| | |
|---|---|
| Features (5) | `etl_thickness_um`, `absorber_thickness_um`, `htl_thickness_um`, `defect_density_cm3`, `acceptor_density_cm3` |
| Targets (4) | `eta_pct`, `voc_V`, `jsc_mAcm2`, `ff_pct` |
| Split | 80/20 train/test, `random_state=42` (same split for ML and NN, so results are directly comparable) |

## Classical ML pipeline (`ml_training/training.py`)

1. Load `dataset_dedup.csv`, drop rows with missing values.
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

1. Same `dataset_dedup.csv`, same 80/20 split.
2. Standardize X and y (scalers fitted on the training fold).
3. Train a dense PyTorch MLP `5 → 128 → 64 → 32 → 4` with BatchNorm, Dropout(0.2),
   ReLU, Adam (lr=1e-3), MSE loss and early stopping (patience 30, max 500 epochs)
   on a 15% validation split.
4. Evaluate in original units: R², RMSE, MAE per target + overall.
5. Save metrics, predictions, weights, scalers and plots to `neural_network_results/`.

## Results (test set, 266 samples — honest, leakage-free)

Overall scores (mean R² across the four targets):

| Model | R² (overall) | RMSE (overall) | MAE (overall) |
|---|---|---|---|
| RandomForest | **0.999917** | 0.000702 | 0.000250 |
| GradientBoosting | 0.999904 | 0.000731 | 0.000318 |
| XGBoost | 0.999247 | 0.000982 | 0.000354 |
| MLP (PyTorch) | 0.997673 | 0.004036 | 0.002890 |
| KNeighbors | 0.996508 | 0.005277 | 0.001242 |
| Ridge | 0.990983 | 0.009664 | 0.007719 |
| LinearRegression | 0.990814 | 0.009758 | 0.007735 |

**Best model: RandomForest (overall R² = 0.999917).** Tree-based ensembles clearly
outperform linear models on this data; the neural network lands between the
ensembles and the linear baselines.

Per-target R² of the best model (RandomForest):

| Target | R² | RMSE | MAE |
|---|---|---|---|
| eta (%) | 0.999944 | 0.001028 | 0.000356 |
| Voc (V) | 0.999847 | 0.000018 | 0.000011 |
| Jsc (mA/cm²) | 0.999930 | 0.001120 | 0.000270 |
| FF (%) | 0.999949 | 0.000642 | 0.000363 |

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
│   ├── dataset.csv                  # 2,944 raw samples (kept for reference)
│   ├── dataset_dedup.csv            # 1,328 unique design points (used for training)
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
- The dataset is Tim's own SCAPS simulation data. The raw file
  (`ml_training/dataset.csv`, 2,944 rows) is committed for reference; training
  uses the deduplicated file (`ml_training/dataset_dedup.csv`, 1,328 unique
  design points) so that test scores measure real generalization, not
  memorization of near-duplicate rows.
