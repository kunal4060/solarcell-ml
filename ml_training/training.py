"""
Solar-cell multi-output regression benchmark.

Predicts Voc, Jsc, FF and efficiency (eta) of a perovskite solar cell from
five SCAPS simulation inputs:
    - ETL thickness [um]
    - Absorber-layer thickness [um]
    - HTL thickness [um]
    - Absorber-layer total defect density [1/cm^3]
    - Absorber-layer acceptor density [1/cm^3]

Methodology
-----------
1. Load dataset.csv, drop rows with missing values.
2. 80/20 train/test split (random_state=42).
3. Train six regressors. Tree ensembles learn the raw features directly;
   linear / distance-based models get a StandardScaler via a Pipeline:
     - XGBoost            (MultiOutputRegressor wrapper)
     - RandomForest       (native multi-output)
     - GradientBoosting   (MultiOutputRegressor wrapper)
     - LinearRegression   (StandardScaler + LinearRegression)
     - Ridge              (StandardScaler + Ridge, alpha=1.0)
     - KNeighbors         (StandardScaler + KNeighborsRegressor, k=5)
4. Evaluate on the held-out test set: R^2, RMSE and MAE per target, plus an
   overall score (mean across the four targets).
5. Persist metrics, test-set predictions and diagnostic plots.

Outputs (all under ml_training/results/)
    model_metrics.csv            per-model, per-target R2/RMSE/MAE + overall rows
    test_set_predictions.csv     actual targets + every model's predictions
    actual_vs_predicted_<model>.png   2x2 parity plots, one panel per target
    model_performance_r2.png / _rmse.png / _mae.png / _overall.png
                                 bar charts comparing the six models

Run:
    python ml_training/training.py
"""

import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "dataset.csv")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "etl_thickness_um",
    "absorber_thickness_um",
    "htl_thickness_um",
    "defect_density_cm3",
    "acceptor_density_cm3",
]
TARGET_COLUMNS = ["eta_pct", "voc_V", "jsc_mAcm2", "ff_pct"]

RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_models():
    """Return the six regressors, scaling where the model needs it."""
    return {
        "XGBoost": MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                tree_method="hist",
            )
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            )
        ),
        "LinearRegression": Pipeline(
            [("scaler", StandardScaler()), ("lr", LinearRegression())]
        ),
        "Ridge": Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
        ),
        "KNeighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=5)),
            ]
        ),
    }


def evaluate(y_true, y_pred):
    """R2 / RMSE / MAE per target."""
    out = {}
    for i, target in enumerate(TARGET_COLUMNS):
        yt, yp = y_true[:, i], y_pred[:, i]
        out[target] = {
            "r2": r2_score(yt, yp),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": mean_absolute_error(yt, yp),
        }
    return out


def parity_plot(y_true, y_pred, model_name):
    """2x2 predicted-vs-actual panels, Aman style (R2/RMSE/MAE titles, no Accuracy).

    Layout: Voc, Jsc (top row), FF, Efficiency (bottom row).
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    plot_order = [
        ("voc_V", "Voc"),
        ("jsc_mAcm2", "Jsc"),
        ("ff_pct", "FF"),
        ("eta_pct", "Efficiency"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (tkey, tname) in zip(axes.ravel(), plot_order):
        i = TARGET_COLUMNS.index(tkey)
        yt, yp = y_true[:, i], y_pred[:, i]
        r2 = r2_score(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = mean_absolute_error(yt, yp)
        ax.scatter(yt, yp, s=40, alpha=0.65, color="#1f77b4", edgecolors="none")
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        pad = (hi - lo) * 0.03
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            color="red",
            linestyle="--",
            linewidth=1.6,
            label="Perfect Prediction",
        )
        ax.set_xlabel(f"Actual {tname}", fontsize=16)
        ax.set_ylabel(f"Predicted {tname}", fontsize=16)
        ax.set_title(
            f"{model_name} - {tname} Actual Vs Predicted\n"
            f"R² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}",
            fontsize=14,
        )
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.4)
        ax.tick_params(labelsize=11)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"actual_vs_predicted_{model_name}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def metric_bar_charts(metrics_df):
    """Bar charts comparing the six models on each metric."""
    overall = metrics_df[metrics_df["target"] == "overall"]
    for metric, fname in [
        ("r2", "model_performance_r2.png"),
        ("rmse", "model_performance_rmse.png"),
        ("mae", "model_performance_mae.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        models = overall["model"].tolist()
        vals = overall[metric].tolist()
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(models)))
        bars = ax.bar(models, vals, color=colors)
        ax.set_title(f"Model comparison — overall {metric.upper()}")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=110)
        plt.close(fig)

    # combined panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ["r2", "rmse", "mae"]):
        vals = overall[metric].tolist()
        ax.bar(overall["model"].tolist(), vals, color=plt.cm.viridis(np.linspace(0.15, 0.85, len(vals))))
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    fig.suptitle("Overall model performance", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "model_performance_overall.png"), dpi=110)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH).dropna().reset_index(drop=True)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMNS].to_numpy(dtype=float)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    models = build_models()
    metric_rows = []
    pred_frame = pd.DataFrame()
    for target in TARGET_COLUMNS:
        pred_frame[f"actual_{target}"] = y_test[:, TARGET_COLUMNS.index(target)]

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test))
        dt = time.time() - t0
        per_target = evaluate(y_test, y_pred)
        print(f"\n{name}  (fit {dt:.1f}s)")
        for target in TARGET_COLUMNS:
            m = per_target[target]
            print(
                f"  {target:10s}  R2={m['r2']:.6f}  RMSE={m['rmse']:.6f}  MAE={m['mae']:.6f}"
            )
            metric_rows.append(
                {"model": name, "target": target, **{k: float(v) for k, v in m.items()}}
            )
        overall = {k: float(np.mean([per_target[t][k] for t in TARGET_COLUMNS])) for k in ("r2", "rmse", "mae")}
        metric_rows.append({"model": name, "target": "overall", **overall})
        print(f"  {'overall':10s}  R2={overall['r2']:.6f}  RMSE={overall['rmse']:.6f}  MAE={overall['mae']:.6f}")
        for target in TARGET_COLUMNS:
            pred_frame[f"pred_{name}_{target}"] = y_pred[:, TARGET_COLUMNS.index(target)]
        parity_plot(y_test, y_pred, name)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "model_metrics.csv"), index=False)
    pred_frame.to_csv(
        os.path.join(RESULTS_DIR, "test_set_predictions.csv"), index=False
    )
    metric_bar_charts(metrics_df)

    best = metrics_df[metrics_df["target"] == "overall"].sort_values("r2", ascending=False).iloc[0]
    print(f"\nBest model: {best['model']}  (overall R2 = {best['r2']:.6f})")
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
