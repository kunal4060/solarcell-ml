"""
Solar-cell regression benchmark — independent implementation.

Predicts Voc, Jsc, FF and eta from 5 device parameters using 6 regressors.
Data: Vinay__dataset_of_3000_samples__1.xlsx (SCAPS-style simulation data).

Usage:
    python train.py --data /path/to/dataset.xlsx --out ./results
"""

import argparse
import json
from pathlib import Path

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# --- dataset schema -------------------------------------------------------
FEATURES = [
    "ETL thickness",
    "Absorber thickness",
    "HTL thickness",
    "Defect density",
    "Acceptor density",
]
TARGETS = ["Voc", "Jsc", "FF", "eta"]

COLUMN_ALIASES = {
    "etl thickness": "ETL thickness",
    "absorber layerthickness": "Absorber thickness",
    "htl thickness": "HTL thickness",
    "absorber layer total defect density": "Defect density",
    "absorber layer acceptor density": "Acceptor density",
    "voc": "Voc",
    "jsc": "Jsc",
    "ff": "FF",
    "eta": "eta",
}


def clean_column(name: str) -> str:
    """Strip units/brackets and map to a canonical feature/target name."""
    base = "".join(ch for ch in str(name).lower() if ch.isalnum() or ch == " ").strip()
    base = " ".join(base.split())
    # drop trailing unit tokens like 'v', 'm', 'm2', 'cm3', '3' leftovers
    for key, canon in COLUMN_ALIASES.items():
        if base.startswith(key):
            return canon
    return str(name).strip()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df.columns = [clean_column(c) for c in df.columns]
    df = df[FEATURES + TARGETS].apply(pd.to_numeric, errors="coerce").dropna()
    return df.reset_index(drop=True)


def get_models() -> dict:
    xgb = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
    )
    return {
        "XGBoost": MultiOutputRegressor(xgb),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, random_state=42, n_jobs=-1),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(random_state=42)),
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "KNN": make_pipeline(StandardScaler(),
                             KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 0.10) -> dict:
    """R2, RMSE, MAE and fraction of predictions within tol relative error."""
    resid = np.abs(y_pred - y_true)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "acc": float(np.mean(resid <= tol * denom)),
    }


def plot_parity(y_true, y_pred, targets, model_name, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, i in zip(axes.flat, range(len(targets))):
        t, p = y_true[:, i], y_pred[:, i]
        ax.scatter(t, p, alpha=0.5, s=12)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        m = metrics(t, p)
        ax.set_title(f"{model_name} / {targets[i]}  "
                     f"R²={m['r2']:.4f} RMSE={m['rmse']:.2e}")
        ax.set_xlabel(f"Actual {targets[i]}")
        ax.set_ylabel(f"Predicted {targets[i]}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"parity_{model_name}.png", dpi=150)
    plt.close(fig)


def plot_comparison(summary: pd.DataFrame, out: Path):
    for col, title in [("r2", "R²"), ("rmse", "RMSE"),
                       ("mae", "MAE"), ("acc", "Accuracy@10%")]:
        piv = summary[summary.target != "overall"].pivot(
            index="model", columns="target", values=col)
        ax = piv.plot(kind="bar", figsize=(10, 5))
        ax.set_title(f"Model comparison — {title}")
        ax.set_ylabel(title)
        plt.tight_layout()
        fig = ax.get_figure()
        fig.savefig(out / f"compare_{col}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default=Path("results"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    X = df[FEATURES].to_numpy()
    y = df[TARGETS].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42)

    rows, preds = [], pd.DataFrame(
        yte, columns=[f"actual_{t}" for t in TARGETS])

    for name, model in get_models().items():
        print(f"[train] {name}", flush=True)
        model.fit(Xtr, ytr)
        yp = np.asarray(model.predict(Xte))

        for j, tgt in enumerate(TARGETS):
            m = metrics(yte[:, j], yp[:, j])
            rows.append({"model": name, "target": tgt, **m})
        m_all = metrics(yte.ravel(), yp.ravel())
        rows.append({"model": name, "target": "overall", **m_all})

        for j, tgt in enumerate(TARGETS):
            preds[f"pred_{name}_{tgt}"] = yp[:, j]
        plot_parity(yte, yp, TARGETS, name, args.out)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "metrics.csv", index=False)
    preds.to_csv(args.out / "test_predictions.csv", index=False)
    plot_comparison(summary, args.out)

    overall = summary[summary.target == "overall"].sort_values(
        "r2", ascending=False)
    print("\n=== Overall (sorted by R²) ===")
    print(overall.to_string(index=False))
    print(f"\nWrote {len(rows)} metric rows, "
          f"{len(preds)} predictions to {args.out}")
    (args.out / "summary.json").write_text(
        overall.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
