"""
PyTorch MLP for solar-cell multi-output regression.

Predicts Voc, Jsc, FF and efficiency (eta) from the five SCAPS simulation
inputs (same dataset.csv as the classical ML pipeline).

Methodology
-----------
1. Load dataset.csv, drop rows with missing values.
2. 80/20 train/test split (random_state=42 — same split as ml_training so
   the classical models and the network are compared on identical data).
3. Standardize X and y with StandardScaler (fitted on the training fold).
4. Train a dense MLP:
       5 -> 128 -> 64 -> 32 -> 4
   with BatchNorm, Dropout(0.2) and ReLU activations, Adam optimizer
   (lr=1e-3), MSE loss, and early stopping on a 15% validation split
   (patience 30, max 500 epochs).
5. Evaluate on the held-out test set in ORIGINAL units: R^2, RMSE and MAE
   per target, plus an overall score (mean across targets).
6. Persist metrics, predictions, weights, scalers and diagnostic plots.

Outputs (all under neural_network_results/)
    nn_model_metrics.csv        per-target R2/RMSE/MAE + overall row
    nn_test_predictions.csv     actual targets + MLP predictions
    training_history.png        train/val loss curves
    actual_vs_predicted.png     2x2 parity plots, one panel per target
    metrics_bar_charts.png      R2/RMSE/MAE bars per target
    neural_network_model.pt     trained weights (state_dict)
    scalers.npz                 StandardScaler params for X and y

Run:
    python nural_network/neural_network_training.py
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "ml_training", "dataset.csv")
RESULTS_DIR = os.path.join(HERE, "..", "neural_network_results")
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
VAL_FRACTION = 0.15
MAX_EPOCHS = 500
PATIENCE = 30
LR = 1e-3
BATCH_SIZE = 256

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class SolarMLP(nn.Module):
    """5 -> 128 -> 64 -> 32 -> 4 dense regressor."""

    def __init__(self, in_dim=5, out_dim=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, stale = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best val loss {best_val:.6f})")
            break
        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train {train_loss:.6f}  val {val_loss:.6f}")

    model.load_state_dict(best_state)
    return history


def plot_history(history):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["val_loss"], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled)")
    ax.set_title("MLP training history")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_history.png"), dpi=110)
    plt.close(fig)


def parity_plot(y_true, y_pred):
    """2x2 parity plot, Aman style (R2/RMSE/MAE titles, no Accuracy)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    plot_order = [
        (0, "Voc"),
        (1, "Jsc"),
        (2, "FF"),
        (3, "Efficiency"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (i, tname) in zip(axes.ravel(), plot_order):
        yt, yp = y_true[:, i], y_pred[:, i]
        r2 = r2_score(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = mean_absolute_error(yt, yp)
        ax.scatter(yt, yp, s=40, alpha=0.7, color="#4682B4", edgecolors="#1a1a2e", linewidths=0.5)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
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
            f"MLP - {tname} Actual Vs Predicted\n"
            f"R² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}",
            fontsize=14,
        )
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.4)
        ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"), dpi=110)
    plt.close(fig)


def metrics_bars(metrics_df):
    per_target = metrics_df[metrics_df["target"] != "overall"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["r2", "rmse", "mae"]):
        ax.bar(
            per_target["target"].tolist(),
            per_target[metric].tolist(),
            color=plt.cm.Greens(np.linspace(0.35, 0.8, len(per_target))),
        )
        ax.set_title(f"MLP {metric.upper()} per target")
        ax.tick_params(axis="x", rotation=15, labelsize=9)
    fig.suptitle("MLP metrics per target", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "metrics_bar_charts.png"), dpi=110)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH).dropna().reset_index(drop=True)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMNS].to_numpy(dtype=float)
    print(f"Dataset: {X.shape[0]} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)
    Xs_train, ys_train = scaler_X.transform(X_train), scaler_y.transform(y_train)
    Xs_test = scaler_X.transform(X_test)

    # train/validation split inside the training fold
    n_val = int(len(Xs_train) * VAL_FRACTION)
    rng = np.random.RandomState(RANDOM_STATE)
    val_idx = rng.choice(len(Xs_train), n_val, replace=False)
    tr_mask = np.ones(len(Xs_train), bool)
    tr_mask[val_idx] = False

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xs_train[tr_mask], dtype=torch.float32),
        torch.tensor(ys_train[tr_mask], dtype=torch.float32),
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xs_train[~tr_mask], dtype=torch.float32),
        torch.tensor(ys_train[~tr_mask], dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = SolarMLP()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP parameters: {n_params}")
    t0 = time.time()
    history = train_model(model, train_loader, val_loader)
    print(f"Training time: {time.time() - t0:.1f}s")

    # evaluate in original units
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.tensor(Xs_test, dtype=torch.float32)).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    rows = []
    for i, target in enumerate(TARGET_COLUMNS):
        yt, yp = y_test[:, i], y_pred[:, i]
        rows.append(
            {
                "model": "MLP",
                "target": target,
                "r2": float(r2_score(yt, yp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                "mae": float(mean_absolute_error(yt, yp)),
            }
        )
    rows.append(
        {
            "model": "MLP",
            "target": "overall",
            "r2": float(np.mean([r["r2"] for r in rows])),
            "rmse": float(np.mean([r["rmse"] for r in rows])),
            "mae": float(np.mean([r["mae"] for r in rows])),
        }
    )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "nn_model_metrics.csv"), index=False)
    print(metrics_df.to_string(index=False))

    pred_df = pd.DataFrame()
    for i, target in enumerate(TARGET_COLUMNS):
        pred_df[f"actual_{target}"] = y_test[:, i]
        pred_df[f"pred_MLP_{target}"] = y_pred[:, i]
    pred_df.to_csv(os.path.join(RESULTS_DIR, "nn_test_predictions.csv"), index=False)

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "neural_network_model.pt"))
    np.savez(
        os.path.join(RESULTS_DIR, "scalers.npz"),
        X_mean=scaler_X.mean_,
        X_scale=scaler_X.scale_,
        y_mean=scaler_y.mean_,
        y_scale=scaler_y.scale_,
    )

    plot_history(history)
    parity_plot(y_test, y_pred)
    metrics_bars(metrics_df)
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
