"""
Custom neural network for solar-cell regression — my own architecture.

MLP: 5 -> 128 -> 64 -> 32 -> 4 with BatchNorm, ReLU, Dropout.
Trained with Adam + ReduceLROnPlateau + early stopping on validation loss.

Usage:
    python train_nn.py --data /path/to/dataset.xlsx --out ./results_nn
"""

import argparse
import json
from pathlib import Path

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
from torch.utils.data import DataLoader, TensorDataset

from train import FEATURES, TARGETS, clean_column  # reuse schema only

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")


class SolarMLP(nn.Module):
    """My architecture: deep MLP with BatchNorm + Dropout."""

    def __init__(self, in_dim=5, out_dim=4, dropout=0.1):
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


def load_tensors(data_path: Path):
    df = pd.read_excel(data_path, header=1)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df.columns = [clean_column(c) for c in df.columns]
    df = df[FEATURES + TARGETS].apply(pd.to_numeric, errors="coerce").dropna()

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGETS].to_numpy(dtype=np.float32)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.15, random_state=42)

    sx, sy = StandardScaler(), StandardScaler()
    Xtr, Xva, Xte = sx.fit_transform(Xtr), sx.transform(Xva), sx.transform(Xte)
    ytr, yva = sy.fit_transform(ytr), sy.transform(yva)

    to_t = lambda a: torch.from_numpy(a)
    train_dl = DataLoader(TensorDataset(to_t(Xtr), to_t(ytr)),
                          batch_size=64, shuffle=True)
    val_dl = DataLoader(TensorDataset(to_t(Xva), to_t(yva)), batch_size=256)
    return train_dl, val_dl, (to_t(Xte), to_t(yte), yte), sy


def train_model(train_dl, val_dl, epochs=400, patience=40, lr=1e-3):
    model = SolarMLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=12)
    loss_fn = nn.MSELoss()

    best, best_state, bad = float("inf"), None, 0
    hist = {"train": [], "val": []}

    for ep in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
        tl /= len(train_dl.dataset)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl += loss_fn(model(xb), yb).item() * len(xb)
        vl /= len(val_dl.dataset)
        sched.step(vl)
        hist["train"].append(tl)
        hist["val"].append(vl)

        if vl < best - 1e-9:
            best, best_state, bad = vl, {k: v.cpu().clone()
                                         for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:3d}  train={tl:.6f}  val={vl:.6f}  "
                  f"lr={opt.param_groups[0]['lr']:.1e}", flush=True)
        if bad >= patience:
            print(f"  early stop at epoch {ep+1}", flush=True)
            break

    model.load_state_dict(best_state)
    return model, hist, best


def metrics_report(y_true, y_pred, tol=0.10):
    out = {}
    for j, tgt in enumerate(TARGETS):
        t, p = y_true[:, j], y_pred[:, j]
        resid = np.abs(p - t)
        out[tgt] = {
            "r2": float(r2_score(t, p)),
            "rmse": float(np.sqrt(mean_squared_error(t, p))),
            "mae": float(mean_absolute_error(t, p)),
            "acc": float(np.mean(resid <= tol * np.maximum(np.abs(t), 1e-8))),
        }
    tall, pall = y_true.ravel(), y_pred.ravel()
    out["overall"] = {
        "r2": float(r2_score(tall, pall)),
        "rmse": float(np.sqrt(mean_squared_error(tall, pall))),
        "mae": float(mean_absolute_error(tall, pall)),
        "acc": float(np.mean(np.abs(pall - tall)
                             <= tol * np.maximum(np.abs(tall), 1e-8))),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default=Path("results_nn"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    train_dl, val_dl, (Xte, yte_t, yte_raw), scaler_y = load_tensors(args.data)
    print(f"Train: {len(train_dl.dataset)}  Val: {len(val_dl.dataset)}  "
          f"Test: {len(Xte)}", flush=True)

    print("Training SolarMLP...", flush=True)
    model, hist, best_val = train_model(train_dl, val_dl)

    # plots: loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["train"], label="train")
    ax.plot(hist["val"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (scaled)")
    ax.set_title("SolarMLP training curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "nn_training_curve.png", dpi=150)
    plt.close(fig)

    # evaluate on test set (inverse transform)
    model.eval()
    with torch.no_grad():
        yp_scaled = model(Xte.to(DEVICE)).cpu().numpy()
    yp = scaler_y.inverse_transform(yp_scaled)

    # parity plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, j, tgt in zip(axes.flat, range(4), TARGETS):
        t, p = yte_raw[:, j], yp[:, j]
        ax.scatter(t, p, alpha=0.5, s=12)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        ax.set_title(f"SolarMLP / {tgt}  R²={r2_score(t, p):.4f}")
        ax.set_xlabel(f"Actual {tgt}")
        ax.set_ylabel(f"Predicted {tgt}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "nn_parity.png", dpi=150)
    plt.close(fig)

    report = metrics_report(yte_raw, yp)
    (args.out / "nn_metrics.json").write_text(json.dumps(report, indent=2))
    torch.save(model.state_dict(), args.out / "solar_mlp.pt")

    print("\n=== SolarMLP (custom NN) test metrics ===")
    for tgt in TARGETS + ["overall"]:
        m = report[tgt]
        print(f"  {tgt:8s} R²={m['r2']:.6f} RMSE={m['rmse']:.6f} "
              f"MAE={m['mae']:.6f} Acc={m['acc']:.4f}")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
