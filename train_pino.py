"""
Dedicated PINO (Physics-Informed Neural Operator) Training & Benchmark Evaluation Script.

Usage:
    python train_pino.py --data data/md_profiles.csv --epochs 100000 --out artifacts/pino/pino
"""

from __future__ import annotations

import argparse
import json
import os, shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from data_utils import (
    OUTPUT_COLUMNS,
    fit_scalers,
    load_dataframe,
    make_center_bc_points,
    make_collocation_points,
    transform,
)
from pino_model import PINOConfig, get_pino_model, train_pino


def parse_args():
    p = argparse.ArgumentParser(description="Train PINO model on MD profiles.")
    p.add_argument("--data", default="data/md_profiles.csv", help="Path to MD profiles CSV")
    p.add_argument("--out", default="artifacts/pino/pino", help="Output directory")
    p.add_argument("--model", default="pino", help="Model architecture (pino, pi-deeponet, xpinn, rar-piqnn, fbpinn)")
    p.add_argument("--epochs", type=int, default=100000, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--collocation", type=int, default=5000, help="Number of collocation points")
    p.add_argument("--print-every", type=int, default=5000, help="Logging interval")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def parity_plot(y_true, y_pred, title, xlabel, ylabel, path):
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, s=12, alpha=0.4, color='#1f77b4', label=f'Test Data (R² = {r2:.4f})')
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Ideal Alignment (y = x)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (R² = {r2:.4f})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    df = load_dataframe(args.data)
    unique_sims = df["sim_id"].unique()
    train_sims = unique_sims[:280]
    test_sims = unique_sims[280:]

    train_df = df[df["sim_id"].isin(train_sims)].reset_index(drop=True)
    test_df = df[df["sim_id"].isin(test_sims)].reset_index(drop=True)

    x_scaler, y_scaler = fit_scalers(train_df)
    x_train, y_train = transform(train_df, x_scaler, y_scaler)
    x_test, y_test_scaled = transform(test_df, x_scaler, y_scaler)

    x_col = make_collocation_points(train_df, x_scaler, n_points=args.collocation, seed=args.seed)
    x_bc = make_center_bc_points(train_df, x_scaler)

    config = PINOConfig(
        hidden_layers=3,
        neurons=20,
        learning_rate=args.lr,
        epochs=args.epochs,
        print_every=args.print_every,
    )

    print(f"=== STARTING {args.model.upper()} TRAINING FOR {args.epochs:,} EPOCHS UNDER ACTIVE SI PHYSICS ===", flush=True)
    model = get_pino_model(args.model, config)

    history, device = train_pino(
        model,
        x_train,
        y_train,
        x_col,
        x_bc,
        config,
        x_scaler.mean_,
        x_scaler.scale_,
        y_scaler.mean_,
        y_scaler.scale_,
        device=args.device,
    )

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(args.out, f"history_{args.model}.csv"), index=False)

    # Save model state dict
    torch.save(model.state_dict(), os.path.join(args.out, f"model_{args.model}.pt"))

    plt.figure(figsize=(7, 5))
    plt.plot(history["total"], label="Total Loss", linewidth=2)
    plt.plot(history["data"], label="Data Loss (MSE)", linestyle="--")
    plt.plot(history["pde"], label="PDE Loss (Active SI Physics)")
    plt.plot(history["bc"], label="BC Loss (Symmetry)", linestyle=":")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{args.model.upper()} Training Loss Trajectory under Active Physics Constraints")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    loss_fig_path = os.path.join(args.out, f"loss_{args.model}.png")
    plt.savefig(loss_fig_path, dpi=300)
    plt.close()

    model.eval()
    with torch.no_grad():
        x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)
        y_pred_scaled = model(x_test_t).cpu().numpy()

    y_test = y_scaler.inverse_transform(y_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    r2_results = {}
    for i, col in enumerate(OUTPUT_COLUMNS):
        r2_results[col] = float(r2_score(y_test[:, i], y_pred[:, i]))

    e = 1.602176634e-19
    rho_e_true = e * (y_test[:, 1] - y_test[:, 2]) * 1e27
    rho_e_pred = e * (y_pred[:, 1] - y_pred[:, 2]) * 1e27
    r2_results["charge_density"] = float(r2_score(rho_e_true, rho_e_pred))

    with open(os.path.join(args.out, f"metrics_{args.model}.json"), "w") as f:
        json.dump(r2_results, f, indent=4)

    test_pred_df = test_df.copy()
    for i, col in enumerate(OUTPUT_COLUMNS):
        test_pred_df[f"{col}_pred"] = y_pred[:, i]
    test_pred_df["rho_e"] = rho_e_true
    test_pred_df["rho_e_pred"] = rho_e_pred
    test_pred_df.to_csv(os.path.join(args.out, f"test_predictions_{args.model}.csv"), index=False)

    # Parity plots
    parity_plot(y_test[:, 3], y_pred[:, 3], f"{args.model.upper()} Velocity Parity", "MD Velocity (m/s)", "PINO Predicted Velocity (m/s)", os.path.join(args.out, f"parity_velocity_{args.model}.png"))
    parity_plot(y_test[:, 0], y_pred[:, 0], f"{args.model.upper()} Water Density Parity", "MD Water Density (kg/m³)", "PINO Predicted Water Density (kg/m³)", os.path.join(args.out, f"parity_water_density_{args.model}.png"))
    parity_plot(rho_e_true, rho_e_pred, f"{args.model.upper()} Net Charge Density Parity", "MD Net Charge Density (C/m³)", "PINO Predicted Net Charge Density (C/m³)", os.path.join(args.out, f"parity_charge_density_{args.model}.png"))

    print(f"\n=== FINAL TEST R2 METRICS ({args.model.upper()} - {args.epochs:,} Epochs) ===", flush=True)
    for k, v in r2_results.items():
        print(f"  {k:25s}: {v:.6f}", flush=True)

    # Sync figures to figure directories
    target_dirs = [
        r"C:\Users\Anvesha\.gemini\antigravity\brain\09ceb6fb-1d42-4d86-97e3-133260bdbf63\figures",
        r"c:\Users\Anvesha\OneDrive\Desktop\TESTING\figures",
        r"c:\Users\Anvesha\OneDrive\Desktop\TESTING\GOAT\Research_Project\graphene_pinn\figures",
    ]
    fig_mappings = [
        (f"loss_{args.model}.png", "fig2_loss_pino.png"),
        (f"loss_{args.model}.png", f"loss_{args.model}.png"),
        (f"parity_velocity_{args.model}.png", "fig3_parity_velocity_pino.png"),
        (f"parity_velocity_{args.model}.png", f"parity_velocity_{args.model}.png"),
        (f"parity_water_density_{args.model}.png", "fig4_parity_water_pino.png"),
        (f"parity_water_density_{args.model}.png", f"parity_water_density_{args.model}.png"),
        (f"parity_charge_density_{args.model}.png", "fig5_parity_charge_pino.png"),
        (f"parity_charge_density_{args.model}.png", f"parity_charge_density_{args.model}.png"),
    ]
    for td in target_dirs:
        os.makedirs(td, exist_ok=True)
        for s_file, d_file in fig_mappings:
            sp = os.path.join(args.out, s_file)
            dp = os.path.join(td, d_file)
            if os.path.exists(sp):
                shutil.copy2(sp, dp)

    print("Synced all active physics plots to figure directories!")


if __name__ == "__main__":
    main()
