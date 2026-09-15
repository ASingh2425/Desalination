from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from data_utils import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    fit_scalers,
    group_train_test_split,
    make_center_bc_points,
    make_collocation_points,
    save_json,
    save_scalers,
    transform,
    validate_dataframe,
    load_dataframe,
)
from pinn_model import ElectroosmoticPINN, PINNConfig, save_checkpoint, train_pinn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="artifacts")
    p.add_argument("--epochs", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--viscosity", type=float, default=8.9e-4)
    p.add_argument("--collocation", type=int, default=5000)
    p.add_argument("--print-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])

    p.add_argument(
        "--loss-mode",
        default="pinn",
        choices=["pinn", "data"],
        help="Loss mode: 'pinn' uses PDE+BC+data; 'data' uses data-only MSE (PDE/BC still computed for logging).",
    )

    p.add_argument(
        "--pde-weight",
        type=float,
        default=1.0,
        help="Weight applied to PDE residual loss when loss-mode is 'pinn'.",
    )

    p.add_argument(
        "--bc-weight",
        type=float,
        default=1.0,
        help="Weight applied to BC loss when loss-mode is 'pinn'.",
    )

    return p.parse_args()


def parity_plot(y_true, y_pred, title, xlabel, ylabel, path):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    df = load_dataframe(args.data)

    print(f"Rows: {len(df)}")
    print(f"MD simulations: {df['sim_id'].nunique()}")

    train_df, test_df = group_train_test_split(
        df,
        test_size=0.30,
        random_state=args.seed,
    )

    print(f"Train simulations: {train_df['sim_id'].nunique()}")
    print(f"Test simulations:  {test_df['sim_id'].nunique()}")

    x_scaler, y_scaler = fit_scalers(train_df)
    save_scalers(x_scaler, y_scaler, args.out)

    x_train, y_train = transform(train_df, x_scaler, y_scaler)
    x_col = make_collocation_points(
        train_df,
        x_scaler,
        n_points=args.collocation,
        seed=args.seed,
    )
    x_bc = make_center_bc_points(train_df, x_scaler)

    config = PINNConfig(
        hidden_layers=3,
        neurons=20,
        learning_rate=args.lr,
        viscosity=args.viscosity,
        epochs=args.epochs,
        print_every=args.print_every,
        dtype=args.dtype,
    )

    model = ElectroosmoticPINN(config)

    history, device = train_pinn(
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
        loss_mode=args.loss_mode,
        pde_weight=args.pde_weight,
        bc_weight=args.bc_weight,
    )

    checkpoint_path = os.path.join(args.out, "model.pt")
    save_checkpoint(model, config, checkpoint_path)

    pd.DataFrame(history).to_csv(
        os.path.join(args.out, "history.csv"), index=False
    )

    # Held-out simulations only.
    x_test, y_test_scaled = transform(test_df, x_scaler, y_scaler)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    x_test_t = torch.as_tensor(x_test, dtype=dtype, device=device)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(x_test_t).cpu().numpy()

    y_test = y_scaler.inverse_transform(y_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    metrics = {}
    for i, name in enumerate(OUTPUT_COLUMNS):
        metrics[name] = float(r2_score(y_test[:, i], y_pred[:, i]))

    e = 1.602176634e-19
    rho_e_true = e * (y_test[:, 1] - y_test[:, 2])
    rho_e_pred = e * (y_pred[:, 1] - y_pred[:, 2])
    metrics["charge_density"] = float(r2_score(rho_e_true, rho_e_pred))

    save_json(metrics, os.path.join(args.out, "metrics.json"))

    result = test_df.copy()
    for i, name in enumerate(OUTPUT_COLUMNS):
        result[f"{name}_pred"] = y_pred[:, i]
    result["rho_e"] = rho_e_true
    result["rho_e_pred"] = rho_e_pred
    result.to_csv(os.path.join(args.out, "test_predictions.csv"), index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(history["total"], label="Total")
    plt.plot(history["data"], label="Data")
    plt.plot(history["pde"], label="PDE")
    plt.plot(history["bc"], label="BC")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN training losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "loss.png"), dpi=200)
    plt.close()

    parity_plot(
        y_test[:, 3], y_pred[:, 3], "Velocity: MD vs PINN",
        "MD velocity", "PINN velocity",
        os.path.join(args.out, "parity_velocity.png"),
    )
    parity_plot(
        y_test[:, 0], y_pred[:, 0], "Water density: MD vs PINN",
        "MD water density", "PINN water density",
        os.path.join(args.out, "parity_water_density.png"),
    )
    parity_plot(
        rho_e_true, rho_e_pred, "Charge density: MD vs PINN",
        "MD charge density", "PINN charge density",
        os.path.join(args.out, "parity_charge_density.png"),
    )

    print("\nTest R²:")
    for key, value in metrics.items():
        print(f"  {key:20s}: {value:.6f}")
    print(f"\nDevice: {device}")
    print(f"Saved everything to: {args.out}")


if __name__ == "__main__":
    main()
