from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from data_utils import (
    OUTPUT_COLUMNS,
    fit_scalers,
    group_train_test_split,
    make_center_bc_points,
    make_collocation_points,
    save_json,
    save_scalers,
    transform,
    validate_dataframe,
)

from pinn_model_m1 import (
    PINNConfig,
    save_checkpoint,
    train_pinn,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True)
    p.add_argument("--out", default="artifacts/m1_residual")

    p.add_argument("--epochs", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--viscosity", type=float, default=8.9e-4)

    p.add_argument("--collocation", type=int, default=5000)
    p.add_argument("--print-every", type=int, default=1000)

    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
    )

    p.add_argument(
        "--model",
        default="all",
        choices=["pi-deeponet", "pino", "rar-piqnn", "xpinn", "fbpinn", "all"],
        help="Which model to train. Use 'all' to train all five models sequentially.",
    )

    return p.parse_args()


def parity_plot(
    y_true,
    y_pred,
    title,
    xlabel,
    ylabel,
    path,
):
    plt.figure(figsize=(6, 5))

    plt.scatter(
        y_true,
        y_pred,
        s=8,
        alpha=0.5,
    )

    mn = min(
        y_true.min(),
        y_pred.min(),
    )

    mx = max(
        y_true.max(),
        y_pred.max(),
    )

    plt.plot(
        [mn, mx],
        [mn, mx],
        linestyle="--",
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
    )
    plt.close()


def main():

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(
        args.out,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load and validate dataset
    # ---------------------------------------------------------

    # Use the data_utils loader which normalizes common MD CSV column names
    # to the canonical names expected by the training code.
    try:
        from data_utils import load_dataframe
        df = load_dataframe(args.data)
    except Exception:
        # Fallback to direct read and validate if loader is unavailable or fails
        df = pd.read_csv(args.data)
        validate_dataframe(df)

    print(f"Rows: {len(df)}")
    print(
        f"MD simulations: "
        f"{df['sim_id'].nunique()}"
    )

    # ---------------------------------------------------------
    # Grouped 70/30 train-test split
    # ---------------------------------------------------------

    train_df, test_df = group_train_test_split(
        df,
        test_size=0.30,
        random_state=args.seed,
    )

    print(
        f"Train simulations: "
        f"{train_df['sim_id'].nunique()}"
    )

    print(
        f"Test simulations:  "
        f"{test_df['sim_id'].nunique()}"
    )

    # ---------------------------------------------------------
    # Fit scalers ONLY on training simulations
    # ---------------------------------------------------------

    x_scaler, y_scaler = fit_scalers(
        train_df
    )

    save_scalers(
        x_scaler,
        y_scaler,
        args.out,
    )

    # ---------------------------------------------------------
    # Training data
    # ---------------------------------------------------------

    x_train, y_train = transform(
        train_df,
        x_scaler,
        y_scaler,
    )

    # ---------------------------------------------------------
    # Physics collocation points
    # ---------------------------------------------------------

    x_col = make_collocation_points(
        train_df,
        x_scaler,
        n_points=args.collocation,
        seed=args.seed,
    )

    # ---------------------------------------------------------
    # Center-plane BC points
    # ---------------------------------------------------------

    x_bc = make_center_bc_points(
        train_df,
        x_scaler,
    )

    # ---------------------------------------------------------
    # PINN configuration
    #
    # Same configuration as original model.
    # ---------------------------------------------------------

    config = PINNConfig(
        hidden_layers=3,
        neurons=20,
        learning_rate=args.lr,
        viscosity=args.viscosity,
        epochs=args.epochs,
        print_every=args.print_every,
        dtype=args.dtype,
    )

    # ---------------------------------------------------------
    # Prepare held-out test transformed data once (shared across models)
    # ---------------------------------------------------------

    x_test, y_test_scaled = transform(
        test_df,
        x_scaler,
        y_scaler,
    )

    # ---------------------------------------------------------
    # Choose models to train
    # ---------------------------------------------------------

    # Allowed model names: pi-deeponet, pino, rar-piqnn, xpinn, fbpinn
    model_choices = [
        "pi-deeponet",
        "pino",
        "rar-piqnn",
        "xpinn",
        "fbpinn",
    ]

    if args.model == "all":
        selected = model_choices
    else:
        selected = [args.model]

    for model_name in selected:
        print(f"\nMODEL: {model_name}")

        # Per-model output directory
        model_out = os.path.join(args.out, model_name)
        os.makedirs(model_out, exist_ok=True)

        # Build model via factory
        from pinn_models import get_model

        model = get_model(model_name, config)

        print(
            "Parameters:",
            sum(p.numel() for p in model.parameters()),
        )

        # Train
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
        )

        # Save model checkpoint
        checkpoint_path = os.path.join(
            model_out,
            f"model_{model_name}.pt",
        )

        save_checkpoint(
            model,
            config,
            checkpoint_path,
        )

        # Save history
        pd.DataFrame(history).to_csv(
            os.path.join(model_out, f"history_{model_name}.csv"),
            index=False,
        )

        # ---------------------------------------------------------
        # Evaluate on held-out simulations for this model
        # ---------------------------------------------------------

        dtype = torch.float64 if args.dtype == "float64" else torch.float32
        x_test_t = torch.as_tensor(x_test, dtype=dtype, device=device)

        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(x_test_t).cpu().numpy()

        y_test = y_scaler.inverse_transform(y_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        # R² metrics
        metrics = {}
        for i, name in enumerate(OUTPUT_COLUMNS):
            metrics[name] = float(r2_score(y_test[:, i], y_pred[:, i]))

        # Charge density
        e = 1.602176634e-19
        rho_e_true = e * (y_test[:, 1] - y_test[:, 2])
        rho_e_pred = e * (y_pred[:, 1] - y_pred[:, 2])
        metrics["charge_density"] = float(r2_score(rho_e_true, rho_e_pred))

        # Save metrics
        save_json(metrics, os.path.join(model_out, f"metrics_{model_name}.json"))

        # Save predictions
        result = test_df.copy()
        for i, name in enumerate(OUTPUT_COLUMNS):
            result[f"{name}_pred"] = y_pred[:, i]
        result["rho_e"] = rho_e_true
        result["rho_e_pred"] = rho_e_pred
        result.to_csv(os.path.join(model_out, f"test_predictions_{model_name}.csv"), index=False)

        # Loss plot
        plt.figure(figsize=(7, 5))
        plt.plot(history["total"], label="Total")
        plt.plot(history["data"], label="Data")
        plt.plot(history["pde"], label="PDE")
        plt.plot(history["bc"], label="BC")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Losses - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_out, f"loss_{model_name}.png"), dpi=200)
        plt.close()

        # Parity plots
        parity_plot(
            y_test[:, 3], y_pred[:, 3], f"Velocity: MD vs {model_name}", "MD velocity", f"{model_name} velocity",
            os.path.join(model_out, f"parity_velocity_{model_name}.png"),
        )
        parity_plot(
            y_test[:, 0], y_pred[:, 0], f"Water density: MD vs {model_name}", "MD water density",
            f"{model_name} water density", os.path.join(model_out, f"parity_water_density_{model_name}.png"),
        )
        parity_plot(
            rho_e_true, rho_e_pred, f"Charge density: MD vs {model_name}", "MD charge density",
            f"{model_name} charge density", os.path.join(model_out, f"parity_charge_density_{model_name}.png"),
        )

        # Print results for this model
        print("\n========================================")
        print(f"{model_name} - TEST RESULTS")
        print("========================================")
        print("\nTest R²:")
        for key, value in metrics.items():
            print(f"  {key:20s}: {value:.6f}")
        print(f"\nDevice: {device}")
        print(f"Saved everything to: {model_out}")


if __name__ == "__main__":
    main()