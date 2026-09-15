from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch

from pinn_model import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/model.pt")
    p.add_argument("--input-scaler", default="artifacts/input_scaler.joblib")
    p.add_argument("--output-scaler", default="artifacts/output_scaler.joblib")
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--H", type=float, required=True)
    p.add_argument("--E", type=float, required=True)
    p.add_argument("--C", type=float, required=True)
    p.add_argument("--points", type=int, default=100)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output", default="artifacts/prediction_profile.csv")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device

    x_scaler = joblib.load(args.input_scaler)
    y_scaler = joblib.load(args.output_scaler)

    model, config = load_checkpoint(args.model, device=device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    model = model.to(dtype=dtype)

    z = np.linspace(-args.H / 2, args.H / 2, args.points)
    raw = pd.DataFrame({
        "sigma": args.sigma,
        "H": args.H,
        "E": args.E,
        "C": args.C,
        "z": z,
    })

    x = x_scaler.transform(raw).astype(np.float64 if config.dtype == "float64" else np.float32)
    x_t = torch.as_tensor(x, dtype=dtype, device=device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_t).cpu().numpy()

    pred = y_scaler.inverse_transform(pred_scaled)

    result = raw.copy()
    result["rho_water"] = pred[:, 0]
    result["n_Na"] = pred[:, 1]
    result["n_Cl"] = pred[:, 2]
    result["u"] = pred[:, 3]

    e = 1.602176634e-19
    result["rho_e"] = e * (result["n_Na"] - result["n_Cl"])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved profile to {args.output}")


if __name__ == "__main__":
    main()
