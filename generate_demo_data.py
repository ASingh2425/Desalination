"""
Generate a synthetic dataset ONLY to test the software pipeline.

This is not the paper's MD dataset and should not be used to reproduce
the paper's scientific results.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/demo_profiles.csv")
    p.add_argument("--simulations", type=int, default=40)
    p.add_argument("--points", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    rows = []

    for sim_id in range(args.simulations):
        H = rng.uniform(1.75e-9, 7.9e-9)
        sigma = -rng.uniform(0.0, 0.12)
        E = rng.uniform(0.2e9, 1.1e9)
        C = rng.uniform(0.2, 1.1)

        z = np.linspace(-H / 2, H / 2, args.points)

        # Synthetic, smooth demonstration profiles.
        x = 2.0 * z / H

        # A simple plug/parabolic hybrid profile.
        u_center = (
            0.25
            + 1.5 * (E / 1e9)
            * (0.5 + 2.5 * abs(sigma))
            * (H / 5e-9)
        )
        u = u_center * (1.0 - 0.15 * x**2)
        u += rng.normal(0, 0.01, size=len(z))

        rho_water = (
            997.0
            + 3.0 * np.exp(-((abs(x) - 1.0) / 0.15) ** 2)
            + rng.normal(0, 0.2, size=len(z))
        )

        # Approximate salt number density.
        n_total = C * 1000.0 * 6.02214076e23
        wall_enrichment = (
            1.0
            + 0.25
            * abs(sigma) / 0.12
            * np.exp(-((abs(x) - 1.0) / 0.2) ** 2)
        )

        n_na = 0.5 * n_total * wall_enrichment
        n_cl = 0.5 * n_total / wall_enrichment

        n_na += rng.normal(
            0,
            0.01 * max(n_total, 1.0),
            size=len(z),
        )
        n_cl += rng.normal(
            0,
            0.01 * max(n_total, 1.0),
            size=len(z),
        )

        n_na = np.maximum(n_na, 1e10)
        n_cl = np.maximum(n_cl, 1e10)

        for j in range(len(z)):
            rows.append(
                {
                    "sim_id": sim_id,
                    "sigma": sigma,
                    "H": H,
                    "E": E,
                    "C": C,
                    "z": z[j],
                    "rho_water": rho_water[j],
                    "n_Na": n_na[j],
                    "n_Cl": n_cl[j],
                    "u": u[j],
                }
            )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)

    print(
        f"Generated {len(df)} rows from "
        f"{args.simulations} synthetic simulations."
    )
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
