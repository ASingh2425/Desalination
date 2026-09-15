from __future__ import annotations

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = [
    "sim_id",
    "sigma",
    "H",
    "E",
    "C",
    "z",
    "rho_water",
    "n_Na",
    "n_Cl",
    "u",
]

INPUT_COLUMNS = ["sigma", "H", "E", "C", "z"]
OUTPUT_COLUMNS = ["rho_water", "n_Na", "n_Cl", "u"]


# Known alternative column names mapping from MD output to required names
COLUMN_ALIASES = {
    "simulation_id": "sim_id",
    "channel_height_nm": "H",
    "ionic_concentration_mol_L": "C",
    "surface_charge_density_C_m2": "sigma",
    "electric_field_V_nm": "E",
    "z_nm": "z",
    "velocity_m_s": "u",
    "water_density_kg_m3": "rho_water",
    "Na_density_number_nm3": "n_Na",
    "Cl_density_number_nm3": "n_Cl",
}


def load_dataframe(path: str) -> pd.DataFrame:
    """Read a CSV and normalize column names to the expected schema.

    This will rename common alternative column names produced by the MD
    preprocessing pipeline to the canonical names used by the training
    code, then validate the result.
    """
    df = pd.read_csv(path)
    # Rename any known aliases that appear
    rename = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)

    # If sim_id is present, try to ensure it's an integer type
    if "sim_id" in df.columns:
        try:
            df["sim_id"] = df["sim_id"].astype(int)
        except Exception:
            # leave as-is if conversion fails
            pass

    validate_dataframe(df)
    return df


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    input_scaler: StandardScaler
    output_scaler: StandardScaler


def validate_dataframe(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
        )

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains NaN values in required columns.")


def group_train_test_split(df: pd.DataFrame, test_size=0.30, random_state=42):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=df["sim_id"].values,
        )
    )

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def fit_scalers(train_df: pd.DataFrame):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(train_df[INPUT_COLUMNS])
    y_scaler.fit(train_df[OUTPUT_COLUMNS])

    return x_scaler, y_scaler


def transform(df, x_scaler, y_scaler):
    x = x_scaler.transform(df[INPUT_COLUMNS]).astype(np.float32)
    y = y_scaler.transform(df[OUTPUT_COLUMNS]).astype(np.float32)
    return x, y


def make_collocation_points(
    df: pd.DataFrame,
    x_scaler: StandardScaler,
    n_points: int = 5000,
    seed: int = 42,
):
    """
    Create collocation points over the operating-condition ranges and
    channel-coordinate range represented in the training data.

    We preserve realistic combinations by sampling existing operating
    conditions and random z values within their channel-height bounds.
    """
    rng = np.random.default_rng(seed)

    sim_conditions = (
        df.groupby("sim_id")[["sigma", "H", "E", "C"]]
        .first()
        .reset_index(drop=True)
    )

    chosen = sim_conditions.iloc[
        rng.integers(0, len(sim_conditions), size=n_points)
    ].copy()

    # For a symmetric channel, use z in [-H/2, H/2].
    chosen["z"] = (
        rng.uniform(-0.5, 0.5, size=n_points) * chosen["H"].to_numpy()
    )

    x = x_scaler.transform(chosen[INPUT_COLUMNS]).astype(np.float32)
    return x


def make_center_bc_points(
    df: pd.DataFrame,
    x_scaler: StandardScaler,
):
    """
    One center-plane point for every training simulation condition:
        z = 0
    """
    conditions = (
        df.groupby("sim_id")[["sigma", "H", "E", "C"]]
        .first()
        .reset_index(drop=True)
    )
    conditions["z"] = 0.0
    return x_scaler.transform(
        conditions[INPUT_COLUMNS]
    ).astype(np.float32)


def save_scalers(x_scaler, y_scaler, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(
        x_scaler,
        os.path.join(output_dir, "input_scaler.joblib"),
    )
    joblib.dump(
        y_scaler,
        os.path.join(output_dir, "output_scaler.joblib"),
    )


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
