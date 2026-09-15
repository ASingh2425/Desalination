import pandas as pd

INPUT_FILE = r"data\md_profiles.csv"
OUTPUT_FILE = r"data\md_profiles_pinn.csv"


df = pd.read_csv(INPUT_FILE)

rename_map = {
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

df = df.rename(columns=rename_map)

required_columns = [
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

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# Keep only the columns required by the PINN.
df = df[required_columns]

# Remove rows with missing values.
df = df.dropna()

df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("Dataset prepared successfully.")
print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())
print()
print(df.head().to_string())
print()
print("Saved to:", OUTPUT_FILE)