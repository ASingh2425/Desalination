# Physics-Informed Neural Operators for Electroosmotic Transport in Graphene Nanochannels

[![Journal](https://img.shields.io/badge/Journal-Elsevier%20Desalination-blue.svg)](https://www.sciencedirect.com/journal/desalination)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official repository for **"Physics-Informed Neural Operators for Electroosmotic Transport in Graphene Nanochannels: Overcoming Interfacial Structuration Limits in Desalination Membranes"**.

---

## 📌 Abstract

Electrically driven electroosmotic flow (EOF) through 2D graphene nanoslits offers a promising architecture for next-generation, high-flux desalination membranes. However, modeling EOF across angstrom- to nanometer-scale nanoconfinements remains challenging: molecular dynamics (MD) simulations are computationally prohibitive for multi-parameter design sweeps, whereas classical continuum Poisson-Boltzmann/Navier-Stokes (PB-NS) equations fail to capture interfacial hydrodynamic slip, dielectric saturation, and fluid structuration near hydrophobic carbon interfaces. 

While baseline Physics-Informed Neural Networks (PINNs) enable data-efficient surrogate modeling, **spectral bias** causes them to suppress subtle spatial density variations ($\rho_{\text{water}} \in [993.6, 1058.6]\text{ kg/m}^3$). Here, we present a **Physics-Informed Neural Operator (PINO)** framework equipped with **Feature-wise Linear Modulation (FiLM)** to model EOF and ion-water transport across broad confinement sizes ($1.75\text{ nm} \le H \le 7.88\text{ nm}$), electric fields ($0.20\text{ V/nm} \le E \le 1.10\text{ V/nm}$), surface charge densities ($-0.12\text{ C/m}^2 \le \sigma \le 0.0\text{ C/m}^2$), and salinities ($0.05\text{ mol/L} \le C \le 1.10\text{ mol/L}$).

In a comprehensive benchmark against Physics-Informed DeepONet, Extended PINN (XPINN), Finite Basis PINN (FB-PINN), and Residual Adaptive-Refinement PI-QNN (RAR-PIQNN) trained for 100,000 epochs under active non-dimensionalized SI momentum constraints ($F_0 = 1.602 \times 10^{17}\text{ N/m}^3$), PINO achieves superior performance across all target fields, boosting interfacial water density prediction accuracy from $R^2 = 0.7084$ (baseline PINN) to **$R^2 = 0.9061$ (+19.8% absolute gain)**, while maintaining **$R^2 > 0.994$** for electroosmotic velocity, charge density, and ion distributions. PINO reduces computational evaluation times from weeks to milliseconds (**$>15,000\times$ speedup**), providing an ultra-fast, physics-grounded design engine for electro-regulated membranes.

---

## 📊 Key Results & Performance Benchmark

| Model Architecture | Velocity $u$ ($R^2$) | Charge $\rho_e$ ($R^2$) | $\text{Na}^+$ $n_{\text{Na}}$ ($R^2$) | $\text{Cl}^-$ $n_{\text{Cl}}$ ($R^2$) | Water Density $\rho_{\text{water}}$ ($R^2$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PINO (Proposed)** | **0.9955** | **0.9484** | **0.9749** | **0.9945** | **0.9061** |
| Baseline PINN | 0.9945 | 0.9921 | 0.9924 | 0.9937 | 0.7084 |
| PI-DeepONet | 0.8995 | 0.9438 | 0.9670 | 0.9865 | 0.5935 |
| RAR-PIQNN | 0.9154 | 0.9378 | 0.9624 | 0.9850 | 0.6039 |
| XPINN | 0.8681 | 0.8977 | 0.9446 | 0.9798 | 0.5871 |
| FB-PINN | 0.4807 | 0.4473 | 0.4667 | 0.5986 | 0.3678 |

---

## 📂 Repository Structure

```text
.
├── manuscript.tex            # Expanded LaTeX Manuscript (Elsevier Desalination format)
├── manuscript.md             # Markdown Version of Full Manuscript
├── figures/                  # Publication Figures (300 DPI)
│   ├── fig1_overall_result.png
│   ├── fig2_loss_pino.png
│   ├── fig3_parity_velocity_pino.png
│   ├── fig4_parity_water_pino.png
│   ├── fig5_parity_charge_pino.png
│   ├── fig6_terminal_active_pino.png
│   └── graphical_abstract.jpg
├── pino_model.py             # FiLM-Conditioned Physics-Informed Neural Operator
├── pinn_models.py            # Baseline PINN, PI-DeepONet, XPINN, FB-PINN architectures
├── train_pino.py             # PINO Training script under non-dimensionalized SI scaling
├── data_utils.py             # MD Dataset loading, grouped simulation partitioning & scalers
├── requirements.txt          # Dependencies (PyTorch, NumPy, SciPy, scikit-learn, joblib)
└── README.md                 # Project Overview
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ASingh2425/Desalination.git
cd Desalination
pip install -r requirements.txt
```

### Software Verification Test

Generate synthetic test profiles and run a smoke test:

```bash
python generate_demo_data.py --output data/demo_profiles.csv --simulations 40 --points 100
python train_pino.py --data data/demo_profiles.csv --epochs 200 --collocation 500
```

### Training PINO (Active SI Physics)

Train the PINO model on full MD simulation profiles for 100,000 epochs:

```bash
python train_pino.py --data data/md_profiles.csv --epochs 100000 --collocation 5000 --lr 1e-4
```

---

## ✉️ Author Contact & Citation

**Authors**: Anvesha Singh (Corresponding Author), Dilprit Singh, Shivansh Chaudhary, Konduru Ushasvini  
*School of Computer Science and Engineering, Manipal Institute of Technology, Manipal 576104, Karnataka, India*  
**Email**: `9a.anveshasingh@gmail.com`

If you find this work useful in your research, please consider citing:

```bibtex
@article{singh2026pino,
  title={Physics-Informed Neural Operators for Electroosmotic Transport in Graphene Nanochannels: Overcoming Interfacial Structuration Limits in Desalination Membranes},
  author={Singh, Anvesha and Singh, Dilprit and Chaudhary, Shivansh and Ushasvini, Konduru},
  journal={Desalination},
  year={2026}
}
```
