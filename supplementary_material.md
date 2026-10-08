# Supplementary Material: Physics-Informed Neural Operators for Electroosmotic Transport in Graphene Nanochannels

> **Journal Target**: *Desalination* (Elsevier, Q1, Impact Factor: ~9.9)  
> **Authors**: Anvesha Singh (Corresponding Author), Dilprit Singh, Shivansh Chaudhary, Konduru Ushasvini  
> **Affiliation**: School of Computer Science and Engineering, Manipal Institute of Technology, Manipal 576104, Karnataka, India  
> **Email**: `9a.anveshasingh@gmail.com`

---

## Table of Contents

- [S1. Atomistic Molecular Dynamics (MD) Simulation Protocols](#s1-atomistic-molecular-dynamics-md-simulation-protocols)
  - [S1.1. Force Field Parameters and Interatomic Potentials](#s11-force-field-parameters-and-interatomic-potentials)
  - [S1.2. Liquid Packing and Piston Force Formulation](#s12-liquid-packing-and-piston-force-formulation)
  - [S1.3. Parameter Space Boundaries](#s13-parameter-space-boundaries)
- [S2. Mathematical Derivations and Model Architectures](#s2-mathematical-derivations-and-model-architectures)
  - [S2.1. Neural Tangent Kernel (NTK) Spectral Bias Derivation](#s21-neural-tangent-kernel-ntk-spectral-bias-derivation)
  - [S2.2. Feature-wise Linear Modulation (FiLM) Mechanism](#s22-feature-wise-linear-modulation-film-mechanism)
- [S3. Non-Dimensionalized SI Physics Derivation](#s3-non-dimensionalized-si-physics-derivation)
- [S4. Comprehensive Multi-Architecture Error Metrics](#s4-comprehensive-multi-architecture-error-metrics)
- [S5. Transport Integrals and Efficiency Metrics](#s5-transport-integrals-and-efficiency-metrics)
- [S6. Supplementary Figures](#s6-supplementary-figures)

---

## S1. Atomistic Molecular Dynamics (MD) Simulation Protocols

### S1.1. Force Field Parameters and Interatomic Potentials
All-atom MD simulations were performed using the GROMACS simulation package. Water molecules were modeled using the Extended Simple Point Charge (SPC/E) rigid three-site model. Non-bonded interatomic interactions for sodium ions ($\text{Na}^+$), chloride ions ($\text{Cl}^-$), and carbon atoms of the pristine graphene sheets ($\text{C}$) were parameterized using the Optimized Potentials for Liquid Simulations (OPLS-AA) force field.

Cross-species non-bonded Lennard-Jones (LJ) interactions were evaluated using standard Lorentz-Berthelot combination rules:

$$\sigma_{ij} = \frac{\sigma_{ii} + \sigma_{jj}}{2}, \quad \varepsilon_{ij} = \sqrt{\varepsilon_{ii} \varepsilon_{jj}} \tag{S1}$$

The 12-6 Lennard-Jones potential $V_{\text{LJ}}(r_{ij})$ is expressed as:

$$V_{\text{LJ}}(r_{ij}) = 4 \varepsilon_{ij} \left[ \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^6 \right] \tag{S2}$$

#### Table S1: Non-bonded force field Lennard-Jones parameters and atomic partial charges

| Atom / Species | Zero-Crossing Diameter $\sigma$ (\AA) | Well Depth $\epsilon$ (kJ/mol) | Partial Charge $q$ ($e$) |
| :--- | :---: | :---: | :---: |
| O ($\text{H}_2\text{O}$, SPC/E) | 3.1660 | 0.6500 | $-0.8476$ |
| H ($\text{H}_2\text{O}$, SPC/E) | 0.0000 | 0.0000 | $+0.4238$ |
| C (Graphene) | 3.5500 | 0.2928 | $\sigma / \text{surface area}$ |
| $\text{Na}^+$ (Sodium) | 2.5830 | 0.4184 | $+1.0000$ |
| $\text{Cl}^-$ (Chloride) | 4.4010 | 0.4184 | $-1.0000$ |

---

### S1.2. Liquid Packing and Piston Force Formulation
To achieve realistic liquid density distributions at ambient pressure ($P = 1.0\text{ bar}$) and temperature ($T = 300\text{ K}$), liquid packing simulations were conducted under an $NP_zT$ ensemble. Rigid graphene slabs acting as pistons exerted a normal pressure force $f_{\text{piston}}$ along the $z$-axis:

$$f_{\text{piston}} = \frac{\Delta P \cdot L_x \cdot L_y}{n_C \cdot m_C} \tag{S3}$$

where $\Delta P = 1.01325 \times 10^5\text{ Pa}$, lateral box dimensions $L_x = L_y = 10.0\text{ nm}$, $n_C$ is the number of graphene carbon atoms per sheet, and $m_C = 12.011\text{ g/mol}$.

---

### S1.3. Parameter Space Boundaries
The reference dataset comprises 400 distinct MD simulation profiles sampled uniformly across 4 operational parameters:
1. **Surface Charge Density ($\sigma$)**: $-0.12\text{ C/m}^2 \le \sigma \le 0.0\text{ C/m}^2$ (5 discrete levels)
2. **Confinement Width ($H$)**: $1.75\text{ nm} \le H \le 7.88\text{ nm}$ (8 discrete levels)
3. **Axial Electric Field ($E$)**: $0.20\text{ V/nm} \le E \le 1.10\text{ V/nm}$ (5 discrete levels)
4. **Salt Concentration ($C$)**: $0.05\text{ mol/L} \le C \le 1.10\text{ mol/L}$ (2 discrete levels)

---

## S2. Mathematical Derivations and Model Architectures

### S2.1. Neural Tangent Kernel (NTK) Spectral Bias Derivation
Consider a standard Multilayer Perceptron (MLP) trained with gradient descent on spatial evaluation coordinates $z$. Let $\mathbf{K}(z, z')$ denote the Neural Tangent Kernel matrix. The spatial Fourier expansion of the target field $y(z)$ is given by:

$$y(z) = \sum_{k=-\infty}^{\infty} c_k e^{i 2\pi k z / H} \tag{S4}$$

During Adam optimization, the residual error $\Delta \hat{y}(k, t) = \hat{y}(k, t) - c_k$ for wavenumber $k$ evolves according to:

$$\frac{d \hat{y}(k, t)}{dt} = -\lambda_k \left( \hat{y}(k, t) - c_k \right) \tag{S5}$$

For standard MLPs with smooth activation functions (e.g., Tanh, Sigmoid, ReLU), the NTK eigenvalues $\lambda_k$ decay power-law or exponentially with spatial frequency:

$$\lambda_k \propto |k|^{-\alpha}, \quad \alpha > 1 \tag{S6}$$

Consequently, low-frequency components ($k \to 0$, such as parabolic bulk velocity profiles) converge within $10^3$ epochs, whereas high-frequency interfacial hydration layers ($\lambda_{\text{hydration}} \approx 0.3\text{ nm}, k \sim 2\pi / \lambda_{\text{hydration}}$) require prohibitively large optimization steps, resulting in spectral smoothing ($R^2 = 0.7084$).

---

### S2.2. Feature-wise Linear Modulation (FiLM) Mechanism
PINO resolves spectral bias by decoupling global operating parameters $\mathbf{c} = [\sigma, H, E, C]^T \in \mathbb{R}^4$ from spatial coordinates $z \in [-H/2, H/2]$. The global branch processes $\mathbf{c}$ through two fully connected layers to generate affine scale $\boldsymbol{\gamma}(\mathbf{c}) \in \mathbb{R}^d$ and shift $\boldsymbol{\beta}(\mathbf{c}) \in \mathbb{R}^d$ vectors:

$$\mathbf{h}_c = \text{ReLU}\left(\mathbf{W}_{c,2} \text{ReLU}\left(\mathbf{W}_{c,1}\mathbf{c} + \mathbf{b}_{c,1}\right) + \mathbf{b}_{c,2}\right) \tag{S7}$$

$$\boldsymbol{\gamma}(\mathbf{c}) = \mathbf{W}_{\gamma} \mathbf{h}_c + \mathbf{b}_{\gamma}, \quad \boldsymbol{\beta}(\mathbf{c}) = \mathbf{W}_{\beta} \mathbf{h}_c + \mathbf{b}_{\beta} \tag{S8}$$

The spatial trunk maps $z$ to initial feature vector $\mathbf{h}_z^{(0)} = \mathbf{W}_z z + \mathbf{b}_z$. FiLM modulates $\mathbf{h}_z^{(0)}$ element-wise:

$$\mathbf{h}_{\text{FiLM}} = \mathbf{h}_z^{(0)} \odot \left( \mathbf{1} + \boldsymbol{\gamma}(\mathbf{c}) \right) + \boldsymbol{\beta}(\mathbf{c}) \tag{S9}$$

#### Table S2: Hyperparameter configurations for all 6 surrogate model architectures

| Hyperparameter | PINO | Baseline PINN (Arya et al.) | DeepONet | XPINN | FB-PINN | RAR-PIQNN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Total Parameters | 25,668 | 1,424 | 32,840 | 4,272 | 8,544 | 12,816 |
| Hidden Layers | 4 | 3 | 4 (Branch/Trunk) | 2 x 3 Sub | 6 Window | 4 Experts |
| Neurons per Layer | 64 | 20 | 64 | 20 | 20 | 32 |
| Activation | Tanh | Tanh | Tanh | Tanh | Tanh | Tanh |
| Learning Rate | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| Training Epochs | 100,000 | 100,000 | 100,000 | 100,000 | 100,000 | 100,000 |
| Collocation Points | 5,000 | 5,000 | 5,000 | 5,000 | 5,000 | 5,000 |

---

## S3. Non-Dimensionalized SI Physics Derivation

The 1D electroosmotic momentum conservation PDE in SI units is given by:

$$\eta \frac{d^2 u(z)}{d z^2} + e \left[ n_{\text{Na,m}^{-3}}(z) - n_{\text{Cl,m}^{-3}}(z) \right] E_{\text{V/m}} = 0 \tag{S10}$$

where dynamic viscosity $\eta = 8.9 \times 10^{-4}\text{ Pa}\cdot\text{s}$ and elementary charge $e = 1.602176634 \times 10^{-19}\text{ C}$.

To eliminate numerical underflow ($10^{-39}$ PDE loss), we non-dimensionalize by dividing by the characteristic electroosmotic force density scale $F_0$:

$$F_0 = \rho_{e,0} \cdot E_0 = \left(e \cdot 10^{27}\text{ m}^{-3}\right) \cdot \left(1.0 \times 10^9\text{ V/m}\right) = 1.602176634 \times 10^{17}\text{ N/m}^3 \tag{S11}$$

The dimensionless PDE residual $\mathcal{R}_{\text{pde}}(z)$ evaluated at collocation points $z_j$ is:

$$\mathcal{R}_{\text{pde}}(z_j) = \frac{\eta \frac{d^2 \hat{u}(z_j)}{d z^2} + e \left[ \hat{n}_{\text{Na}}(z_j) - \hat{n}_{\text{Cl}}(z_j) \right] E}{F_0} \tag{S12}$$

Mid-plane symmetry requires:

$$\mathcal{R}_{\text{bc}} = \left. \frac{\partial \hat{u}}{\partial z} \right|_{z=0} = 0 \tag{S13}$$

Total non-dimensional composite loss function:

$$\mathcal{L}_{\text{total}}(\boldsymbol{\theta}) = \mathcal{L}_{\text{data}}(\boldsymbol{\theta}) + 1.0 \cdot \mathcal{L}_{\text{pde}}(\boldsymbol{\theta}) + 1.0 \cdot \mathcal{L}_{\text{bc}}(\boldsymbol{\theta}) \tag{S14}$$

---

## S4. Comprehensive Multi-Architecture Error Metrics

#### Table S3: Detailed quantitative error metrics on 120 held-out test simulation channels (12,000 spatial evaluation points)

| Model | Metric | Velocity $u$ | Charge $\rho_e$ | $\text{Na}^+$ $n_{\text{Na}}$ | $\text{Cl}^-$ $n_{\text{Cl}}$ | Water $\rho_{\text{water}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PINO (Proposed)** | $R^2$<br/>MAE<br/>Rel $L_2$ | **0.9955**<br/>0.0084<br/>0.0421 | **0.9484**<br/>0.0215<br/>0.1812 | **0.9749**<br/>0.0112<br/>0.1145 | **0.9945**<br/>0.0078<br/>0.0521 | **0.9061**<br/>0.0145<br/>0.0984 |
| **Arya et al.** [22] | $R^2$<br/>MAE<br/>Rel $L_2$ | 0.9945<br/>0.0092<br/>0.0485 | 0.9921<br/>0.0085<br/>0.0712 | 0.9924<br/>0.0079<br/>0.0685 | 0.9937<br/>0.0081<br/>0.0594 | 0.7084<br/>0.0412<br/>0.2854 |
| **PI-DeepONet** | $R^2$<br/>MAE<br/>Rel $L_2$ | 0.8995<br/>0.0412<br/>0.2114 | 0.9438<br/>0.0245<br/>0.1984 | 0.9670<br/>0.0185<br/>0.1485 | 0.9865<br/>0.0142<br/>0.0985 | 0.5935<br/>0.0521<br/>0.3842 |
| **RAR-PIQNN** | $R^2$<br/>MAE<br/>Rel $L_2$ | 0.9154<br/>0.0385<br/>0.1945 | 0.9378<br/>0.0268<br/>0.2085 | 0.9624<br/>0.0194<br/>0.1584 | 0.9850<br/>0.0151<br/>0.1042 | 0.6039<br/>0.0504<br/>0.3751 |
| **XPINN** | $R^2$<br/>MAE<br/>Rel $L_2$ | 0.8681<br/>0.0512<br/>0.2485 | 0.8977<br/>0.0385<br/>0.2645 | 0.9446<br/>0.0264<br/>0.1984 | 0.9798<br/>0.0185<br/>0.1285 | 0.5871<br/>0.0548<br/>0.3985 |
| **FB-PINN** | $R^2$<br/>MAE<br/>Rel $L_2$ | 0.4807<br/>0.1425<br/>0.5842 | 0.4473<br/>0.1284<br/>0.6124 | 0.4667<br/>0.1145<br/>0.5984 | 0.5986<br/>0.0984<br/>0.4851 | 0.3678<br/>0.0894<br/>0.6214 |

---

## S5. Transport Integrals and Efficiency Metrics

Volumetric water flow rate $Q_{\text{water}}$ ($\text{m}^3/\text{s}$):

$$Q_{\text{water}} = W_{\text{channel}} \int_{-H/2}^{H/2} u(z) dz \tag{S15}$$

where lateral channel width $W_{\text{channel}} = 10.0\text{ nm}$.

Ionic current density $J_{\text{ionic}}$ ($\text{A/m}^2$):

$$J_{\text{ionic}} = e \int_{-H/2}^{H/2} \left[ n_{\text{Na}}(z) \mu_{\text{Na}} + n_{\text{Cl}}(z) \mu_{\text{Cl}} \right] E \, dz \tag{S16}$$

where mobilities $\mu_{\text{Na}} = 5.19 \times 10^{-8}\text{ m}^2/(\text{V}\cdot\text{s})$ and $\mu_{\text{Cl}} = 7.91 \times 10^{-8}\text{ m}^2/(\text{V}\cdot\text{s})$.

Electrical power dissipation $W_{\text{electric}}$ ($\text{W}$):

$$W_{\text{electric}} = E \cdot L_x \cdot J_{\text{ionic}} \cdot L_y \tag{S17}$$

Specific water production rate $P_{\text{spec}}$ ($\text{m}^3/\text{kWh}$):

$$P_{\text{spec}} = \frac{Q_{\text{water}} \cdot 3.6 \times 10^6}{W_{\text{electric}}} \tag{S18}$$

---

## S6. Supplementary Figures

![Supplementary Figure S1: Clean Vertical Methodology Flowchart](file:///c:/Users/Anvesha/OneDrive/Desktop/TESTING/GOAT/Research_Project/graphene_pinn/figures/fig_methodology_flowchart_vertical_clean.png)  
**Supplementary Figure S1**: Detailed non-overlapping vertical methodology flowchart illustrating the 7-stage PINO modeling pipeline, dual-stream FiLM feature conditioning, non-dimensionalized SI physics regularization, and decision convergence loopbacks.

![Supplementary Figure S2: PINO Training Loss Convergence Trajectory](file:///c:/Users/Anvesha/OneDrive/Desktop/TESTING/GOAT/Research_Project/graphene_pinn/figures/fig2_loss_pino.png)  
**Supplementary Figure S2**: Log-scale convergence trajectories of composite loss $\mathcal{L}_{\text{total}}$, data MSE $\mathcal{L}_{\text{data}}$, non-dimensionalized momentum PDE loss $\mathcal{L}_{\text{pde}}$, and symmetry boundary condition loss $\mathcal{L}_{\text{bc}}$ during 100,000 training epochs of the PINO model.
