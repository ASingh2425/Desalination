"""
Physics-Informed Neural Operator (PINO) Model & Training Pipeline.

This module provides dedicated PINO Neural Operator architectures (PINO, PI-DeepONet, 
RAR-PIQNN, XPINN, FBPINN) and the non-dimensionalized SI physics-informed training loop
for electro-hydrodynamic flows in graphene nanochannels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

E_CHARGE = 1.602176634e-19
F_CHARACTERISTIC = 1.602176634e17  # Characteristic electroosmotic force scale (N/m^3)


@dataclass
class PINOConfig:
    hidden_layers: int = 3
    neurons: int = 20
    learning_rate: float = 1e-4
    viscosity: float = 8.9e-4
    epochs: int = 20_000
    print_every: int = 1000
    dtype: str = "float32"


class PINO(nn.Module):
    """
    Physics-Informed Neural Operator (PINO) Architecture.

    Uses a conditioner network for global operating conditions (sigma, H, E, C)
    and a trunk network for local spatial coordinate (z) with FiLM modulation.
    """

    def __init__(self, config: PINOConfig | None = None, hidden: int = 64, layers: int = 4):
        super().__init__()
        self.config = config or PINOConfig()

        # Conditioner for global conditions (sigma, H, E, C)
        self.cond_net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # FiLM generators for scale and shift parameters
        self.scale_gen = nn.Linear(hidden, hidden)
        self.shift_gen = nn.Linear(hidden, hidden)

        # Trunk network for local coordinate z with feature modulation
        trunk = []
        trunk.append(nn.Linear(1, hidden))
        for _ in range(layers - 1):
            trunk.append(nn.ReLU())
            trunk.append(nn.Linear(hidden, hidden))
        self.trunk = nn.ModuleList(trunk)

        # Output head mapping features to 4 outputs [rho_water, n_Na, n_Cl, u]
        self.head = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cond = x[:, :4]
        z = x[:, 4:5]

        c = self.cond_net(cond)
        scale = self.scale_gen(c)
        shift = self.shift_gen(c)

        h = z
        h = self.trunk[0](h)
        h = h * (1.0 + scale) + shift
        for layer in self.trunk[1:]:
            h = layer(h)

        out = self.head(h)
        return out


class PIDeepONet(nn.Module):
    """Physics-Informed DeepONet Operator architecture."""

    def __init__(self, config: PINOConfig | None = None, branch_dim: int = 64, trunk_dim: int = 64):
        super().__init__()
        self.config = config or PINOConfig()

        self.branch = nn.Sequential(
            nn.Linear(4, branch_dim),
            nn.Tanh(),
            nn.Linear(branch_dim, branch_dim),
            nn.Tanh(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(1, trunk_dim),
            nn.Tanh(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.Tanh(),
        )

        self.trunk_proj = None
        if trunk_dim != branch_dim:
            self.trunk_proj = nn.Linear(trunk_dim, branch_dim)
            nn.init.xavier_uniform_(self.trunk_proj.weight)
            nn.init.zeros_(self.trunk_proj.bias)

        self.head = nn.Linear(branch_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cond = x[:, :4]
        z = x[:, 4:5]

        b = self.branch(cond)
        t = self.trunk(z)

        if self.trunk_proj is not None:
            t = self.trunk_proj(t)

        features = b * t
        out = self.head(features)
        return out


class RAR_PIQNN(nn.Module):
    """Residual Adaptive-Refinement Physics-Informed Neural Operator."""

    def __init__(self, config: PINOConfig | None = None, experts: int = 4, hidden: int = 48):
        super().__init__()
        self.config = config or PINOConfig()
        self.experts = nn.ModuleList()
        self.centers = torch.linspace(-0.5, 0.5, experts).unsqueeze(1)

        for _ in range(experts):
            net = nn.Sequential(
                nn.Linear(5, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 4),
            )
            self.experts.append(net)

    def _window(self, z: torch.Tensor, center: torch.Tensor, width: float = 0.5) -> torch.Tensor:
        return torch.exp(-((z - center.to(z.device)) ** 2) / (2 * width * width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[:, 4:5]
        outs, weights = [], []
        for i, net in enumerate(self.experts):
            out = net(x)
            w = self._window(z, self.centers[i])
            outs.append(out * w)
            weights.append(w)

        denom = torch.clamp(torch.sum(torch.stack(weights, dim=0), dim=0), min=1e-6)
        combined = torch.sum(torch.stack(outs, dim=0), dim=0) / denom
        return combined


class XPINN(nn.Module):
    """Extended Domain-Decomposition PIN Operator."""

    def __init__(self, config: PINOConfig | None = None, hidden: int = 64):
        super().__init__()
        self.config = config or PINOConfig()

        self.left = nn.Sequential(
            nn.Linear(5, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),
        )

        self.right = nn.Sequential(
            nn.Linear(5, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[:, 4:5]
        s = torch.sigmoid(z * 50.0)
        return (1.0 - s) * self.left(x) + s * self.right(x)


class FBPINN(nn.Module):
    """Finite Basis Physics-Informed Neural Operator."""

    def __init__(self, config: PINOConfig | None = None, windows: int = 6, hidden: int = 48):
        super().__init__()
        self.config = config or PINOConfig()
        self.windows = windows
        self.workers = nn.ModuleList()
        centers = torch.linspace(-0.5, 0.5, windows)
        self.register_buffer("_centers", centers)

        for _ in range(windows):
            net = nn.Sequential(
                nn.Linear(5, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 4),
            )
            self.workers.append(net)

    def _bump(self, z: torch.Tensor, center: torch.Tensor, width: float = 0.35) -> torch.Tensor:
        d = torch.abs(z - center.to(z.device)) / width
        w = torch.clamp(0.5 * (1 + torch.cos(torch.clamp(d, 0.0, 1.0) * torch.pi)), min=0.0)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[:, 4:5]
        outs, weights = [], []
        for i, net in enumerate(self.workers):
            c = self._centers[i:i+1]
            w = self._bump(z, c)
            weights.append(w)
            outs.append(net(x) * w)

        denom = torch.clamp(torch.sum(torch.stack(weights, dim=0), dim=0), min=1e-6)
        combined = torch.sum(torch.stack(outs, dim=0), dim=0) / denom
        return combined


def get_pino_model(name: str, config: PINOConfig | None = None) -> nn.Module:
    """Factory returning PINO/Operator model instance by name."""
    n = name.lower()
    if n in ("pino", "pino_model"):
        return PINO(config)
    if n in ("pi-deeponet", "deeponet", "pi_deeponet"):
        return PIDeepONet(config)
    if n in ("rar-piqnn", "rar", "piqnn"):
        return RAR_PIQNN(config)
    if n in ("xpinn", "x-pinn"):
        return XPINN(config)
    if n in ("fbpinn", "fb-pinn"):
        return FBPINN(config)
    raise ValueError(f"Unknown operator model name: {name}")


def _column(x: torch.Tensor, index: int) -> torch.Tensor:
    return x[:, index:index + 1]


def pino_pde_residual(
    model: nn.Module,
    x: torch.Tensor,
    viscosity: float,
    input_mean: torch.Tensor,
    input_scale: torch.Tensor,
    output_mean: torch.Tensor,
    output_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates electroosmotic Navier-Stokes PDE residual in SI units normalized by F0."""
    x = x.clone().detach().requires_grad_(True)
    y_scaled = model(x)
    u_scaled = _column(y_scaled, 3)

    du_scaled_dx = torch.autograd.grad(
        u_scaled.sum(),
        x,
        create_graph=True,
        retain_graph=True,
    )[0]
    du_scaled_dzscaled = _column(du_scaled_dx, 4)

    d2u_scaled_dx = torch.autograd.grad(
        du_scaled_dzscaled.sum(),
        x,
        create_graph=True,
        retain_graph=True,
    )[0]
    d2u_scaled_dzscaled2 = _column(d2u_scaled_dx, 4)

    z_scale_m = input_scale[4:5] * 1e-9  # nm to m
    u_scale_m = output_scale[3:4]        # m/s

    du_dz = du_scaled_dzscaled
    d2u_dz2_SI = (u_scale_m / torch.square(z_scale_m)) * d2u_scaled_dzscaled2

    n_na_m3 = (_column(y_scaled, 1) * output_scale[1:2] + output_mean[1:2]) * 1e27  # nm^-3 to m^-3
    n_cl_m3 = (_column(y_scaled, 2) * output_scale[2:3] + output_mean[2:3]) * 1e27

    rho_e_SI = E_CHARGE * (n_na_m3 - n_cl_m3)  # C/m^3
    electric_field_SI = (_column(x, 2) * input_scale[2:3] + input_mean[2:3]) * 1e9  # V/m

    residual = (viscosity * d2u_dz2_SI + rho_e_SI * electric_field_SI) / F_CHARACTERISTIC
    return residual, du_dz, y_scaled


def pino_losses(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_collocation: torch.Tensor,
    x_bc: torch.Tensor,
    viscosity: float,
    input_mean: torch.Tensor,
    input_scale: torch.Tensor,
    output_mean: torch.Tensor,
    output_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    y_pred = model(x_data)
    l_data = torch.sum(torch.mean(torch.square(y_data - y_pred), dim=0))

    residual, _, _ = pino_pde_residual(
        model, x_collocation, viscosity, input_mean, input_scale, output_mean, output_scale
    )
    l_pde = torch.mean(torch.square(residual))

    _, du_dz_bc, _ = pino_pde_residual(
        model, x_bc, viscosity, input_mean, input_scale, output_mean, output_scale
    )
    l_bc = torch.mean(torch.square(du_dz_bc))

    return {
        "total": l_data + l_pde + l_bc,
        "data": l_data,
        "pde": l_pde,
        "bc": l_bc,
    }


def train_pino(
    model: nn.Module,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_collocation: np.ndarray,
    x_bc: np.ndarray,
    config: PINOConfig,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    output_mean: np.ndarray,
    output_scale: np.ndarray,
    device: str = "auto",
    loss_mode: str = "pinn",
    pde_weight: float = 1.0,
    bc_weight: float = 1.0,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_obj = torch.device(device)
    model = model.to(device_obj)

    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    model = model.to(dtype=dtype)

    x_data_t = torch.as_tensor(x_data, dtype=dtype, device=device_obj)
    y_data_t = torch.as_tensor(y_data, dtype=dtype, device=device_obj)
    x_col_t = torch.as_tensor(x_collocation, dtype=dtype, device=device_obj)
    x_bc_t = torch.as_tensor(x_bc, dtype=dtype, device=device_obj)

    input_mean_t = torch.as_tensor(input_mean, dtype=dtype, device=device_obj)
    input_scale_t = torch.as_tensor(input_scale, dtype=dtype, device=device_obj)
    output_mean_t = torch.as_tensor(output_mean, dtype=dtype, device=device_obj)
    output_scale_t = torch.as_tensor(output_scale, dtype=dtype, device=device_obj)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = {"total": [], "data": [], "pde": [], "bc": []}

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        losses = pino_losses(
            model,
            x_data_t,
            y_data_t,
            x_col_t,
            x_bc_t,
            config.viscosity,
            input_mean_t,
            input_scale_t,
            output_mean_t,
            output_scale_t,
        )

        if loss_mode == "data":
            total = losses["data"]
        else:
            total = losses["data"] + pde_weight * losses["pde"] + bc_weight * losses["bc"]

        losses["total"] = total
        losses["total"].backward()
        optimizer.step()

        values = {k: float(v.detach().cpu()) for k, v in losses.items()}
        for k in history:
            history[k].append(values[k])

        if epoch == 1 or epoch % config.print_every == 0:
            print(
                f"Epoch {epoch:>7d}/{config.epochs} | "
                f"total={values['total']:.6e} | "
                f"data={values['data']:.6e} | "
                f"pde={values['pde']:.6e} | "
                f"bc={values['bc']:.6e}",
                flush=True
            )

    return history, device
