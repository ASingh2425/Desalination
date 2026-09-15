"""
PyTorch PINN reconstruction for electroosmotic flow in graphene nanochannels.

This implementation intentionally uses PyTorch instead of TensorFlow so that
it can run in the user's Python 3.14 environment. The scientific model is
kept aligned with the paper reconstruction:
    inputs:  sigma, H, E, C, z
    outputs: rho_water, n_Na, n_Cl, u
    network: 3 hidden layers x 20 neurons, tanh, linear 4-output layer
    loss: data MSE + PDE residual MSE + center-plane BC MSE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn

E_CHARGE = 1.602176634e-19
F_CHARACTERISTIC = 1.602176634e17  # Characteristic electroosmotic force scale (N/m^3)


@dataclass
class PINNConfig:
    hidden_layers: int = 3
    neurons: int = 20
    learning_rate: float = 1e-4
    viscosity: float = 8.9e-4
    epochs: int = 100_000
    print_every: int = 1000
    dtype: str = "float32"


class ElectroosmoticPINN(nn.Module):
    def __init__(self, config: PINNConfig | None = None):
        super().__init__()
        self.config = config or PINNConfig()

        layers = []
        in_features = 5
        for i in range(self.config.hidden_layers):
            layers.append(nn.Linear(in_features, self.config.neurons))
            layers.append(nn.Tanh())
            in_features = self.config.neurons
        layers.append(nn.Linear(in_features, 4))
        self.network = nn.Sequential(*layers)

        # Match the common Xavier/Glorot initialization used by the original
        # TensorFlow/Keras implementation.
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _column(x: torch.Tensor, index: int) -> torch.Tensor:
    return x[:, index:index + 1]


def pde_residual(
    model: nn.Module,
    x: torch.Tensor,
    viscosity: float,
    input_mean: torch.Tensor,
    input_scale: torch.Tensor,
    output_mean: torch.Tensor,
    output_scale: torch.Tensor,
):
    """
    Physical PDE residual:

        eta * d2u/dz2 + rho_e * E = 0

    The network receives standardized [sigma, H, E, C, z].
    The chain rule converts the derivative of standardized velocity with
    respect to standardized z back to physical du/dz and d2u/dz2.
    """
    # A leaf tensor is required for autograd. Do not modify the caller's tensor.
    x = x.clone().detach().requires_grad_(True)

    y_scaled = model(x)
    u_scaled = _column(y_scaled, 3)

    # Because samples are independent, differentiating the sum of the scalar
    # outputs gives the per-row derivative for this feed-forward network.
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

    z_scale_m = input_scale[4:5] * 1e-9  # convert z scale from nm to m
    u_scale_m = output_scale[3:4]        # m/s

    du_dz = du_scaled_dzscaled  # dimensionless derivative for BC loss in standardized units
    d2u_dz2_SI = (u_scale_m / torch.square(z_scale_m)) * d2u_scaled_dzscaled2

    n_na_m3 = (_column(y_scaled, 1) * output_scale[1:2] + output_mean[1:2]) * 1e27  # convert nm^-3 to m^-3
    n_cl_m3 = (_column(y_scaled, 2) * output_scale[2:3] + output_mean[2:3]) * 1e27  # convert nm^-3 to m^-3

    rho_e_SI = E_CHARGE * (n_na_m3 - n_cl_m3)  # C / m^3

    electric_field_SI = (_column(x, 2) * input_scale[2:3] + input_mean[2:3]) * 1e9  # convert V/nm to V/m

    residual = (viscosity * d2u_dz2_SI + rho_e_SI * electric_field_SI) / F_CHARACTERISTIC

    return residual, du_dz, y_scaled


def data_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # Sum of four MSE terms, following the paper description.
    per_output = torch.mean(torch.square(y_true - y_pred), dim=0)
    return torch.sum(per_output)


def bc_loss(du_dz: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.square(du_dz))


def pinn_losses(
    model,
    x_data,
    y_data,
    x_collocation,
    x_bc,
    viscosity,
    input_mean,
    input_scale,
    output_mean,
    output_scale,
):
    y_pred = model(x_data)
    l_data = data_loss(y_data, y_pred)

    residual, _, _ = pde_residual(
        model,
        x_collocation,
        viscosity,
        input_mean,
        input_scale,
        output_mean,
        output_scale,
    )
    l_pde = torch.mean(torch.square(residual))

    _, du_dz_bc, _ = pde_residual(
        model,
        x_bc,
        viscosity,
        input_mean,
        input_scale,
        output_mean,
        output_scale,
    )
    l_bc = bc_loss(du_dz_bc)

    return {
        "total": l_data + l_pde + l_bc,
        "data": l_data,
        "pde": l_pde,
        "bc": l_bc,
    }


def train_pinn(
    model,
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_collocation: np.ndarray,
    x_bc: np.ndarray,
    config: PINNConfig,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    output_mean: np.ndarray,
    output_scale: np.ndarray,
    device: str = "auto",
    loss_mode: str = "pinn",  # 'pinn' or 'data' (data-only)
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

        losses = pinn_losses(
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

        # Support data-only training where only the data-matching MSE is used
        # to form the optimization objective while still computing PDE/BC
        # components for monitoring.
        if loss_mode == "data":
            total = losses["data"]
        else:
            total = (
                losses["data"]
                + pde_weight * losses["pde"]
                + bc_weight * losses["bc"]
            )

        # Replace total in the dict so logging and history reflect the value
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
                f"bc={values['bc']:.6e}"
            )

    return history, device


def save_checkpoint(model, config: PINNConfig, path: str):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = PINNConfig(**checkpoint["config"])
    model = ElectroosmoticPINN(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config
