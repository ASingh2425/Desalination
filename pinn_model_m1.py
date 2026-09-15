"""
Modification 1:
Residual/skip-connected architecture for the graphene nanochannel PINN.

This file keeps the original:
    inputs:  sigma, H, E, C, z
    outputs: rho_water, n_Na, n_Cl, u
    PDE: eta*d2u/dz2 + rho_e*E = 0
    BC: center-plane du/dz = 0

The ONLY modification is the neural-network architecture:
    Original: 3 sequential Dense(20)-Tanh layers
    M1:       3 residual blocks with 20 hidden features
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


class ResidualBlock(nn.Module):
    """
    Residual block:

        h_out = h + F(h)

    The input and output dimensions are both `features`.
    """

    def __init__(self, features: int):
        super().__init__()

        self.linear1 = nn.Linear(features, features)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(features, features)

        # Xavier initialization, matching the original model.
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return self.activation(residual + x)


class ElectroosmoticPINN_M1(nn.Module):
    """
    M1 residual PINN.

    Inputs:
        sigma, H, E, C, z

    Outputs:
        rho_water, n_Na, n_Cl, u
    """

    def __init__(self, config: PINNConfig | None = None):
        super().__init__()

        self.config = config or PINNConfig()

        # Input projection.
        self.input_layer = nn.Linear(
            5,
            self.config.neurons,
        )

        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)

        self.input_activation = nn.Tanh()

        # Three residual blocks, matching the original 3 hidden layers.
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(self.config.neurons)
                for _ in range(self.config.hidden_layers)
            ]
        )

        # Four physical outputs.
        self.output_layer = nn.Linear(
            self.config.neurons,
            4,
        )

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_layer(x)
        x = self.input_activation(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


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
    """

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

    d2u_scaled_dzscaled2 = _column(
        d2u_scaled_dx,
        4,
    )

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


def data_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> torch.Tensor:

    per_output = torch.mean(
        torch.square(y_true - y_pred),
        dim=0,
    )

    return torch.sum(per_output)


def bc_loss(
    du_dz: torch.Tensor,
) -> torch.Tensor:

    return torch.mean(
        torch.square(du_dz)
    )


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

    l_data = data_loss(
        y_data,
        y_pred,
    )

    residual, _, _ = pde_residual(
        model,
        x_collocation,
        viscosity,
        input_mean,
        input_scale,
        output_mean,
        output_scale,
    )

    l_pde = torch.mean(
        torch.square(residual)
    )

    _, du_dz_bc, _ = pde_residual(
        model,
        x_bc,
        viscosity,
        input_mean,
        input_scale,
        output_mean,
        output_scale,
    )

    l_bc = bc_loss(
        du_dz_bc
    )

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
):
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device_obj = torch.device(device)

    model = model.to(device_obj)

    dtype = (
        torch.float64
        if config.dtype == "float64"
        else torch.float32
    )

    model = model.to(dtype=dtype)

    x_data_t = torch.as_tensor(
        x_data,
        dtype=dtype,
        device=device_obj,
    )

    y_data_t = torch.as_tensor(
        y_data,
        dtype=dtype,
        device=device_obj,
    )

    x_col_t = torch.as_tensor(
        x_collocation,
        dtype=dtype,
        device=device_obj,
    )

    x_bc_t = torch.as_tensor(
        x_bc,
        dtype=dtype,
        device=device_obj,
    )

    input_mean_t = torch.as_tensor(
        input_mean,
        dtype=dtype,
        device=device_obj,
    )

    input_scale_t = torch.as_tensor(
        input_scale,
        dtype=dtype,
        device=device_obj,
    )

    output_mean_t = torch.as_tensor(
        output_mean,
        dtype=dtype,
        device=device_obj,
    )

    output_scale_t = torch.as_tensor(
        output_scale,
        dtype=dtype,
        device=device_obj,
    )

    # M1 intentionally keeps Adam.
    # Optimizer modification is M5 and will be tested separately.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    history = {
        "total": [],
        "data": [],
        "pde": [],
        "bc": [],
    }

    for epoch in range(
        1,
        config.epochs + 1,
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

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

        losses["total"].backward()

        optimizer.step()

        values = {
            k: float(
                v.detach().cpu()
            )
            for k, v in losses.items()
        }

        for k in history:
            history[k].append(
                values[k]
            )

        if (
            epoch == 1
            or epoch % config.print_every == 0
        ):
            print(
                f"Epoch {epoch:>7d}/"
                f"{config.epochs} | "
                f"total={values['total']:.6e} | "
                f"data={values['data']:.6e} | "
                f"pde={values['pde']:.6e} | "
                f"bc={values['bc']:.6e}"
            )

    return history, device


def save_checkpoint(
    model,
    config: PINNConfig,
    path: str,
):
    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "config":
                config.__dict__,
        },
        path,
    )


def load_checkpoint(
    path: str,
    device: str = "cpu",
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    config = PINNConfig(
        **checkpoint["config"]
    )

    model = ElectroosmoticPINN_M1(
        config
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, config