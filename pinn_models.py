from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

# Reuse the PINNConfig defined for other models (same fields expected)
from pinn_model_m1 import PINNConfig


class PIDeepONet(nn.Module):
    """A minimal Physics-Informed DeepONet.

    Branch net processes the global condition vector (sigma,H,E,C) and trunk
    net processes the local coordinate z. The final output is the dot
    (inner) product of branch and trunk features, then a small head maps to
    physical outputs.
    """

    def __init__(self, config: PINNConfig | None = None, branch_dim: int = 64, trunk_dim: int = 64):
        super().__init__()
        self.config = config or PINNConfig()

        # Branch: processes operating conditions (sigma,H,E,C) -> branch_dim
        self.branch = nn.Sequential(
            nn.Linear(4, branch_dim),
            nn.Tanh(),
            nn.Linear(branch_dim, branch_dim),
            nn.Tanh(),
        )

        # Trunk: processes z coordinate -> trunk_dim
        self.trunk = nn.Sequential(
            nn.Linear(1, trunk_dim),
            nn.Tanh(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.Tanh(),
        )

        # If branch and trunk dims differ, add a small projection from trunk_dim->branch_dim
        self.trunk_proj = None
        if trunk_dim != branch_dim:
            self.trunk_proj = nn.Linear(trunk_dim, branch_dim)
            nn.init.xavier_uniform_(self.trunk_proj.weight)
            nn.init.zeros_(self.trunk_proj.bias)

        # Final linear head mapping inner product to 4 outputs
        self.head = nn.Linear(branch_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 5] -> [sigma,H,E,C,z]
        cond = x[:, :4]
        z = x[:, 4:5]

        b = self.branch(cond)  # [N, branch_dim]
        t = self.trunk(z)      # [N, trunk_dim]

        # If dims differ, project trunk to branch_dim using learned projection if present
        if self.trunk_proj is not None:
            t = self.trunk_proj(t)

        # inner product per sample -> [N, branch_dim] * [N, branch_dim] -> [N, branch_dim]
        features = b * t
        out = self.head(features)
        return out


class PINO(nn.Module):
    """A lightweight Physics-Informed Neural Operator inspired model.

    Implements a conditioner for the global operating conditions and a trunk
    network for the local coordinate, combined with FiLM-style modulation.
    """

    def __init__(self, config: PINNConfig | None = None, hidden: int = 64, layers: int = 4):
        super().__init__()
        self.config = config or PINNConfig()

        # Conditioner for global conditions
        self.cond_net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # FiLM generators
        self.scale_gen = nn.Linear(hidden, hidden)
        self.shift_gen = nn.Linear(hidden, hidden)

        # Trunk network for z with modulation
        trunk = []
        trunk.append(nn.Linear(1, hidden))
        for _ in range(layers - 1):
            trunk.append(nn.Tanh())
            trunk.append(nn.Linear(hidden, hidden))
        self.trunk = nn.ModuleList(trunk)

        self.head = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cond = x[:, :4]
        z = x[:, 4:5]

        c = self.cond_net(cond)  # [N, hidden]
        scale = self.scale_gen(c)
        shift = self.shift_gen(c)

        h = z
        # First linear
        h = self.trunk[0](h)
        # Apply FiLM
        h = h * (1.0 + scale) + shift
        # Remaining trunk layers
        for layer in self.trunk[1:]:
            h = layer(h)

        out = self.head(h)
        return out


class RAR_PIQNN(nn.Module):
    """Residual Adaptive-Refinement like PI-QNN (simple partitioned experts).

    This model splits the z domain into K overlapping experts. Each expert
    is a small PINN-like MLP and outputs are blended with smooth window
    functions so that the combined output is continuous.
    """

    def __init__(self, config: PINNConfig | None = None, experts: int = 4, hidden: int = 48):
        super().__init__()
        self.config = config or PINNConfig()
        self.experts = nn.ModuleList()
        self.centers = torch.linspace(-0.5, 0.5, experts).unsqueeze(1)  # relative centers

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
        # Smooth bump using Gaussian window
        return torch.exp(-((z - center.to(z.device)) ** 2) / (2 * width * width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[:, 4:5]
        outs = []
        weights = []
        for i, net in enumerate(self.experts):
            out = net(x)
            w = self._window(z, self.centers[i])
            outs.append(out * w)
            weights.append(w)

        denom = torch.sum(torch.stack(weights, dim=0), dim=0)
        denom = torch.clamp(denom, min=1e-6)
        combined = torch.sum(torch.stack(outs, dim=0), dim=0) / denom
        return combined


class XPINN(nn.Module):
    """Domain-decomposition XPINN with two subdomains split at z=0.

    Two sub-networks handle z<0 and z>=0; final output is blended by a smooth
    partition of unity so gradients remain defined.
    """

    def __init__(self, config: PINNConfig | None = None, hidden: int = 64):
        super().__init__()
        self.config = config or PINNConfig()

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
        # Smooth indicator: sigmoid around 0 with small slope for smoothness
        s = torch.sigmoid(z * 50.0)
        left_out = self.left(x)
        right_out = self.right(x)
        return (1.0 - s) * left_out + s * right_out


class FBPINN(nn.Module):
    """A simple overlapping-windowed FBPINN with partition-of-unity.

    The domain in z is divided into M windows; each window has a local
    network that sees the global input but is multiplied by a compactly
    supported bump function. This is a basic FBPINN-like construction.
    """

    def __init__(self, config: PINNConfig | None = None, windows: int = 6, hidden: int = 48):
        super().__init__()
        self.config = config or PINNConfig()
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
        # Compact-ish smooth bump using raised cosine
        d = torch.abs(z - center.to(z.device)) / width
        w = torch.clamp(0.5 * (1 + torch.cos(torch.clamp(d, 0.0, 1.0) * torch.pi)), min=0.0)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[:, 4:5]
        outs = []
        weights = []
        for i, net in enumerate(self.workers):
            c = self._centers[i:i+1]
            w = self._bump(z, c)
            weights.append(w)
            outs.append(net(x) * w)

        denom = torch.sum(torch.stack(weights, dim=0), dim=0)
        denom = torch.clamp(denom, min=1e-6)
        combined = torch.sum(torch.stack(outs, dim=0), dim=0) / denom
        return combined


def get_model(name: str, config: PINNConfig | None = None) -> nn.Module:
    """Factory returning a model by name.

    Supported names (case-insensitive):
      - 'pi-deeponet' or 'deepONet'
      - 'pino'
      - 'rar-piqnn' or 'rar'
      - 'xpinn' or 'x-pinn'
      - 'fbpinn' or 'fb-pinn'
    """
    n = name.lower()
    if n in ("pi-deeponet", "deeponet", "pi_deeponet"):
        return PIDeepONet(config)
    if n == "pino":
        return PINO(config)
    if n in ("rar-piqnn", "rar", "piqnn"):
        return RAR_PIQNN(config)
    if n in ("xpinn", "x-pinn"):
        return XPINN(config)
    if n in ("fbpinn", "fb-pinn"):
        return FBPINN(config)
    raise ValueError(f"Unknown model name: {name}")
