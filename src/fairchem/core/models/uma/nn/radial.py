"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _FrozenLinearInputPrefixFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, detached_inputs, input_prefix, weight, bias):
        ctx.save_for_backward(weight)
        ctx.prefix = input_prefix.shape[1]
        return torch.nn.functional.linear(detached_inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        grad_prefix = torch.mm(grad_output, weight[:, : ctx.prefix])
        return None, grad_prefix, None, None


def _frozen_linear_input_prefix(inputs, weight, bias, prefix):
    return _FrozenLinearInputPrefixFunction.apply(
        inputs.detach(), inputs[:, :prefix], weight, bias
    )


@torch.jit.script
def gaussian(x: torch.Tensor, mean, std) -> torch.Tensor:
    a = (2 * math.pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class PolynomialEnvelope(torch.nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff.
    """

    def __init__(self, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = 1 + (d_scaled**self.p) * (
            self.a + d_scaled * (self.b + self.c * d_scaled)
        )
        return torch.where(d_scaled < 1, env_val, 0)


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset, persistent=False)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RadialMLP(nn.Module):
    """
    Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels
    """

    def __init__(self, channels_list) -> None:
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue

            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]

            if i == len(channels_list) - 1:
                break

            modules.append(nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)
        self.first_linear_grad_prefix: int | None = None

    def configure_first_linear_grad_prefix(
        self, prefix: int, expected_input_features: int
    ) -> None:
        first_linear = self.net[0]
        if not isinstance(first_linear, nn.Linear):
            raise TypeError("radial first layer must be Linear")
        if first_linear.in_features != expected_input_features:
            raise ValueError("radial first-linear input width does not match x_edge")
        if not 0 < prefix < expected_input_features:
            raise ValueError("prefix must be between zero and the input width")
        self.first_linear_grad_prefix = prefix

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.first_linear_grad_prefix is None:
            return self.net(inputs)
        first_linear = self.net[0]
        hidden = _frozen_linear_input_prefix(
            inputs,
            first_linear.weight,
            first_linear.bias,
            self.first_linear_grad_prefix,
        )
        for index in range(1, len(self.net)):
            hidden = self.net[index](hidden)
        return hidden
