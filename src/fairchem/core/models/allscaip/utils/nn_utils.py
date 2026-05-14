from __future__ import annotations

import torch.nn as nn

from fairchem.core.models.escaip.utils.nn_utils import Activation, get_linear


def get_feedforward(
    hidden_dim: int,
    activation: Activation | None,
    hidden_layer_multiplier: int,
    bias: bool = False,
    dropout: float = 0.0,
    input_dim: int | None = None,
    output_dim: int | None = None,
):
    """
    Build a feedforward layer with optional activation function.
    """
    if hidden_layer_multiplier == 0:
        return get_linear(
            in_features=hidden_dim if input_dim is None else input_dim,
            out_features=hidden_dim if output_dim is None else output_dim,
            bias=bias,
            activation=None,
            dropout=dropout,
        )
    return nn.Sequential(
        get_linear(
            in_features=hidden_dim if input_dim is None else input_dim,
            out_features=hidden_dim * hidden_layer_multiplier,
            bias=bias,
            activation=activation,
            dropout=dropout,
        ),
        get_linear(
            in_features=hidden_dim * hidden_layer_multiplier,
            out_features=hidden_dim if output_dim is None else output_dim,
            bias=bias,
            activation=None,
            dropout=dropout,
        ),
    )
