"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.profiler import record_function

from fairchem.core.common import gp_utils
from fairchem.core.models.puma.nn.activation import (
    GateActivation,
    SeparableS2Activation,
)
from fairchem.core.models.puma.nn.layer_norm import (
    get_normalization_layer,
)
from fairchem.core.models.puma.nn.radial import PolynomialEnvelope
from fairchem.core.models.puma.nn.so2_layers import SO2_Convolution
from fairchem.core.models.puma.nn.so3_layers import SO3_Linear


class Edgewise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list,
        mappingReduced,
        SO3_grid,
        cutoff,
        act_type="gate",
    ):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.act_type = act_type

        if self.act_type == "gate":
            self.act = GateActivation(
                lmax=self.lmax, mmax=self.mmax, num_channels=self.hidden_channels
            )
            extra_m0_output_channels = self.lmax * self.hidden_channels
        elif self.act_type == "s2":
            # NOTE: this is the only place where the SO3 grid of the edges (lmax/mmax) is used
            self.act = SeparableS2Activation(
                lmax=self.lmax, mmax=self.mmax, SO3_grid=self.SO3_grid
            )
            extra_m0_output_channels = self.hidden_channels
        else:
            raise ValueError(f"Unknown activation type {self.act_type}")

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list,
            extra_m0_output_channels=extra_m0_output_channels,
        )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.sphere_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )

        self.cutoff = cutoff
        self.envelope = PolynomialEnvelope(exponent=5)

        self.out_mask = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )

    def forward(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
        node_offset: int = 0,
    ):
        if gp_utils.initialized():
            x_full = gp_utils.gather_from_model_parallel_region_sum_grad(x, dim=0)
            x_source = x_full[edge_index[0]]
            x_target = x_full[edge_index[1]]
        else:
            x_source = x[edge_index[0]]
            x_target = x[edge_index[1]]

        x_message = torch.cat((x_source, x_target), dim=2)
        # NOTE if strictly local
        # x_message = x[edge_index[0]]

        with record_function("SO2Conv"):
            # Rotate the irreps to align with the edge
            x_message = torch.bmm(wigner[:, self.out_mask, :], x_message)

            # SO2 convolution
            x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
            x_message = self.act(x_0_gating, x_message)
            x_message = self.so2_conv_2(x_message, x_edge)

            # envelope
            dist_scaled = edge_distance / self.cutoff
            env = self.envelope(dist_scaled)
            x_message = x_message * env.view(-1, 1, 1)

            # Rotate back the irreps
            x_message = torch.bmm(wigner_inv[:, :, self.out_mask], x_message)

        # Compute the sum of the incoming neighboring messages for each target node
        new_embedding = torch.zeros(
            (x.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )

        new_embedding.index_add_(0, edge_index[1] - node_offset, x_message)

        return new_embedding


class SpectralAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        self.scalar_mlp = nn.Sequential(
            nn.Linear(
                self.sphere_channels,
                self.lmax * self.hidden_channels,
                bias=True,
            ),
            nn.SiLU(),
        )

        self.so3_linear_1 = SO3_Linear(
            self.sphere_channels, self.hidden_channels, lmax=self.lmax
        )
        self.act = GateActivation(
            lmax=self.lmax, mmax=self.lmax, num_channels=self.hidden_channels
        )
        self.so3_linear_2 = SO3_Linear(
            self.hidden_channels, self.sphere_channels, lmax=self.lmax
        )

    def forward(self, x):
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))
        x = self.so3_linear_1(x)
        x = self.act(gating_scalars, x)
        x = self.so3_linear_2(x)
        return x


class GridAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        self.grid_mlp = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.sphere_channels, bias=False),
        )

    def forward(self, x):
        # Project to grid
        x_grid = self.SO3_grid["lmax_lmax"].to_grid(x, self.lmax, self.lmax)
        # Perform point-wise operations
        x_grid = self.grid_mlp(x_grid)
        # Project back to spherical harmonic coefficients
        x = self.SO3_grid["lmax_lmax"].from_grid(x_grid, self.lmax, self.lmax)
        return x


class eSCNMD_Block(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced,
        SO3_grid,
        edge_channels_list: list[int],
        cutoff: float,
        norm_type: str,
        act_type: str,
        ff_type: str,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.norm_1 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )

        self.edge_wise = Edgewise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            cutoff=cutoff,
            act_type=act_type,
        )

        self.norm_2 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )

        if ff_type == "spectral":
            self.atom_wise = SpectralAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )
        elif ff_type == "grid":
            self.atom_wise = GridAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )

    def forward(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
        sys_node_embedding=None,
        node_offset: int = 0,
    ):
        x_res = x
        x = self.norm_1(x)

        if sys_node_embedding is not None:
            x[:, 0, :] = x[:, 0, :] + sys_node_embedding

        with record_function("edgewise"):
            x = self.edge_wise(
                x,
                x_edge,
                edge_distance,
                edge_index,
                wigner,
                wigner_inv,
                node_offset,
            )
            x = x + x_res

        x_res = x
        x = self.norm_2(x)

        with record_function("atomwise"):
            x = self.atom_wise(x)
            x = x + x_res

        return x
