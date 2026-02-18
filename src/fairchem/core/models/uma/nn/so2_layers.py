"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import Linear

from .radial import RadialMLP

if TYPE_CHECKING:
    from fairchem.core.models.uma.common.so3 import CoefficientMapping


class SO2_m_Conv(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
    """

    def __init__(
        self,
        m: int,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
    ) -> None:
        super().__init__()

        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax

        assert self.mmax >= m
        num_coefficents = self.lmax - m + 1
        num_channels = num_coefficents * self.sphere_channels

        self.out_channels_half = self.m_output_channels * (
            num_channels // self.sphere_channels
        )
        self.fc = Linear(
            num_channels,
            2 * self.out_channels_half,
            bias=False,
        )
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_m = self.fc(x_m)
        x_r_0, x_i_0, x_r_1, x_i_1 = x_m.reshape(
            x_m.shape[0], -1, self.out_channels_half
        ).split(1, dim=1)
        x_m_r = x_r_0 - x_i_1  # x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r_1 + x_i_0  # x_r[:, 1] + x_i[:, 0]
        return (
            x_m_r.view(x_m.shape[0], -1, self.m_output_channels),
            x_m_i.view(x_m.shape[0], -1, self.m_output_channels),
        )


class SO2_m_Conv_Block(torch.nn.Module):
    """
    SO(2) Conv with block matrix formulation for m > 0.

    Uses a single larger GEMM instead of batched GEMM:
        [x_real, x_imag] @ [[W1, -W2], [W2, W1]]^T -> [out_real, out_imag]

    This can be 1.3-1.5x faster than the standard batched approach
    due to better tensor core utilization.

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
    """

    def __init__(
        self,
        m: int,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
    ) -> None:
        super().__init__()

        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax

        assert self.mmax >= m
        num_coefficents = self.lmax - m + 1
        num_channels = num_coefficents * self.sphere_channels

        self.out_channels_half = self.m_output_channels * (
            num_channels // self.sphere_channels
        )
        self.in_size = num_channels
        self.num_l = self.out_channels_half // self.m_output_channels

        # Original weight layout: [2*out_half, in_size]
        # where W1=[:out_half], W2=[out_half:]
        # Store fc for weight access (used by other parts of codebase)
        self.fc = Linear(
            num_channels,
            2 * self.out_channels_half,
            bias=False,
        )
        self.fc.weight.data.mul_(1 / math.sqrt(2))

        # Register buffer for cached W_block (built lazily on first
        # forward)
        self.register_buffer("_w_block", None, persistent=False)

    @torch.no_grad()
    def _build_w_block(self) -> None:
        """
        Build and cache the block matrix from fc weights.

        This converts from standard weight layout:
            W1 = fc.weight[:out_half]  (real-to-real, imag-to-imag)
            W2 = fc.weight[out_half:]  (cross terms)

        To block matrix:
            [[W1, -W2], [W2, W1]]

        Called lazily on first forward or explicitly after weight
        loading.
        """
        W1, W2 = self.fc.weight.split(
            self.out_channels_half, dim=0
        )  # Each [out_half, in_size]
        self._w_block = torch.cat(
            [
                torch.cat([W1, -W2], dim=1),
                torch.cat([W2, W1], dim=1),
            ],
            dim=0,
        ).contiguous()  # [2*out_half, 2*in_size]

    def forward(self, x_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using block matrix GEMM.

        Args:
            x_m: [E, 2, in_size] where dim 1 is [real, imag]

        Returns:
            (out_real, out_imag): each [E, num_l, m_output_channels]
        """
        # Use cached W_block if available, otherwise build it
        if self._w_block is None:
            self._build_w_block()

        # Flatten [E, 2, in_size] -> [E, 2*in_size]
        x_cat = x_m.flatten(1)

        # Single GEMM: [E, 2*in] @ [2*in, 2*out_half] = [E, 2*out_half]
        out_cat = x_cat @ self._w_block.T

        # View to [E, 2, num_l, m_output_channels] and unbind
        return out_cat.view(-1, 2, self.num_l, self.m_output_channels).unbind(1)

    def forward_fused(self, x_m: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning fused real/imag output (no unbind).

        Args:
            x_m: [E, 2, in_size] where dim 1 is [real, imag]

        Returns:
            out: [E, 2*num_l, m_output_channels] - real and imag
                interleaved in dim 1
        """
        if self._w_block is None:
            self._build_w_block()

        out_cat = x_m.flatten(1) @ self._w_block.T
        return out_cat.view(-1, 2 * self.num_l, self.m_output_channels)


class SO2_Conv1_WithRadialBlock(torch.nn.Module):
    """
    Specialized first SO2 convolution with:
    - Precomputed radial embeddings (x * radial)
    - Extra m0 output channels for gating
    - Block matrix GEMM for m>0

    REQUIRES: unify_radial_mlp=True (radial from UnifiedRadialMLP)
    No conditional branches - optimized for torch.compile.

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
        mappingReduced (CoefficientMapping): Used to extract a subset
            of m components
        extra_m0_output_channels (int): Extra output channels for
            gating
        edge_channels_list (list[int]): Edge embedding channels
            (for checkpoint compatibility only)
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        extra_m0_output_channels: int,
        edge_channels_list: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.m_output_channels = m_output_channels
        self.extra_m0_output_channels = extra_m0_output_channels

        num_channels_m0 = (lmax + 1) * sphere_channels

        # SO(2) convolution for m = 0 (includes extra output channels
        # for gating)
        m0_output_channels = m_output_channels * (lmax + 1) + extra_m0_output_channels
        self.fc_m0 = Linear(num_channels_m0, m0_output_channels)

        # SO(2) convolution for non-zero m using block GEMM
        self.so2_m_conv = nn.ModuleList()
        for m in range(1, mmax + 1):
            self.so2_m_conv.append(
                SO2_m_Conv_Block(
                    m,
                    sphere_channels,
                    m_output_channels,
                    lmax,
                    mmax,
                )
            )

        # Split sizes for input x (by m)
        self.m_split_sizes = [mappingReduced.m_size[0]] + [
            s * 2 for s in mappingReduced.m_size[1:]
        ]

        # Split sizes for radial (by m) - used by UnifiedRadialMLP
        self.edge_split_sizes = [self.fc_m0.in_features] + [
            mod.fc.in_features for mod in self.so2_m_conv
        ]

        # rad_func: NOT used in forward(), but required for checkpoint
        # loading. Weights are loaded here and extracted by
        # UnifiedRadialMLP.
        if edge_channels_list is not None:
            edge_channels_list = copy.deepcopy(edge_channels_list)
            edge_channels_list.append(sum(self.edge_split_sizes))
            self.rad_func = RadialMLP(edge_channels_list)

    def forward(
        self,
        x: torch.Tensor,
        precomputed_radial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with precomputed radial.

        Args:
            x: Input features [E, coeffs, channels]
            precomputed_radial: Radial weights from UnifiedRadialMLP
                [E, rad_features]
                REQUIRED - this class does not have its own rad_func

        Returns:
            (out, x_0_extra): out is [E, coeffs, m_out_channels]
                              x_0_extra is [E, extra_m0_output_channels]
        """
        # Split precomputed radial by m
        x_edge_by_m = precomputed_radial.split(self.edge_split_sizes, dim=1)
        x_by_m = x.split(self.m_split_sizes, dim=1)

        # m=0: apply radial, linear, split gating extras
        x_0 = x_by_m[0].flatten(1) * x_edge_by_m[0]
        x_0 = self.fc_m0(x_0)
        x_0_extra, x_0 = x_0.split(
            [
                self.extra_m0_output_channels,
                self.fc_m0.out_features - self.extra_m0_output_channels,
            ],
            dim=-1,
        )
        out = [x_0.view(-1, self.lmax + 1, self.m_output_channels)]

        # m>0: apply radial, block GEMM (fused output avoids unbind +
        # cat overhead)
        for m in range(1, self.mmax + 1):
            x_m = x_by_m[m].view(-1, 2, self.so2_m_conv[m - 1].in_size) * x_edge_by_m[
                m
            ].unsqueeze(1)
            out.append(self.so2_m_conv[m - 1].forward_fused(x_m))

        return torch.cat(out, dim=1), x_0_extra


class SO2_Conv2_InternalBlock(torch.nn.Module):
    """
    Specialized second SO2 convolution with:
    - Internal weights only (no radial)
    - No extra m0 output channels
    - Block matrix GEMM for m>0

    Simpler forward pass - no radial multiply, no gating split.

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
        mappingReduced (CoefficientMapping): Used to extract a subset
            of m components
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
    ) -> None:
        super().__init__()
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax

        num_channels_m0 = (lmax + 1) * sphere_channels

        # SO(2) convolution for m = 0 (no extras)
        self.fc_m0 = Linear(num_channels_m0, m_output_channels * (lmax + 1))

        # SO(2) convolution for non-zero m using block GEMM
        self.so2_m_conv = nn.ModuleList()
        for m in range(1, mmax + 1):
            self.so2_m_conv.append(
                SO2_m_Conv_Block(
                    m,
                    sphere_channels,
                    m_output_channels,
                    lmax,
                    mmax,
                )
            )

        # Split sizes for input x (by m)
        self.m_split_sizes = [mappingReduced.m_size[0]] + [
            s * 2 for s in mappingReduced.m_size[1:]
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - internal weights only, no radial.

        Args:
            x: [E, coeffs, channels]

        Returns:
            out: [E, coeffs, m_output_channels]
        """
        x_by_m = x.split(self.m_split_sizes, dim=1)

        # m=0: just linear
        x_0 = x_by_m[0].flatten(1)
        x_0 = self.fc_m0(x_0)
        out = [x_0.view(-1, self.lmax + 1, self.m_output_channels)]

        # m>0: just block GEMM (fused output avoids unbind + cat
        # overhead)
        for m in range(1, self.mmax + 1):
            x_m = x_by_m[m].view(-1, 2, self.so2_m_conv[m - 1].in_size)
            out.append(self.so2_m_conv[m - 1].forward_fused(x_m))

        return torch.cat(out, dim=1)


class SO2_Convolution(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    This is the general-purpose SO2 convolution used by the baseline
    path. For optimized inference, use SO2_Conv1_WithRadialBlock and
    SO2_Conv2_InternalBlock.

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
        mappingReduced (CoefficientMapping): Used to extract a subset of m components
        internal_weights (bool):    If True, not using radial function to multiply inputs features
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
        extra_m0_output_channels (int): If not None, return `out_embedding` and `extra_m0_features` (Tensor).
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        internal_weights: bool = True,
        edge_channels_list: list[int] | None = None,
        extra_m0_output_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.internal_weights = internal_weights
        self.extra_m0_output_channels = extra_m0_output_channels

        num_channels_m0 = (self.lmax + 1) * self.sphere_channels

        # SO(2) convolution for m = 0
        m0_output_channels = self.m_output_channels * (
            num_channels_m0 // self.sphere_channels
        )
        if self.extra_m0_output_channels is not None:
            m0_output_channels = m0_output_channels + self.extra_m0_output_channels
        self.fc_m0 = Linear(num_channels_m0, m0_output_channels)
        num_channels_rad = self.fc_m0.in_features

        # SO(2) convolution for non-zero m
        self.so2_m_conv = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.so2_m_conv.append(
                SO2_m_Conv(
                    m,
                    self.sphere_channels,
                    self.m_output_channels,
                    self.lmax,
                    self.mmax,
                )
            )
            num_channels_rad = num_channels_rad + self.so2_m_conv[-1].fc.in_features

        # Embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            assert edge_channels_list is not None
            edge_channels_list = copy.deepcopy(edge_channels_list)
            edge_channels_list.append(int(num_channels_rad))
            # This can moved outside of SO2 conv and into Edgewise
            self.rad_func = RadialMLP(edge_channels_list)

        self.m_split_sizes = [self.mappingReduced.m_size[0]] + (
            torch.tensor(self.mappingReduced.m_size[1:]) * 2
        ).tolist()
        self.edge_split_sizes = [self.fc_m0.in_features] + [
            mod.fc.in_features for mod in self.so2_m_conv
        ]

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for SO2 convolution.

        Args:
            x: Input features [E, coeffs, channels]
            x_edge: Edge embeddings [E, edge_features] - used if
                rad_func is not None

        Returns:
            Output features, and optionally extra m0 features for
            gating
        """
        # Compute radial weights if needed
        if self.rad_func is not None:
            x_edge_by_m = self.rad_func(x_edge).split(self.edge_split_sizes, dim=1)

        x_by_m = x.split(self.m_split_sizes, dim=1)
        num_edges = x.shape[0]

        # m=0: only real values (no imaginary)
        x_0 = x_by_m[0].view(num_edges, -1)
        if self.rad_func is not None:
            x_0 = x_0 * x_edge_by_m[0]
        x_0 = self.fc_m0(x_0)

        # Extract extra m0 features for gating
        if self.extra_m0_output_channels is not None:
            x_0_extra, x_0 = x_0.split(
                [
                    self.extra_m0_output_channels,
                    self.fc_m0.out_features - self.extra_m0_output_channels,
                ],
                dim=-1,
            )

        out = [x_0.view(num_edges, -1, self.m_output_channels)]

        # m > 0: complex values (real + imaginary)
        for m in range(1, self.mmax + 1):
            x_m = x_by_m[m].view(num_edges, 2, -1)
            if self.rad_func is not None:
                x_m = x_m * x_edge_by_m[m].unsqueeze(1)
            x_m = self.so2_m_conv[m - 1](x_m)
            out.extend(x_m)

        out = torch.cat(out, dim=1)

        if self.extra_m0_output_channels is not None:
            return out, x_0_extra
        return out


class SO2_Linear(torch.nn.Module):
    """
    SO(2) Linear: Perform SO(2) linear for all m (orders).

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)
        mappingReduced (CoefficientMapping): Used to extract a subset of m components
        internal_weights (bool):    If True, not using radial function to multiply inputs features
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
    """

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        internal_weights: bool = False,
        edge_channels_list: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.internal_weights = internal_weights
        self.edge_channels_list = copy.deepcopy(edge_channels_list)

        num_channels_m0 = (self.lmax + 1) * self.sphere_channels

        # SO(2) linear for m = 0
        self.fc_m0 = Linear(
            num_channels_m0,
            self.m_output_channels * (num_channels_m0 // self.sphere_channels),
        )
        num_channels_rad = self.fc_m0.in_features

        # SO(2) linear for non-zero m
        self.so2_m_fc = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            num_coefficents = self.lmax - m + 1
            num_in_channels = num_coefficents * self.sphere_channels
            fc = Linear(
                num_in_channels,
                self.m_output_channels * (num_in_channels // self.sphere_channels),
                bias=False,
            )
            num_channels_rad = num_channels_rad + fc.in_features
            self.so2_m_fc.append(fc)

        # Embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            assert self.edge_channels_list is not None
            self.edge_channels_list.append(int(num_channels_rad))
            self.rad_func = RadialMLP(self.edge_channels_list)

    def forward(self, x: torch.Tensor, x_edge: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out = []

        # Reshape the spherical harmonics based on m (order)
        x = torch.einsum("nac,ba->nbc", x, self.mappingReduced.to_m)

        # radial function
        if self.rad_func is not None:
            x_edge = self.rad_func(x_edge)
        offset_rad = 0

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_0 = x.narrow(1, 0, self.mappingReduced.m_size[0])
        x_0 = x_0.reshape(batch_size, -1)
        if self.rad_func is not None:
            x_edge_0 = x_edge.narrow(1, 0, self.fc_m0.in_features)
            x_0 = x_0 * x_edge_0
        x_0 = self.fc_m0(x_0)
        x_0 = x_0.view(batch_size, -1, self.m_output_channels)
        out.append(x_0)
        offset_rad = offset_rad + self.fc_m0.in_features

        # Compute the values for the m > 0 coefficients
        offset = self.mappingReduced.m_size[0]
        for m in range(1, self.mmax + 1):
            # Get the m order coefficients
            x_m = x.narrow(1, offset, 2 * self.mappingReduced.m_size[m])
            x_m = x_m.reshape(batch_size, 2, -1)
            if self.rad_func is not None:
                x_edge_m = x_edge.narrow(
                    1, offset_rad, self.so2_m_fc[m - 1].in_features
                )
                x_edge_m = x_edge_m.reshape(
                    batch_size, 1, self.so2_m_fc[m - 1].in_features
                )
                x_m = x_m * x_edge_m

            # Perform SO(2) linear
            x_m = self.so2_m_fc[m - 1](x_m)
            x_m = x_m.view(batch_size, -1, self.m_output_channels)
            out.append(x_m)

            offset = offset + 2 * self.mappingReduced.m_size[m]
            offset_rad = offset_rad + self.so2_m_fc[m - 1].in_features

        out = torch.cat(out, dim=1)

        # Reshape the spherical harmonics based on l (degree)
        out = torch.einsum("nac,ab->nbc", out, self.mappingReduced.to_m)
        return out
