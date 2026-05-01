"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.profiler import record_function
from typing_extensions import Literal

from fairchem.core.common import gp_utils
from fairchem.core.models.uma.graph_parallel import (
    GPContext,
    all_to_all_collect,
    all_to_all_collect_compiled,
    finish_all_to_all_collect,
    start_all_to_all_collect,
)
from fairchem.core.models.uma.nn.activation import (
    GateActivation,
    SeparableS2Activation_M,
)
from fairchem.core.models.uma.nn.layer_norm import (
    get_normalization_layer,
)
from fairchem.core.models.uma.nn.mole import MOLE
from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear

if TYPE_CHECKING:
    from fairchem.core.models.uma.common.so3 import CoefficientMapping, SO3_Grid
    from fairchem.core.models.uma.nn.execution_backends import ExecutionBackend


def set_mole_ac_start_index(module: nn.Module, index: int) -> None:
    for submodule in module.modules():
        if isinstance(submodule, MOLE):
            submodule.global_mole_tensors.ac_start_idx = index


class Edgewise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list: list[int],
        mappingReduced: CoefficientMapping,
        SO3_grid: SO3_Grid,
        cutoff: float,
        # Enables activation checkpointing of edges in
        # activation_checkpoint_chunk_size size edge blocks
        activation_checkpoint_chunk_size: int | None,
        backend: ExecutionBackend,
        act_type: Literal["gate", "s2"] = "gate",
        use_overlap_gp: bool = False,
        use_p2p_gp: bool = False,
    ):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.activation_checkpoint_chunk_size = activation_checkpoint_chunk_size
        self.use_overlap_gp = use_overlap_gp
        self.use_p2p_gp = use_p2p_gp
        self.backend = backend

        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        self.act_type = act_type

        if self.act_type == "gate":
            self.act = GateActivation(
                lmax=self.lmax,
                mmax=self.mmax,
                num_channels=self.hidden_channels,
                m_prime=True,
            )
            extra_m0_output_channels = self.lmax * self.hidden_channels
        elif self.act_type == "s2":
            # NOTE: this is the only place where the SO3 grid of the
            # edges (lmax/mmax) is used
            self.act = SeparableS2Activation_M(
                lmax=self.lmax,
                mmax=self.mmax,
                SO3_grid=self.SO3_grid,
                to_m=self.mappingReduced.to_m,
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
            edge_channels_list=copy.deepcopy(edge_channels_list),
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

    def forward(
        self,
        x,
        x_edge,
        edge_index,
        wigner,
        wigner_inv_envelope,
        total_atoms_across_gp_ranks,
        node_offset: int = 0,
        gp_ctx: GPContext | None = None,
        send_indices: torch.Tensor | None = None,
    ):
        """
        Forward pass with support for both all-gather and all-to-all GP.

        When gp_ctx is provided, uses all-to-all to collect only the
        needed remote embeddings. Otherwise falls back to all-gather.

        When use_overlap_gp is True and in eval mode, overlaps
        communication with local edge computation for better latency.
        """
        # Check if we should use the overlapped path:
        # - gp_ctx must be provided (all-to-all mode)
        # - use_overlap_gp must be enabled
        # - must NOT be in training mode (overlap path doesn't support autograd)
        # - must NOT need gradients (autograd forces/stress require
        #   autograd-compatible communication, overlap path doesn't provide this)
        # - must NOT use activation checkpointing (incompatible with edge split)
        # - must have both local and boundary edges
        needs_grad_for_overlap = torch.is_grad_enabled() and (
            x.requires_grad if isinstance(x, torch.Tensor) else False
        )
        use_overlap = (
            self.use_overlap_gp
            and gp_ctx is not None
            and gp_utils.initialized()
            and not self.training
            and not needs_grad_for_overlap
            and self.activation_checkpoint_chunk_size is None
            and gp_ctx.local_edge_mask is not None
            and gp_ctx.num_local_edges > 0
            and gp_ctx.num_boundary_edges > 0
        )

        if use_overlap:
            return self._forward_overlap(
                x, x_edge, wigner, wigner_inv_envelope, gp_ctx, send_indices
            )

        if gp_ctx is not None and gp_utils.initialized():
            # All-to-all path: collect only needed remote embeddings.
            # When x requires grad (autograd forces/stress), we use the
            # autograd-compatible functional collective so gradients flow
            # through the communication. Both autograd and non-autograd
            # variants are compile-friendly (no graph break).
            needs_grad = torch.is_grad_enabled() and x.requires_grad
            if not self.training:
                # Eval path: compile-friendly functional collectives.
                # Selects autograd variant when gradients are needed
                # (e.g., UMA-S with direct_forces=False).
                with record_function("a2a_collect_compiled"):
                    x_received = all_to_all_collect_compiled(
                        x, gp_ctx, send_indices, autograd=needs_grad
                    )
                    x_full = torch.cat([x, x_received], dim=0)
                    edge_index_local = gp_ctx.edge_index_local
            else:
                # Training path: uses AllToAllCollect autograd.Function
                # which always supports backward.
                with record_function("a2a_collect"):
                    x_received = all_to_all_collect(x, gp_ctx, send_indices)
                    x_full = torch.cat([x, x_received], dim=0)
                    edge_index_local = gp_ctx.edge_index_local
            # In local space, node_offset is 0
            local_node_offset = 0
        elif gp_utils.initialized():
            # Legacy all-gather path
            with record_function("allgather_collect"):
                x_full = gp_utils.gather_from_model_parallel_region_sum_grad(
                    x, total_atoms_across_gp_ranks
                )
            edge_index_local = edge_index
            local_node_offset = node_offset
        else:
            x_full = x
            edge_index_local = edge_index
            local_node_offset = node_offset

        if self.activation_checkpoint_chunk_size is None:
            return self.forward_chunk(
                x_full,
                x.shape[0],
                x_edge,
                edge_index_local,
                wigner,
                wigner_inv_envelope,
                local_node_offset,
            )
        edge_index_partitions = edge_index_local.split(
            self.activation_checkpoint_chunk_size, dim=1
        )
        wigner_partitions = wigner.split(self.activation_checkpoint_chunk_size, dim=0)
        wigner_inv_partitions = wigner_inv_envelope.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        x_edge_partitions = x_edge.split(self.activation_checkpoint_chunk_size, dim=0)
        new_embeddings = []
        # when chunking, we need to keep track of the start index
        # of the chunk and give this information to the mole layers
        ac_mole_start_idx = 0

        for idx in range(len(edge_index_partitions)):
            new_embeddings.append(
                torch.utils.checkpoint.checkpoint(
                    self.forward_chunk,
                    x_full,
                    x.shape[0],
                    x_edge_partitions[idx],
                    edge_index_partitions[idx],
                    wigner_partitions[idx],
                    wigner_inv_partitions[idx],
                    local_node_offset,
                    ac_mole_start_idx,
                    use_reentrant=False,
                )
            )
            ac_mole_start_idx += edge_index_partitions[idx].shape[1]

            if len(new_embeddings) > 8:
                new_embeddings = [torch.stack(new_embeddings).sum(axis=0)]
        return torch.stack(new_embeddings).sum(axis=0)

    def _forward_overlap(
        self,
        x,
        x_edge,
        wigner,
        wigner_inv_envelope,
        gp_ctx: GPContext,
        send_indices: torch.Tensor | None,
    ):
        """
        Overlapped communication-computation forward pass.

        Overlaps the all-to-all communication with local edge
        computation for better inference latency. Only used in
        eval mode (no autograd through the communication).

        Edges are pre-sorted in build_gp_context (local edges first,
        boundary edges last via edge_reorder). This allows using
        compile-friendly split() instead of boolean indexing.

        Steps:
        1. Start async all-to-all to exchange boundary embeddings.
        2. Compute local edges (both endpoints are local atoms)
           while communication is in flight.
        3. Wait for communication to complete.
        4. Compute boundary edges (source is remote).
        5. Sum local + boundary contributions.
        """
        edge_index_local = gp_ctx.edge_index_local
        num_local_atoms = x.shape[0]
        n_local = gp_ctx.num_local_edges

        # Split pre-sorted per-edge data: local first, boundary last.
        # No boolean indexing — compile-friendly.
        local_edge_idx = edge_index_local[:, :n_local]
        boundary_edge_idx = edge_index_local[:, n_local:]
        local_x_edge = x_edge[:n_local]
        boundary_x_edge = x_edge[n_local:]
        local_wigner = wigner[:n_local]
        boundary_wigner = wigner[n_local:]
        local_wigner_inv = wigner_inv_envelope[:n_local]
        boundary_wigner_inv = wigner_inv_envelope[n_local:]

        # Step 1: Start async all-to-all
        with record_function("a2a_collect_async_start"):
            recv_buf, work_handles = start_all_to_all_collect(x, gp_ctx, send_indices)

        # Step 2: Compute local edges while comm is in flight
        with record_function("local_edges"):
            local_contribution = self.forward_chunk(
                x,
                num_local_atoms,
                local_x_edge,
                local_edge_idx,
                local_wigner,
                local_wigner_inv,
                0,
            )

        # Step 3: Wait for communication
        with record_function("a2a_collect_async_wait"):
            x_received = finish_all_to_all_collect(recv_buf, work_handles)
            x_full = torch.cat([x, x_received], dim=0)

        # Step 4: Compute boundary edges
        with record_function("boundary_edges"):
            boundary_contribution = self.forward_chunk(
                x_full,
                num_local_atoms,
                boundary_x_edge,
                boundary_edge_idx,
                boundary_wigner,
                boundary_wigner_inv,
                0,
            )

        # Step 5: Sum contributions
        return local_contribution + boundary_contribution

    def forward_chunk(
        self,
        x_full,
        x_original_shape,
        x_edge,
        edge_index,
        wigner,
        wigner_inv_envelope,
        node_offset: int = 0,
        ac_mole_start_idx: int = 0,
    ):
        # here we need to update the ac_start_idx of the mole layers
        # under here for this chunking to work properly with MoLE
        set_mole_ac_start_index(self, ac_mole_start_idx)

        with record_function("SO2Conv"):
            x_message = self.backend.node_to_edge_wigner_permute(
                x_full, edge_index, wigner
            )
            x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
            x_message = self.act(x_0_gating, x_message)
            x_message = self.so2_conv_2(x_message)
            new_embedding = self.backend.permute_wigner_inv_edge_to_node(
                x_message,
                wigner_inv_envelope,
                edge_index,
                x_original_shape,
                node_offset,
            )

        # reset ac start index
        set_mole_ac_start_index(self, 0)
        return new_embedding


class SpectralAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid: SO3_Grid,
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
        SO3_grid: SO3_Grid,
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
        mappingReduced: CoefficientMapping,
        SO3_grid: SO3_Grid,
        edge_channels_list: list[int],
        cutoff: float,
        norm_type: Literal["layer_norm", "layer_norm_sh", "rms_norm_sh"],
        act_type: Literal["gate", "s2"],
        ff_type: Literal["spectral", "grid"],
        activation_checkpoint_chunk_size: int | None,
        backend: ExecutionBackend,
        use_overlap_gp: bool = False,
        use_p2p_gp: bool = False,
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
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
            backend=backend,
            use_overlap_gp=use_overlap_gp,
            use_p2p_gp=use_p2p_gp,
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
        edge_index,
        wigner,
        wigner_inv_envelope,
        total_atoms_across_gp_ranks,
        sys_node_embedding=None,
        node_offset: int = 0,
        gp_ctx: GPContext | None = None,
        send_indices: torch.Tensor | None = None,
    ):
        x_res = x
        x = self.norm_1(x)

        if sys_node_embedding is not None:
            x[:, 0, :] = x[:, 0, :] + sys_node_embedding

        with record_function("edgewise"):
            x = self.edge_wise(
                x,
                x_edge,
                edge_index,
                wigner,
                wigner_inv_envelope,
                total_atoms_across_gp_ranks=total_atoms_across_gp_ranks,
                node_offset=node_offset,
                gp_ctx=gp_ctx,
                send_indices=send_indices,
            )
            x = x + x_res

        x_res = x
        x = self.norm_2(x)

        with record_function("atomwise"):
            x = self.atom_wise(x)
            x = x + x_res

        return x
