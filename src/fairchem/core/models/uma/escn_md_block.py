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
from torch import distributed as dist
from torch.profiler import record_function
from typing_extensions import Literal

from fairchem.core.common import gp_utils
from fairchem.core.common.gp_utils import (
    get_gp_rank,
    size_list_fn,
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
        # Enables activation checkpointing of edges in
        # activation_checkpoint_chunk_size size edge blocks
        activation_checkpoint_chunk_size: int | None,
        act_type: Literal["gate", "s2"] = "gate",
    ):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.activation_checkpoint_chunk_size = activation_checkpoint_chunk_size

        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
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
            # NOTE: this is the only place where the SO3 grid of the edges (lmax/mmax) is used
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

        self.out_mask = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )

    def forward_gp_single(
        self,
        x,
        x_edge,
        edge_envelope,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        natoms,
        sizes,
        padded_size,
        edge_splits,
        gloo_backend,
        node_offset: int = 0,
    ):
        size_list = size_list_fn(natoms, gp_utils.get_gp_world_size())
        out = self.forward_chunk(
            x,
            x_edge,
            edge_envelope,
            edge_index,
            wigner_and_M_mapping,
            wigner_and_M_mapping_inv,
            natoms_local=size_list[
                gp_utils.get_gp_rank()
            ],  # required for compile not to break
            sizes=None,
            padded_size=0,
            edge_splits=None,
            gloo_backend=gloo_backend,
            node_offset=node_offset,
        )

        if gloo_backend:
            all_atoms, _ = (
                gp_utils.gather_from_model_parallel_region_sum_grad_async_gloo(
                    out, size_list, False
                )
            )
            # need to deal with padding
            all_atoms_splits = all_atoms.split(max(size_list), dim=0)
            return torch.cat(
                [
                    all_atoms_splits[idx][: size_list[idx]]
                    for idx in range(len(size_list))
                ]
            )
        all_atoms = gp_utils.gather_from_model_parallel_region_sum_grad_noasync(
            out, natoms
        )
        return all_atoms

    def forward_gp_staggered(
        self,
        x,
        x_edge,
        edge_envelope,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        natoms,
        sizes,
        padded_size,
        edge_splits,
        gloo_backend,
        node_offset: int = 0,
    ):
        group = gp_utils.get_gp_group()
        rank = get_gp_rank()
        world_size = dist.get_world_size(group=group)

        n_chunks = sizes.shape[1]
        rank_sizes = sizes[rank].tolist()
        chunk_sizes = [sizes[:, chunk_idx].tolist() for chunk_idx in range(n_chunks)]
        # distribute into subchunks
        results_async = []
        rank_offset = node_offset

        wigner_and_M_mapping_partitions = wigner_and_M_mapping.split(edge_splits, dim=0)
        wigner_and_M_mapping_inv_partitions = wigner_and_M_mapping_inv.split(
            edge_splits, dim=0
        )

        x_edge_partitions = x_edge.split(edge_splits, dim=0)
        edge_envelope_partitions = edge_envelope.split(edge_splits, dim=0)
        edge_index_partitions = edge_index.split(edge_splits, dim=1)

        _local_node_offset = 0
        # preallocated_output=torch.zeros((padded_size*n_chunks,*x.shape[1:]),device=x.device,dtype=x.dtype)
        for chunk_idx in range(n_chunks):
            _global_node_offset = rank_offset + _local_node_offset

            out = self.forward_chunk(
                x,
                x_edge_partitions[chunk_idx],
                edge_envelope_partitions[chunk_idx],
                edge_index_partitions[chunk_idx],
                wigner_and_M_mapping_partitions[chunk_idx],
                wigner_and_M_mapping_inv_partitions[chunk_idx],
                natoms_local=rank_sizes[chunk_idx],  # required for compile not to break
                sizes=sizes,
                padded_size=padded_size,
                edge_splits=edge_splits,
                gloo_backend=gloo_backend,
                node_offset=_global_node_offset,
            )
            if gloo_backend:
                out_global_async = (
                    gp_utils.gather_from_model_parallel_region_sum_grad_async_gloo(
                        out,
                        chunk_sizes[chunk_idx],
                        True,  # [padded_size for _ in range(world_size)]
                    )
                )
            else:
                out_global_async = (
                    gp_utils.gather_from_model_parallel_region_sum_grad_async(
                        out,
                        True,  # [padded_size for _ in range(world_size)]
                        natoms,
                    )
                )
            with record_function("STAG1"):
                results_async.append(
                    (
                        chunk_sizes[chunk_idx],
                        out,
                        *out_global_async,
                    )
                )
            with record_function("STAG2"):
                _local_node_offset += rank_sizes[chunk_idx]

        # wait for async ops to finish
        results_aync_merged = []
        for size_list, local_out, all_atoms_padded, handle in results_async:
            if handle is not None:
                handle.wait()
            if gloo_backend:
                # need to deal with padding
                all_atoms_splits = all_atoms_padded.split(max(size_list), dim=0)
                all_atoms = torch.cat(
                    [
                        all_atoms_splits[idx][: size_list[idx]]
                        for idx in range(len(size_list))
                    ]
                )
            else:
                all_atoms = all_atoms_padded

            all_atoms_split = list(all_atoms.split(size_list))
            all_atoms_split[rank] = local_out
            results_aync_merged.append(all_atoms_split)

        # # locally reconstruct full atom embeddings
        full_list_async = []
        for rank_idx in range(world_size):
            for chunk_idx in range(n_chunks):
                full_list_async.append(results_aync_merged[chunk_idx][rank_idx])
        return torch.cat(full_list_async, dim=0)

    def forward_checkpoint(
        self,
        x,
        x_edge,
        edge_envelope,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        natoms,
        sizes,
        padded_size,
        edge_splits,
        gloo_backend,
        node_offset: int = 0,
    ):
        edge_index_partitions = edge_index.split(
            self.activation_checkpoint_chunk_size, dim=1
        )
        wigner_partitions = wigner_and_M_mapping.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        wigner_inv_partitions = wigner_and_M_mapping_inv.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        edge_envelope_parititons = edge_envelope.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        x_edge_partitions = x_edge.split(self.activation_checkpoint_chunk_size, dim=0)
        new_embeddings = []
        # when chunking, we need to keep track of the start index of the chunk and give this information
        # to the mole layers
        ac_mole_start_idx = 0
        for idx in range(len(edge_index_partitions)):
            new_embeddings.append(
                torch.utils.checkpoint.checkpoint(
                    self.forward_chunk,
                    x,
                    x_edge_partitions[idx],
                    edge_envelope_parititons[idx],
                    edge_index_partitions[idx],
                    wigner_partitions[idx],
                    wigner_inv_partitions[idx],
                    x.shape[0],
                    sizes,
                    padded_size,
                    edge_splits,
                    gloo_backend,
                    node_offset,
                    ac_mole_start_idx,
                    use_reentrant=False,
                )
            )
            ac_mole_start_idx += edge_index_partitions[idx].shape[1]

            if len(new_embeddings) > 8:
                new_embeddings = [torch.stack(new_embeddings).sum(axis=0)]
        return torch.stack(new_embeddings).sum(axis=0)

    def forward(
        self,
        x,
        x_edge,
        edge_envelope,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        natoms,
        sizes,
        padded_size,
        edge_splits,
        gloo_backend,
        node_offset: int = 0,
    ):
        forward_func = self.forward_chunk
        if gp_utils.initialized():
            # forward_func = self.forward_gp_staggered
            forward_func = self.forward_gp_single
        elif self.activation_checkpoint_chunk_size is not None:
            forward_func = self.forward_checkpoint
        return forward_func(
            x,
            x_edge,
            edge_envelope,
            edge_index,
            wigner_and_M_mapping,
            wigner_and_M_mapping_inv,
            x.shape[0],
            sizes,
            padded_size,
            edge_splits,
            gloo_backend,
            node_offset=node_offset,
        )

    def forward_chunk(
        self,
        x_full,
        x_edge,
        edge_envelope,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        natoms_local,
        sizes,
        padded_size,
        edge_splits,
        gloo_backend,
        node_offset: int = 0,
        ac_mole_start_idx: int = 0,
    ):
        # here we need to update the ac_start_idx of the mole layers under here for this chunking to
        # work properly with MoLE together
        set_mole_ac_start_index(self, ac_mole_start_idx)

        x_source = x_full[edge_index[0]]
        x_target = x_full[edge_index[1]]

        x_message = torch.cat((x_source, x_target), dim=2)

        with record_function("SO2Conv"):
            # Rotate the irreps to align with the edge
            x_message = torch.bmm(wigner_and_M_mapping, x_message)

            # SO2 convolution
            x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)

            # M-prime...
            x_message = self.act(x_0_gating, x_message)

            x_message = self.so2_conv_2(x_message, x_edge)

            # envelope
            x_message = x_message * edge_envelope

            # Rotate back the irreps
            x_message = torch.bmm(wigner_and_M_mapping_inv, x_message)

        # Compute the sum of the incoming neighboring messages for each target node
        new_embedding = torch.zeros(
            (natoms_local,) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )

        new_embedding.index_add_(0, edge_index[1] - node_offset, x_message)
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
        norm_type: Literal["layer_norm", "layer_norm_sh", "rms_norm_sh"],
        act_type: Literal["gate", "s2"],
        ff_type: Literal["spectral", "grid"],
        activation_checkpoint_chunk_size: int | None,
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
            act_type=act_type,
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
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
        edge_distance_envelope,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        natoms,
        sizes,
        padded_size,
        edge_splits,
        gloo_backend,
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
                edge_distance_envelope,
                edge_index,
                wigner_and_M_mapping,
                wigner_and_M_mapping_inv,
                natoms,
                sizes,
                padded_size,
                edge_splits,
                gloo_backend,
                node_offset,
            )
            x = x + x_res

        x_res = x
        x = self.norm_2(x)

        with record_function("atomwise"):
            x = self.atom_wise(x)
            x = x + x_res

        return x
