"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.profiler import record_function as _record_function
from typing_extensions import Literal

from fairchem.core.common import gp_utils
from fairchem.core.models.uma.nn.activation import (
    GateActivation,
    SeparableS2Activation_M,
)
from fairchem.core.models.uma.nn.execution_backends import (
    BaseExecutionBackend,
    ExecutionMode,
    get_execution_backend,
)
from fairchem.core.models.uma.nn.layer_norm import (
    get_normalization_layer,
)
from fairchem.core.models.uma.nn.mole import MOLE
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear

if TYPE_CHECKING:
    from fairchem.core.models.uma.common.so3 import (
        CoefficientMapping,
        SO3_Grid,
    )


@contextmanager
def _compile_safe_record_function(name: str):
    """
    Profiler wrapper that's safe for torch.compile.

    torch.dynamo has issues with record_function context manager
    bytecode transformation. This wrapper provides a no-op fallback
    during compilation while preserving profiling functionality during
    normal execution.
    """
    if torch.compiler.is_compiling():
        yield
    else:
        with _record_function(name):
            yield


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
        act_type: Literal["gate", "s2"] = "gate",
        # UMA-S execution mode:
        #   "baseline" - Reference PyTorch (SO2_Convolution)
        #   "fast_triton" - Triton GPU kernels
        #   "fast_cpp" - C++ CPU kernels (future)
        #   "fast_pytorch" - Optimized PyTorch
        execution_mode: str = "baseline",
        triton_backward_impl: str = "triton",
    ):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.activation_checkpoint_chunk_size = activation_checkpoint_chunk_size

        # UMA-S execution mode
        self.execution_mode = ExecutionMode(execution_mode)

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

        # Create execution backend - handles SO2 module creation and
        # rotation ops. Backend encapsulates all mode-specific logic
        # (Baseline vs Fast modes)
        self.backend: BaseExecutionBackend = get_execution_backend(
            mode=self.execution_mode,
            sphere_channels=self.sphere_channels,
            hidden_channels=self.hidden_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
            extra_m0_output_channels=extra_m0_output_channels,
            edge_channels_list=edge_channels_list,
            backward_impl=triton_backward_impl,
        )

        # Register hook to remap old checkpoint keys
        # (so2_conv_* -> backend.so2_conv_*)
        # This provides backward compatibility without duplicate
        # module registration (duplicate registration would break
        # MOLE conversion which walks the module tree)
        self._register_load_state_dict_pre_hook(self._remap_so2_keys)

    @staticmethod
    def _remap_so2_keys(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Remap old checkpoint keys: so2_conv_* -> backend.so2_conv_*
        """
        keys_to_remap = []
        for key in list(state_dict.keys()):
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                if suffix.startswith(("so2_conv_1.", "so2_conv_2.")):
                    new_key = f"{prefix}backend.{suffix}"
                    keys_to_remap.append((key, new_key))

        for old_key, new_key in keys_to_remap:
            state_dict[new_key] = state_dict.pop(old_key)

    @property
    def so2_conv_1(self):
        """
        Access to so2_conv_1 for MOLE conversion and external
        references.
        """
        return self.backend.so2_conv_1

    @property
    def so2_conv_2(self):
        """
        Access to so2_conv_2 for MOLE conversion and external
        references.
        """
        return self.backend.so2_conv_2

    def forward(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        total_atoms_across_gp_ranks,
        node_offset: int = 0,
        precomputed_radial: torch.Tensor | None = None,
    ):
        # we perform the all gather upfront once during each forward
        # call so we don't need to repeat this multiple times during
        # activation checkpointing.
        if gp_utils.initialized():
            x_full = gp_utils.gather_from_model_parallel_region_sum_grad(
                x, total_atoms_across_gp_ranks
            )
        else:
            x_full = x

        if self.activation_checkpoint_chunk_size is None:
            return self.forward_chunk(
                x_full,
                x.shape[0],
                x_edge,
                edge_distance,
                edge_index,
                wigner_and_M_mapping,
                wigner_and_M_mapping_inv,
                node_offset,
                0,  # ac_mole_start_idx
                precomputed_radial,
            )
        edge_index_partitions = edge_index.split(
            self.activation_checkpoint_chunk_size, dim=1
        )
        wigner_partitions = wigner_and_M_mapping.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        wigner_inv_partitions = wigner_and_M_mapping_inv.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        edge_distance_parititons = edge_distance.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        x_edge_partitions = x_edge.split(self.activation_checkpoint_chunk_size, dim=0)
        new_embeddings = []
        # when chunking, we need to keep track of the start index of
        # the chunk and give this information to the mole layers
        ac_mole_start_idx = 0

        for idx in range(len(edge_index_partitions)):
            new_embeddings.append(
                torch.utils.checkpoint.checkpoint(
                    self.forward_chunk,
                    x_full,
                    x.shape[0],
                    x_edge_partitions[idx],
                    edge_distance_parititons[idx],
                    edge_index_partitions[idx],
                    wigner_partitions[idx],
                    wigner_inv_partitions[idx],
                    node_offset,
                    ac_mole_start_idx,
                    use_reentrant=False,
                )
            )
            ac_mole_start_idx += edge_index_partitions[idx].shape[1]

            if len(new_embeddings) > 8:
                new_embeddings = [torch.stack(new_embeddings).sum(axis=0)]
        return torch.stack(new_embeddings).sum(axis=0)

    def forward_chunk(
        self,
        x_full,
        x_original_shape,
        x_edge,
        edge_distance,
        edge_index,
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,  # Envelope is pre-fused
        node_offset: int = 0,
        ac_mole_start_idx: int = 0,
        precomputed_radial: torch.Tensor | None = None,
    ):
        # here we need to update the ac_start_idx of the mole layers
        # under here for this chunking to work properly with MoLE
        # together
        set_mole_ac_start_index(self, ac_mole_start_idx)

        with _compile_safe_record_function("SO2Conv"):
            # All operations delegated to backend -
            # no conditionals here!
            backend = self.backend

            # Gather + rotate L->M
            x_message = backend.gather_rotate(x_full, edge_index, wigner_and_M_mapping)

            # SO2 convolution 1 (with radial modulation)
            x_message, x_0_gating = backend.conv1(x_message, x_edge, precomputed_radial)

            # M-prime activation
            x_message = self.act(x_0_gating, x_message)

            # SO2 convolution 2 (internal weights)
            x_message = backend.conv2(x_message, x_edge)

            # Rotate back M->L
            x_message = backend.rotate_back(x_message, wigner_and_M_mapping_inv)

        # Compute the sum of the incoming neighboring messages for
        # each target node
        new_embedding = torch.zeros(
            (x_original_shape,) + x_message.shape[1:],
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
            self.sphere_channels,
            self.hidden_channels,
            lmax=self.lmax,
        )
        self.act = GateActivation(
            lmax=self.lmax,
            mmax=self.lmax,
            num_channels=self.hidden_channels,
        )
        self.so3_linear_2 = SO3_Linear(
            self.hidden_channels,
            self.sphere_channels,
            lmax=self.lmax,
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
            nn.Linear(
                self.sphere_channels,
                self.hidden_channels,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                self.hidden_channels,
                self.hidden_channels,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                self.hidden_channels,
                self.sphere_channels,
                bias=False,
            ),
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
        # UMA-S execution mode
        execution_mode: str = "baseline",
        triton_backward_impl: str = "triton",
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
            execution_mode=execution_mode,
            triton_backward_impl=triton_backward_impl,
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
        wigner_and_M_mapping,
        wigner_and_M_mapping_inv,
        total_atoms_across_gp_ranks,
        sys_node_embedding=None,
        node_offset: int = 0,
        precomputed_radial: torch.Tensor | None = None,
    ):
        x_res = x
        x = self.norm_1(x)

        if sys_node_embedding is not None:
            x[:, 0, :] = x[:, 0, :] + sys_node_embedding

        with _compile_safe_record_function("edgewise"):
            x = self.edge_wise(
                x,
                x_edge,
                edge_distance,
                edge_index,
                wigner_and_M_mapping,
                wigner_and_M_mapping_inv,
                total_atoms_across_gp_ranks=total_atoms_across_gp_ranks,
                node_offset=node_offset,
                precomputed_radial=precomputed_radial,
            )
            x = x + x_res

        x_res = x
        x = self.norm_2(x)

        with _compile_safe_record_function("atomwise"):
            x = self.atom_wise(x)
            x = x + x_res

        return x
