"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

import torch

from fairchem.core.models.uma.escn_md_block import Edgewise
from fairchem.core.models.uma.nn.embedding import EdgeDegreeEmbedding
from fairchem.core.models.uma.nn.unified_radial import UnifiedRadialMLP

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import (
        InferenceSettings,
    )

__all__ = [
    "ExecutionMode",
    "ExecutionBackend",
    "UMASFastPytorchBackend",
    "UMASFastGPUBackend",
    "UMASFastGPUEdgeDegreeEmbedding",
    "UMASFastEdgewise",
    "UMASFastGPUEdgewise",
    "get_execution_backend",
    "maybe_update_settings_backend",
]

# Indices for m=0 spherical harmonic coefficients in L-major ordering (lmax=2)
_M0_COL_INDICES_L_ORDER = [0, 2, 6]


class UMASFastGPUEdgeDegreeEmbedding(EdgeDegreeEmbedding):
    """
    EdgeDegreeEmbedding variant for UMASFastGPUBackend.

    Uses L-ordered wigner_inv (not M-ordered) for m=0 column indexing.
    Created at inference time via from_instance().
    """

    @classmethod
    def from_instance(cls, src: EdgeDegreeEmbedding) -> UMASFastGPUEdgeDegreeEmbedding:
        """Create GPU variant from existing EdgeDegreeEmbedding."""
        new = cls.__new__(cls)
        torch.nn.Module.__init__(new)

        # Config
        new.sphere_channels = src.sphere_channels
        new.lmax = src.lmax
        new.mmax = src.mmax
        new.activation_checkpoint_chunk_size = src.activation_checkpoint_chunk_size
        new.rescale_factor = src.rescale_factor

        # Computed
        new.m_0_num_coefficients = src.m_0_num_coefficients
        new.m_all_num_coefficents = src.m_all_num_coefficents

        # References (shared, not owned)
        new.mappingReduced = src.mappingReduced
        new.backend = src.backend

        # Submodule (transfer ownership)
        new.rad_func = src.rad_func

        return new

    def edge_degree_scatter(
        self,
        x: torch.Tensor,
        radial_output: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_index: torch.Tensor,
        node_offset: int = 0,
    ) -> torch.Tensor:
        """
        Edge degree embedding with L-ordered wigner indexing.

        Uses _M0_COL_INDICES_L_ORDER [0, 2, 6] instead of [:3] slice
        because UMASFastGPUBackend.prepare_wigner passes through raw
        L-ordered wigner matrices (doesn't apply M-mapping).
        """
        radial = radial_output.reshape(
            -1, self.m_0_num_coefficients, self.sphere_channels
        )

        # Select m=0 columns from L-ordered wigner_inv
        wigner_inv_m0 = wigner_inv[:, :, _M0_COL_INDICES_L_ORDER]
        x_edge_embedding = torch.bmm(wigner_inv_m0, radial)

        x_edge_embedding = x_edge_embedding.to(x.dtype)

        return x.index_add(
            0,
            edge_index[1] - node_offset,
            x_edge_embedding / self.rescale_factor,
        )


class UMASFastEdgewise(Edgewise):
    """
    Edgewise variant with block-diagonal SO2 convolutions.

    Base class for fast execution backends. Created at inference time
    via from_instance().
    """

    @classmethod
    def from_instance(cls, src: Edgewise) -> UMASFastEdgewise:
        """Create fast variant from existing Edgewise.

        Converts SO2_Convolution modules to block-diagonal variants
        and transfers all state from the source instance.
        """
        from fairchem.core.models.uma.nn.so2_layers import (
            convert_so2_conv1,
            convert_so2_conv2,
        )

        new = cls.__new__(cls)
        torch.nn.Module.__init__(new)

        # Config
        new.sphere_channels = src.sphere_channels
        new.hidden_channels = src.hidden_channels
        new.lmax = src.lmax
        new.mmax = src.mmax
        new.activation_checkpoint_chunk_size = src.activation_checkpoint_chunk_size
        new.act_type = src.act_type

        # References (shared, not owned)
        new.mappingReduced = src.mappingReduced
        new.SO3_grid = src.SO3_grid
        new.backend = src.backend

        # Submodules - convert SO2 layers to block-diagonal variants
        new.act = src.act
        new.so2_conv_1 = convert_so2_conv1(src.so2_conv_1)
        new.so2_conv_2 = convert_so2_conv2(src.so2_conv_2)

        return new


class UMASFastGPUEdgewise(UMASFastEdgewise):
    """
    Edgewise variant for UMASFastGPUBackend.

    Extends UMASFastEdgewise with Triton kernels for wigner operations.
    """

    def node_to_edge_wigner_permute(
        self,
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        from fairchem.core.models.uma.triton import (
            UMASFastGPUNodeToEdgeWignerPermute,
        )

        return UMASFastGPUNodeToEdgeWignerPermute.apply(x_full, edge_index, wigner)

    def permute_wigner_inv_edge_to_node(
        self,
        x_message: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_offset: int = 0,
    ) -> torch.Tensor:
        from fairchem.core.models.uma.triton import (
            UMASFastGPUPermuteWignerInvEdgeToNode,
        )

        # Rotate M->L using Triton kernel
        x_rotated = UMASFastGPUPermuteWignerInvEdgeToNode.apply(x_message, wigner_inv)
        # Scatter to nodes
        new_embedding = torch.zeros(
            (num_nodes,) + x_rotated.shape[1:],
            dtype=x_rotated.dtype,
            device=x_rotated.device,
        )
        new_embedding.index_add_(0, edge_index[1] - node_offset, x_rotated)
        return new_embedding


class ExecutionMode(str, Enum):
    """
    Execution mode for model inference.
    """

    GENERAL = "general"
    UMAS_FAST_PYTORCH = "umas_fast_pytorch"
    UMAS_FAST_GPU = "umas_fast_gpu"


class ExecutionBackend:
    """
    Parameterless function dispatch for execution modes.

    Provides default PyTorch implementations for rotation and scatter
    operations. Subclass and override methods with optimized kernels
    (e.g. Triton) for specific execution modes.

    All methods are static — backends carry no instance state.

    Methods (override for optimization):
        - prepare_model_for_inference: Apply backend-specific model transforms
    """

    @staticmethod
    def validate(
        model: torch.nn.Module,
        settings: InferenceSettings | None = None,
    ) -> None:
        """
        Validate that model and settings are compatible with this backend.

        Called during model construction (settings=None) and before
        first inference (settings provided).

        Args:
            model: The backbone model to validate.
            settings: Inference settings, or None at construction time.

        Raises:
            ValueError: If incompatible with this backend.
        """

    @staticmethod
    def prepare_model_for_inference(model: torch.nn.Module) -> None:
        """
        Prepare a model for inference with backend-specific transforms.

        Called once during prepare_for_inference. Override in subclasses
        to apply model transformations (e.g. SO2 block conversion).

        Args:
            model: The backbone model to prepare.
        """

    @staticmethod
    def get_layer_radial_emb(
        x_edge: torch.Tensor,
        model: torch.nn.Module,
    ) -> list[torch.Tensor]:
        """
        Get edge embeddings for each layer.

        Default implementation returns the same raw x_edge for all layers.
        SO2_Convolution will compute rad_func(x_edge) internally.

        Override in fast backends to precompute radials.

        Args:
            x_edge: Edge embeddings [E, edge_features]
            model: The backbone model

        Returns:
            List of edge embeddings, one per layer
        """
        return [x_edge] * len(model.blocks)

    @staticmethod
    def prepare_wigner(
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        mappingReduced,
        coefficient_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transform raw Wigner matrices for this backend.

        Default: Apply coefficient selection (if mmax != lmax) and
        pre-compose with M-mapping via einsum.

        Args:
            wigner: Raw Wigner matrices [E, L, L]
            wigner_inv: Raw inverse Wigner matrices [E, L, L]
            mappingReduced: CoefficientMapping with to_m matrix
            coefficient_index: Indices for mmax != lmax selection,
                or None if mmax == lmax.

        Returns:
            Transformed (wigner, wigner_inv) ready for this backend.
        """
        if coefficient_index is not None:
            wigner = wigner.index_select(1, coefficient_index)
            wigner_inv = wigner_inv.index_select(2, coefficient_index)

        wigner = torch.einsum(
            "mk,nkj->nmj",
            mappingReduced.to_m.to(wigner.dtype),
            wigner,
        )
        wigner_inv = torch.einsum(
            "njk,mk->njm",
            wigner_inv,
            mappingReduced.to_m.to(wigner_inv.dtype),
        )
        return wigner, wigner_inv


class UMASFastPytorchBackend(ExecutionBackend):
    """
    Optimized PyTorch backend using block-diagonal SO2 convolutions.

    Requires merge_mole=True and activation_checkpointing=False.
    """

    @staticmethod
    def validate(
        model: torch.nn.Module,
        settings: InferenceSettings | None = None,
    ) -> None:
        """
        Validate that settings are compatible with fast pytorch mode.
        """
        # Check activation_checkpointing from model (chunk_size is None when disabled)
        if model.edge_degree_embedding.activation_checkpoint_chunk_size is not None:
            raise ValueError(
                "UMASFastPytorchBackend requires activation_checkpointing=False"
            )
        # Also reject if user tries to enable it via inference settings
        if settings is not None and settings.activation_checkpointing:
            raise ValueError(
                "UMASFastPytorchBackend requires activation_checkpointing=False"
            )

    @staticmethod
    def prepare_model_for_inference(model: torch.nn.Module) -> None:
        """
        Convert Edgewise modules to UMASFastEdgewise with block-diagonal
        SO2 convolutions and create unified radial MLP.
        """
        # Replace Edgewise modules with fast variants (includes SO2 conversion)
        for block in model.blocks:
            block.edge_wise = UMASFastEdgewise.from_instance(block.edge_wise)

        # Create unified radial MLP for batched computation
        rad_funcs = [block.edge_wise.so2_conv_1.rad_func for block in model.blocks]
        model._unified_radial_mlp = UnifiedRadialMLP(rad_funcs)

    @staticmethod
    def get_layer_radial_emb(
        x_edge: torch.Tensor,
        model: torch.nn.Module,
    ) -> list[torch.Tensor]:
        """
        Compute radial embeddings for all layers using batched UnifiedRadialMLP.

        Args:
            x_edge: Edge embeddings [E, edge_features]
            model: The backbone model with _unified_radial_mlp

        Returns:
            List of radial embeddings, one per layer [E, radial_features]
        """
        return model._unified_radial_mlp(x_edge)


class UMASFastGPUBackend(UMASFastPytorchBackend):
    """
    GPU-optimized backend: SO2 block conversion + Triton kernels.

    Replaces EdgeDegreeEmbedding and Edgewise modules with GPU-optimized
    variants that include Triton-accelerated wigner operations and
    block-diagonal SO2 convolutions.

    Requires lmax==2, mmax==2, and merge_mole=True.

    Note: sphere_channels % 128 == 0 gives optimal GPU utilization.
    Smaller values work but with reduced efficiency.
    """

    @staticmethod
    def validate(
        model: torch.nn.Module,
        settings: InferenceSettings | None = None,
    ) -> None:
        UMASFastPytorchBackend.validate(model, settings)
        if not torch.cuda.is_available():
            raise ValueError("umas_fast_gpu requires CUDA")
        if model.lmax != 2 or model.mmax != 2:
            raise ValueError("umas_fast_gpu requires lmax==2 and mmax==2")
        if settings is not None and not settings.merge_mole:
            raise ValueError("umas_fast_gpu requires merge_mole=True")

    @staticmethod
    def prepare_model_for_inference(model: torch.nn.Module) -> None:
        """
        Prepare model for GPU-optimized inference.

        Replaces EdgeDegreeEmbedding with UMASFastGPUEdgeDegreeEmbedding
        for L-ordered wigner indexing, and replaces Edgewise with
        UMASFastGPUEdgewise which includes SO2 block conversion and
        Triton-accelerated wigner operations.

        Also creates unified radial MLP for batched computation.
        """
        # Replace edge_degree_embedding with GPU variant
        model.edge_degree_embedding = UMASFastGPUEdgeDegreeEmbedding.from_instance(
            model.edge_degree_embedding
        )

        # Replace Edgewise modules with GPU variants (includes SO2 conversion)
        for block in model.blocks:
            block.edge_wise = UMASFastGPUEdgewise.from_instance(block.edge_wise)

        # Create unified radial MLP for batched computation
        rad_funcs = [block.edge_wise.so2_conv_1.rad_func for block in model.blocks]
        model._unified_radial_mlp = UnifiedRadialMLP(rad_funcs)

    @staticmethod
    def prepare_wigner(
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        mappingReduced,
        coefficient_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Passthrough — Triton kernels handle L-to-M internally
        return wigner, wigner_inv


_EXECUTION_BACKENDS: dict[ExecutionMode, type[ExecutionBackend]] = {
    ExecutionMode.GENERAL: ExecutionBackend,
    ExecutionMode.UMAS_FAST_PYTORCH: UMASFastPytorchBackend,
    ExecutionMode.UMAS_FAST_GPU: UMASFastGPUBackend,
}


def get_execution_backend(
    mode: ExecutionMode | str = ExecutionMode.GENERAL,
) -> ExecutionBackend:
    """
    Factory function to create the appropriate execution backend.

    Args:
        mode: Execution mode (enum or string). Defaults to GENERAL.

    Returns:
        Configured execution backend instance
    """
    if isinstance(mode, str):
        mode = ExecutionMode(mode)

    if mode not in _EXECUTION_BACKENDS:
        available = [m.value for m in _EXECUTION_BACKENDS]
        raise ValueError(f"Unknown execution mode: {mode}. Available: {available}")
    return _EXECUTION_BACKENDS[mode]()


def maybe_update_settings_backend(
    settings: InferenceSettings,
    model: torch.nn.Module,
) -> InferenceSettings:
    """
    Update inference settings to use UMAS_FAST_GPU if conditions are met.

    Sets execution_mode to UMAS_FAST_GPU if:
    - execution_mode is not already set
    - UMASFastGPUBackend.validate passes for the model and settings

    Args:
        settings: Current inference settings.
        model: The backbone model to validate.

    Returns:
        Updated inference settings with the appropriate execution mode.
    """
    if settings.execution_mode is not None:
        return settings

    try:
        UMASFastGPUBackend.validate(model, settings)
        return replace(settings, execution_mode=ExecutionMode.UMAS_FAST_GPU)
    except ValueError:
        return settings
