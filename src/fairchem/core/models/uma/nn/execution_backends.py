"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

from fairchem.core.models.uma.nn.unified_radial import UnifiedRadialMLP

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

__all__ = [
    "ExecutionMode",
    "ExecutionBackend",
    "UMASFastPytorchBackend",
    "UMASFastGPUBackend",
    "get_execution_backend",
]

# Indices for m=0 spherical harmonic coefficients in L-major ordering (lmax=2)
_M0_COL_INDICES_L_ORDER = [0, 2, 6]


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
        - gather_rotate: Gather node features and rotate L->M
        - rotate_back: Rotate M->L
        - edge_degree_scatter: Rotate radial and scatter to nodes
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

    @staticmethod
    def gather_rotate(
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather node features and rotate L->M.

        Default: PyTorch gather + BMM.

        Args:
            x_full: Node features [N, L, C]
            edge_index: Edge indices [2, E]
            wigner: Wigner rotation matrices [E, M, L] or [E, M, 2L]

        Returns:
            Rotated edge messages [E, M, 2C]
        """
        x_source = x_full[edge_index[0]]
        x_target = x_full[edge_index[1]]
        x_message = torch.cat((x_source, x_target), dim=2)
        return torch.bmm(wigner, x_message)

    @staticmethod
    def rotate_back(
        x: torch.Tensor,
        wigner_inv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotate M->L.

        Default: PyTorch BMM.

        Args:
            x: Message features [E, M, C]
            wigner_inv: Inverse Wigner matrices [E, L, M]

        Returns:
            Rotated features [E, L, C]
        """
        return torch.bmm(wigner_inv, x)

    @staticmethod
    def edge_degree_scatter(
        x: torch.Tensor,
        radial_output: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_index: torch.Tensor,
        m_0_num_coefficients: int,
        sphere_channels: int,
        rescale_factor: float,
        node_offset: int = 0,
    ) -> torch.Tensor:
        """
        Edge degree embedding: rotate radial and scatter to nodes.

        Default: PyTorch BMM + index_add.

        Args:
            x: Node features [N, L, C] to update
            radial_output: RadialMLP output [E, m0 * C]
            wigner_inv: Wigner inverse with envelope pre-fused
                [E, L, m0] or [E, L, L]
            edge_index: Edge indices [2, E]
            m_0_num_coefficients: Number of m=0 coefficients
                (3 for lmax=2)
            sphere_channels: Number of channels C
            rescale_factor: Aggregation rescale factor
            node_offset: Node offset for graph parallelism

        Returns:
            Updated node features [N, L, C]
        """
        # Reshape radial output: [E, m0*C] -> [E, m0, C]
        radial = radial_output.reshape(-1, m_0_num_coefficients, sphere_channels)

        # Slice wigner to m=0 columns and rotate:
        # [E, L, m0] @ [E, m0, C] -> [E, L, C]
        wigner_inv_m0 = wigner_inv[:, :, :m_0_num_coefficients]
        x_edge_embedding = torch.bmm(wigner_inv_m0, radial)

        # Type cast if needed
        x_edge_embedding = x_edge_embedding.to(x.dtype)

        # Scatter to destination nodes with rescaling
        return x.index_add(
            0,
            edge_index[1] - node_offset,
            x_edge_embedding / rescale_factor,
        )


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
        if settings is not None and settings.activation_checkpointing:
            raise ValueError(
                "UMASFastPytorchBackend requires " "activation_checkpointing=False"
            )

    @staticmethod
    def prepare_model_for_inference(model: torch.nn.Module) -> None:
        """
        Convert SO2_Convolution modules to block-diagonal GEMM variants.

        Replaces so2_conv_1 with SO2_Conv1_WithRadialBlock and
        so2_conv_2 with SO2_Conv2_InternalBlock in each block's
        Edgewise module.
        """
        from fairchem.core.models.uma.nn.so2_layers import (
            convert_so2_conv1,
            convert_so2_conv2,
        )

        for block in model.blocks:
            block.edge_wise.so2_conv_1 = convert_so2_conv1(block.edge_wise.so2_conv_1)
            block.edge_wise.so2_conv_2 = convert_so2_conv2(block.edge_wise.so2_conv_2)

    @staticmethod
    def get_layer_radial_emb(
        x_edge: torch.Tensor,
        model: torch.nn.Module,
    ) -> list[torch.Tensor]:
        """
        Compute radial embeddings for all layers.

        Args:
            x_edge: Edge embeddings [E, edge_features]
            model: The backbone model with blocks containing rad_funcs

        Returns:
            List of radial embeddings, one per layer [E, radial_features]
        """
        return [block.edge_wise.so2_conv_1.rad_func(x_edge) for block in model.blocks]


class UMASFastGPUBackend(UMASFastPytorchBackend):
    """
    GPU-optimized backend: SO2 block conversion + Triton kernels.

    Extends UMASFastPytorchBackend with Triton-accelerated
    gather_rotate, rotate_back, and edge_degree_scatter.
    Requires lmax==2, mmax==2, sphere_channels divisible by 128,
    and merge_mole=True.
    """

    @staticmethod
    def validate(
        model: torch.nn.Module,
        settings: InferenceSettings | None = None,
    ) -> None:
        UMASFastPytorchBackend.validate(model, settings)
        from fairchem.core.models.uma.triton import HAS_TRITON

        if not HAS_TRITON:
            raise ValueError("umas_fast_gpu requires Triton")
        if model.lmax != 2 or model.mmax != 2:
            raise ValueError("umas_fast_gpu requires lmax==2 and mmax==2")
        if model.sphere_channels % 128 != 0:
            raise ValueError("sphere_channels must be divisible by 128")
        if settings is not None and not settings.merge_mole:
            raise ValueError("umas_fast_gpu requires merge_mole=True")

    @staticmethod
    def prepare_wigner(
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        mappingReduced,
        coefficient_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Passthrough — Triton kernels handle L-to-M internally
        return wigner, wigner_inv

    @staticmethod
    def gather_rotate(
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        from fairchem.core.models.uma.triton import (
            FusedEdgeGatherWignerL2MTritonBwdEmitFunction,
        )

        return FusedEdgeGatherWignerL2MTritonBwdEmitFunction.apply(
            x_full, edge_index, wigner
        )

    @staticmethod
    def rotate_back(
        x: torch.Tensor,
        wigner_inv: torch.Tensor,
    ) -> torch.Tensor:
        from fairchem.core.models.uma.triton.wigner_ops import (
            FusedMToLThenWignerLmax2Function,
        )

        return FusedMToLThenWignerLmax2Function.apply(x, wigner_inv)

    @staticmethod
    def edge_degree_scatter(
        x: torch.Tensor,
        radial_output: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_index: torch.Tensor,
        m_0_num_coefficients: int,
        sphere_channels: int,
        rescale_factor: float,
        node_offset: int = 0,
    ) -> torch.Tensor:
        radial = radial_output.reshape(-1, m_0_num_coefficients, sphere_channels)

        # Select m=0 columns from L-ordered wigner_inv
        wigner_inv_m0 = wigner_inv[:, :, _M0_COL_INDICES_L_ORDER]
        x_edge_embedding = torch.bmm(wigner_inv_m0, radial)

        x_edge_embedding = x_edge_embedding.to(x.dtype)

        return x.index_add(
            0,
            edge_index[1] - node_offset,
            x_edge_embedding / rescale_factor,
        )

    @staticmethod
    def prepare_model_for_inference(model: torch.nn.Module) -> None:
        """
        Prepare model for inference: SO2 block conversion + unified radial.

        Calls parent class for SO2 conversion, then creates a UnifiedRadialMLP
        from all radial functions for batched computation.
        """
        UMASFastPytorchBackend.prepare_model_for_inference(model)
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
