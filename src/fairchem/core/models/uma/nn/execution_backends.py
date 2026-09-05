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

from fairchem.core.models.uma.nn.unified_radial import UnifiedRadialMLP

if TYPE_CHECKING:
    from fairchem.core.models.uma.nn.activation import GateActivation
    from fairchem.core.units.mlip_unit.api.inference import (
        InferenceSettings,
    )

__all__ = [
    "ExecutionMode",
    "ExecutionBackend",
    "UMASFastPytorchBackend",
    "UMASFastGPUBackend",
    "get_execution_backend",
    "maybe_update_settings_backend",
]

# Indices for m=0 spherical harmonic coefficients in L-major ordering (lmax=2)
_M0_COL_INDICES_L_ORDER = [0, 2, 6]


def _dense_l2_wigner(wigner: torch.Tensor) -> torch.Tensor:
    if wigner.ndim == 3:
        if wigner.shape[1:] != (9, 9):
            raise ValueError("wigner must have shape [E, 35] or [E, 9, 9]")
        return wigner
    if wigner.ndim != 2 or wigner.shape[1] != 35:
        raise ValueError("wigner must have shape [E, 35] or [E, 9, 9]")
    num_edges = wigner.shape[0]
    return (
        torch.nn.functional.pad(wigner[:, :1].view(num_edges, 1, 1), (0, 8, 0, 8))
        + torch.nn.functional.pad(wigner[:, 1:10].view(num_edges, 3, 3), (1, 5, 1, 5))
        + torch.nn.functional.pad(wigner[:, 10:].view(num_edges, 5, 5), (4, 0, 4, 0))
    )


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
        - node_to_edge_wigner_permute: Gather node features and rotate L->M
        - permute_wigner_inv_edge_to_node: Rotate M->L and scatter to nodes
        - edge_degree_scatter: Rotate radial and scatter to nodes
        - prepare_model_for_inference: Apply backend-specific model transforms
    """

    # Whether this backend exposes the fused edgewise SO2 path (producer conv1
    # pack + consumer conv2 inv fusion).
    supports_fused_edgewise: bool = False

    @staticmethod
    def validate(
        lmax: int,
        mmax: int,
        settings: InferenceSettings,
    ) -> None:
        """
        Validate that model parameters and settings are compatible with this backend.

        Called before first inference.

        Args:
            lmax: Maximum degree of spherical harmonics.
            mmax: Maximum order of spherical harmonics.
            settings: Inference settings.

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
    def node_to_edge_wigner_permute(
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
    def permute_wigner_inv_edge_to_node(
        x_message: torch.Tensor,
        wigner_inv: torch.Tensor,
        scatter_target: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Rotate M->L and scatter edge messages to nodes.

        Default: PyTorch BMM + index_add.

        Args:
            x_message: Edge message features [E, M, C]
            wigner_inv: Inverse Wigner matrices [E, L, M]
            scatter_target: Pre-computed local target indices [E]
                for scattering into node output tensor.
            num_nodes: Total number of nodes (output size)

        Returns:
            Node embeddings [N, L, C] accumulated from edge messages
        """
        # Rotate M->L
        x_rotated = torch.bmm(wigner_inv, x_message)
        # Scatter to nodes
        new_embedding = torch.zeros(
            (num_nodes,) + x_rotated.shape[1:],
            dtype=x_rotated.dtype,
            device=x_rotated.device,
        )
        new_embedding.index_add_(0, scatter_target, x_rotated)
        return new_embedding

    @staticmethod
    def edge_degree_scatter(
        x: torch.Tensor,
        radial_output: torch.Tensor,
        wigner_inv: torch.Tensor,
        scatter_target: torch.Tensor,
        m_0_num_coefficients: int,
        sphere_channels: int,
        rescale_factor: float,
    ) -> torch.Tensor:
        """
        Edge degree embedding: rotate radial and scatter to nodes.

        Default: PyTorch BMM + index_add.

        Args:
            x: Node features [N, L, C] to update
            radial_output: RadialMLP output [E, m0 * C]
            wigner_inv: Wigner inverse with envelope pre-fused
                [E, L, m0] or [E, L, L]
            scatter_target: Pre-computed local target indices [E]
                for scattering into node output tensor.
            m_0_num_coefficients: Number of m=0 coefficients
                (3 for lmax=2)
            sphere_channels: Number of channels C
            rescale_factor: Aggregation rescale factor

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
            scatter_target,
            x_edge_embedding / rescale_factor,
        )


class UMASFastPytorchBackend(ExecutionBackend):
    """
    Optimized PyTorch backend using block-diagonal SO2 convolutions.

    Requires merge_mole=True and activation_checkpointing=False.
    """

    @staticmethod
    def validate(
        lmax: int,
        mmax: int,
        settings: InferenceSettings,
    ) -> None:
        """
        Validate that settings are compatible with fast pytorch mode.
        """
        # Also reject if user tries to enable it via inference settings
        if settings is not None and settings.activation_checkpointing:
            raise ValueError(
                "UMASFastPytorchBackend requires activation_checkpointing=False"
            )
        if (
            settings is not None
            and settings.fp16_radial_fc2_blocks
            and settings.predict_untrained_hessian
        ):
            raise ValueError("fp16_radial_fc2_blocks does not support Hessians")

    @staticmethod
    def prepare_model_for_inference(model: torch.nn.Module) -> None:
        """
        Convert SO2_Convolution modules to block-diagonal GEMM variants
        and create unified radial MLP for batched computation.

        Replaces so2_conv_1 with SO2_Conv1_WithRadialBlock and
        so2_conv_2 with SO2_Conv2_InternalBlock in each block's
        Edgewise module. Then creates a UnifiedRadialMLP from all
        radial functions for efficient batched computation.
        """
        from fairchem.core.models.uma.nn.so2_layers import (
            convert_so2_conv1,
            convert_so2_conv2,
        )

        settings = getattr(model, "_inference_settings", None)
        fp16_radial_fc2_blocks = tuple(getattr(settings, "fp16_radial_fc2_blocks", ()))
        if fp16_radial_fc2_blocks and model.regress_config.hessian:
            raise ValueError("fp16_radial_fc2_blocks does not support Hessians")
        if fp16_radial_fc2_blocks and max(fp16_radial_fc2_blocks) >= len(model.blocks):
            raise ValueError(
                "fp16_radial_fc2_blocks contains an index outside model.blocks"
            )

        for block in model.blocks:
            block.edge_wise.so2_conv_1 = convert_so2_conv1(block.edge_wise.so2_conv_1)
            block.edge_wise.so2_conv_2 = convert_so2_conv2(block.edge_wise.so2_conv_2)

        # Create unified radial MLP for batched computation
        rad_funcs = [block.edge_wise.so2_conv_1.rad_func for block in model.blocks]
        model._unified_radial_mlp = UnifiedRadialMLP(rad_funcs)
        model._unified_radial_mlp.configure_fp16_fc2(fp16_radial_fc2_blocks)

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

    Extends UMASFastPytorchBackend with Triton-accelerated
    node_to_edge_wigner_permute, permute_wigner_inv_edge_to_node, and edge_degree_scatter.
    Requires lmax==2, mmax==2, and merge_mole=True.

    Note: sphere_channels % 128 == 0 gives optimal GPU utilization.
    Smaller values work but with reduced efficiency.
    """

    # Expose the fused edgewise path: producer conv1 pack + consumer conv2 inv.
    supports_fused_edgewise: bool = True

    @staticmethod
    def validate(
        lmax: int,
        mmax: int,
        settings: InferenceSettings,
    ) -> None:
        UMASFastPytorchBackend.validate(lmax, mmax, settings)
        if not torch.cuda.is_available():
            raise ValueError("umas_fast_gpu requires CUDA")
        if lmax != 2 or mmax != 2:
            raise ValueError("umas_fast_gpu requires lmax==2 and mmax==2")
        if not settings.merge_mole:
            raise ValueError("umas_fast_gpu requires merge_mole=True")
        if settings.predict_untrained_hessian and settings.hessian_vmap:
            raise ValueError(
                "umas_fast_gpu does not support hessian_vmap=True; "
                "set hessian_vmap=False"
            )

    @staticmethod
    def prepare_wigner(
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        mappingReduced,
        coefficient_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if wigner.shape[-1] == 35:
            return wigner, wigner_inv

        def pack_blocks(value: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                (
                    value[:, :1, :1].flatten(1),
                    value[:, 1:4, 1:4].flatten(1),
                    value[:, 4:9, 4:9].flatten(1),
                ),
                dim=1,
            )

        return pack_blocks(wigner), pack_blocks(wigner_inv)

    @staticmethod
    def node_to_edge_wigner_permute(
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        from fairchem.core.models.uma.triton import (
            UMASFastGPUNodeToEdgeWignerPermute,
        )

        return UMASFastGPUNodeToEdgeWignerPermute.apply(
            x_full, edge_index, _dense_l2_wigner(wigner)
        )

    @staticmethod
    def permute_wigner_inv_edge_to_node(
        x_message: torch.Tensor,
        wigner_inv: torch.Tensor,
        scatter_target: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        from fairchem.core.models.uma.triton import (
            UMASFastGPUPermuteWignerInvEdgeToNode,
        )

        # Rotate M->L using Triton kernel
        x_rotated = UMASFastGPUPermuteWignerInvEdgeToNode.apply(
            x_message, _dense_l2_wigner(wigner_inv)
        )
        # Scatter to nodes
        new_embedding = torch.zeros(
            (num_nodes,) + x_rotated.shape[1:],
            dtype=x_rotated.dtype,
            device=x_rotated.device,
        )
        new_embedding.index_add_(0, scatter_target, x_rotated)
        return new_embedding

    @staticmethod
    def edge_degree_scatter(
        x: torch.Tensor,
        radial_output: torch.Tensor,
        wigner_inv: torch.Tensor,
        scatter_target: torch.Tensor,
        m_0_num_coefficients: int,
        sphere_channels: int,
        rescale_factor: float,
    ) -> torch.Tensor:
        radial = radial_output.reshape(-1, m_0_num_coefficients, sphere_channels)

        if wigner_inv.shape[-1] == 35:
            wigner = wigner_inv.reshape(-1, 35)
            x_edge_embedding = torch.cat(
                (
                    wigner[:, 0:1, None] * radial[:, 0:1],
                    wigner[:, (2, 5, 8), None] * radial[:, 1:2],
                    wigner[:, (12, 17, 22, 27, 32), None] * radial[:, 2:3],
                ),
                dim=1,
            )
            return x.index_add(
                0,
                scatter_target,
                x_edge_embedding.to(x.dtype) / rescale_factor,
            )

        # Select m=0 columns from L-ordered wigner_inv
        wigner_inv_m0 = wigner_inv[:, :, _M0_COL_INDICES_L_ORDER]
        x_edge_embedding = torch.bmm(wigner_inv_m0, radial)

        x_edge_embedding = x_edge_embedding.to(x.dtype)

        return x.index_add(
            0,
            scatter_target,
            x_edge_embedding / rescale_factor,
        )

    @staticmethod
    def fused_node_to_edge_conv1_pack(
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
        radial: torch.Tensor,
        sphere_channels: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Producer-side fusion: emit conv1's scaled + GEMM-packed buffers directly.

        Fuses the node_to_edge gather, block-diagonal Wigner rotation, L->M
        permute, and conv1 per-m radial scale/pack into one op. The [E,9,2C]
        x_message intermediate never materializes.

        Args:
            x_full: Node features [N, 9, C] (L-major).
            edge_index: Edge indices [2, E].
            wigner: Compact Wigner rotation blocks [E, 35].
            radial: Per-layer conv1 radial embedding [E, 6*2C] (rad_func applied).
            sphere_channels: Number of channels C.

        Returns:
            (m0, m1, m2) GEMM-ready packed buffers for conv1.
        """
        from fairchem.core.models.uma.triton import wigner_conv1_fused_op

        return wigner_conv1_fused_op(
            x_full, edge_index, wigner, radial, sphere_channels
        )

    @staticmethod
    def gate_activation(
        x0_full: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        channels: int,
        activation: GateActivation,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            channels > 0
            and channels & (channels - 1) == 0
            and x0_full.dtype == torch.float32
            and x1.dtype == torch.float32
            and x2.dtype == torch.float32
        ):
            from fairchem.core.models.uma.triton import packed_gate_op

            return packed_gate_op(x0_full, x1, x2, channels)

        gating, x0 = x0_full.split((2 * channels, 3 * channels), dim=-1)
        return activation.forward_m_blocks(gating, (x0, x1, x2))

    @staticmethod
    def fused_conv2_inv_edge_to_node(
        g0: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        wigner_inv_envelope: torch.Tensor,
        scatter_target: torch.Tensor,
        num_nodes: int,
        sphere_channels: int,
    ) -> torch.Tensor:
        """
        Consumer-side fusion: unpack conv2 GEMM buffers + inv-rotate + scatter.

        Fuses the M->L unpack, inverse-Wigner rotation, and node scatter of the
        three conv2 block-GEMM outputs (g0,g1,g2) without materializing an
        [E,9,C] intermediate.

        Args:
            g0: conv2 fc_m0 output [E, 3C].
            g1: conv2 m=1 block-GEMM output [E, 4C].
            g2: conv2 m=2 block-GEMM output [E, 2C].
            wigner_inv_envelope: Compact inverse Wigner blocks with the
                envelope pre-fused [E, 35].
            scatter_target: Pre-computed local target indices [E] for
                scattering into the node output tensor. In the non-GP case
                this is ``edge_index[1]``; under GP it is the caller's
                local-partition remap (see Edgewise.forward).
            num_nodes: Total number of nodes (output size).
            sphere_channels: Number of channels C.

        Returns:
            Node embeddings [N, 9, C] accumulated from edge messages.
        """
        from fairchem.core.models.uma.triton import wigner_inv_conv2_scatter_op

        return wigner_inv_conv2_scatter_op(
            g0,
            g1,
            g2,
            wigner_inv_envelope,
            scatter_target,
            num_nodes,
            sphere_channels,
        )


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
    model_config: dict,
) -> InferenceSettings:
    """
    Update inference settings to use UMAS_FAST_GPU if conditions are met.

    Sets execution_mode to UMAS_FAST_GPU if:
    - execution_mode is not already set
    - UMASFastGPUBackend.validate passes for the model and settings

    Args:
        settings: Current inference settings.
        model_config: The model configuration dictionary to validate.

    Returns:
        Updated inference settings with the appropriate execution mode.
    """
    if settings.execution_mode is not None:
        return settings

    try:
        lmax = model_config["backbone"]["lmax"]
        mmax = model_config["backbone"]["mmax"]
        UMASFastGPUBackend.validate(lmax, mmax, settings)
        return replace(settings, execution_mode=ExecutionMode.UMAS_FAST_GPU)
    except (ValueError, KeyError):
        return settings
