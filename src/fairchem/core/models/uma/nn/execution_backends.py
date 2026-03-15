"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING

import torch

# Enable expandable segments for the CUDA caching allocator to reduce
# memory fragmentation and eliminate periodic GC stalls during inference.
# Must be set before the first CUDA allocation.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Enable coordinate descent tuning for inductor-generated kernels
torch._inductor.config.coordinate_descent_tuning = True
# Enable aggressive fusion of inductor ops
torch._inductor.config.aggressive_fusion = True
# Reduce inductor compilation overhead
torch._inductor.config.triton.unique_kernel_names = False
# Allow inductor to reorder nodes for better locality
torch._inductor.config.reorder_for_locality = True
# Disable size asserts in generated code (small speedup)
torch._inductor.config.size_asserts = False
# Disable automatic dynamic shapes — use static shapes for faster code
torch._dynamo.config.automatic_dynamic_shapes = False

from fairchem.core.models.uma.nn.unified_radial import UnifiedRadialMLP

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

__all__ = [
    "ExecutionMode",
    "ExecutionBackend",
    "UMASFastPytorchBackend",
    "UMASFastGPUBackend",
    "UMASFastCPUBackend",
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
    UMAS_FAST_CPU = "umas_fast_cpu"


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
        edge_index: torch.Tensor,
        num_nodes: int,
        node_offset: int = 0,
    ) -> torch.Tensor:
        """
        Rotate M->L and scatter edge messages to nodes.

        Default: PyTorch BMM + index_add.

        Args:
            x_message: Edge message features [E, M, C]
            wigner_inv: Inverse Wigner matrices [E, L, M]
            edge_index: Edge indices [2, E]
            num_nodes: Total number of nodes (output size)
            node_offset: Offset for node indices (for chunking)

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
        new_embedding.index_add_(0, edge_index[1] - node_offset, x_rotated)
        return new_embedding

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

        for block in model.blocks:
            block.edge_wise.so2_conv_1 = convert_so2_conv1(block.edge_wise.so2_conv_1)
            block.edge_wise.so2_conv_2 = convert_so2_conv2(block.edge_wise.so2_conv_2)

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


class _EdgeDegreeScatterFunction(torch.autograd.Function):
    """
    Custom autograd for edge_degree_scatter to reduce backward overhead.

    Fuses: W_m0 @ radial / rescale + scatter into a single autograd node
    instead of separate bmm → to → div → index_add nodes.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        radial: torch.Tensor,
        wigner_inv_m0: torch.Tensor,
        edge_index_1: torch.Tensor,
        rescale_factor: float,
    ) -> torch.Tensor:
        x_edge = torch.bmm(wigner_inv_m0, radial)
        x_edge = x_edge.to(x.dtype)
        inv_rescale = 1.0 / rescale_factor
        result = x.index_add(0, edge_index_1, x_edge * inv_rescale)
        ctx.save_for_backward(radial, wigner_inv_m0, edge_index_1)
        ctx.inv_rescale = inv_rescale
        ctx.num_nodes = x.shape[0]
        return result

    @staticmethod
    def backward(ctx, grad_out):
        radial, wigner_inv_m0, edge_index_1 = ctx.saved_tensors
        inv_rescale = ctx.inv_rescale

        # grad of index_add w.r.t. x is identity
        grad_x = grad_out

        # grad of index_add w.r.t. src is gather
        grad_edge = grad_out[edge_index_1] * inv_rescale  # [E, 9, C]
        grad_edge = grad_edge.to(radial.dtype)

        # grad_radial = W_m0^T @ grad_edge  → [E, m0, C]
        grad_radial = torch.bmm(wigner_inv_m0.transpose(1, 2), grad_edge)

        # grad_wigner_m0 = grad_edge @ radial^T → [E, 9, m0]
        grad_wigner_m0 = torch.bmm(grad_edge, radial.transpose(1, 2))

        return grad_x, grad_radial, grad_wigner_m0, None, None


class UMASFastGPUBackend(UMASFastPytorchBackend):
    """
    GPU-optimized backend: SO2 block conversion + Triton kernels.

    Extends UMASFastPytorchBackend with Triton-accelerated
    node_to_edge_wigner_permute, permute_wigner_inv_edge_to_node, and edge_degree_scatter.
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
    def prepare_wigner(
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        mappingReduced,
        coefficient_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Passthrough — Triton kernels handle L-to-M internally
        return wigner, wigner_inv

    @staticmethod
    def node_to_edge_wigner_permute(
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        from fairchem.core.models.uma.triton import (
            UMASFastGPUNodeToEdgeWignerPermute,
        )

        return UMASFastGPUNodeToEdgeWignerPermute.apply(x_full, edge_index, wigner)

    @staticmethod
    def permute_wigner_inv_edge_to_node(
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
        wigner_inv_m0 = wigner_inv[:, :, _M0_COL_INDICES_L_ORDER]

        return _EdgeDegreeScatterFunction.apply(
            x, radial, wigner_inv_m0, edge_index[1] - node_offset, rescale_factor
        )


class UMASFastCPUBackend(UMASFastPytorchBackend):
    """
    CPU-optimized backend: SO2 block conversion + CPU kernels.

    Extends UMASFastPytorchBackend with CPU-optimized
    node_to_edge_wigner_permute and permute_wigner_inv_edge_to_node
    that use block-diagonal Wigner structure with L↔M permutation
    (same math as the GPU Triton kernels, but using PyTorch CPU ops).

    Requires lmax==2, mmax==2, and merge_mole=True.
    """

    @staticmethod
    def validate(
        model: torch.nn.Module,
        settings: InferenceSettings | None = None,
    ) -> None:
        UMASFastPytorchBackend.validate(model, settings)
        if model.lmax != 2 or model.mmax != 2:
            raise ValueError("umas_fast_cpu requires lmax==2 and mmax==2")
        if settings is not None and not settings.merge_mole:
            raise ValueError("umas_fast_cpu requires merge_mole=True")

    @staticmethod
    def prepare_wigner(
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        mappingReduced,
        coefficient_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Passthrough — CPU kernels handle L-to-M internally
        return wigner, wigner_inv

    @staticmethod
    def node_to_edge_wigner_permute(
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        from fairchem.core.models.uma.cpu import (
            CPUNodeToEdgeWignerPermuteFunction,
        )

        return CPUNodeToEdgeWignerPermuteFunction.apply(x_full, edge_index, wigner)

    @staticmethod
    def permute_wigner_inv_edge_to_node(
        x_message: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_offset: int = 0,
    ) -> torch.Tensor:
        from fairchem.core.models.uma.cpu import (
            CPUPermuteWignerInvEdgeToNodeFunction,
        )

        # Rotate M->L using CPU kernel
        x_rotated = CPUPermuteWignerInvEdgeToNodeFunction.apply(x_message, wigner_inv)
        # Scatter to nodes
        new_embedding = torch.zeros(
            (num_nodes,) + x_rotated.shape[1:],
            dtype=x_rotated.dtype,
            device=x_rotated.device,
        )
        new_embedding.index_add_(0, edge_index[1] - node_offset, x_rotated)
        return new_embedding

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
        wigner_inv_m0 = wigner_inv[:, :, _M0_COL_INDICES_L_ORDER]

        return _EdgeDegreeScatterFunction.apply(
            x, radial, wigner_inv_m0, edge_index[1] - node_offset, rescale_factor
        )


_EXECUTION_BACKENDS: dict[ExecutionMode, type[ExecutionBackend]] = {
    ExecutionMode.GENERAL: ExecutionBackend,
    ExecutionMode.UMAS_FAST_PYTORCH: UMASFastPytorchBackend,
    ExecutionMode.UMAS_FAST_GPU: UMASFastGPUBackend,
    ExecutionMode.UMAS_FAST_CPU: UMASFastCPUBackend,
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
