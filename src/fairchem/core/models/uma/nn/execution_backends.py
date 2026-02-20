"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

__all__ = [
    "ExecutionMode",
    "ExecutionBackend",
    "UMASFastPytorchBackend",
    "TritonBackend",
    "TritonAtomicBackend",
    "TritonPytorchBwdBackend",
    "TritonRecomputeBackend",
    "get_execution_backend",
]


class ExecutionMode(str, Enum):
    """
    Execution mode for model inference.
    """

    GENERAL = "general"
    UMAS_FAST_PYTORCH = "umas_fast_pytorch"
    TRITON_SCATTER_ADD = "triton_scatter_add"
    TRITON_ATOMIC = "triton_atomic"
    TRITON_PYTORCH_BWD = "triton_pytorch_bwd"
    TRITON_RECOMPUTE = "triton_recompute"


# Set of all triton execution modes for easy membership testing
TRITON_MODES = frozenset(
    {
        ExecutionMode.TRITON_SCATTER_ADD,
        ExecutionMode.TRITON_ATOMIC,
        ExecutionMode.TRITON_PYTORCH_BWD,
        ExecutionMode.TRITON_RECOMPUTE,
    }
)


class ExecutionBackend:
    """
    Parameterless function dispatch for execution modes.

    Provides default PyTorch implementations for rotation and scatter
    operations. Subclass and override methods with optimized kernels
    (e.g. Triton) for specific execution modes.

    All methods are static — backends carry no instance state.

    Methods (override for optimization):
        - prepare_wigner: Transform raw Wigner matrices for this backend
        - gather_rotate: Gather node features and rotate L->M
        - rotate_back: Rotate M->L
        - gate_activation: Apply gating activation
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
            "mk,nkj->nmj", mappingReduced.to_m.to(wigner.dtype), wigner
        )
        wigner_inv = torch.einsum(
            "njk,mk->njm", wigner_inv, mappingReduced.to_m.to(wigner_inv.dtype)
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
    def gate_activation(
        x_0_gating: torch.Tensor,
        x_message: torch.Tensor,
        act: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Apply gating activation.

        Default: Delegates to the provided activation module.

        Args:
            x_0_gating: Gating scalars
            x_message: Message features
            act: Activation module (e.g. GateActivation)

        Returns:
            Activated message features
        """
        return act(x_0_gating, x_message)

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
                "UMASFastPytorchBackend requires activation_checkpointing=False"
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


class TritonBackend(ExecutionBackend):
    """
    Triton-accelerated backend using fused kernels for gather+rotate.

    Uses raw Wigner matrices (handles M-mapping internally in kernels).
    Requires lmax=mmax=2, sphere_channels % 128 == 0, and no activation
    checkpointing.

    Default backward uses two-phase scatter_add (no atomics).
    Subclasses override only ``gather_rotate`` to select a different
    backward strategy.

    All methods are static — no instance state.
    """

    @staticmethod
    def validate(
        model: torch.nn.Module,
        settings: InferenceSettings | None = None,
    ) -> None:
        """
        Validate that Triton is available and model/settings are compatible.
        """
        from fairchem.core.models.uma.triton import HAS_TRITON

        if not HAS_TRITON:
            raise ValueError(
                "Triton is required for TritonBackend but is not installed."
            )
        if model.lmax != 2 or model.mmax != 2:
            raise ValueError("Triton backends require lmax=mmax=2")
        if model.sphere_channels % 128 != 0:
            raise ValueError("Triton backends require sphere_channels divisible by 128")
        if settings is not None and settings.activation_checkpointing:
            raise ValueError("TritonBackend requires activation_checkpointing=False")

    @staticmethod
    def prepare_wigner(
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        mappingReduced,
        coefficient_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return raw Wigner matrices unchanged.

        Triton kernels handle M-mapping internally.
        """
        return wigner, wigner_inv

    @staticmethod
    def gather_rotate(
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather node features and rotate L->M using Triton.

        Uses scatter_add (two-phase, no atomics) backward by default.
        """
        from fairchem.core.models.uma.triton import (
            FusedEdgeGatherWignerL2MTritonV2BwdFunction,
        )

        return FusedEdgeGatherWignerL2MTritonV2BwdFunction.apply(
            x_full, edge_index, wigner
        )

    @staticmethod
    def rotate_back(
        x: torch.Tensor,
        wigner_inv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotate M->L using Triton.
        """
        from fairchem.core.models.uma.triton import m_to_l_then_wigner_lmax2

        return m_to_l_then_wigner_lmax2(x, wigner_inv)

    @staticmethod
    def gate_activation(
        x_0_gating: torch.Tensor,
        x_message: torch.Tensor,
        act: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Fused gate activation using Triton.
        """
        from fairchem.core.models.uma.triton import gate_activation_triton

        return gate_activation_triton(x_0_gating, x_message)

    @staticmethod
    def prepare_model_for_inference(model: torch.nn.Module) -> None:
        """
        Convert SO2_Convolution modules to block-diagonal GEMM variants.
        """
        from fairchem.core.models.uma.nn.so2_layers import (
            convert_so2_conv1,
            convert_so2_conv2,
        )

        for block in model.blocks:
            block.edge_wise.so2_conv_1 = convert_so2_conv1(block.edge_wise.so2_conv_1)
            block.edge_wise.so2_conv_2 = convert_so2_conv2(block.edge_wise.so2_conv_2)


class TritonAtomicBackend(TritonBackend):
    """
    Triton backend using atomic-based backward.

    Overrides only ``gather_rotate`` to use the atomic scatter backward.
    """

    @staticmethod
    def gather_rotate(
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather node features and rotate L->M using Triton.

        Uses atomic-based Triton backward.
        """
        from fairchem.core.models.uma.triton import (
            FusedEdgeGatherWignerL2MTritonBwdFunction,
        )

        return FusedEdgeGatherWignerL2MTritonBwdFunction.apply(
            x_full, edge_index, wigner
        )


class TritonPytorchBwdBackend(TritonBackend):
    """
    Triton backend using PyTorch backward.

    Triton forward, PyTorch backward. Overrides only ``gather_rotate``.
    """

    @staticmethod
    def gather_rotate(
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather node features and rotate L->M using Triton.

        Uses PyTorch backward.
        """
        from fairchem.core.models.uma.triton import (
            FusedEdgeGatherWignerL2MPyTorchBwdFunction,
        )

        return FusedEdgeGatherWignerL2MPyTorchBwdFunction.apply(
            x_full, edge_index, wigner
        )


class TritonRecomputeBackend(TritonBackend):
    """
    Triton backend using memory-optimized recompute backward.

    Saves ~670MB for typical 2000 node, 74K edge graphs by
    recomputing edge features in backward instead of saving them.
    Overrides only ``gather_rotate``.
    """

    @staticmethod
    def gather_rotate(
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather node features and rotate L->M using Triton.

        Uses memory-optimized recompute backward.
        """
        from fairchem.core.models.uma.triton import (
            FusedEdgeGatherWignerL2MRecomputeFunction,
        )

        return FusedEdgeGatherWignerL2MRecomputeFunction.apply(
            x_full, edge_index, wigner
        )


_EXECUTION_BACKENDS: dict[ExecutionMode, type[ExecutionBackend]] = {
    ExecutionMode.GENERAL: ExecutionBackend,
    ExecutionMode.UMAS_FAST_PYTORCH: UMASFastPytorchBackend,
    ExecutionMode.TRITON_SCATTER_ADD: TritonBackend,
    ExecutionMode.TRITON_ATOMIC: TritonAtomicBackend,
    ExecutionMode.TRITON_PYTORCH_BWD: TritonPytorchBwdBackend,
    ExecutionMode.TRITON_RECOMPUTE: TritonRecomputeBackend,
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
