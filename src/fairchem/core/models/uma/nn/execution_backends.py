"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from fairchem.core.models.uma.common.so3 import CoefficientMapping


__all__ = [
    "ExecutionMode",
    "BaseExecutionBackend",
    "EdgeDegreeBackend",
    "BaselineBackend",
    "FastPytorchBackend",
    "TritonBackend",
    "CppBackend",
    "get_execution_backend",
    "is_fast_mode",
    "uses_raw_wigner",
]


class ExecutionMode(str, Enum):
    """
    Execution mode for model inference.
    """

    BASELINE = "baseline"
    FAST_TRITON = "fast_triton"
    FAST_CPP = "fast_cpp"
    FAST_PYTORCH = "fast_pytorch"


# Check for Triton availability
try:
    from fairchem.core.models.uma.nn.triton import (
        HAS_TRITON,
        triton_gather_rotate_l2m,
        triton_rotate_m2l,
    )
    from fairchem.core.models.uma.triton import gate_activation_triton
except ImportError:
    HAS_TRITON = False
    triton_gather_rotate_l2m = None
    triton_rotate_m2l = None
    gate_activation_triton = None


# Check for C++ kernels availability (placeholder)
HAS_CPP_KERNELS = False
cpp_gather_rotate_l2m = None
cpp_rotate_m2l = None


class BaseExecutionBackend(nn.Module, ABC):
    """
    Abstract base class for execution backends.

    Each backend creates its own SO2 convolution modules and provides
    unified method signatures for forward operations.

    Methods:
        ABSTRACT (must override):
            - conv1: Backend-specific SO2 convolution with radial
            - conv2: Backend-specific SO2 convolution with internal weights

        DEFAULT (override only for optimization):
            - gather_rotate: PyTorch gather + BMM (Triton/Cpp override)
            - rotate_back: PyTorch BMM (Triton/Cpp override)
            - gate_activation: PyTorch sigmoid + silu (Triton overrides)
            - edge_degree_scatter: PyTorch BMM + index_add

    Attributes:
        wigner_format: "raw" or "m_mapped"
        mode: The ExecutionMode this backend implements
    """

    wigner_format: str = "m_mapped"
    mode: ExecutionMode

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by each backend
    # =========================================================================

    @abstractmethod
    def conv1(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
        precomputed_radial: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        First SO2 convolution with radial modulation.

        Args:
            x: Input features [E, M, C]
            x_edge: Edge embeddings [E, D] (used by baseline)
            precomputed_radial: Precomputed radial weights [E, R]

        Returns:
            (output, gating): output [E, M, C'], gating [E, G]
        """
        ...

    @abstractmethod
    def conv2(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
    ) -> torch.Tensor:
        """
        Second SO2 convolution (internal weights).

        Args:
            x: Input features [E, M, C]
            x_edge: Edge embeddings [E, D] (used by baseline)

        Returns:
            Output features [E, M, C']
        """
        ...

    # =========================================================================
    # DEFAULT IMPLEMENTATIONS - Override in subclasses for optimized versions
    # =========================================================================

    def gather_rotate(
        self,
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

    def rotate_back(
        self,
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

    def gate_activation(
        self,
        gating_scalars: torch.Tensor,
        input_tensors: torch.Tensor,
        lmax: int = 2,
    ) -> torch.Tensor:
        """
        Apply gated activation.

        Default: PyTorch sigmoid + silu.

        Args:
            gating_scalars: [E, lmax * C] gating scalars
            input_tensors: [E, (lmax+1)^2, C] input tensors
            lmax: Maximum L value (default 2 for UMA-S)

        Returns:
            Gated output [E, (lmax+1)^2, C]
        """
        num_channels = input_tensors.shape[2]
        gate = torch.sigmoid(gating_scalars).view(
            gating_scalars.shape[0], lmax, num_channels
        )
        # expand_index for lmax=2, m_prime=True: [0, 1, 0, 1, 0, 1, 1, 1]
        expand_index = torch.tensor([0, 1, 0, 1, 0, 1, 1, 1], device=gate.device)
        gate = torch.index_select(gate, dim=1, index=expand_index)

        scalars = input_tensors[:, :1]
        vectors = input_tensors[:, 1:]
        scalars = torch.nn.functional.silu(scalars)
        vectors = vectors * gate

        return torch.cat((scalars, vectors), dim=1)

    def edge_degree_scatter(
        self,
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
            edge_index: Edge indices [2, E]
            m_0_num_coefficients: Number of m=0 coefficients
            sphere_channels: Number of channels C
            rescale_factor: Aggregation rescale factor
            node_offset: Node offset for graph parallelism

        Returns:
            Updated node features [N, L, C]
        """
        # Reshape radial output: [E, m0*C] -> [E, m0, C]
        radial = radial_output.reshape(-1, m_0_num_coefficients, sphere_channels)

        # Slice wigner to m=0 columns and rotate
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


class EdgeDegreeBackend(BaseExecutionBackend):
    """
    Lightweight backend for EdgeDegreeEmbedding only.

    Only provides edge_degree_scatter - no SO2 convolutions created.
    """

    wigner_format = "m_mapped"
    mode = ExecutionMode.BASELINE

    def __init__(self):
        super().__init__()

    def conv1(self, x, x_edge, precomputed_radial):
        raise NotImplementedError("EdgeDegreeBackend does not support conv1")

    def conv2(self, x, x_edge):
        raise NotImplementedError("EdgeDegreeBackend does not support conv2")

    # Uses inherited defaults:
    # - gather_rotate
    # - rotate_back
    # - gate_activation
    # - edge_degree_scatter


class BaselineBackend(BaseExecutionBackend):
    """
    Reference PyTorch implementation using SO2_Convolution.

    Uses standard SO2_Convolution with internal radial MLP.
    Edge embeddings are passed to conv1/conv2 for on-the-fly radial
    computation.
    """

    wigner_format = "m_mapped"
    mode = ExecutionMode.BASELINE

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        extra_m0_output_channels: int,
        edge_channels_list: list[int],
    ):
        super().__init__()

        from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution

        self.so2_conv_1 = SO2_Convolution(
            2 * sphere_channels,
            hidden_channels,
            lmax,
            mmax,
            mappingReduced,
            internal_weights=False,
            edge_channels_list=copy.deepcopy(edge_channels_list),
            extra_m0_output_channels=extra_m0_output_channels,
        )
        self.so2_conv_2 = SO2_Convolution(
            hidden_channels,
            sphere_channels,
            lmax,
            mmax,
            mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )

    def conv1(self, x, x_edge, precomputed_radial):
        # Baseline uses x_edge for on-the-fly radial computation
        return self.so2_conv_1(x, x_edge)

    def conv2(self, x, x_edge):
        # Baseline uses x_edge
        return self.so2_conv_2(x, x_edge)

    # gather_rotate: inherited from BaseExecutionBackend
    # rotate_back: inherited from BaseExecutionBackend
    # gate_activation: inherited from BaseExecutionBackend
    # edge_degree_scatter: inherited from BaseExecutionBackend


class FastPytorchBackend(BaseExecutionBackend):
    """
    Optimized PyTorch using specialized SO2 classes.

    Uses SO2_Conv1_WithRadialBlock and SO2_Conv2_InternalBlock which
    expect precomputed radial weights from unified radial MLP.
    """

    wigner_format = "m_mapped"
    mode = ExecutionMode.FAST_PYTORCH

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        extra_m0_output_channels: int,
        edge_channels_list: list[int],
    ):
        super().__init__()

        from fairchem.core.models.uma.nn.so2_layers import (
            SO2_Conv1_WithRadialBlock,
            SO2_Conv2_InternalBlock,
        )

        self.so2_conv_1 = SO2_Conv1_WithRadialBlock(
            2 * sphere_channels,
            hidden_channels,
            lmax,
            mmax,
            mappingReduced,
            extra_m0_output_channels=extra_m0_output_channels,
            edge_channels_list=copy.deepcopy(edge_channels_list),
        )
        self.so2_conv_2 = SO2_Conv2_InternalBlock(
            hidden_channels,
            sphere_channels,
            lmax,
            mmax,
            mappingReduced,
        )

    def conv1(self, x, x_edge, precomputed_radial):
        # Fast mode uses precomputed radial
        return self.so2_conv_1(x, precomputed_radial)

    def conv2(self, x, x_edge):
        # Fast mode: no args needed (internal weights)
        return self.so2_conv_2(x)

    # gather_rotate: inherited from BaseExecutionBackend
    # rotate_back: inherited from BaseExecutionBackend
    # gate_activation: inherited from BaseExecutionBackend
    # edge_degree_scatter: inherited from BaseExecutionBackend


class TritonBackend(BaseExecutionBackend):
    """
    Triton GPU kernels with specialized SO2 classes.

    Uses Triton kernels for rotation operations and specialized SO2
    classes that expect precomputed radial weights.

    Requires: GPU with Triton, lmax=mmax=2, sphere_channels % 128 == 0
    """

    wigner_format = "raw"
    mode = ExecutionMode.FAST_TRITON

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        extra_m0_output_channels: int,
        edge_channels_list: list[int],
        backward_impl: str = "scatter_add",
    ):
        super().__init__()

        if not HAS_TRITON:
            raise RuntimeError(
                "TritonBackend requires Triton but it's not available. "
                "Install with: pip install triton"
            )

        from fairchem.core.models.uma.nn.so2_layers import (
            SO2_Conv1_WithRadialBlock,
            SO2_Conv2_InternalBlock,
        )

        self.backward_impl = backward_impl

        self.so2_conv_1 = SO2_Conv1_WithRadialBlock(
            2 * sphere_channels,
            hidden_channels,
            lmax,
            mmax,
            mappingReduced,
            extra_m0_output_channels=extra_m0_output_channels,
            edge_channels_list=copy.deepcopy(edge_channels_list),
        )
        self.so2_conv_2 = SO2_Conv2_InternalBlock(
            hidden_channels,
            sphere_channels,
            lmax,
            mmax,
            mappingReduced,
        )

    def gather_rotate(self, x_full, edge_index, wigner):
        return triton_gather_rotate_l2m(
            x_full,
            edge_index,
            wigner,
            backward_impl=self.backward_impl,
        )

    def conv1(self, x, x_edge, precomputed_radial):
        return self.so2_conv_1(x, precomputed_radial)

    def conv2(self, x, x_edge):
        return self.so2_conv_2(x)

    def rotate_back(self, x, wigner_inv):
        return triton_rotate_m2l(x, wigner_inv)

    def gate_activation(self, gating_scalars, input_tensors, lmax=2):
        # Use Triton kernel for maximum performance
        return gate_activation_triton(gating_scalars, input_tensors)

    # edge_degree_scatter: inherited from BaseExecutionBackend


class CppBackend(BaseExecutionBackend):
    """
    C++ CPU kernels with specialized SO2 classes.

    Placeholder for future C++ CPU kernel implementation.
    Uses specialized SO2 classes with C++ rotation kernels.
    """

    wigner_format = "raw"
    mode = ExecutionMode.FAST_CPP

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        extra_m0_output_channels: int,
        edge_channels_list: list[int],
    ):
        super().__init__()

        if not HAS_CPP_KERNELS:
            raise RuntimeError(
                "CppBackend requires C++ kernels but they're not "
                "available. "
                "Build with: python setup.py build_ext --inplace"
            )

        from fairchem.core.models.uma.nn.so2_layers import (
            SO2_Conv1_WithRadialBlock,
            SO2_Conv2_InternalBlock,
        )

        self.so2_conv_1 = SO2_Conv1_WithRadialBlock(
            2 * sphere_channels,
            hidden_channels,
            lmax,
            mmax,
            mappingReduced,
            extra_m0_output_channels=extra_m0_output_channels,
            edge_channels_list=copy.deepcopy(edge_channels_list),
        )
        self.so2_conv_2 = SO2_Conv2_InternalBlock(
            hidden_channels,
            sphere_channels,
            lmax,
            mmax,
            mappingReduced,
        )

    def gather_rotate(self, x_full, edge_index, wigner):
        return cpp_gather_rotate_l2m(x_full, edge_index, wigner)

    def conv1(self, x, x_edge, precomputed_radial):
        return self.so2_conv_1(x, precomputed_radial)

    def conv2(self, x, x_edge):
        return self.so2_conv_2(x)

    def rotate_back(self, x, wigner_inv):
        return cpp_rotate_m2l(x, wigner_inv)

    # gate_activation: inherited from BaseExecutionBackend
    # edge_degree_scatter: inherited from BaseExecutionBackend


def get_execution_backend(
    mode: ExecutionMode | str,
    sphere_channels: int,
    hidden_channels: int,
    lmax: int,
    mmax: int,
    mappingReduced: CoefficientMapping,
    extra_m0_output_channels: int,
    edge_channels_list: list[int],
    backward_impl: str = "triton",
) -> BaseExecutionBackend:
    """
    Factory function to create the appropriate execution backend.

    Args:
        mode: Execution mode (enum or string)
        sphere_channels: Number of sphere channels
        hidden_channels: Number of hidden channels
        lmax: Maximum L value
        mmax: Maximum M value
        mappingReduced: Coefficient mapping for SO2 convolutions
        extra_m0_output_channels: Extra m=0 channels for gating
        edge_channels_list: Edge channel dimensions for radial MLP
        backward_impl: Backward implementation for Triton

    Returns:
        Configured execution backend instance
    """
    if isinstance(mode, str):
        mode = ExecutionMode(mode)

    common_args = dict(
        sphere_channels=sphere_channels,
        hidden_channels=hidden_channels,
        lmax=lmax,
        mmax=mmax,
        mappingReduced=mappingReduced,
        extra_m0_output_channels=extra_m0_output_channels,
        edge_channels_list=edge_channels_list,
    )

    if mode == ExecutionMode.BASELINE:
        return BaselineBackend(**common_args)
    elif mode == ExecutionMode.FAST_PYTORCH:
        return FastPytorchBackend(**common_args)
    elif mode == ExecutionMode.FAST_TRITON:
        return TritonBackend(**common_args, backward_impl=backward_impl)
    elif mode == ExecutionMode.FAST_CPP:
        return CppBackend(**common_args)
    else:
        raise ValueError(f"Unknown execution mode: {mode}")


def is_fast_mode(mode: ExecutionMode | str) -> bool:
    """
    Check if mode uses specialized SO2 classes (fast modes).
    """
    if isinstance(mode, str):
        mode = ExecutionMode(mode)
    return mode in (
        ExecutionMode.FAST_TRITON,
        ExecutionMode.FAST_CPP,
        ExecutionMode.FAST_PYTORCH,
    )


def uses_raw_wigner(mode: ExecutionMode | str) -> bool:
    """
    Check if mode uses raw Wigner matrices (no M-mapping pre-applied).
    """
    if isinstance(mode, str):
        mode = ExecutionMode(mode)
    return mode in (ExecutionMode.FAST_TRITON, ExecutionMode.FAST_CPP)
