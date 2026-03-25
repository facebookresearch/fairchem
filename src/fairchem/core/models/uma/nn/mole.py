"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from contextlib import suppress
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

fairchem_cpp_found = False
with suppress(ModuleNotFoundError):
    import fairchem_cpp  # try to use DGL if available

    fairchem_cpp_found = True

# Optional cuBLAS grouped GEMM support
_cublas_available = False
try:
    import numpy as np
    import nvmath.bindings.cublas as cublas

    _cublas_available = True
except ImportError:
    np = None


def interval_intersection(interval1, interval2):
    """
    Compute intersection of two intervals [a, b] and [c, d]
    Returns None if no intersection, otherwise returns [start, end]
    """
    a, b = interval1
    c, d = interval2

    start = max(a, c)
    end = min(b, d)

    if start <= end:
        return [start, end]
    else:
        return None  # No intersection


def _softmax(x):
    return torch.softmax(x, dim=1) + 0.005


def _pnorm(x):
    return torch.nn.functional.normalize(x.abs() + 2 / x.shape[0], p=1.0, dim=1)


def norm_str_to_fn(act):
    if act == "softmax":
        return _softmax
    elif act == "pnorm":
        return _pnorm
    else:
        raise ValueError


@dataclass
class MOLEGlobals:
    # the linear coefficient for each expert
    expert_mixing_coefficients: torch.Tensor
    # if the input contains N separate systems, then the sizes represent the number of atoms in each system
    # this is used to for the MoLE to assign the correct parameters for each system
    mole_sizes: torch.Tensor
    # when using activation checkpointing, the inputs are chunked and given piecemeal so the start idx must be
    # updated each time the chunked operation happens. It's better to make this an input but in order for
    # the MolE interface to maintain functional equivalence to the Linear layer interface, this extra info
    # needs to be added here instead. (TODO: is there a cleaner way to do this?)
    ac_start_idx: int = 0
    # cuBLAS handle for grouped GEMM (created once, reused)
    _cublas_handle: int | None = None
    # Keep numpy arrays alive across a full forward+backward pass to prevent
    # GC while cuBLAS async kernels may still be reading from host buffers.
    # Cleared at each new batch via invalidate_pad_cache.
    _cublas_keepalive: list | None = None

    def get_cublas_keepalive(self):
        """Get or create the keepalive list for cuBLAS numpy arrays.

        Numpy arrays passed to cuBLAS sgemm_grouped_batched are read
        asynchronously. We must keep them alive until the next batch
        boundary (when invalidate_pad_cache is called).
        """
        if self._cublas_keepalive is None:
            self._cublas_keepalive = []
        return self._cublas_keepalive

    def get_cublas_handle(self):
        """Get or create a cuBLAS handle for grouped GEMM.

        Sets TF32 math mode to match PyTorch's float32_matmul_precision('high').
        """
        if self._cublas_handle is None and _cublas_available:
            self._cublas_handle = cublas.create()
            cublas.set_stream(
                self._cublas_handle,
                torch.cuda.current_stream().cuda_stream,
            )
            # Match PyTorch TF32 behavior
            cublas.set_math_mode(
                self._cublas_handle,
                cublas.Math.TF32_TENSOR_OP_MATH,
            )
        return self._cublas_handle


def init_linear(num_experts, use_bias, out_features, in_features):
    k = math.sqrt(1.0 / in_features)
    weights = nn.Parameter(
        k * 2 * (torch.rand(num_experts, out_features, in_features) - 0.5)
    )
    bias = nn.Parameter(k * 2 * (torch.rand(out_features) - 0.5)) if use_bias else None
    return weights, bias


class MOLEDGL(torch.nn.Module):
    def __init__(
        self,
        num_experts,
        in_features,
        out_features,
        global_mole_tensors,
        bias: bool,
    ):
        super().__init__()

        assert global_mole_tensors is not None
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.weights, self.bias = init_linear(
            num_experts, bias, out_features, in_features
        )

        self.global_mole_tensors = global_mole_tensors

    def forward(self, x):
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->bio",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )
        x_shape = x.shape
        if x.ndim == 2:
            r = fairchem_cpp.ops.segment_mm(
                x, weights, self.global_mole_tensors.mole_sizes
            )
        elif x.ndim == 3:
            r = fairchem_cpp.ops.segment_mm(
                x.reshape(-1, x_shape[-1]),
                weights,
                self.global_mole_tensors.mole_sizes * x_shape[1],
            ).reshape(*x_shape[:-1], -1)
        else:
            raise ValueError("x.ndim not in (2,3) not allowed")
        if self.bias is not None:
            r += self.bias
        return r


def _cublas_grouped_gemm(handle, x, weights, sizes, keepalive_list=None):
    """Raw cuBLAS grouped GEMM: C[s:s+m] = X[s:s+m] @ W[i] for each segment.

    x must be contiguous (total_rows, K). weights must be contiguous (num_seg, K, N).
    Returns C (total_rows, N), NOT connected to autograd.

    keepalive_list: if provided, numpy arrays are appended to this list to prevent
    GC while async cuBLAS kernels may still be reading them. The caller is
    responsible for clearing this list after synchronization (e.g., at the start
    of each new batch via MOLEGlobals.invalidate_pad_cache).
    """
    num_seg = len(sizes)
    K = x.shape[1]
    N = weights.shape[2]

    C = torch.empty(x.shape[0], N, device=x.device, dtype=x.dtype)

    transa = np.zeros(num_seg, dtype=np.int32)
    transb = np.zeros(num_seg, dtype=np.int32)
    m_arr = np.full(num_seg, N, dtype=np.int32)
    n_arr = np.array(sizes, dtype=np.int32)
    k_arr = np.full(num_seg, K, dtype=np.int32)
    alpha_arr = np.full(num_seg, 1.0, dtype=np.float32)
    beta_arr = np.full(num_seg, 0.0, dtype=np.float32)
    lda_arr = np.full(num_seg, N, dtype=np.int32)
    ldb_arr = np.full(num_seg, K, dtype=np.int32)
    ldc_arr = np.full(num_seg, N, dtype=np.int32)
    group_size = np.ones(num_seg, dtype=np.int32)
    a_ptrs = np.empty(num_seg, dtype=np.int64)
    b_ptrs = np.empty(num_seg, dtype=np.int64)
    c_ptrs = np.empty(num_seg, dtype=np.int64)

    start = 0
    for i in range(num_seg):
        mi = sizes[i]
        a_ptrs[i] = weights[i].data_ptr()
        b_ptrs[i] = x[start].data_ptr()
        c_ptrs[i] = C[start].data_ptr()
        start += mi

    # cuBLAS sgemm_grouped_batched reads host arrays asynchronously — if Python
    # garbage-collects them before the kernel finishes, we get illegal memory
    # access. Append to the caller's keepalive list so they survive until the
    # next batch boundary (when invalidate_pad_cache clears the list).
    arrays = (
        transa,
        transb,
        m_arr,
        n_arr,
        k_arr,
        alpha_arr,
        beta_arr,
        lda_arr,
        ldb_arr,
        ldc_arr,
        group_size,
        a_ptrs,
        b_ptrs,
        c_ptrs,
    )
    if keepalive_list is not None:
        keepalive_list.append(arrays)

    cublas.sgemm_grouped_batched(
        handle,
        transa.ctypes.data,
        transb.ctypes.data,
        m_arr.ctypes.data,
        n_arr.ctypes.data,
        k_arr.ctypes.data,
        alpha_arr.ctypes.data,
        a_ptrs.ctypes.data,
        lda_arr.ctypes.data,
        b_ptrs.ctypes.data,
        ldb_arr.ctypes.data,
        beta_arr.ctypes.data,
        c_ptrs.ctypes.data,
        ldc_arr.ctypes.data,
        num_seg,
        group_size.ctypes.data,
    )
    return C


class _SegmentMMGroupedFn(torch.autograd.Function):
    """Autograd-compatible segment_mm via cuBLAS grouped GEMM.

    Forward: C[s:s+m] = X[s:s+m] @ W[i]  (grouped GEMM, single kernel)
    Backward: dX[s:s+m] = dC[s:s+m] @ W[i]^T  (also grouped GEMM)
    """

    @staticmethod
    def forward(ctx, x, weights, sizes_list, handle, keepalive_list):
        # Ensure contiguous for cuBLAS
        x_c = x.contiguous() if not x.is_contiguous() else x
        w_c = weights.contiguous() if not weights.is_contiguous() else weights

        cublas.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        C = _cublas_grouped_gemm(handle, x_c, w_c, sizes_list, keepalive_list)

        ctx.save_for_backward(x_c, w_c)
        ctx.sizes_list = sizes_list
        ctx.handle = handle
        ctx.keepalive_list = keepalive_list
        return C

    @staticmethod
    def backward(ctx, grad_output):
        x_c, w_c = ctx.saved_tensors
        sizes_list = ctx.sizes_list
        handle = ctx.handle
        keepalive_list = ctx.keepalive_list

        grad_output = (
            grad_output.contiguous() if not grad_output.is_contiguous() else grad_output
        )

        # dX[s:s+m] = dC[s:s+m] @ W[i]^T
        # W[i] is (K, N), W[i]^T is (N, K)
        w_t = w_c.transpose(1, 2).contiguous()  # (num_seg, N, K)

        cublas.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        grad_x = _cublas_grouped_gemm(
            handle, grad_output, w_t, sizes_list, keepalive_list
        )

        # grad_weights not needed for inference-only (forces only need grad w.r.t. x)
        return grad_x, None, None, None, None


class MOLEGroupedGemm(torch.nn.Module):
    """MOLE using cuBLAS sgemm_grouped_batched — zero padding waste, single kernel.

    Uses nvmath.bindings.cublas grouped GEMM to dispatch all variable-size
    segment GEMMs in a single kernel launch. No padding, no C++ extension needed.
    Wrapped in torch.autograd.Function for correct force computation.

    Advantages over DGL: pure Python (no C++ build), uses cuBLAS grouped API.
    Advantages over padded: zero padding waste at any size variance.
    """

    def __init__(
        self,
        num_experts,
        in_features,
        out_features,
        global_mole_tensors,
        bias: bool,
    ):
        super().__init__()
        assert global_mole_tensors is not None
        assert _cublas_available, "nvmath.bindings.cublas required for grouped_gemm"
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weights, self.bias = init_linear(
            num_experts, bias, out_features, in_features
        )
        self.global_mole_tensors = global_mole_tensors

    def forward(self, x):
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->bio",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )
        x_shape = x.shape
        sizes = self.global_mole_tensors.mole_sizes
        handle = self.global_mole_tensors.get_cublas_handle()
        keepalive = self.global_mole_tensors.get_cublas_keepalive()

        if x.ndim == 2:
            sizes_list = [int(s) for s in sizes.tolist()]
            r = _SegmentMMGroupedFn.apply(x, weights, sizes_list, handle, keepalive)
        elif x.ndim == 3:
            sizes_list = [int(s) for s in (sizes * x_shape[1]).tolist()]
            r = _SegmentMMGroupedFn.apply(
                x.reshape(-1, x_shape[-1]),
                weights,
                sizes_list,
                handle,
                keepalive,
            ).reshape(*x_shape[:-1], -1)
        else:
            raise ValueError("x.ndim not in (2,3) not allowed")
        if self.bias is not None:
            r += self.bias
        return r


class MOLE(torch.nn.Module):
    def __init__(
        self,
        num_experts,
        in_features,
        out_features,
        global_mole_tensors: MOLEGlobals,
        bias: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.weights, self.bias = init_linear(
            num_experts, bias, out_features, in_features
        )

        self.global_mole_tensors = global_mole_tensors

    def merged_linear_layer(self):
        linear = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
        ).to(self.weights.device)

        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->boi",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )

        with torch.no_grad():
            linear.weight.copy_(weights[0])
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear

    def forward(self, x):
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->boi",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )

        out = []
        ac_start_idx = self.global_mole_tensors.ac_start_idx
        assert len(self.global_mole_tensors.mole_sizes) > 0
        # TODO: precompute these if needed but they should be small and on cpu
        start_idxs = [0] + torch.cumsum(
            self.global_mole_tensors.mole_sizes, dim=0
        ).tolist()
        mole_intervals = list(zip(start_idxs, start_idxs[1:]))

        # Because activation checkpointing can chunk the inputs, we need to only compute
        # the mole_size intervals that overlap with the current chunks
        # for example if mole_sizes = [10,10,15]
        # start_idxs -> [0,10,20,35]
        # mole_intervals -> [(0,10),(10,20),(20,35)]
        # if the input segment is (5,15) then we compute the following 2 segments
        # (5,10),(10,15)
        input_segment = (ac_start_idx, ac_start_idx + x.shape[0])

        for n, mole_segment in enumerate(mole_intervals):
            interval_overlap = interval_intersection(input_segment, mole_segment)
            if interval_overlap is not None:
                start = interval_overlap[0] - ac_start_idx
                end = interval_overlap[1] - ac_start_idx
                out.append(F.linear(x[start:end], weights[n], bias=self.bias))

        result = torch.concatenate(out, dim=0)
        assert (
            result.shape[0] == x.shape[0]
        ), f"result shape {result.shape}, does not match input shape {x.shape} at dim 0"
        return result
