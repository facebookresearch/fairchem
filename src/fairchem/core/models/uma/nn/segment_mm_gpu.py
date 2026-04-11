"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import atexit
import threading
from contextlib import suppress

import numpy as np
import torch

try:
    from cuda.bindings.runtime import cudaDataType
    from nvmath.bindings import cublas

    _HAS_CUBLAS = True
except Exception:
    cudaDataType = None
    cublas = None
    _HAS_CUBLAS = False

# ---------------------------------------------------------------------------
# cuBLAS handle management
# ---------------------------------------------------------------------------

_HANDLE_LOCK = threading.Lock()
_CUBLAS_HANDLES: dict[int, int] = {}


def _get_cached_handle(device_index: int) -> int:
    with _HANDLE_LOCK:
        handle = _CUBLAS_HANDLES.get(device_index)
        if handle is None:
            handle = cublas.create()
            _CUBLAS_HANDLES[device_index] = handle
        return handle


@atexit.register
def _destroy_cached_handles() -> None:
    if not _HAS_CUBLAS:
        return
    with _HANDLE_LOCK:
        for handle in _CUBLAS_HANDLES.values():
            with suppress(Exception):
                cublas.destroy(handle)
        _CUBLAS_HANDLES.clear()


# ---------------------------------------------------------------------------
# GPU dispatch via cuBLAS gemm_ex
# ---------------------------------------------------------------------------


def _segment_mm_forward_gpu(
    A: torch.Tensor,
    B: torch.Tensor,
    seglen: torch.Tensor,
    *,
    b_trans: bool,
) -> torch.Tensor:
    """
    GPU forward dispatch via cublas.gemm_ex.

    Identical col-major mapping to the old segmentmm module.
    """
    A = A.contiguous()
    B = B.contiguous()
    num_seg = int(seglen.numel())
    B_k = int(B.shape[1])
    B_n = int(B.shape[2])
    out_cols = B_k if b_trans else B_n
    C = torch.empty(A.shape[0], out_cols, dtype=A.dtype, device=A.device).contiguous()
    if num_seg == 0:
        return C

    data_type = int(cudaDataType.CUDA_R_64F)
    compute_type = int(cublas.ComputeType.COMPUTE_64F)
    algo = (
        int(cublas.GemmAlgo.DEFAULT_TENSOR_OP)
        if hasattr(cublas.GemmAlgo, "DEFAULT_TENSOR_OP")
        else int(cublas.GemmAlgo.DEFAULT)
    )
    alpha = np.array([1.0], dtype=np.float64)
    beta = np.array([0.0], dtype=np.float64)

    elem_size = A.element_size()
    A_base = A.data_ptr()
    B_base = B.data_ptr()
    C_base = C.data_ptr()

    op_n = int(cublas.Operation.N)
    op_t = int(cublas.Operation.T)

    A_off = 0
    B_off = 0
    C_off = 0

    device_index = (
        A.device.index if A.device.index is not None else torch.cuda.current_device()
    )
    handle = _get_cached_handle(device_index)
    cublas.set_stream(
        handle,
        torch.cuda.current_stream(device=A.device).cuda_stream,
    )
    old_mode = int(cublas.get_pointer_mode(handle))
    host_mode = int(cublas.PointerMode.HOST)
    if old_mode != host_mode:
        cublas.set_pointer_mode(handle, host_mode)

    try:
        for rel in range(num_seg):
            m = int(seglen[rel].item())
            n = B_n
            k = B_k
            ldb = n
            lda = k
            ldc = n
            trans_b = op_n

            if b_trans:
                trans_b = op_t
                ldb = n
                lda = n
                ldc = k
                n, k = k, n

            cublas.gemm_ex(
                handle,
                trans_b,
                op_n,
                n,
                m,
                k,
                alpha.ctypes.data,
                B_base + B_off * elem_size,
                data_type,
                ldb,
                A_base + A_off * elem_size,
                data_type,
                lda,
                beta.ctypes.data,
                C_base + C_off * elem_size,
                data_type,
                ldc,
                compute_type,
                algo,
            )

            A_off += m * k
            B_off += k * n
            C_off += m * n
    finally:
        if old_mode != host_mode:
            cublas.set_pointer_mode(handle, old_mode)

    return C


def _segment_mm_backward_b_gpu(
    A: torch.Tensor,
    dC: torch.Tensor,
    seglen: torch.Tensor,
) -> torch.Tensor:
    """
    GPU backward-B dispatch via cublas.gemm_ex.

    dB_i = A_i^T @ dC_i
    """
    A = A.contiguous()
    dC = dC.contiguous()
    num_seg = int(seglen.numel())
    K = int(A.shape[1])
    N = int(dC.shape[1])
    dB = torch.empty(num_seg, K, N, dtype=A.dtype, device=A.device).contiguous()
    if num_seg == 0:
        return dB

    data_type = int(cudaDataType.CUDA_R_64F)
    compute_type = int(cublas.ComputeType.COMPUTE_64F)
    algo = (
        int(cublas.GemmAlgo.DEFAULT_TENSOR_OP)
        if hasattr(cublas.GemmAlgo, "DEFAULT_TENSOR_OP")
        else int(cublas.GemmAlgo.DEFAULT)
    )
    alpha = np.array([1.0], dtype=np.float64)
    beta = np.array([0.0], dtype=np.float64)

    elem_size = A.element_size()
    A_base = A.data_ptr()
    dC_base = dC.data_ptr()
    dB_base = dB.data_ptr()

    op_n = int(cublas.Operation.N)
    op_t = int(cublas.Operation.T)

    A_off = 0
    dC_off = 0
    dB_off = 0

    device_index = (
        A.device.index if A.device.index is not None else torch.cuda.current_device()
    )
    handle = _get_cached_handle(device_index)
    cublas.set_stream(
        handle,
        torch.cuda.current_stream(device=A.device).cuda_stream,
    )
    old_mode = int(cublas.get_pointer_mode(handle))
    host_mode = int(cublas.PointerMode.HOST)
    if old_mode != host_mode:
        cublas.set_pointer_mode(handle, host_mode)

    try:
        for rel in range(num_seg):
            seg_k = int(seglen[rel].item())

            cublas.gemm_ex(
                handle,
                op_n,
                op_t,
                N,
                K,
                seg_k,
                alpha.ctypes.data,
                dC_base + dC_off * elem_size,
                data_type,
                N,
                A_base + A_off * elem_size,
                data_type,
                K,
                beta.ctypes.data,
                dB_base + dB_off * elem_size,
                data_type,
                N,
                compute_type,
                algo,
            )

            dC_off += N * seg_k
            A_off += K * seg_k
            dB_off += N * K
    finally:
        if old_mode != host_mode:
            cublas.set_pointer_mode(handle, old_mode)

    return dB
