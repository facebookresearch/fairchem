"""
Driver for the DGL ``segment_mm`` operation using nvmath cuBLAS bindings.

This module provides a torch-autograd-compatible ``segment_mm`` function
whose signature matches the historical ``fairchem_cpp.ops.segment_mm``
op (``A, B, seglen_A``). It is the runtime backend used by ``MOLEDGL``
when ``mole_layer_type=dgl`` is requested via the model config.

Two cuBLAS dispatch paths are exposed:

* ``use_grouped_gemm=True``  (default, when supported): a single
  ``cublasGemmGroupedBatchedEx`` call covering all per-segment GEMMs.
* ``use_grouped_gemm=False`` : a Python loop of ``cublasGemmEx`` calls,
  one per segment. Useful when grouped batching is unavailable or as a
  numerical / perf reference.

Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""

from __future__ import annotations

import atexit
import threading
from contextlib import contextmanager

import numpy as np
import torch

try:
    from cuda.bindings.runtime import cudaDataType
    from nvmath.bindings import cublas

    _HAS_NVMATH = True
except Exception:
    cudaDataType = None
    cublas = None
    _HAS_NVMATH = False

__all__ = ["SEGMENTMM", "segment_mm", "USE_GROUPED_GEMM"]

# Default GEMM dispatch for MOLEDGL. Module-level so callers can flip it
# without threading a kwarg through every call site. Per-call override is
# supported via the ``use_grouped_gemm`` kwarg on ``segment_mm``.
USE_GROUPED_GEMM = True


_HANDLE_LOCK = threading.Lock()
_CUBLAS_HANDLES: dict[int, int] = {}


def _ensure_nvmath_available() -> None:
    if not _HAS_NVMATH:
        raise RuntimeError(
            "segment_mm requires nvmath-python and cuda-bindings at runtime. "
            "Install via `pip install nvmath-python` (and a CUDA-enabled "
            "PyTorch build)."
        )


def _get_cached_handle(device_index: int) -> int:
    with _HANDLE_LOCK:
        handle = _CUBLAS_HANDLES.get(device_index)
        if handle is None:
            handle = cublas.create()
            _CUBLAS_HANDLES[device_index] = handle
        return handle


@atexit.register
def _destroy_cached_handles() -> None:
    if not _HAS_NVMATH:
        return
    with _HANDLE_LOCK:
        for handle in _CUBLAS_HANDLES.values():
            try:
                cublas.destroy(handle)
            except Exception:
                # Best effort cleanup at process teardown.
                pass
        _CUBLAS_HANDLES.clear()


@contextmanager
def _host_pointer_mode(handle: int):
    old_mode = int(cublas.get_pointer_mode(handle))
    host_mode = int(cublas.PointerMode.HOST)
    if old_mode != host_mode:
        cublas.set_pointer_mode(handle, host_mode)
    try:
        yield
    finally:
        if old_mode != host_mode:
            cublas.set_pointer_mode(handle, old_mode)


def _dtype_to_cublas_grouped(A: torch.Tensor) -> tuple[int, int]:
    dtype = A.dtype
    if dtype == torch.float32:
        return int(cudaDataType.CUDA_R_32F), int(cublas.ComputeType.COMPUTE_32F)
    if dtype == torch.float16:
        return int(cudaDataType.CUDA_R_16F), int(cublas.ComputeType.COMPUTE_32F)
    if dtype == torch.bfloat16:
        return int(cudaDataType.CUDA_R_16BF), int(cublas.ComputeType.COMPUTE_32F)
    raise TypeError(f"Unsupported dtype for nvmath segment_mm: {dtype}")


def _dtype_to_cublas_gemm_ex(A: torch.Tensor) -> tuple[int, int, np.dtype]:
    dtype = A.dtype
    if dtype == torch.float32:
        return int(cudaDataType.CUDA_R_32F), int(cublas.ComputeType.COMPUTE_32F), np.float32
    if dtype == torch.float16:
        # Match cublasGemmEx behavior for half kernels.
        return int(cudaDataType.CUDA_R_16F), int(cublas.ComputeType.COMPUTE_16F), np.float16
    if dtype == torch.bfloat16:
        return int(cudaDataType.CUDA_R_16BF), int(cublas.ComputeType.COMPUTE_32F), np.float32
    raise TypeError(f"Unsupported dtype for nvmath segment_mm: {dtype}")


def _gemm_ex_algo() -> int:
    if hasattr(cublas.GemmAlgo, "DEFAULT_TENSOR_OP"):
        return int(cublas.GemmAlgo.DEFAULT_TENSOR_OP)
    if hasattr(cublas.GemmAlgo, "DEFAULT"):
        return int(cublas.GemmAlgo.DEFAULT)
    return int(cublas.GemmAlgo.ALGO0)


def _prepare_seglen(seglen_A: torch.Tensor) -> torch.Tensor:
    """Coerce a segment-length tensor to a contiguous int32 CPU tensor.

    The cuBLAS grouped/looped paths read segment lengths host-side to
    construct the per-segment problem descriptors, so the tensor is moved
    to CPU here if necessary. Callers may pass either a CPU or a CUDA
    tensor; we accept both for caller convenience (the UMA backbone
    publishes ``mole_sizes`` on CPU but the per-dataset head router
    publishes it on CUDA).
    """
    if seglen_A.device.type != "cpu":
        seglen_A = seglen_A.detach().to("cpu")
    return seglen_A.to(dtype=torch.int32).contiguous()


def _validate_forward_inputs(A: torch.Tensor, B: torch.Tensor, seglen_A: torch.Tensor) -> None:
    if A.device.type != "cuda" or B.device.type != "cuda":
        raise ValueError("segment_mm is CUDA-only in this nvmath driver.")
    if A.dim() != 2:
        raise ValueError("segment_mm expects A to be a 2D tensor.")
    if B.dim() != 3:
        raise ValueError("segment_mm expects B to be a 3D tensor.")
    if A.device != B.device:
        raise ValueError("A and B must be on the same device.")
    if int(seglen_A.numel()) != int(B.shape[0]):
        raise ValueError("seglen_A length must match B.shape[0].")


def _segment_mm_grouped(A: torch.Tensor, B: torch.Tensor, seglen_A: torch.Tensor, *, b_trans: bool) -> torch.Tensor:
    seg = _prepare_seglen(seglen_A)
    num_rel = int(seg.numel())
    if num_rel != int(B.shape[0]):
        raise ValueError("seglen_A length must match B.shape[0].")
    out_cols = B.shape[1] if b_trans else B.shape[2]
    C = torch.empty((A.shape[0], out_cols), device=A.device, dtype=A.dtype).contiguous()
    if num_rel == 0:
        return C

    data_type, compute_type = _dtype_to_cublas_grouped(A)
    transa: list[int] = []
    transb: list[int] = []
    m: list[int] = []
    n: list[int] = []
    k: list[int] = []
    lda: list[int] = []
    ldb: list[int] = []
    ldc: list[int] = []
    group_size = [1] * num_rel
    A_ptrs: list[int] = []
    B_ptrs: list[int] = []
    C_ptrs: list[int] = []

    elem_size = A.element_size()
    A_base = A.data_ptr()
    B_base = B.data_ptr()
    C_base = C.data_ptr()

    A_offset = 0
    B_offset = 0
    C_offset = 0
    m_offset = 0

    B_n = int(B.size(2))
    B_k = int(B.size(1))

    for rel in range(num_rel):
        seg_m = int(seg[rel].item())
        if seg_m < 0:
            raise ValueError("Segment length must be non-negative.")
        if m_offset + seg_m > A.size(0):
            raise ValueError("Segment index out of bounds of A.shape[0].")

        # Keep the same col-major mapping as the C++ extension implementation.
        A_ptrs.append(B_base + B_offset * elem_size)
        B_ptrs.append(A_base + A_offset * elem_size)
        C_ptrs.append(C_base + C_offset * elem_size)

        if not b_trans:
            transa.append(int(cublas.Operation.N))
            transb.append(int(cublas.Operation.N))
            m.append(B_n)
            n.append(seg_m)
            k.append(B_k)
            lda.append(B_n)
            ldb.append(B_k)
            ldc.append(B_n)
            A_offset += seg_m * B_k
            B_offset += B_k * B_n
            C_offset += seg_m * B_n
        else:
            transa.append(int(cublas.Operation.T))
            transb.append(int(cublas.Operation.N))
            m.append(B_k)
            n.append(seg_m)
            k.append(B_n)
            lda.append(B_n)
            ldb.append(B_n)
            ldc.append(B_k)
            A_offset += seg_m * B_n
            B_offset += B_k * B_n
            C_offset += seg_m * B_k
        m_offset += seg_m

    dA_ptrs = torch.tensor(A_ptrs, dtype=torch.int64, device=A.device)
    dB_ptrs = torch.tensor(B_ptrs, dtype=torch.int64, device=A.device)
    dC_ptrs = torch.tensor(C_ptrs, dtype=torch.int64, device=A.device)
    alpha = np.ones((num_rel,), dtype=np.float32)
    beta = np.zeros((num_rel,), dtype=np.float32)

    device_index = A.device.index if A.device.index is not None else torch.cuda.current_device()
    handle = _get_cached_handle(device_index)
    cublas.set_stream(handle, torch.cuda.current_stream(device=A.device).cuda_stream)
    with _host_pointer_mode(handle):
        cublas.gemm_grouped_batched_ex_64(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha.ctypes.data,
            dA_ptrs.data_ptr(),
            data_type,
            lda,
            dB_ptrs.data_ptr(),
            data_type,
            ldb,
            beta.ctypes.data,
            dC_ptrs.data_ptr(),
            data_type,
            ldc,
            num_rel,
            group_size,
            compute_type,
        )

    return C


def _segment_mm_gemm_ex(A: torch.Tensor, B: torch.Tensor, seglen_A: torch.Tensor, *, b_trans: bool) -> torch.Tensor:
    seg = _prepare_seglen(seglen_A)
    num_rel = int(seg.numel())
    if num_rel != int(B.shape[0]):
        raise ValueError("seglen_A length must match B.shape[0].")
    out_cols = B.shape[1] if b_trans else B.shape[2]
    C = torch.empty((A.shape[0], out_cols), device=A.device, dtype=A.dtype).contiguous()
    if num_rel == 0:
        return C

    data_type, compute_type, scale_dtype = _dtype_to_cublas_gemm_ex(A)
    algo = _gemm_ex_algo()
    alpha = np.array([1.0], dtype=scale_dtype)
    beta = np.array([0.0], dtype=scale_dtype)

    elem_size = A.element_size()
    A_base = A.data_ptr()
    B_base = B.data_ptr()
    C_base = C.data_ptr()

    A_offset = 0
    B_offset = 0
    C_offset = 0
    m_offset = 0

    B_n = int(B.size(2))
    B_k = int(B.size(1))
    op_n = int(cublas.Operation.N)
    op_t = int(cublas.Operation.T)

    device_index = A.device.index if A.device.index is not None else torch.cuda.current_device()
    handle = _get_cached_handle(device_index)
    cublas.set_stream(handle, torch.cuda.current_stream(device=A.device).cuda_stream)
    with _host_pointer_mode(handle):
        for rel in range(num_rel):
            m = int(seg[rel].item())
            if m < 0:
                raise ValueError("Segment length must be non-negative.")
            if m_offset + m > A.size(0):
                raise ValueError("Segment index out of bounds of A.shape[0].")

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
                B_base + B_offset * elem_size,
                data_type,
                ldb,
                A_base + A_offset * elem_size,
                data_type,
                lda,
                beta.ctypes.data,
                C_base + C_offset * elem_size,
                data_type,
                ldc,
                compute_type,
                algo,
            )

            A_offset += m * k
            B_offset += k * n
            C_offset += m * n
            m_offset += m

    return C


def _segment_mm_backward_b_grouped(A: torch.Tensor, dC: torch.Tensor, seglen_A: torch.Tensor) -> torch.Tensor:
    seg = _prepare_seglen(seglen_A)
    num_rel = int(seg.numel())
    dB = torch.empty((num_rel, A.size(1), dC.size(1)), device=A.device, dtype=A.dtype).contiguous()
    if num_rel == 0:
        return dB

    data_type, compute_type = _dtype_to_cublas_grouped(A)
    transa = [int(cublas.Operation.N)] * num_rel
    transb = [int(cublas.Operation.T)] * num_rel
    m: list[int] = []
    n: list[int] = []
    k: list[int] = []
    lda: list[int] = []
    ldb: list[int] = []
    ldc: list[int] = []
    group_size = [1] * num_rel
    A_ptrs: list[int] = []
    B_ptrs: list[int] = []
    C_ptrs: list[int] = []

    elem_size = A.element_size()
    A_base = A.data_ptr()
    dC_base = dC.data_ptr()
    dB_base = dB.data_ptr()

    A_offset = 0
    dC_offset = 0
    dB_offset = 0
    k_offset = 0

    m_const = int(dC.size(1))
    n_const = int(A.size(1))

    for rel in range(num_rel):
        seg_k = int(seg[rel].item())
        if k_offset + seg_k > A.size(0):
            raise ValueError("Segment index out of bounds of A.shape[0].")

        m.append(m_const)
        n.append(n_const)
        k.append(seg_k)
        lda.append(m_const)
        ldb.append(n_const)
        ldc.append(m_const)

        A_ptrs.append(dC_base + dC_offset * elem_size)
        B_ptrs.append(A_base + A_offset * elem_size)
        C_ptrs.append(dB_base + dB_offset * elem_size)

        dC_offset += m_const * seg_k
        A_offset += n_const * seg_k
        dB_offset += m_const * n_const
        k_offset += seg_k

    dA_ptrs = torch.tensor(A_ptrs, dtype=torch.int64, device=A.device)
    dB_ptrs = torch.tensor(B_ptrs, dtype=torch.int64, device=A.device)
    dC_ptrs = torch.tensor(C_ptrs, dtype=torch.int64, device=A.device)
    alpha = np.ones((num_rel,), dtype=np.float32)
    beta = np.zeros((num_rel,), dtype=np.float32)

    device_index = A.device.index if A.device.index is not None else torch.cuda.current_device()
    handle = _get_cached_handle(device_index)
    cublas.set_stream(handle, torch.cuda.current_stream(device=A.device).cuda_stream)
    with _host_pointer_mode(handle):
        cublas.gemm_grouped_batched_ex_64(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha.ctypes.data,
            dA_ptrs.data_ptr(),
            data_type,
            lda,
            dB_ptrs.data_ptr(),
            data_type,
            ldb,
            beta.ctypes.data,
            dC_ptrs.data_ptr(),
            data_type,
            ldc,
            num_rel,
            group_size,
            compute_type,
        )

    return dB


def _segment_mm_backward_b_gemm_ex(A: torch.Tensor, dC: torch.Tensor, seglen_A: torch.Tensor) -> torch.Tensor:
    seg = _prepare_seglen(seglen_A)
    num_rel = int(seg.numel())
    dB = torch.empty((num_rel, A.size(1), dC.size(1)), device=A.device, dtype=A.dtype).contiguous()
    if num_rel == 0:
        return dB

    data_type, compute_type, scale_dtype = _dtype_to_cublas_gemm_ex(A)
    algo = _gemm_ex_algo()
    alpha = np.array([1.0], dtype=scale_dtype)
    beta = np.array([0.0], dtype=scale_dtype)

    elem_size = A.element_size()
    A_base = A.data_ptr()
    dC_base = dC.data_ptr()
    dB_base = dB.data_ptr()

    A_offset = 0
    dC_offset = 0
    dB_offset = 0
    k_offset = 0

    m_const = int(dC.size(1))
    n_const = int(A.size(1))
    op_n = int(cublas.Operation.N)
    op_t = int(cublas.Operation.T)

    device_index = A.device.index if A.device.index is not None else torch.cuda.current_device()
    handle = _get_cached_handle(device_index)
    cublas.set_stream(handle, torch.cuda.current_stream(device=A.device).cuda_stream)
    with _host_pointer_mode(handle):
        for rel in range(num_rel):
            k = int(seg[rel].item())
            if k < 0:
                raise ValueError("Segment length must be non-negative.")
            if k_offset + k > A.size(0):
                raise ValueError("Segment index out of bounds of A.shape[0].")

            cublas.gemm_ex(
                handle,
                op_n,
                op_t,
                m_const,
                n_const,
                k,
                alpha.ctypes.data,
                dC_base + dC_offset * elem_size,
                data_type,
                m_const,
                A_base + A_offset * elem_size,
                data_type,
                n_const,
                beta.ctypes.data,
                dB_base + dB_offset * elem_size,
                data_type,
                m_const,
                compute_type,
                algo,
            )

            dC_offset += m_const * k
            A_offset += n_const * k
            dB_offset += m_const * n_const
            k_offset += k

    return dB


def _segment_mm_forward_dispatch(
    A: torch.Tensor,
    B: torch.Tensor,
    seglen_A: torch.Tensor,
    *,
    b_trans: bool,
    use_grouped_gemm: bool,
) -> torch.Tensor:
    if use_grouped_gemm:
        return _segment_mm_grouped(A, B, seglen_A, b_trans=b_trans)
    return _segment_mm_gemm_ex(A, B, seglen_A, b_trans=b_trans)


def _segment_mm_backward_b_dispatch(
    A: torch.Tensor,
    dC: torch.Tensor,
    seglen_A: torch.Tensor,
    *,
    use_grouped_gemm: bool,
) -> torch.Tensor:
    if use_grouped_gemm:
        return _segment_mm_backward_b_grouped(A, dC, seglen_A)
    return _segment_mm_backward_b_gemm_ex(A, dC, seglen_A)


class SEGMENTMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        seglen_A: torch.Tensor,
        use_grouped_gemm: bool,
    ) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        seglen_A = _prepare_seglen(seglen_A)
        _validate_forward_inputs(A, B, seglen_A)
        C = _segment_mm_forward_dispatch(
            A,
            B,
            seglen_A,
            b_trans=False,
            use_grouped_gemm=use_grouped_gemm,
        )
        ctx.backward_cache = (A, B, seglen_A, use_grouped_gemm)
        return C

    @staticmethod
    def backward(ctx, dZ: torch.Tensor):
        dZ = dZ.contiguous()
        A, B, seglen_A, use_grouped_gemm = ctx.backward_cache
        A_grad = B_grad = None
        if ctx.needs_input_grad[0]:
            A_grad = _segment_mm_forward_dispatch(
                dZ,
                B,
                seglen_A,
                b_trans=True,
                use_grouped_gemm=use_grouped_gemm,
            )
        if ctx.needs_input_grad[1]:
            B_grad = _segment_mm_backward_b_dispatch(
                A,
                dZ,
                seglen_A,
                use_grouped_gemm=use_grouped_gemm,
            )
        return A_grad, B_grad, None, None


def segment_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    seglen_A: torch.Tensor,
    use_grouped_gemm: bool | None = None,
) -> torch.Tensor:
    """Segment-wise matrix multiplication with autograd support.

    Args:
        A: 2D tensor of shape ``(N, K)`` (concatenated per-segment inputs).
        B: 3D tensor of shape ``(num_segments, K, M)``.
        seglen_A: 1D tensor of length ``num_segments`` giving the row count
            of each segment in ``A``. May be on CPU or CUDA; will be moved
            to CPU internally as cuBLAS reads segment lengths host-side.
        use_grouped_gemm: If ``None`` (default), uses the module-level
            ``USE_GROUPED_GEMM``. If ``True``, dispatches via
            ``cublasGemmGroupedBatchedEx`` (single launch). If ``False``,
            uses a Python loop of ``cublasGemmEx`` calls.

    Returns:
        2D tensor of shape ``(N, M)`` containing the per-segment products.

    The signature is intentionally compatible with the legacy
    ``fairchem_cpp.ops.segment_mm`` op so this can serve as a drop-in
    backend for ``MOLEDGL.forward``.
    """
    _ensure_nvmath_available()
    if use_grouped_gemm is None:
        use_grouped_gemm = USE_GROUPED_GEMM
    if A.dtype != B.dtype:
        B = B.to(A.dtype)
    _validate_forward_inputs(A, B, seglen_A)
    return SEGMENTMM.apply(A, B, seglen_A, use_grouped_gemm)
