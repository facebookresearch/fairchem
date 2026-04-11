"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

try:
    from nvmath.bindings.nvpl import blas as nvpl_blas

    _HAS_NVPL = True
except Exception:
    nvpl_blas = None
    _HAS_NVPL = False

if _HAS_NVPL:
    _COL_MAJOR = int(nvpl_blas.ORDER.ColMajor)
    _NVPL_NO_TRANS = int(nvpl_blas.TRANSPOSE.NoTrans)
    _NVPL_TRANS = int(nvpl_blas.TRANSPOSE.Trans)


def _segment_mm_forward_cpu(
    A: torch.Tensor,
    B: torch.Tensor,
    seglen: torch.Tensor,
    *,
    b_trans: bool,
) -> torch.Tensor:
    """
    CPU forward dispatch via nvpl.blas.dgemm.

    Same col-major pointer mapping as the GPU cuBLAS path.
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

    elem_size = A.element_size()
    A_base = A.data_ptr()
    B_base = B.data_ptr()
    C_base = C.data_ptr()

    A_off = 0
    B_off = 0
    C_off = 0

    for rel in range(num_seg):
        seg_m = int(seglen[rel].item())

        if not b_trans:
            nvpl_blas.dgemm(
                _COL_MAJOR,
                _NVPL_NO_TRANS,
                _NVPL_NO_TRANS,
                B_n,
                seg_m,
                B_k,
                1.0,
                B_base + B_off * elem_size,
                B_n,
                A_base + A_off * elem_size,
                B_k,
                0.0,
                C_base + C_off * elem_size,
                B_n,
            )
            A_off += seg_m * B_k
            B_off += B_k * B_n
            C_off += seg_m * B_n
        else:
            nvpl_blas.dgemm(
                _COL_MAJOR,
                _NVPL_TRANS,
                _NVPL_NO_TRANS,
                B_k,
                seg_m,
                B_n,
                1.0,
                B_base + B_off * elem_size,
                B_n,
                A_base + A_off * elem_size,
                B_n,
                0.0,
                C_base + C_off * elem_size,
                B_k,
            )
            A_off += seg_m * B_n
            B_off += B_k * B_n
            C_off += seg_m * B_k

    return C


def _segment_mm_backward_b_cpu(
    A: torch.Tensor,
    dC: torch.Tensor,
    seglen: torch.Tensor,
) -> torch.Tensor:
    """
    CPU backward-B dispatch via nvpl.blas.dgemm.

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

    elem_size = A.element_size()
    A_base = A.data_ptr()
    dC_base = dC.data_ptr()
    dB_base = dB.data_ptr()

    A_off = 0
    dC_off = 0
    dB_off = 0

    for rel in range(num_seg):
        seg_k = int(seglen[rel].item())

        nvpl_blas.dgemm(
            _COL_MAJOR,
            _NVPL_NO_TRANS,
            _NVPL_TRANS,
            N,
            K,
            seg_k,
            1.0,
            dC_base + dC_off * elem_size,
            N,
            A_base + A_off * elem_size,
            K,
            0.0,
            dB_base + dB_off * elem_size,
            N,
        )

        dC_off += N * seg_k
        A_off += K * seg_k
        dB_off += N * K

    return dB
