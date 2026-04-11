"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.models.uma.nn.segment_mm import (
    segment_mm_double_backward,
    segment_mm_ref,
)
from fairchem.core.models.uma.nn.segment_mm_gpu import (
    _HAS_CUBLAS,
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_inputs(
    seglen_list,
    K,
    N,
    dtype=torch.float64,
    device="cpu",
    requires_grad=True,
):
    """
    Create ragged inputs for segment_mm.
    """
    seglen = torch.tensor(seglen_list, dtype=torch.int32)
    total_rows = sum(seglen_list)
    A = torch.randn(
        total_rows,
        K,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    B = torch.randn(
        len(seglen_list),
        K,
        N,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return A, B, seglen


def _run_forward_test(device):
    seglen_list = [3, 7, 2, 5]
    K, N = 4, 6
    A, B, seglen = _make_inputs(seglen_list, K, N, device=device, requires_grad=False)

    C_ref = segment_mm_ref(A, B, seglen)
    C_db = segment_mm_double_backward(A, B, seglen)

    assert torch.allclose(C_ref, C_db, atol=1e-12), (
        f"Forward mismatch on {device}: "
        f"max diff = {(C_ref - C_db).abs().max().item()}"
    )


def _run_first_backward_test(device):
    seglen_list = [3, 7, 2, 5]
    K, N = 4, 6

    A_ref, B_ref, seglen = _make_inputs(seglen_list, K, N, device=device)
    C_ref = segment_mm_ref(A_ref, B_ref, seglen)
    C_ref.sum().backward()

    A_db = A_ref.data.clone().requires_grad_(True)
    B_db = B_ref.data.clone().requires_grad_(True)
    C_db = segment_mm_double_backward(A_db, B_db, seglen)
    C_db.sum().backward()

    assert torch.allclose(A_ref.grad, A_db.grad, atol=1e-12), (
        f"A.grad mismatch on {device}: "
        f"max diff = {(A_ref.grad - A_db.grad).abs().max().item()}"
    )
    assert torch.allclose(B_ref.grad, B_db.grad, atol=1e-12), (
        f"B.grad mismatch on {device}: "
        f"max diff = {(B_ref.grad - B_db.grad).abs().max().item()}"
    )


def _run_double_backward_test(device):
    seglen_list = [3, 7, 2, 5]
    K, N = 4, 6

    # Reference
    A_ref, B_ref, seglen = _make_inputs(seglen_list, K, N, device=device)
    C_ref = segment_mm_ref(A_ref, B_ref, seglen)
    gA_ref, gB_ref = torch.autograd.grad(C_ref.sum(), (A_ref, B_ref), create_graph=True)
    (gA_ref.sum() + gB_ref.sum()).backward()
    A_ref_g2 = A_ref.grad.clone()
    B_ref_g2 = B_ref.grad.clone()

    # Double-backward version
    A_db = A_ref.data.clone().requires_grad_(True)
    B_db = B_ref.data.clone().requires_grad_(True)
    C_db = segment_mm_double_backward(A_db, B_db, seglen)
    gA_db, gB_db = torch.autograd.grad(C_db.sum(), (A_db, B_db), create_graph=True)
    (gA_db.sum() + gB_db.sum()).backward()

    assert torch.allclose(A_ref_g2, A_db.grad, atol=1e-12), (
        f"A 2nd-order grad mismatch on {device}: "
        f"max diff = {(A_ref_g2 - A_db.grad).abs().max().item()}"
    )
    assert torch.allclose(B_ref_g2, B_db.grad, atol=1e-12), (
        f"B 2nd-order grad mismatch on {device}: "
        f"max diff = {(B_ref_g2 - B_db.grad).abs().max().item()}"
    )


def _run_gradgradcheck_test(device):
    seglen_list = [3, 7, 2, 5]
    K, N = 4, 6
    A, B, seglen = _make_inputs(seglen_list, K, N, device=device)

    def func(a, b):
        return segment_mm_double_backward(a, b, seglen)

    assert torch.autograd.gradgradcheck(
        func, (A, B), atol=1e-5, rtol=1e-4
    ), f"gradgradcheck failed on {device}"


def _run_all_tests(device):
    label = device.upper()
    _run_forward_test(device)
    print(f"  PASSED [{label}]: forward matches ref")
    _run_first_backward_test(device)
    print(f"  PASSED [{label}]: first backward matches ref")
    _run_double_backward_test(device)
    print(f"  PASSED [{label}]: double backward matches ref")
    _run_gradgradcheck_test(device)
    print(f"  PASSED [{label}]: gradgradcheck")


if __name__ == "__main__":
    print("=== CPU tests (nvpl.blas) ===")
    _run_all_tests("cpu")

    if torch.cuda.is_available() and _HAS_CUBLAS:
        print("\n=== GPU tests (cuBLAS) ===")
        _run_all_tests("cuda")
    else:
        print("\nSkipping GPU tests (no CUDA available)")

    print("\nAll tests passed!")
