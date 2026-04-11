"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.nn.mole import MOLE, MOLEFairchemCpp, MOLEGlobals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mole_pair(
    num_experts=4,
    in_features=8,
    out_features=6,
    atoms_per_system=(3, 7, 2, 5),
    device="cpu",
    dtype=torch.float64,
):
    """
    Create a matched MOLE (pytorch ref) and MOLEFairchemCpp (fairchem_cpp) pair
    sharing the same weights, bias, and MOLEGlobals.

    Args:
        atoms_per_system: Tuple of per-system atom counts (ragged).
    """
    num_systems = len(atoms_per_system)
    total_atoms = sum(atoms_per_system)
    mole_sizes = torch.tensor(atoms_per_system, dtype=torch.int32)
    coefficients = torch.randn(
        num_systems, num_experts, device=device, dtype=dtype
    ).softmax(dim=1)

    global_tensors = MOLEGlobals(
        expert_mixing_coefficients=coefficients,
        mole_sizes=mole_sizes,
    )

    ref = MOLE(
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        global_mole_tensors=global_tensors,
        bias=True,
    ).to(device=device, dtype=dtype)

    cpp = MOLEFairchemCpp(
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        global_mole_tensors=global_tensors,
        bias=True,
    ).to(device=device, dtype=dtype)

    # Share weights and bias so outputs are directly comparable
    with torch.no_grad():
        cpp.weights.copy_(ref.weights)
        cpp.bias.copy_(ref.bias)

    x = torch.randn(total_atoms, in_features, device=device, dtype=dtype)
    return ref, cpp, x


def _default_dtype(device):
    return torch.float32 if device == "cuda" else torch.float64


def _default_atol(device):
    return 1e-5 if device == "cuda" else 1e-12


def _run_forward_test(device):
    ref, cpp, x = _make_mole_pair(device=device, dtype=_default_dtype(device))
    atol = _default_atol(device)

    y_ref = ref(x)
    y_cpp = cpp(x)

    assert torch.allclose(y_ref, y_cpp, atol=atol), (
        f"Forward mismatch on {device}: "
        f"max diff = {(y_ref - y_cpp).abs().max().item()}"
    )


def _run_first_backward_test(device):
    ref, cpp, x_base = _make_mole_pair(device=device, dtype=_default_dtype(device))
    atol = _default_atol(device)

    x_ref = x_base.clone().requires_grad_(True)
    y_ref = ref(x_ref)
    y_ref.sum().backward()

    x_cpp = x_base.clone().requires_grad_(True)
    y_cpp = cpp(x_cpp)
    y_cpp.sum().backward()

    assert torch.allclose(x_ref.grad, x_cpp.grad, atol=atol), (
        f"x.grad mismatch on {device}: "
        f"max diff = {(x_ref.grad - x_cpp.grad).abs().max().item()}"
    )
    assert torch.allclose(ref.weights.grad, cpp.weights.grad, atol=atol), (
        f"weights.grad mismatch on {device}: "
        f"max diff = {(ref.weights.grad - cpp.weights.grad).abs().max().item()}"
    )
    assert torch.allclose(ref.bias.grad, cpp.bias.grad, atol=atol), (
        f"bias.grad mismatch on {device}: "
        f"max diff = {(ref.bias.grad - cpp.bias.grad).abs().max().item()}"
    )


def _run_double_backward_test(device):
    ref, cpp, x_base = _make_mole_pair(device=device, dtype=_default_dtype(device))

    # Reference: double backward through both x and weights
    x_ref = x_base.clone().requires_grad_(True)
    y_ref = ref(x_ref)
    grads_ref = torch.autograd.grad(
        y_ref.sum(), [x_ref, ref.weights], create_graph=True
    )
    sum(g.sum() for g in grads_ref).backward()
    x_ref_g2 = x_ref.grad.clone()
    w_ref_g2 = ref.weights.grad.clone()

    # fairchem_cpp: double backward through both x and weights
    x_cpp = x_base.clone().requires_grad_(True)
    y_cpp = cpp(x_cpp)
    grads_cpp = torch.autograd.grad(
        y_cpp.sum(), [x_cpp, cpp.weights], create_graph=True
    )
    sum(g.sum() for g in grads_cpp).backward()

    atol = _default_atol(device)
    assert torch.allclose(x_ref_g2, x_cpp.grad, atol=atol), (
        f"x 2nd-order grad mismatch on {device}: "
        f"max diff = {(x_ref_g2 - x_cpp.grad).abs().max().item()}"
    )
    assert torch.allclose(w_ref_g2, cpp.weights.grad, atol=atol), (
        f"weights 2nd-order grad mismatch on {device}: "
        f"max diff = {(w_ref_g2 - cpp.weights.grad).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# CPU tests — fairchem_cpp CPU path is pure PyTorch loop, all passes work
# ---------------------------------------------------------------------------


def test_mole_vs_mole_fairchem_cpp_forward_cpu():
    _run_forward_test("cpu")


def test_mole_vs_mole_fairchem_cpp_first_backward_cpu():
    _run_first_backward_test("cpu")


def test_mole_vs_mole_fairchem_cpp_double_backward_cpu():
    _run_double_backward_test("cpu")


# ---------------------------------------------------------------------------
# GPU tests — CUDA segment_mm with three-level autograd for double backward
# ---------------------------------------------------------------------------


@pytest.mark.gpu()
def test_mole_vs_mole_fairchem_cpp_forward_gpu():
    _run_forward_test("cuda")


@pytest.mark.gpu()
def test_mole_vs_mole_fairchem_cpp_first_backward_gpu():
    _run_first_backward_test("cuda")


@pytest.mark.gpu()
def test_mole_vs_mole_fairchem_cpp_double_backward_gpu():
    _run_double_backward_test("cuda")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CPU tests (MOLE vs MOLEFairchemCpp) ===")
    for fn in [
        _run_forward_test,
        _run_first_backward_test,
        _run_double_backward_test,
    ]:
        fn("cpu")
        print(f"  PASSED [CPU]: {fn.__name__}")

    if torch.cuda.is_available():
        print("\n=== GPU tests (MOLE vs MOLEFairchemCpp) ===")
        for fn in [
            _run_forward_test,
            _run_first_backward_test,
            _run_double_backward_test,
        ]:
            fn("cuda")
            print(f"  PASSED [GPU]: {fn.__name__}")
    else:
        print("\nSkipping GPU tests (no CUDA available)")

    print("\nDone.")
