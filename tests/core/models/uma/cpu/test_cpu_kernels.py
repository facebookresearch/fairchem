"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for umas_fast_cpu C++ kernels.

Verifies forward correctness against PyTorch reference and
backward correctness by comparing C++ gradients against PyTorch
reference gradients.
"""

from __future__ import annotations

import pytest
import torch

# C++ kernels are float32 only; use float32 with relaxed tolerance
DTYPE = torch.float32


def _make_block_diag_wigner(E, dtype=DTYPE):
    """
    Create a proper block-diagonal Wigner matrix [E, 9, 9].
    """
    W = torch.zeros(E, 9, 9, dtype=dtype)
    W[:, 0, 0] = torch.randn(E, dtype=dtype)
    W[:, 1:4, 1:4] = torch.randn(E, 3, 3, dtype=dtype)
    W[:, 4:9, 4:9] = torch.randn(E, 5, 5, dtype=dtype)
    return W


def _pytorch_ref_n2e(x, edge_index, wigner):
    """
    PyTorch reference for node_to_edge_wigner_permute.
    """
    L_TO_M = [0, 2, 6, 3, 7, 1, 5, 8, 4]
    E = edge_index.shape[1]
    xs = x[edge_index[0]]
    xt = x[edge_index[1]]
    xcat = torch.cat((xs, xt), dim=2)
    W = wigner.reshape(E, 9, 9)
    rot = torch.empty_like(xcat)
    rot[:, 0:1, :] = W[:, 0:1, 0:1] * xcat[:, 0:1, :]
    rot[:, 1:4, :] = torch.bmm(W[:, 1:4, 1:4], xcat[:, 1:4, :])
    rot[:, 4:9, :] = torch.bmm(W[:, 4:9, 4:9], xcat[:, 4:9, :])
    return rot[:, L_TO_M, :]


def _pytorch_ref_pwi(x, wigner_inv):
    """
    PyTorch reference for permute_wigner_inv.
    """
    M_TO_L = [0, 5, 1, 3, 8, 6, 2, 4, 7]
    E = x.shape[0]
    xl = x[:, M_TO_L, :]
    W = wigner_inv.reshape(E, 9, 9)
    out = torch.empty_like(xl)
    out[:, 0:1, :] = W[:, 0:1, 0:1] * xl[:, 0:1, :]
    out[:, 1:4, :] = torch.bmm(W[:, 1:4, 1:4], xl[:, 1:4, :])
    out[:, 4:9, :] = torch.bmm(W[:, 4:9, 4:9], xl[:, 4:9, :])
    return out


class TestNodeToEdgeWignerPermute:
    """
    Test CPUNodeToEdgeWignerPermuteFunction.
    """

    @pytest.fixture()
    def setup(self):
        torch.manual_seed(42)
        N, E, C = 10, 30, 16
        x = torch.randn(N, 9, C, dtype=DTYPE, requires_grad=True)
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,)),
                torch.randint(0, N, (E,)),
            ]
        ).to(torch.int64)
        wigner = _make_block_diag_wigner(E).requires_grad_(True)
        return x, edge_index, wigner

    def test_forward_matches_reference(self, setup):
        from fairchem.core.models.uma.cpu.ops import (
            CPUNodeToEdgeWignerPermuteFunction,
        )

        x, edge_index, wigner = setup

        ref = _pytorch_ref_n2e(x, edge_index, wigner)
        cpp = CPUNodeToEdgeWignerPermuteFunction.apply(
            x, edge_index, wigner
        )
        assert torch.allclose(ref, cpp, atol=1e-5), (
            f"Forward mismatch: max diff = {(ref - cpp).abs().max()}"
        )

    def test_backward_grad_x(self, setup):
        """
        Verify grad_x from C++ matches PyTorch reference backward.
        """
        from fairchem.core.models.uma.cpu.ops import (
            CPUNodeToEdgeWignerPermuteFunction,
        )

        x, edge_index, wigner = setup

        # C++ backward
        x_cpp = x.detach().clone().requires_grad_(True)
        w_cpp = wigner.detach().clone().requires_grad_(True)
        out_cpp = CPUNodeToEdgeWignerPermuteFunction.apply(
            x_cpp, edge_index, w_cpp
        )
        grad_out = torch.randn_like(out_cpp)
        out_cpp.backward(grad_out)

        # PyTorch reference backward
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = wigner.detach().clone().requires_grad_(True)
        out_ref = _pytorch_ref_n2e(x_ref, edge_index, w_ref)
        out_ref.backward(grad_out)

        assert torch.allclose(x_cpp.grad, x_ref.grad, atol=1e-4), (
            f"grad_x mismatch: max diff = "
            f"{(x_cpp.grad - x_ref.grad).abs().max()}"
        )

    def test_backward_grad_wigner(self, setup):
        """
        Verify grad_wigner from C++ matches PyTorch reference
        (critical for forces).
        """
        from fairchem.core.models.uma.cpu.ops import (
            CPUNodeToEdgeWignerPermuteFunction,
        )

        x, edge_index, wigner = setup

        # C++ backward
        x_cpp = x.detach().clone().requires_grad_(True)
        w_cpp = wigner.detach().clone().requires_grad_(True)
        out_cpp = CPUNodeToEdgeWignerPermuteFunction.apply(
            x_cpp, edge_index, w_cpp
        )
        grad_out = torch.randn_like(out_cpp)
        out_cpp.backward(grad_out)

        # PyTorch reference backward
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = wigner.detach().clone().requires_grad_(True)
        out_ref = _pytorch_ref_n2e(x_ref, edge_index, w_ref)
        out_ref.backward(grad_out)

        assert torch.allclose(w_cpp.grad, w_ref.grad, atol=1e-4), (
            f"Wigner grad mismatch: max diff = "
            f"{(w_cpp.grad - w_ref.grad).abs().max()}"
        )

    def test_wigner_grad_is_nonzero(self, setup):
        """
        Verify wigner gradient is computed (needed for forces).
        """
        from fairchem.core.models.uma.cpu.ops import (
            CPUNodeToEdgeWignerPermuteFunction,
        )

        x, edge_index, wigner = setup

        out = CPUNodeToEdgeWignerPermuteFunction.apply(
            x, edge_index, wigner
        )
        loss = out.sum()
        loss.backward()

        assert wigner.grad is not None, "Wigner gradient is None!"
        assert wigner.grad.abs().sum() > 0, (
            "Wigner gradient is all zeros!"
        )


class TestPermuteWignerInv:
    """
    Test CPUPermuteWignerInvEdgeToNodeFunction.
    """

    @pytest.fixture()
    def setup(self):
        torch.manual_seed(42)
        E, C = 30, 16
        x = torch.randn(E, 9, C, dtype=DTYPE, requires_grad=True)
        wigner_inv = _make_block_diag_wigner(E).requires_grad_(True)
        return x, wigner_inv

    def test_forward_matches_reference(self, setup):
        from fairchem.core.models.uma.cpu.ops import (
            CPUPermuteWignerInvEdgeToNodeFunction,
        )

        x, wigner_inv = setup

        ref = _pytorch_ref_pwi(x, wigner_inv)
        cpp = CPUPermuteWignerInvEdgeToNodeFunction.apply(
            x, wigner_inv
        )
        assert torch.allclose(ref, cpp, atol=1e-5), (
            f"Forward mismatch: max diff = {(ref - cpp).abs().max()}"
        )

    def test_backward_grad_x(self, setup):
        """
        Verify grad_x from C++ matches PyTorch reference backward.
        """
        from fairchem.core.models.uma.cpu.ops import (
            CPUPermuteWignerInvEdgeToNodeFunction,
        )

        x, wigner_inv = setup

        # C++ backward
        x_cpp = x.detach().clone().requires_grad_(True)
        w_cpp = wigner_inv.detach().clone().requires_grad_(True)
        out_cpp = CPUPermuteWignerInvEdgeToNodeFunction.apply(
            x_cpp, w_cpp
        )
        grad_out = torch.randn_like(out_cpp)
        out_cpp.backward(grad_out)

        # PyTorch reference backward
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = wigner_inv.detach().clone().requires_grad_(True)
        out_ref = _pytorch_ref_pwi(x_ref, w_ref)
        out_ref.backward(grad_out)

        assert torch.allclose(x_cpp.grad, x_ref.grad, atol=1e-4), (
            f"grad_x mismatch: max diff = "
            f"{(x_cpp.grad - x_ref.grad).abs().max()}"
        )

    def test_backward_grad_wigner_inv(self, setup):
        """
        Verify wigner_inv gradient from C++ matches PyTorch reference
        (critical for forces).
        """
        from fairchem.core.models.uma.cpu.ops import (
            CPUPermuteWignerInvEdgeToNodeFunction,
        )

        x, wigner_inv = setup

        # C++ backward
        x_cpp = x.detach().clone().requires_grad_(True)
        w_cpp = wigner_inv.detach().clone().requires_grad_(True)
        out_cpp = CPUPermuteWignerInvEdgeToNodeFunction.apply(
            x_cpp, w_cpp
        )
        grad_out = torch.randn_like(out_cpp)
        out_cpp.backward(grad_out)

        # PyTorch reference backward
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = wigner_inv.detach().clone().requires_grad_(True)
        out_ref = _pytorch_ref_pwi(x_ref, w_ref)
        out_ref.backward(grad_out)

        assert torch.allclose(w_cpp.grad, w_ref.grad, atol=1e-4), (
            f"Wigner_inv grad mismatch: max diff = "
            f"{(w_cpp.grad - w_ref.grad).abs().max()}"
        )

    def test_wigner_grad_is_nonzero(self, setup):
        """
        Verify wigner inverse gradient is computed (needed for forces).
        """
        from fairchem.core.models.uma.cpu.ops import (
            CPUPermuteWignerInvEdgeToNodeFunction,
        )

        x, wigner_inv = setup

        out = CPUPermuteWignerInvEdgeToNodeFunction.apply(
            x, wigner_inv
        )
        loss = out.sum()
        loss.backward()

        assert wigner_inv.grad is not None, (
            "Wigner inv gradient is None!"
        )
        assert wigner_inv.grad.abs().sum() > 0, (
            "Wigner inv gradient is all zeros!"
        )
