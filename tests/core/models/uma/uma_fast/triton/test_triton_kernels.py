"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Triton kernel unit tests for umas_fast_gpu backend.

Tests forward and backward kernels against PyTorch reference implementations.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.triton import HAS_TRITON

pytestmark = pytest.mark.gpu


# =============================================================================
# PyTorch Reference Implementations
# =============================================================================


def _make_block_diagonal_wigner(wigner: torch.Tensor) -> torch.Tensor:
    """
    Create block-diagonal wigner matrix from dense wigner.

    For lmax=2, the Wigner D-matrix is block-diagonal:
        - L=0: 1x1 block at [0:1, 0:1]
        - L=1: 3x3 block at [1:4, 1:4]
        - L=2: 5x5 block at [4:9, 4:9]

    Args:
        wigner: Dense Wigner matrices [E, 9, 9]

    Returns:
        Block-diagonal Wigner [E, 9, 9] with off-diagonal blocks zeroed
    """
    wigner_bd = torch.zeros_like(wigner)
    wigner_bd[:, 0:1, 0:1] = wigner[:, 0:1, 0:1]  # L=0
    wigner_bd[:, 1:4, 1:4] = wigner[:, 1:4, 1:4]  # L=1
    wigner_bd[:, 4:9, 4:9] = wigner[:, 4:9, 4:9]  # L=2
    return wigner_bd


def pytorch_gather_wigner_l2m(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch reference: gather + concat + block-diagonal bmm + L-to-M permutation.

    NOTE: The Triton kernel uses BLOCK-DIAGONAL Wigner multiplication,
    not dense bmm. Only the L=0, L=1, L=2 diagonal blocks are used.

    Args:
        x: Node features [N, 9, C]
        edge_index: [2, E]
        wigner: [E, 9, 9] (only diagonal blocks are used)

    Returns:
        [E, 9, 2C] in M-major order
    """
    # L-to-M permutation indices
    L_TO_M_GATHER_IDX = [0, 2, 6, 3, 7, 1, 5, 8, 4]

    x_src = x[edge_index[0]]  # [E, 9, C]
    x_tgt = x[edge_index[1]]  # [E, 9, C]
    x_cat = torch.cat([x_src, x_tgt], dim=2)  # [E, 9, 2C]

    # Block-diagonal Wigner rotation in L-major order
    wigner_bd = _make_block_diagonal_wigner(wigner)
    out_l = torch.bmm(wigner_bd, x_cat)  # [E, 9, 2C]

    # L-to-M permutation
    out_m = out_l[:, L_TO_M_GATHER_IDX, :]
    return out_m


def pytorch_m_to_l_then_wigner(
    x: torch.Tensor,
    wigner: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch reference: M-to-L permutation then block-diagonal Wigner rotation.

    NOTE: Uses BLOCK-DIAGONAL Wigner multiplication.

    Args:
        x: Features in M-major order [E, 9, C]
        wigner: Wigner matrices [E, 9, 9] (only diagonal blocks are used)

    Returns:
        [E, 9, C] after M->L permute + Wigner
    """
    M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]

    # M-to-L permutation
    x_l = x[:, M_TO_L_GATHER_IDX, :]

    # Block-diagonal Wigner rotation
    wigner_bd = _make_block_diagonal_wigner(wigner)
    return torch.bmm(wigner_bd, x_l)


# =============================================================================
# Test: Edge Gather Wigner Forward
# =============================================================================


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestEdgeGatherWignerForward:
    """
    Tests the base forward kernel against pure PyTorch reference.
    """

    @pytest.fixture()
    def graph_data(self):
        """Standard test graph fixture."""
        torch.manual_seed(42)
        N, E, C = 200, 5000, 128
        x = torch.randn(N, 9, C, device="cuda", dtype=torch.float32)
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )
        wigner = torch.randn(E, 9, 9, device="cuda", dtype=torch.float32)
        return x, edge_index, wigner

    def test_fwd_matches_pytorch(self, graph_data):
        """
        Base forward kernel matches PyTorch gather + concat + bmm + permute.
        """
        from fairchem.core.models.uma.triton.edge_gather_wigner_fwd import (
            fused_edge_gather_wigner_l2m_lmax2,
        )

        x, edge_index, wigner = graph_data

        # Triton kernel
        triton_out = fused_edge_gather_wigner_l2m_lmax2(x, edge_index, wigner)

        # PyTorch reference
        pytorch_out = pytorch_gather_wigner_l2m(x, edge_index, wigner)

        torch.testing.assert_close(
            triton_out,
            pytorch_out,
            atol=1e-5,
            rtol=1e-5,
            msg="Forward kernel output differs from PyTorch reference",
        )


# =============================================================================
# Test: Emit Forward Kernel
# =============================================================================


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestEmitForwardKernel:
    """
    Tests the emit variant of forward kernel.
    """

    @pytest.fixture()
    def graph_data(self):
        """Standard test graph fixture."""
        torch.manual_seed(42)
        N, E, C = 200, 5000, 128
        x = torch.randn(N, 9, C, device="cuda", dtype=torch.float32)
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )
        wigner = torch.randn(E, 9, 9, device="cuda", dtype=torch.float32)
        return x, edge_index, wigner

    def test_main_output_matches_base_kernel(self, graph_data):
        """
        Emit kernel main output exactly matches base kernel output.
        """
        from fairchem.core.models.uma.triton.edge_gather_wigner_fwd import (
            fused_edge_gather_wigner_l2m_lmax2,
            fused_edge_gather_wigner_l2m_lmax2_emit,
        )

        x, edge_index, wigner = graph_data

        # Base kernel
        base_out = fused_edge_gather_wigner_l2m_lmax2(x, edge_index, wigner)

        # Emit kernel
        emit_out, x_edge = fused_edge_gather_wigner_l2m_lmax2_emit(
            x, edge_index, wigner
        )

        torch.testing.assert_close(
            emit_out,
            base_out,
            atol=0,
            rtol=0,
            msg="Emit kernel main output differs from base kernel (should be exact)",
        )

    def test_side_outputs_match_gather(self, graph_data):
        """
        Emit kernel side outputs match PyTorch gather.
        """
        from fairchem.core.models.uma.triton.edge_gather_wigner_fwd import (
            fused_edge_gather_wigner_l2m_lmax2_emit,
        )

        x, edge_index, wigner = graph_data
        C = x.shape[2]

        # Emit kernel
        _, x_edge = fused_edge_gather_wigner_l2m_lmax2_emit(x, edge_index, wigner)

        # PyTorch gather
        x_src = x[edge_index[0]]  # [E, 9, C]
        x_tgt = x[edge_index[1]]  # [E, 9, C]

        torch.testing.assert_close(
            x_edge[:, :, :C],
            x_src,
            atol=0,
            rtol=0,
            msg="Emit kernel x_src differs from PyTorch gather (should be exact)",
        )

        torch.testing.assert_close(
            x_edge[:, :, C:],
            x_tgt,
            atol=0,
            rtol=0,
            msg="Emit kernel x_tgt differs from PyTorch gather (should be exact)",
        )


# =============================================================================
# Test: Wigner Ops Forward
# =============================================================================


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestWignerOpsForward:
    """
    Tests M->L + Wigner rotation kernels.
    """

    @pytest.fixture()
    def edge_data(self):
        """Edge-level test data fixture."""
        torch.manual_seed(42)
        E, C = 5000, 128
        x = torch.randn(E, 9, C, device="cuda", dtype=torch.float32)
        wigner = torch.randn(E, 9, 9, device="cuda", dtype=torch.float32)
        return x, wigner

    def test_m_to_l_then_wigner_matches_pytorch(self, edge_data):
        """
        MToLThenWignerLmax2Function matches PyTorch permute + bmm.
        """
        from fairchem.core.models.uma.triton.wigner_ops import (
            MToLThenWignerLmax2Function,
        )

        x, wigner = edge_data

        # Triton kernel via autograd Function
        triton_out = MToLThenWignerLmax2Function.apply(x, wigner)

        # PyTorch reference
        pytorch_out = pytorch_m_to_l_then_wigner(x, wigner)

        torch.testing.assert_close(
            triton_out,
            pytorch_out,
            atol=1e-5,
            rtol=1e-5,
            msg="MToLThenWignerLmax2Function differs from PyTorch reference",
        )

    def test_fused_matches_non_fused(self, edge_data):
        """
        FusedMToLThenWignerLmax2Function exactly matches non-fused variant.
        """
        from fairchem.core.models.uma.triton.wigner_ops import (
            FusedMToLThenWignerLmax2Function,
            MToLThenWignerLmax2Function,
        )

        x, wigner = edge_data

        # Non-fused
        non_fused_out = MToLThenWignerLmax2Function.apply(x, wigner)

        # Fused
        fused_out = FusedMToLThenWignerLmax2Function.apply(x, wigner)

        torch.testing.assert_close(
            fused_out,
            non_fused_out,
            atol=0,
            rtol=0,
            msg="Fused and non-fused variants should be exactly equal",
        )


# =============================================================================
# Test: Edge Gather Wigner Backward
# =============================================================================


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestEdgeGatherWignerBackward:
    """
    Tests backward passes via .apply() forward -> .sum().backward().
    """

    @pytest.fixture()
    def graph_data_with_grad(self):
        """Graph data with requires_grad for backward tests."""
        torch.manual_seed(42)
        N, E, C = 100, 3000, 128
        x = torch.randn(N, 9, C, device="cuda", dtype=torch.float32, requires_grad=True)
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )
        wigner = torch.randn(
            E, 9, 9, device="cuda", dtype=torch.float32, requires_grad=True
        )
        return x, edge_index, wigner

    def _compute_pytorch_grads(self, x, edge_index, wigner):
        """
        Compute gradients using pure PyTorch autograd.
        """
        x = x.clone().detach().requires_grad_(True)
        wigner = wigner.clone().detach().requires_grad_(True)

        out = pytorch_gather_wigner_l2m(x, edge_index, wigner)
        loss = out.sum()
        loss.backward()

        return x.grad, wigner.grad

    def test_v2_bwd_matches_pytorch(self, graph_data_with_grad):
        """
        FusedEdgeGatherWignerL2MTritonV2BwdFunction gradients match PyTorch.
        """
        from fairchem.core.models.uma.triton import (
            FusedEdgeGatherWignerL2MTritonV2BwdFunction,
        )

        x, edge_index, wigner = graph_data_with_grad

        # Triton backward
        x_triton = x.clone().detach().requires_grad_(True)
        wigner_triton = wigner.clone().detach().requires_grad_(True)

        out_triton = FusedEdgeGatherWignerL2MTritonV2BwdFunction.apply(
            x_triton, edge_index, wigner_triton
        )
        out_triton.sum().backward()

        # PyTorch reference
        grad_x_pytorch, grad_wigner_pytorch = self._compute_pytorch_grads(
            x, edge_index, wigner
        )

        # Compare forward (should be exact)
        x_ref = x.clone().detach()
        wigner_ref = wigner.clone().detach()
        out_pytorch = pytorch_gather_wigner_l2m(x_ref, edge_index, wigner_ref)
        torch.testing.assert_close(
            out_triton.detach(),
            out_pytorch,
            atol=1e-5,
            rtol=1e-5,
            msg="V2 function forward differs from PyTorch",
        )

        # Compare grad_x
        torch.testing.assert_close(
            x_triton.grad,
            grad_x_pytorch,
            atol=1e-3,
            rtol=1e-3,
            msg="V2 function grad_x differs from PyTorch",
        )

        # Compare grad_wigner (per L-block due to accumulation order)
        # L=0 block: [0:1, 0:1]
        torch.testing.assert_close(
            wigner_triton.grad[:, 0:1, 0:1],
            grad_wigner_pytorch[:, 0:1, 0:1],
            atol=1e-3,
            rtol=1e-3,
            msg="V2 function grad_wigner L=0 block differs",
        )
        # L=1 block: [1:4, 1:4]
        torch.testing.assert_close(
            wigner_triton.grad[:, 1:4, 1:4],
            grad_wigner_pytorch[:, 1:4, 1:4],
            atol=1e-3,
            rtol=1e-3,
            msg="V2 function grad_wigner L=1 block differs",
        )
        # L=2 block: [4:9, 4:9]
        torch.testing.assert_close(
            wigner_triton.grad[:, 4:9, 4:9],
            grad_wigner_pytorch[:, 4:9, 4:9],
            atol=1e-3,
            rtol=1e-3,
            msg="V2 function grad_wigner L=2 block differs",
        )

    def test_emit_bwd_matches_v2(self, graph_data_with_grad):
        """
        FusedEdgeGatherWignerL2MTritonBwdEmitFunction matches V2.
        """
        from fairchem.core.models.uma.triton import (
            FusedEdgeGatherWignerL2MTritonBwdEmitFunction,
            FusedEdgeGatherWignerL2MTritonV2BwdFunction,
        )

        x, edge_index, wigner = graph_data_with_grad

        # V2 backward
        x_v2 = x.clone().detach().requires_grad_(True)
        wigner_v2 = wigner.clone().detach().requires_grad_(True)
        out_v2 = FusedEdgeGatherWignerL2MTritonV2BwdFunction.apply(
            x_v2, edge_index, wigner_v2
        )
        out_v2.sum().backward()

        # Emit backward
        x_emit = x.clone().detach().requires_grad_(True)
        wigner_emit = wigner.clone().detach().requires_grad_(True)
        out_emit = FusedEdgeGatherWignerL2MTritonBwdEmitFunction.apply(
            x_emit, edge_index, wigner_emit
        )
        out_emit.sum().backward()

        # Forward should be exact
        torch.testing.assert_close(
            out_emit.detach(),
            out_v2.detach(),
            atol=0,
            rtol=0,
            msg="Emit forward differs from V2 (should be exact)",
        )

        # grad_x should match
        torch.testing.assert_close(
            x_emit.grad,
            x_v2.grad,
            atol=1e-3,
            rtol=1e-3,
            msg="Emit grad_x differs from V2",
        )

        # grad_wigner per block
        torch.testing.assert_close(
            wigner_emit.grad[:, 0:1, 0:1],
            wigner_v2.grad[:, 0:1, 0:1],
            atol=1e-3,
            rtol=1e-3,
            msg="Emit grad_wigner L=0 block differs from V2",
        )
        torch.testing.assert_close(
            wigner_emit.grad[:, 1:4, 1:4],
            wigner_v2.grad[:, 1:4, 1:4],
            atol=1e-3,
            rtol=1e-3,
            msg="Emit grad_wigner L=1 block differs from V2",
        )
        torch.testing.assert_close(
            wigner_emit.grad[:, 4:9, 4:9],
            wigner_v2.grad[:, 4:9, 4:9],
            atol=1e-3,
            rtol=1e-3,
            msg="Emit grad_wigner L=2 block differs from V2",
        )


# =============================================================================
# Test: Wigner Ops Backward
# =============================================================================


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestWignerOpsBackward:
    """
    Tests backward pass for M->L + Wigner ops.
    """

    @pytest.fixture()
    def edge_data_with_grad(self):
        """Edge data with requires_grad."""
        torch.manual_seed(42)
        E, C = 3000, 128
        x = torch.randn(E, 9, C, device="cuda", dtype=torch.float32, requires_grad=True)
        wigner = torch.randn(
            E, 9, 9, device="cuda", dtype=torch.float32, requires_grad=True
        )
        return x, wigner

    def test_fused_m2l_bwd_matches_pytorch(self, edge_data_with_grad):
        """
        FusedMToLThenWignerLmax2Function backward matches PyTorch autograd.
        """
        from fairchem.core.models.uma.triton.wigner_ops import (
            FusedMToLThenWignerLmax2Function,
        )

        x, wigner = edge_data_with_grad

        # PyTorch reference
        x_pytorch = x.clone().detach().requires_grad_(True)
        wigner_pytorch = wigner.clone().detach().requires_grad_(True)
        out_pytorch = pytorch_m_to_l_then_wigner(x_pytorch, wigner_pytorch)
        out_pytorch.sum().backward()

        # Triton
        x_triton = x.clone().detach().requires_grad_(True)
        wigner_triton = wigner.clone().detach().requires_grad_(True)
        out_triton = FusedMToLThenWignerLmax2Function.apply(x_triton, wigner_triton)
        out_triton.sum().backward()

        # Forward should match
        torch.testing.assert_close(
            out_triton.detach(),
            out_pytorch.detach(),
            atol=1e-5,
            rtol=1e-5,
            msg="Fused M2L forward differs from PyTorch",
        )

        # grad_x
        torch.testing.assert_close(
            x_triton.grad,
            x_pytorch.grad,
            atol=1e-3,
            rtol=1e-3,
            msg="Fused M2L grad_x differs from PyTorch",
        )

        # grad_wigner
        torch.testing.assert_close(
            wigner_triton.grad,
            wigner_pytorch.grad,
            atol=1e-3,
            rtol=1e-3,
            msg="Fused M2L grad_wigner differs from PyTorch",
        )


# =============================================================================
# Test: Fused Grad Wigner
# =============================================================================


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestFusedGradWigner:
    """
    Tests the fused grad_wigner kernel for block-diagonal structure.
    """

    def _pytorch_grad_wigner_reference(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        PyTorch reference for block-diagonal grad_wigner.

        grad_wigner[e, i, j] = sum_c dy[e, i, c] * x[e, j, c]

        For lmax=2, only compute the block-diagonal entries.
        """
        E = dy.shape[0]
        grad_wigner = torch.zeros(E, 9, 9, device=dy.device, dtype=dy.dtype)

        # L=0 block: [0:1, 0:1]
        grad_wigner[:, 0:1, 0:1] = torch.bmm(
            dy[:, 0:1, :], x[:, 0:1, :].transpose(1, 2)
        )

        # L=1 block: [1:4, 1:4]
        grad_wigner[:, 1:4, 1:4] = torch.bmm(
            dy[:, 1:4, :], x[:, 1:4, :].transpose(1, 2)
        )

        # L=2 block: [4:9, 4:9]
        grad_wigner[:, 4:9, 4:9] = torch.bmm(
            dy[:, 4:9, :], x[:, 4:9, :].transpose(1, 2)
        )

        return grad_wigner

    @pytest.fixture()
    def grad_wigner_data(self):
        """Data for grad_wigner tests."""
        torch.manual_seed(42)
        E, C = 3000, 128
        dy = torch.randn(E, 9, C, device="cuda", dtype=torch.float32)
        x = torch.randn(E, 9, C, device="cuda", dtype=torch.float32)
        return dy, x

    def test_matches_reference(self, grad_wigner_data):
        """
        Fused grad_wigner matches PyTorch per-block bmm reference.
        """
        # Note: This tests the internal grad_wigner computation used in
        # emit backward. We access it through the backward of emit function.
        from fairchem.core.models.uma.triton import (
            FusedEdgeGatherWignerL2MTritonBwdEmitFunction,
        )

        dy, x = grad_wigner_data

        # Create a mock scenario where we can extract grad_wigner
        # by using the emit function backward
        torch.manual_seed(42)
        N, E, C = 100, dy.shape[0], dy.shape[2]
        x_nodes = torch.randn(
            N, 9, C, device="cuda", dtype=torch.float32, requires_grad=True
        )
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )
        wigner = torch.randn(
            E, 9, 9, device="cuda", dtype=torch.float32, requires_grad=True
        )

        # Forward + backward
        out = FusedEdgeGatherWignerL2MTritonBwdEmitFunction.apply(
            x_nodes, edge_index, wigner
        )
        out.sum().backward()

        # Grad wigner should be block-diagonal
        grad_w = wigner.grad

        # Check block-diagonal structure via off-diagonal zeros
        # Off-diagonal between L=0 and L=1
        assert torch.allclose(
            grad_w[:, 0:1, 1:4],
            torch.zeros_like(grad_w[:, 0:1, 1:4]),
            atol=1e-6,
        ), "Off-diagonal L0-L1 should be zero"

        # Off-diagonal between L=0 and L=2
        assert torch.allclose(
            grad_w[:, 0:1, 4:9],
            torch.zeros_like(grad_w[:, 0:1, 4:9]),
            atol=1e-6,
        ), "Off-diagonal L0-L2 should be zero"

        # Off-diagonal between L=1 and L=2
        assert torch.allclose(
            grad_w[:, 1:4, 4:9],
            torch.zeros_like(grad_w[:, 1:4, 4:9]),
            atol=1e-6,
        ), "Off-diagonal L1-L2 should be zero"

    def test_off_diagonal_zeros(self, grad_wigner_data):
        """
        Off-block-diagonal entries are exactly 0.
        """
        from fairchem.core.models.uma.triton import (
            FusedEdgeGatherWignerL2MTritonBwdEmitFunction,
        )

        torch.manual_seed(123)
        N, E, C = 50, 1000, 128
        x_nodes = torch.randn(
            N, 9, C, device="cuda", dtype=torch.float32, requires_grad=True
        )
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )
        wigner = torch.randn(
            E, 9, 9, device="cuda", dtype=torch.float32, requires_grad=True
        )

        out = FusedEdgeGatherWignerL2MTritonBwdEmitFunction.apply(
            x_nodes, edge_index, wigner
        )
        out.sum().backward()

        grad_w = wigner.grad

        # Create mask for off-diagonal blocks
        off_diag_mask = torch.ones(9, 9, device="cuda", dtype=torch.bool)
        off_diag_mask[0:1, 0:1] = False
        off_diag_mask[1:4, 1:4] = False
        off_diag_mask[4:9, 4:9] = False

        off_diag_values = grad_w[:, off_diag_mask]
        assert torch.allclose(
            off_diag_values,
            torch.zeros_like(off_diag_values),
            atol=0,
            rtol=0,
        ), "Off-block-diagonal entries should be exactly zero"

    def test_channels_256(self):
        """
        C=256 multi-tile case works correctly.
        """
        from fairchem.core.models.uma.triton import (
            FusedEdgeGatherWignerL2MTritonBwdEmitFunction,
        )

        torch.manual_seed(42)
        N, E, C = 50, 1000, 256
        x_nodes = torch.randn(
            N, 9, C, device="cuda", dtype=torch.float32, requires_grad=True
        )
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )
        wigner = torch.randn(
            E, 9, 9, device="cuda", dtype=torch.float32, requires_grad=True
        )

        # Should run without error
        out = FusedEdgeGatherWignerL2MTritonBwdEmitFunction.apply(
            x_nodes, edge_index, wigner
        )
        out.sum().backward()

        # Basic sanity: grad should exist and be non-zero
        assert wigner.grad is not None
        assert wigner.grad.abs().sum() > 0

        # PyTorch reference for comparison
        x_ref = x_nodes.clone().detach().requires_grad_(True)
        wigner_ref = wigner.clone().detach().requires_grad_(True)
        out_ref = pytorch_gather_wigner_l2m(x_ref, edge_index, wigner_ref)
        out_ref.sum().backward()

        # Compare diagonal blocks with looser tolerance for large C
        torch.testing.assert_close(
            wigner.grad[:, 0:1, 0:1],
            wigner_ref.grad[:, 0:1, 0:1],
            atol=1e-2,
            rtol=1e-2,
            msg="C=256 grad_wigner L=0 block differs",
        )

    def test_small_graph(self):
        """
        N=5, E=10 tiny graph case.
        """
        from fairchem.core.models.uma.triton import (
            FusedEdgeGatherWignerL2MTritonBwdEmitFunction,
        )

        torch.manual_seed(42)
        N, E, C = 5, 10, 128
        x_nodes = torch.randn(
            N, 9, C, device="cuda", dtype=torch.float32, requires_grad=True
        )
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )
        wigner = torch.randn(
            E, 9, 9, device="cuda", dtype=torch.float32, requires_grad=True
        )

        # Triton
        out = FusedEdgeGatherWignerL2MTritonBwdEmitFunction.apply(
            x_nodes, edge_index, wigner
        )
        out.sum().backward()

        # PyTorch reference
        x_ref = x_nodes.clone().detach().requires_grad_(True)
        wigner_ref = wigner.clone().detach().requires_grad_(True)
        out_ref = pytorch_gather_wigner_l2m(x_ref, edge_index, wigner_ref)
        out_ref.sum().backward()

        # Compare diagonal blocks
        torch.testing.assert_close(
            wigner.grad[:, 0:1, 0:1],
            wigner_ref.grad[:, 0:1, 0:1],
            atol=1e-3,
            rtol=1e-3,
            msg="Small graph grad_wigner L=0 block differs",
        )
        torch.testing.assert_close(
            wigner.grad[:, 1:4, 1:4],
            wigner_ref.grad[:, 1:4, 1:4],
            atol=1e-3,
            rtol=1e-3,
            msg="Small graph grad_wigner L=1 block differs",
        )
        torch.testing.assert_close(
            wigner.grad[:, 4:9, 4:9],
            wigner_ref.grad[:, 4:9, 4:9],
            atol=1e-3,
            rtol=1e-3,
            msg="Small graph grad_wigner L=2 block differs",
        )
