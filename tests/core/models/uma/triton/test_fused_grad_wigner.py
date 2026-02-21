"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.triton import HAS_TRITON

pytestmark = pytest.mark.gpu


def _reference_grad_wigner(grad_output, x, edge_index):
    """
    PyTorch reference for grad_wigner computation.

    Computes grad_wigner = sum_c (M_to_L(grad_src) * x_src + M_to_L(grad_tgt) * x_tgt)
    for block-diagonal entries only.
    """
    M_TO_L = [0, 5, 1, 3, 8, 6, 2, 4, 7]
    grad_l = grad_output[:, M_TO_L, :]

    C = x.shape[2]

    grad_l_src = grad_l[:, :, :C]
    grad_l_tgt = grad_l[:, :, C:]
    x_src = x[edge_index[0]]
    x_tgt = x[edge_index[1]]

    E = edge_index.shape[1]
    grad_wigner = torch.zeros(E, 9, 9, device=x.device, dtype=torch.float32)

    # L=0 block (1x1)
    grad_wigner[:, 0, 0] = (grad_l_src[:, 0, :] * x_src[:, 0, :]).sum(-1) + (
        grad_l_tgt[:, 0, :] * x_tgt[:, 0, :]
    ).sum(-1)

    # L=1 block (3x3)
    grad_wigner[:, 1:4, 1:4] = torch.bmm(
        grad_l_src[:, 1:4, :].float(),
        x_src[:, 1:4, :].float().transpose(1, 2),
    ) + torch.bmm(
        grad_l_tgt[:, 1:4, :].float(),
        x_tgt[:, 1:4, :].float().transpose(1, 2),
    )

    # L=2 block (5x5)
    grad_wigner[:, 4:9, 4:9] = torch.bmm(
        grad_l_src[:, 4:9, :].float(),
        x_src[:, 4:9, :].float().transpose(1, 2),
    ) + torch.bmm(
        grad_l_tgt[:, 4:9, :].float(),
        x_tgt[:, 4:9, :].float().transpose(1, 2),
    )

    return grad_wigner


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestFusedGradWigner:
    """
    Tests for fused_grad_wigner Triton kernel vs PyTorch reference.
    """

    @pytest.fixture()
    def graph_data(self):
        """
        Create test graph data with known dimensions.
        """
        torch.manual_seed(42)
        N, E, C = 200, 5000, 128
        x = torch.randn(N, 9, C, device="cuda", dtype=torch.float32)
        grad_output = torch.randn(E, 9, 2 * C, device="cuda", dtype=torch.float32)
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )
        return x, grad_output, edge_index

    def test_matches_reference(self, graph_data):
        """
        Fused Triton kernel matches PyTorch reference within tolerance.
        """
        from fairchem.core.models.uma.triton.edge_gather_wigner_bwd import (
            fused_grad_wigner,
        )

        x, grad_output, edge_index = graph_data

        result_triton = fused_grad_wigner(grad_output, x, edge_index)
        result_triton_3d = result_triton.view(-1, 9, 9)

        result_ref = _reference_grad_wigner(grad_output, x, edge_index)

        # L=0 block
        torch.testing.assert_close(
            result_triton_3d[:, 0, 0],
            result_ref[:, 0, 0],
            atol=1e-3,
            rtol=1e-3,
        )
        # L=1 block
        torch.testing.assert_close(
            result_triton_3d[:, 1:4, 1:4],
            result_ref[:, 1:4, 1:4],
            atol=1e-3,
            rtol=1e-3,
        )
        # L=2 block
        torch.testing.assert_close(
            result_triton_3d[:, 4:9, 4:9],
            result_ref[:, 4:9, 4:9],
            atol=1e-3,
            rtol=1e-3,
        )

    def test_off_diagonal_zeros(self, graph_data):
        """
        Non-block-diagonal entries must be exactly zero.
        """
        from fairchem.core.models.uma.triton.edge_gather_wigner_bwd import (
            fused_grad_wigner,
        )

        x, grad_output, edge_index = graph_data

        result = fused_grad_wigner(grad_output, x, edge_index).view(-1, 9, 9)

        mask = torch.ones(9, 9, dtype=torch.bool, device="cuda")
        mask[0, 0] = False
        mask[1:4, 1:4] = False
        mask[4:9, 4:9] = False

        assert (result[:, mask] == 0).all()

    def test_channels_256(self):
        """
        Works with C=256 (requires channel tiling, C > BLOCK_C=128).
        """
        from fairchem.core.models.uma.triton.edge_gather_wigner_bwd import (
            fused_grad_wigner,
        )

        torch.manual_seed(42)
        N, E, C = 100, 2000, 256
        x = torch.randn(N, 9, C, device="cuda", dtype=torch.float32)
        grad_output = torch.randn(E, 9, 2 * C, device="cuda", dtype=torch.float32)
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )

        result_triton = fused_grad_wigner(grad_output, x, edge_index).view(-1, 9, 9)
        result_ref = _reference_grad_wigner(grad_output, x, edge_index)

        torch.testing.assert_close(
            result_triton[:, 4:9, 4:9],
            result_ref[:, 4:9, 4:9],
            atol=1e-2,
            rtol=1e-2,
        )

    def test_small_graph(self):
        """
        Works with tiny graph (fewer edges than SM count).
        """
        from fairchem.core.models.uma.triton.edge_gather_wigner_bwd import (
            fused_grad_wigner,
        )

        torch.manual_seed(42)
        N, E, C = 5, 10, 128
        x = torch.randn(N, 9, C, device="cuda", dtype=torch.float32)
        grad_output = torch.randn(E, 9, 2 * C, device="cuda", dtype=torch.float32)
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )

        result_triton = fused_grad_wigner(grad_output, x, edge_index).view(-1, 9, 9)
        result_ref = _reference_grad_wigner(grad_output, x, edge_index)

        torch.testing.assert_close(
            result_triton[:, 1:4, 1:4],
            result_ref[:, 1:4, 1:4],
            atol=1e-3,
            rtol=1e-3,
        )


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestAllFusedAutogradFunction:
    """
    Tests for FusedEdgeGatherWignerL2MAllFusedBwdFunction.
    """

    def test_backward_grad_x_matches_v2(self):
        """
        grad_x from AllFused matches V2 scatter_add backward.
        """
        from fairchem.core.models.uma.triton.edge_gather_wigner_bwd import (
            FusedEdgeGatherWignerL2MAllFusedBwdFunction,
            FusedEdgeGatherWignerL2MTritonV2BwdFunction,
        )

        torch.manual_seed(42)
        N, E, C = 100, 3000, 128

        x = torch.randn(
            N,
            9,
            C,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        wigner = torch.randn(
            E,
            9,
            9,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )

        # Forward + backward with AllFused
        x1 = x.detach().clone().requires_grad_(True)
        w1 = wigner.detach().clone().requires_grad_(True)
        out1 = FusedEdgeGatherWignerL2MAllFusedBwdFunction.apply(x1, edge_index, w1)
        loss1 = out1.sum()
        loss1.backward()

        # Forward + backward with V2 (reference)
        x2 = x.detach().clone().requires_grad_(True)
        w2 = wigner.detach().clone().requires_grad_(True)
        out2 = FusedEdgeGatherWignerL2MTritonV2BwdFunction.apply(x2, edge_index, w2)
        loss2 = out2.sum()
        loss2.backward()

        # Forward outputs should match exactly (same kernel)
        torch.testing.assert_close(out1, out2, atol=0, rtol=0)

        # grad_x should match
        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-3, rtol=1e-3)

        # grad_wigner should match (block-diagonal entries)
        torch.testing.assert_close(
            w1.grad[:, 0, 0],
            w2.grad[:, 0, 0],
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            w1.grad[:, 1:4, 1:4],
            w2.grad[:, 1:4, 1:4],
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            w1.grad[:, 4:9, 4:9],
            w2.grad[:, 4:9, 4:9],
            atol=1e-3,
            rtol=1e-3,
        )


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestEmitForwardKernel:
    """
    Tests that the emit forward kernel produces correct side outputs.
    """

    def test_side_outputs_match_gather(self):
        """
        x_edge from emit kernel matches x[edge_index[0/1]] concatenated.
        """
        from fairchem.core.models.uma.triton.edge_gather_wigner_fwd import (
            fused_edge_gather_wigner_l2m_lmax2_emit,
        )

        torch.manual_seed(42)
        N, E, C = 200, 5000, 128
        x = torch.randn(N, 9, C, device="cuda", dtype=torch.float32)
        wigner = torch.randn(E, 9, 9, device="cuda", dtype=torch.float32)
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )

        out_emit, x_edge = fused_edge_gather_wigner_l2m_lmax2_emit(
            x, edge_index, wigner
        )

        # Side outputs must exactly match PyTorch gather
        torch.testing.assert_close(x_edge[:, :, :C], x[edge_index[0]], atol=0, rtol=0)
        torch.testing.assert_close(x_edge[:, :, C:], x[edge_index[1]], atol=0, rtol=0)

    def test_main_output_matches_base_kernel(self):
        """
        Main output from emit kernel matches base kernel exactly.
        """
        from fairchem.core.models.uma.triton.edge_gather_wigner_fwd import (
            fused_edge_gather_wigner_l2m_lmax2,
            fused_edge_gather_wigner_l2m_lmax2_emit,
        )

        torch.manual_seed(42)
        N, E, C = 200, 5000, 128
        x = torch.randn(N, 9, C, device="cuda", dtype=torch.float32)
        wigner = torch.randn(E, 9, 9, device="cuda", dtype=torch.float32)
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )

        out_base = fused_edge_gather_wigner_l2m_lmax2(x, edge_index, wigner)
        out_emit, _ = fused_edge_gather_wigner_l2m_lmax2_emit(x, edge_index, wigner)

        torch.testing.assert_close(out_emit, out_base, atol=0, rtol=0)


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestEmitAutogradFunction:
    """
    Tests for FusedEdgeGatherWignerL2MTritonBwdEmitFunction.
    """

    def test_backward_matches_v2(self):
        """
        Emit backward produces same grad_x and grad_wigner as V2.
        """
        from fairchem.core.models.uma.triton.edge_gather_wigner_bwd import (
            FusedEdgeGatherWignerL2MTritonBwdEmitFunction,
            FusedEdgeGatherWignerL2MTritonV2BwdFunction,
        )

        torch.manual_seed(42)
        N, E, C = 100, 3000, 128

        x = torch.randn(N, 9, C, device="cuda", dtype=torch.float32, requires_grad=True)
        wigner = torch.randn(
            E, 9, 9, device="cuda", dtype=torch.float32, requires_grad=True
        )
        edge_index = torch.stack(
            [
                torch.randint(0, N, (E,), device="cuda"),
                torch.randint(0, N, (E,), device="cuda"),
            ]
        )

        # Forward + backward with Emit
        x1 = x.detach().clone().requires_grad_(True)
        w1 = wigner.detach().clone().requires_grad_(True)
        out1 = FusedEdgeGatherWignerL2MTritonBwdEmitFunction.apply(x1, edge_index, w1)
        loss1 = out1.sum()
        loss1.backward()

        # Forward + backward with V2 (reference)
        x2 = x.detach().clone().requires_grad_(True)
        w2 = wigner.detach().clone().requires_grad_(True)
        out2 = FusedEdgeGatherWignerL2MTritonV2BwdFunction.apply(x2, edge_index, w2)
        loss2 = out2.sum()
        loss2.backward()

        # Forward outputs should match (same underlying kernel logic)
        torch.testing.assert_close(out1, out2, atol=0, rtol=0)

        # grad_x should match
        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-3, rtol=1e-3)

        # grad_wigner should match (block-diagonal entries)
        torch.testing.assert_close(
            w1.grad[:, 0, 0],
            w2.grad[:, 0, 0],
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            w1.grad[:, 1:4, 1:4],
            w2.grad[:, 1:4, 1:4],
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            w1.grad[:, 4:9, 4:9],
            w2.grad[:, 4:9, 4:9],
            atol=1e-3,
            rtol=1e-3,
        )
