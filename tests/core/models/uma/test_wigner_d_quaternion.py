"""
Tests for quaternion-based Wigner D matrix computation.

Tests verify:
1. Correct edge → +Y mapping
2. Agreement with axis-angle and Euler implementations
3. Gradient stability
4. Real-arithmetic equivalence to complex arithmetic
5. torch.compile compatibility

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fairchem.core.models.uma.common.rotation import (
    eulers_to_wigner,
    init_edge_rot_euler_angles,
)
from fairchem.core.models.uma.common.wigner_d_quaternion import (
    quaternion_wigner,
    quaternion_wigner_real,
    quaternion_to_ra_rb,
    quaternion_to_ra_rb_real,
    wigner_d_matrix_complex,
    wigner_d_matrix_real,
    wigner_d_complex_to_real_blockwise,
    wigner_d_pair_to_real_blockwise,
    precompute_wigner_coefficients_symmetric,
    precompute_U_blocks_euler_aligned,
    precompute_U_blocks_euler_aligned_real,
)
from fairchem.core.models.uma.common.wigner_d_axis_angle import axis_angle_wigner


@pytest.fixture()
def lmax():
    return 3


@pytest.fixture()
def dtype():
    return torch.float64


@pytest.fixture()
def device():
    return torch.device("cpu")


@pytest.fixture()
def Jd_matrices(lmax, dtype, device):
    """Load the J matrices used by the Euler angle approach."""
    test_dir = Path(__file__).parent
    repo_root = test_dir.parent.parent.parent.parent
    jd_path = repo_root / "src" / "fairchem" / "core" / "models" / "uma" / "Jd.pt"

    if jd_path.exists():
        Jd = torch.load(jd_path, map_location=device, weights_only=True)
        return [J.to(dtype=dtype) for J in Jd[: lmax + 1]]
    else:
        pytest.skip(f"Jd.pt not found at {jd_path}")


# =============================================================================
# Test Edge Sets
# =============================================================================

# Standard test edges: ±X, ±Y, ±Z, diagonal, and 3 random
STANDARD_TEST_EDGES = [
    ([1.0, 0.0, 0.0], "+X"),
    ([-1.0, 0.0, 0.0], "-X"),
    ([0.0, 1.0, 0.0], "+Y"),
    ([0.0, -1.0, 0.0], "-Y"),
    ([0.0, 0.0, 1.0], "+Z"),
    ([0.0, 0.0, -1.0], "-Z"),
    ([1.0, 1.0, 1.0], "diagonal"),
    ([0.3, 0.5, 0.8], "random1"),
    ([0.7, -0.2, 0.4], "random2"),
    ([-0.4, 0.6, -0.3], "random3"),
]


# =============================================================================
# Test Core Properties
# =============================================================================


class TestQuaternionWigner:
    """Tests for quaternion_wigner function."""

    @pytest.mark.parametrize("edge,desc", STANDARD_TEST_EDGES)
    def test_edge_aligns_to_y_axis(self, dtype, device, edge, desc):
        """Edge vector should align to +Y axis."""
        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        edge_t = torch.nn.functional.normalize(edge_t, dim=-1)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = quaternion_wigner(edge_t, 1, gamma=gamma)
        D_l1 = D[0, 1:4, 1:4]
        result = D_l1 @ edge_t[0]

        assert torch.allclose(result, y_axis, atol=1e-5), (
            f"Edge {desc} did not align to +Y, got {result}"
        )

    def test_orthogonality_and_determinant(self, lmax, dtype, device):
        """Wigner D matrices are orthogonal with determinant 1."""
        torch.manual_seed(42)
        edges = torch.randn(10, 3, dtype=dtype, device=device)
        gamma = torch.rand(10, dtype=dtype, device=device) * 6.28

        D, D_inv = quaternion_wigner(edges, lmax, gamma=gamma)

        size = (lmax + 1) ** 2
        I = torch.eye(size, dtype=dtype, device=device)

        for i in range(10):
            # Check orthogonality
            product = D[i] @ D[i].T
            assert torch.allclose(product, I, atol=1e-5), f"Not orthogonal for edge {i}"

            # Check determinant
            det = torch.linalg.det(D[i])
            assert torch.allclose(det, torch.ones_like(det), atol=1e-5)

            # Check inverse
            assert torch.allclose(D_inv[i], D[i].T, atol=1e-10)


# =============================================================================
# Test Agreement with Other Implementations
# =============================================================================


class TestImplementationAgreement:
    """Tests for agreement between quaternion and other implementations."""

    def test_matches_axis_angle(self, lmax, dtype, device):
        """quaternion_wigner matches axis_angle_wigner."""
        torch.manual_seed(42)
        edges = torch.randn(50, 3, dtype=dtype, device=device)
        gamma = torch.rand(50, dtype=dtype, device=device) * 6.28

        D_quat, _ = quaternion_wigner(edges, lmax, gamma=gamma)
        D_axis, _ = axis_angle_wigner(edges, lmax, gamma=gamma)

        max_err = (D_quat - D_axis).abs().max().item()
        assert max_err < 1e-9, f"quaternion differs from axis_angle by {max_err}"

    def test_matches_euler(self, lmax, dtype, device, Jd_matrices):
        """Both quaternion and Euler correctly map edge → +Y."""
        test_edges = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.3, 0.5, 0.8],
        ]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

        for edge in test_edges:
            edge_t = torch.tensor([edge], dtype=dtype, device=device)
            edge_norm = torch.nn.functional.normalize(edge_t, dim=-1)

            # Get Euler angles for consistent gamma
            euler_angles = init_edge_rot_euler_angles(edge_t)
            alpha, _, gamma_e = euler_angles

            # Euler approach
            wigner_euler = eulers_to_wigner(euler_angles, 0, lmax, Jd_matrices)

            # Quaternion approach with matching gamma
            wigner_quat, _ = quaternion_wigner(edge_t, lmax, gamma=alpha + gamma_e)

            # Both l=1 blocks should map edge → +Y
            euler_l1 = wigner_euler[0, 1:4, 1:4]
            quat_l1 = wigner_quat[0, 1:4, 1:4]

            assert torch.allclose(euler_l1 @ edge_norm[0], y_axis, atol=1e-5)
            assert torch.allclose(quat_l1 @ edge_norm[0], y_axis, atol=1e-5)

            # l=1 blocks should match
            assert torch.allclose(euler_l1, quat_l1, atol=1e-5), (
                f"l=1 blocks differ for edge {edge}"
            )


# =============================================================================
# Test Gradient Stability
# =============================================================================


class TestGradientStability:
    """Tests for gradient stability."""

    @pytest.mark.parametrize("edge,desc", STANDARD_TEST_EDGES)
    def test_gradient_flow(self, lmax, dtype, device, edge, desc):
        """Gradients flow without NaN/Inf and are reasonably bounded."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device, requires_grad=True)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = quaternion_wigner(edge_t, lmax, gamma=gamma)
        loss = D.sum()
        loss.backward()

        grad = edge_t.grad
        assert grad is not None, f"No gradient for {desc}"
        assert not torch.isnan(grad).any(), f"NaN gradient for {desc}"
        assert not torch.isinf(grad).any(), f"Inf gradient for {desc}"
        assert grad.abs().max() < 1000, f"Gradient too large for {desc}: {grad.abs().max()}"

    @pytest.mark.parametrize("epsilon", [1e-4, 1e-6, 1e-8])
    def test_near_y_axis_gradients(self, lmax, dtype, device, epsilon):
        """Gradients remain bounded near ±Y axis."""
        for sign in [1.0, -1.0]:
            edge = torch.tensor(
                [[epsilon, sign * 1.0, epsilon]], dtype=dtype, device=device, requires_grad=True
            )
            edge_norm = torch.nn.functional.normalize(edge, dim=-1)

            D, _ = quaternion_wigner(edge_norm, lmax)
            D.sum().backward()

            assert not torch.isnan(edge.grad).any(), f"NaN gradient near {'+'if sign>0 else '-'}Y (eps={epsilon})"
            assert edge.grad.abs().max() < 1000, (
                f"Gradient too large near {'+'if sign>0 else '-'}Y (eps={epsilon}): {edge.grad.abs().max()}"
            )


# =============================================================================
# Test Real-Arithmetic Equivalence
# =============================================================================


class TestRealArithmeticEquivalence:
    """Tests verifying real-arithmetic functions match complex-arithmetic functions."""

    def test_quaternion_to_ra_rb_real_matches_complex(self, dtype, device):
        """quaternion_to_ra_rb_real produces same values as quaternion_to_ra_rb."""
        torch.manual_seed(42)
        q = torch.randn(50, 4, dtype=dtype, device=device)
        q = torch.nn.functional.normalize(q, dim=-1)

        # Complex version
        Ra, Rb = quaternion_to_ra_rb(q)

        # Real version
        ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)

        # Compare
        assert torch.allclose(Ra.real, ra_re, atol=1e-12)
        assert torch.allclose(Ra.imag, ra_im, atol=1e-12)
        assert torch.allclose(Rb.real, rb_re, atol=1e-12)
        assert torch.allclose(Rb.imag, rb_im, atol=1e-12)

    def test_wigner_d_matrix_real_matches_complex(self, lmax, dtype, device):
        """wigner_d_matrix_real produces same values as wigner_d_matrix_complex."""
        torch.manual_seed(42)
        q = torch.randn(20, 4, dtype=dtype, device=device)
        q = torch.nn.functional.normalize(q, dim=-1)

        coeffs = precompute_wigner_coefficients_symmetric(lmax, dtype=dtype, device=device)

        # Complex version
        Ra, Rb = quaternion_to_ra_rb(q)
        D_complex = wigner_d_matrix_complex(Ra, Rb, coeffs)

        # Real version
        ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)
        D_re, D_im = wigner_d_matrix_real(ra_re, ra_im, rb_re, rb_im, coeffs)

        # Compare
        assert torch.allclose(D_complex.real, D_re, atol=1e-10)
        assert torch.allclose(D_complex.imag, D_im, atol=1e-10)

    def test_wigner_d_pair_to_real_matches_blockwise(self, lmax, dtype, device):
        """wigner_d_pair_to_real_blockwise produces same result as complex version."""
        torch.manual_seed(42)
        q = torch.randn(20, 4, dtype=dtype, device=device)
        q = torch.nn.functional.normalize(q, dim=-1)

        coeffs = precompute_wigner_coefficients_symmetric(lmax, dtype=dtype, device=device)
        U_blocks = precompute_U_blocks_euler_aligned(lmax, dtype=dtype, device=device)
        U_blocks_real = precompute_U_blocks_euler_aligned_real(lmax, dtype=dtype, device=device)

        # Complex version
        Ra, Rb = quaternion_to_ra_rb(q)
        D_complex = wigner_d_matrix_complex(Ra, Rb, coeffs)
        D_real_complex = wigner_d_complex_to_real_blockwise(D_complex, U_blocks, lmax)

        # Real version
        ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)
        D_re, D_im = wigner_d_matrix_real(ra_re, ra_im, rb_re, rb_im, coeffs)
        D_real_from_pair = wigner_d_pair_to_real_blockwise(D_re, D_im, U_blocks_real, lmax)

        # Compare
        assert torch.allclose(D_real_complex, D_real_from_pair, atol=1e-9)

    def test_quaternion_wigner_real_matches_quaternion_wigner(self, lmax, dtype, device):
        """quaternion_wigner_real produces same result as quaternion_wigner."""
        torch.manual_seed(42)
        edges = torch.randn(50, 3, dtype=dtype, device=device)
        gamma = torch.rand(50, dtype=dtype, device=device) * 6.28

        # Complex version
        D_complex, D_inv_complex = quaternion_wigner(edges, lmax, gamma=gamma)

        # Real version
        D_real, D_inv_real = quaternion_wigner_real(edges, lmax, gamma=gamma)

        # Compare
        max_err = (D_complex - D_real).abs().max().item()
        assert max_err < 1e-9, f"quaternion_wigner_real differs from quaternion_wigner by {max_err}"

        max_err_inv = (D_inv_complex - D_inv_real).abs().max().item()
        assert max_err_inv < 1e-9, f"inverse matrices differ by {max_err_inv}"


class TestRealArithmeticGradients:
    """Tests verifying gradients match between real and complex arithmetic."""

    def test_gradient_equivalence(self, lmax, dtype, device):
        """Gradients from quaternion_wigner_real match quaternion_wigner."""
        torch.manual_seed(42)
        edges_complex = torch.randn(10, 3, dtype=dtype, device=device, requires_grad=True)
        edges_real = edges_complex.detach().clone().requires_grad_(True)
        gamma = torch.rand(10, dtype=dtype, device=device) * 6.28

        # Complex version gradient
        D_complex, _ = quaternion_wigner(edges_complex, lmax, gamma=gamma)
        loss_complex = D_complex.sum()
        loss_complex.backward()

        # Real version gradient
        D_real, _ = quaternion_wigner_real(edges_real, lmax, gamma=gamma)
        loss_real = D_real.sum()
        loss_real.backward()

        # Compare gradients
        grad_diff = (edges_complex.grad - edges_real.grad).abs().max().item()
        assert grad_diff < 1e-8, f"Gradient difference: {grad_diff}"


# =============================================================================
# Test torch.compile Compatibility
# =============================================================================


class TestTorchCompileCompatibility:
    """Tests for torch.compile compatibility of real-arithmetic functions."""

    @pytest.mark.skipif(
        not hasattr(torch, "_dynamo"),
        reason="torch.compile not available"
    )
    def test_quaternion_wigner_real_compiles(self, lmax, dtype, device):
        """quaternion_wigner_real should compile without graph breaks."""
        import torch._dynamo as dynamo

        edges = torch.randn(10, 3, dtype=dtype, device=device)
        gamma = torch.rand(10, dtype=dtype, device=device) * 6.28

        # Define function to compile
        def fn(edge_vec, lmax_val, g):
            return quaternion_wigner_real(edge_vec, lmax_val, gamma=g)

        # Try to compile and run
        try:
            compiled_fn = torch.compile(fn, fullgraph=True)
            D, D_inv = compiled_fn(edges, lmax, gamma)

            # Verify output matches uncompiled version
            D_ref, D_inv_ref = quaternion_wigner_real(edges, lmax, gamma=gamma)
            assert torch.allclose(D, D_ref, atol=1e-10)
        except Exception as e:
            # If fullgraph=True fails, check with explanation
            explanation = dynamo.explain(fn)(edges, lmax, gamma)
            pytest.fail(
                f"torch.compile failed with fullgraph=True. "
                f"Graph break count: {explanation.graph_break_count}. "
                f"Error: {e}"
            )

    @pytest.mark.skipif(
        not hasattr(torch, "_dynamo"),
        reason="torch.compile not available"
    )
    def test_wigner_d_matrix_real_compiles(self, lmax, dtype, device):
        """wigner_d_matrix_real should compile without graph breaks."""
        import torch._dynamo as dynamo

        q = torch.randn(10, 4, dtype=dtype, device=device)
        q = torch.nn.functional.normalize(q, dim=-1)
        coeffs = precompute_wigner_coefficients_symmetric(lmax, dtype=dtype, device=device)

        def fn(quaternions, coeff_dict):
            ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(quaternions)
            return wigner_d_matrix_real(ra_re, ra_im, rb_re, rb_im, coeff_dict)

        try:
            compiled_fn = torch.compile(fn, fullgraph=True)
            D_re, D_im = compiled_fn(q, coeffs)

            # Verify output
            D_re_ref, D_im_ref = fn(q, coeffs)
            assert torch.allclose(D_re, D_re_ref, atol=1e-10)
            assert torch.allclose(D_im, D_im_ref, atol=1e-10)
        except Exception as e:
            explanation = dynamo.explain(fn)(q, coeffs)
            pytest.fail(
                f"torch.compile failed. Graph break count: {explanation.graph_break_count}. "
                f"Error: {e}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
