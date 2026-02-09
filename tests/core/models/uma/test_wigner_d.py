"""
Tests for Wigner D matrix computation.

Tests verify:
1. Mathematical correctness (orthogonality, determinant, edge -> +Y)
2. Agreement between all entry point functions
3. Agreement with Euler-based rotation.py
4. Gradient stability
5. Real-arithmetic equivalence to complex arithmetic
6. torch.compile compatibility
7. Range functions (lmin support)
8. Specialized kernels (l=2 polynomial, l=3/4 matmul)

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fairchem.core.models.uma.common.rotation import (
    eulers_to_wigner,
    init_edge_rot_euler_angles,
    wigner_D,
)
from fairchem.core.models.uma.common.wigner_d_matexp import axis_angle_wigner
from fairchem.core.models.uma.common.wigner_d_hybrid import (
    axis_angle_wigner_hybrid,
)
from fairchem.core.models.uma.common.wigner_d_polynomial import (
    axis_angle_wigner_polynomial,
)
from fairchem.core.models.uma.common.quaternion_wigner_utils import (
    quaternion_to_ra_rb,
    quaternion_to_ra_rb_real,
    wigner_d_matrix_complex,
    wigner_d_matrix_real,
    wigner_d_complex_to_real,
    wigner_d_pair_to_real,
    precompute_wigner_coefficients,
    precompute_U_blocks_euler_aligned,
    precompute_U_blocks_euler_aligned_real,
)


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

# Standard test edges: +/-X, +/-Y, +/-Z, diagonal, and 3 random
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
# Test Core Wigner D Properties
# =============================================================================


class TestWignerDProperties:
    """Tests for mathematical properties of Wigner D matrices."""

    @pytest.mark.parametrize(
        "edge,desc",
        [
            ([1.0, 0.0, 0.0], "X-axis"),
            ([0.0, 1.0, 0.0], "+Y-axis"),
            ([0.0, -1.0, 0.0], "-Y-axis"),
            ([0.0, 0.0, 1.0], "Z-axis"),
            ([1.0, 1.0, 1.0], "diagonal"),
        ],
    )
    def test_orthogonality_and_determinant(self, lmax, dtype, device, edge, desc):
        """Wigner D matrices are orthogonal with determinant 1."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, D_inv = axis_angle_wigner_hybrid(edge_t, lmax, gamma=gamma)

        size = (lmax + 1) ** 2
        I = torch.eye(size, dtype=dtype, device=device)

        # Check orthogonality
        product = D[0] @ D[0].T
        assert torch.allclose(product, I, atol=1e-5), f"Not orthogonal for {desc}"

        # Check determinant
        det = torch.linalg.det(D[0])
        assert torch.allclose(
            det, torch.tensor(1.0, dtype=dtype, device=device), atol=1e-5
        ), f"det != 1 for {desc}"

    @pytest.mark.parametrize("edge,desc", STANDARD_TEST_EDGES)
    def test_edge_to_y(self, lmax, dtype, device, edge, desc):
        """The l=1 block rotates edge -> +Y."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        edge_t = torch.nn.functional.normalize(edge_t, dim=-1)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = axis_angle_wigner_hybrid(edge_t, lmax, gamma=gamma)
        D_l1 = D[0, 1:4, 1:4]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        result = D_l1 @ edge_t[0]

        assert torch.allclose(result, y_axis, atol=1e-5), (
            f"Edge {edge} did not map to +Y, got {result}"
        )

    def test_composition_law(self, lmax, dtype, device):
        """D(R1) @ D(R2) = D(R1 @ R2) - the fundamental group composition property."""
        torch.manual_seed(123)
        n_samples = 10

        # Generate two random rotations (gamma is randomized internally when not specified)
        edges1 = torch.randn(n_samples, 3, dtype=dtype, device=device)
        edges2 = torch.randn(n_samples, 3, dtype=dtype, device=device)

        D1, _ = axis_angle_wigner_hybrid(edges1, lmax)
        D2, _ = axis_angle_wigner_hybrid(edges2, lmax)

        # Compose the Wigner D matrices
        D_product = D1 @ D2

        # The l=1 block of D_product is the composed rotation matrix R1 @ R2
        R_composed = D_product[:, 1:4, 1:4]

        # From R_composed, extract edge (second row, since R @ edge = +Y means edge = R^T @ +Y)
        edge_composed = R_composed[:, 1, :]

        # Compute D for edge_composed with gamma=0 to get the canonical alignment rotation
        D_canonical, _ = axis_angle_wigner_hybrid(
            edge_composed, lmax, gamma=torch.zeros(n_samples, dtype=dtype, device=device)
        )
        R_canonical = D_canonical[:, 1:4, 1:4]

        # The composed rotation is R_composed = R_gamma @ R_canonical
        # So R_gamma = R_composed @ R_canonical^T
        R_gamma = R_composed @ R_canonical.transpose(-1, -2)

        # R_gamma is rotation around Y by gamma:
        # [[cos gamma, 0, sin gamma], [0, 1, 0], [-sin gamma, 0, cos gamma]]
        # So cos(gamma) = R_gamma[0, 0] and sin(gamma) = R_gamma[0, 2]
        gamma_composed = torch.atan2(R_gamma[:, 0, 2], R_gamma[:, 0, 0])

        # Compute D for the composed rotation
        D_composed, _ = axis_angle_wigner_hybrid(edge_composed, lmax, gamma=gamma_composed)

        # Check that the product equals the composed Wigner D
        max_err = (D_product - D_composed).abs().max().item()
        assert max_err < 1e-9, f"Composition law failed: max error = {max_err}"


# =============================================================================
# Test Entry Point Agreement
# =============================================================================


class TestEntryPointAgreement:
    """Tests for agreement between all Wigner D entry point functions."""

    def test_all_methods_match(self, lmax, dtype, device):
        """All Wigner D implementations produce identical results."""
        torch.manual_seed(42)
        edges = torch.randn(50, 3, dtype=dtype, device=device)
        gamma = torch.rand(50, dtype=dtype, device=device) * 6.28

        D_matexp, _ = axis_angle_wigner(edges, lmax, gamma=gamma)
        D_hybrid, _ = axis_angle_wigner_hybrid(edges, lmax, gamma=gamma)
        D_poly, _ = axis_angle_wigner_polynomial(edges, lmax, gamma=gamma)

        assert (D_matexp - D_hybrid).abs().max() < 1e-9, "hybrid differs from matexp"
        assert (D_matexp - D_poly).abs().max() < 1e-9, "polynomial differs from matexp"

    def test_real_methods_match_complex(self, lmax, dtype, device):
        """Real-arithmetic methods match complex-arithmetic methods."""
        torch.manual_seed(42)
        edges = torch.randn(50, 3, dtype=dtype, device=device)
        gamma = torch.rand(50, dtype=dtype, device=device) * 6.28

        D_hybrid, _ = axis_angle_wigner_hybrid(edges, lmax, gamma=gamma)
        D_hybrid_real, _ = axis_angle_wigner_hybrid(edges, lmax, gamma=gamma, use_real_arithmetic=True)
        D_poly, _ = axis_angle_wigner_polynomial(edges, lmax, gamma=gamma)
        D_poly_real, _ = axis_angle_wigner_polynomial(edges, lmax, gamma=gamma, use_real_arithmetic=True)

        assert (D_hybrid - D_hybrid_real).abs().max() < 1e-9, "hybrid_real differs from hybrid"
        assert (D_poly - D_poly_real).abs().max() < 1e-9, "polynomial_real differs from polynomial"


# =============================================================================
# Test Euler Agreement
# =============================================================================


class TestEulerAgreement:
    """Tests for agreement with Euler-based rotation.py."""

    def test_matches_euler_code(self, lmax, dtype, device, Jd_matrices):
        """Axis-angle with use_euler_gamma matches Euler implementation exactly."""
        test_edges = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.3, 0.5, 0.8],
        ]

        for edge in test_edges:
            edge_t = torch.tensor([edge], dtype=dtype, device=device)

            # Compute with axis-angle using Euler gamma
            D_axis, _ = axis_angle_wigner(edge_t, lmax, use_euler_gamma=True)

            # Get Euler angles from production code, zero out random gamma
            gamma, beta, alpha = init_edge_rot_euler_angles(edge_t)
            gamma_zero = torch.zeros_like(gamma)

            for ell in range(lmax + 1):
                start = ell * ell
                end = start + 2 * ell + 1
                D_euler = wigner_D(ell, gamma_zero, beta, alpha, Jd_matrices)
                D_axis_block = D_axis[0, start:end, start:end]

                assert torch.allclose(D_euler, D_axis_block, atol=1e-10), (
                    f"l={ell} mismatch for edge {edge}"
                )

    def test_blend_region_matches_euler(self, lmax, dtype, device, Jd_matrices):
        """Blend region edges (ey in [-0.9, -0.7]) match Euler with correct gamma."""
        blend_region_edges = [
            # yz-plane (ex=0), middle of blend
            ([0.0, -0.8, 0.6], 0.0),
            # xy-plane (ez=0), middle of blend
            ([0.6, -0.8, 0.0], 3.1415926535897931),
            # yz-plane, deeper in blend
            ([0.0, -0.8499922481060458, 0.5267951956497234], 0.0),
            # xy-plane, at blend boundary (ey=-0.9)
            ([0.43589807987318724, -0.8999960355261952, 0.0], 1.5707963267948966),
            # general edge, near blend boundary
            ([0.1000022780778422, -0.7500170855838165, 0.6538148940729324], -0.1320878539772946),
        ]

        for edge, gamma in blend_region_edges:
            edge_t = torch.tensor([edge], dtype=dtype, device=device)
            gamma_t = torch.tensor([gamma], dtype=dtype, device=device)

            # Compute with axis-angle hybrid using pre-computed gamma
            D_hybrid, _ = axis_angle_wigner_hybrid(edge_t, lmax, gamma=gamma_t)

            # Get Euler angles from production code, zero out random gamma
            _, beta, alpha = init_edge_rot_euler_angles(edge_t)
            gamma_zero = torch.zeros(1, dtype=dtype, device=device)

            for ell in range(lmax + 1):
                start = ell * ell
                end = start + 2 * ell + 1
                D_euler = wigner_D(ell, gamma_zero, beta, alpha, Jd_matrices)
                D_hybrid_block = D_hybrid[0, start:end, start:end]

                assert torch.allclose(D_euler[0], D_hybrid_block, atol=1e-10), (
                    f"l={ell} mismatch for blend region edge {edge}"
                )


# =============================================================================
# Test Gradient Stability
# =============================================================================


class TestGradientStability:
    """Tests for gradient stability including near singularities."""

    @pytest.mark.parametrize("edge,desc", STANDARD_TEST_EDGES)
    def test_gradient_flow(self, lmax, dtype, device, edge, desc):
        """Gradients flow without NaN/Inf and are reasonably bounded."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device, requires_grad=True)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = axis_angle_wigner_hybrid(edge_t, lmax, gamma=gamma)
        loss = D.sum()
        loss.backward()

        grad = edge_t.grad
        assert not torch.isnan(grad).any(), f"NaN gradient for {desc}"
        assert not torch.isinf(grad).any(), f"Inf gradient for {desc}"
        assert grad.abs().max() < 1000, f"Gradient too large for {desc}: {grad.abs().max()}"

    @pytest.mark.parametrize("epsilon", [1e-4, 1e-6, 1e-8])
    def test_near_singularity_correctness(self, lmax, dtype, device, epsilon):
        """Edges near +/-Y still correctly map to +Y with bounded gradients."""
        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

        for sign in [1.0, -1.0]:
            edge = torch.tensor(
                [[epsilon, sign * 1.0, 0.0]], dtype=dtype, device=device, requires_grad=True
            )
            edge_norm = torch.nn.functional.normalize(edge, dim=-1)

            D, _ = axis_angle_wigner_hybrid(edge_norm, lmax)
            D_l1 = D[0, 1:4, 1:4]
            result = D_l1 @ edge_norm[0]

            # Check maps to Y
            assert torch.allclose(result, y_axis, atol=1e-5), (
                f"Near {'+'if sign>0 else '-'}Y edge (eps={epsilon}) did not map to +Y"
            )

            # Check gradients are valid and bounded
            D.sum().backward()
            assert not torch.isnan(edge.grad).any()
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

        coeffs = precompute_wigner_coefficients(lmax, dtype=dtype, device=device)

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
        """wigner_d_pair_to_real produces same result as complex version."""
        torch.manual_seed(42)
        q = torch.randn(20, 4, dtype=dtype, device=device)
        q = torch.nn.functional.normalize(q, dim=-1)

        coeffs = precompute_wigner_coefficients(lmax, dtype=dtype, device=device)
        U_blocks = precompute_U_blocks_euler_aligned(lmax, dtype=dtype, device=device)
        U_blocks_real = precompute_U_blocks_euler_aligned_real(lmax, dtype=dtype, device=device)

        # Complex version
        Ra, Rb = quaternion_to_ra_rb(q)
        D_complex = wigner_d_matrix_complex(Ra, Rb, coeffs)
        D_real_complex = wigner_d_complex_to_real(D_complex, U_blocks, lmax)

        # Real version
        ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)
        D_re, D_im = wigner_d_matrix_real(ra_re, ra_im, rb_re, rb_im, coeffs)
        D_real_from_pair = wigner_d_pair_to_real(D_re, D_im, U_blocks_real, lmax)

        # Compare
        assert torch.allclose(D_real_complex, D_real_from_pair, atol=1e-9)


class TestRealArithmeticGradients:
    """Tests verifying gradients match between real and complex arithmetic."""

    def test_gradient_equivalence(self, lmax, dtype, device):
        """Gradients from real-arithmetic methods match complex-arithmetic methods."""
        torch.manual_seed(42)
        edges_complex = torch.randn(10, 3, dtype=dtype, device=device, requires_grad=True)
        edges_real = edges_complex.detach().clone().requires_grad_(True)
        gamma = torch.rand(10, dtype=dtype, device=device) * 6.28

        # Complex version gradient
        D_complex, _ = axis_angle_wigner_hybrid(edges_complex, lmax, gamma=gamma)
        loss_complex = D_complex.sum()
        loss_complex.backward()

        # Real version gradient
        D_real, _ = axis_angle_wigner_hybrid(edges_real, lmax, gamma=gamma, use_real_arithmetic=True)
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
    def test_hybrid_real_compiles(self, lmax, dtype, device):
        """axis_angle_wigner_hybrid with use_real_arithmetic=True should compile without graph breaks."""
        import torch._dynamo as dynamo

        edges = torch.randn(10, 3, dtype=dtype, device=device)
        gamma = torch.rand(10, dtype=dtype, device=device) * 6.28

        # Define function to compile
        def fn(edge_vec, lmax_val, g):
            return axis_angle_wigner_hybrid(edge_vec, lmax_val, gamma=g, use_real_arithmetic=True)

        # Try to compile and run
        try:
            compiled_fn = torch.compile(fn, fullgraph=True)
            D, D_inv = compiled_fn(edges, lmax, gamma)

            # Verify output matches uncompiled version
            D_ref, D_inv_ref = axis_angle_wigner_hybrid(edges, lmax, gamma=gamma, use_real_arithmetic=True)
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
    def test_polynomial_real_compiles(self, lmax, dtype, device):
        """axis_angle_wigner_polynomial with use_real_arithmetic should compile without graph breaks."""
        import torch._dynamo as dynamo

        edges = torch.randn(10, 3, dtype=dtype, device=device)
        gamma = torch.rand(10, dtype=dtype, device=device) * 6.28

        def fn(edge_vec, lmax_val, g):
            return axis_angle_wigner_polynomial(edge_vec, lmax_val, gamma=g, use_real_arithmetic=True)

        try:
            compiled_fn = torch.compile(fn, fullgraph=True)
            D, D_inv = compiled_fn(edges, lmax, gamma)

            D_ref, D_inv_ref = axis_angle_wigner_polynomial(edges, lmax, gamma=gamma, use_real_arithmetic=True)
            assert torch.allclose(D, D_ref, atol=1e-10)
        except Exception as e:
            explanation = dynamo.explain(fn)(edges, lmax, gamma)
            pytest.fail(
                f"torch.compile failed. Graph break count: {explanation.graph_break_count}. "
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
        coeffs = precompute_wigner_coefficients(lmax, dtype=dtype, device=device)

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


# =============================================================================
# Test Range Functions (for hybrid lmin support)
# =============================================================================


class TestRangeFunctions:
    """Tests for the lmin-based range functions."""

    def test_range_matches_full(self, dtype, device):
        """Range Wigner D computation matches full computation for l >= lmin."""
        lmin, lmax = 3, 5
        torch.manual_seed(42)

        # Create quaternions
        q = torch.randn(30, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)
        Ra, Rb = quaternion_to_ra_rb(q)

        # Full computation
        coeffs_full = precompute_wigner_coefficients(lmax, dtype, device)
        U_blocks_full = precompute_U_blocks_euler_aligned(lmax, dtype, device)
        D_complex_full = wigner_d_matrix_complex(Ra, Rb, coeffs_full)
        D_real_full = wigner_d_complex_to_real(D_complex_full, U_blocks_full, lmax)

        # Range computation
        coeffs_range = precompute_wigner_coefficients(lmax, dtype, device, lmin=lmin)
        U_blocks_range = precompute_U_blocks_euler_aligned(lmax, dtype, device, lmin=lmin)
        D_complex_range = wigner_d_matrix_complex(Ra, Rb, coeffs_range)
        D_real_range = wigner_d_complex_to_real(D_complex_range, U_blocks_range, lmax, lmin=lmin)

        # Extract l >= lmin from full
        block_offset = lmin * lmin
        D_full_subset = D_real_full[:, block_offset:, block_offset:]

        # Compare
        max_err = (D_full_subset - D_real_range).abs().max().item()
        assert max_err < 1e-12, f"Range differs from full by {max_err}"


# =============================================================================
# Test Specialized Kernels
# =============================================================================


class TestSpecializedKernels:
    """Tests for the specialized l=2 polynomial and l=3/4 matmul kernels."""

    def test_l2_einsum_matches_matexp(self, dtype, device):
        """l=2 einsum kernel matches matrix exponential method."""
        from fairchem.core.models.uma.common.wigner_d_matexp import (
            quaternion_to_wigner_d_l2_einsum,
            quaternion_to_axis_angle,
            get_so3_generators,
        )

        torch.manual_seed(42)
        n_samples = 500
        q = torch.randn(n_samples, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        # Einsum method
        D_einsum = quaternion_to_wigner_d_l2_einsum(q)

        # Matrix exponential method
        axis, angle = quaternion_to_axis_angle(q)
        generators = get_so3_generators(2, dtype, device)
        K_x, K_y, K_z = generators['K_x'][2], generators['K_y'][2], generators['K_z'][2]
        K = (
            axis[:, 0:1, None, None] * K_x +
            axis[:, 1:2, None, None] * K_y +
            axis[:, 2:3, None, None] * K_z
        ).squeeze(1)
        D_matexp = torch.linalg.matrix_exp(angle[:, None, None] * K)

        max_err = (D_einsum - D_matexp).abs().max().item()
        assert max_err < 1e-10, f"l=2 einsum differs from matexp by {max_err}"

    def test_l3_matmul_matches_matexp(self, dtype, device):
        """l=3 matmul kernel matches matrix exponential method."""
        from fairchem.core.models.uma.common.wigner_d_matexp import (
            quaternion_to_wigner_d_l3_matmul,
            quaternion_to_axis_angle,
            get_so3_generators,
        )

        torch.manual_seed(42)
        n_samples = 100
        q = torch.randn(n_samples, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        # Matmul method
        D_matmul = quaternion_to_wigner_d_l3_matmul(q)

        # Matrix exponential method
        axis, angle = quaternion_to_axis_angle(q)
        generators = get_so3_generators(3, dtype, device)
        K_x, K_y, K_z = generators['K_x'][3], generators['K_y'][3], generators['K_z'][3]
        K = (
            axis[:, 0:1, None, None] * K_x +
            axis[:, 1:2, None, None] * K_y +
            axis[:, 2:3, None, None] * K_z
        ).squeeze(1)
        D_matexp = torch.linalg.matrix_exp(angle[:, None, None] * K)

        max_err = (D_matmul - D_matexp).abs().max().item()
        assert max_err < 1e-9, f"l=3 matmul differs from matexp by {max_err}"

    def test_l4_matmul_matches_matexp(self, dtype, device):
        """l=4 matmul kernel matches matrix exponential method."""
        from fairchem.core.models.uma.common.wigner_d_matexp import (
            quaternion_to_wigner_d_l4_matmul,
            quaternion_to_axis_angle,
            get_so3_generators,
        )

        torch.manual_seed(42)
        n_samples = 100
        q = torch.randn(n_samples, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        # Matmul method
        D_matmul = quaternion_to_wigner_d_l4_matmul(q)

        # Matrix exponential method
        axis, angle = quaternion_to_axis_angle(q)
        generators = get_so3_generators(4, dtype, device)
        K_x, K_y, K_z = generators['K_x'][4], generators['K_y'][4], generators['K_z'][4]
        K = (
            axis[:, 0:1, None, None] * K_x +
            axis[:, 1:2, None, None] * K_y +
            axis[:, 2:3, None, None] * K_z
        ).squeeze(1)
        D_matexp = torch.linalg.matrix_exp(angle[:, None, None] * K)

        max_err = (D_matmul - D_matexp).abs().max().item()
        assert max_err < 1e-9, f"l=4 matmul differs from matexp by {max_err}"

    def test_kernels_orthogonality(self, dtype, device):
        """Specialized kernels produce orthogonal matrices."""
        from fairchem.core.models.uma.common.wigner_d_matexp import (
            quaternion_to_wigner_d_l2_einsum,
            quaternion_to_wigner_d_l3_matmul,
            quaternion_to_wigner_d_l4_matmul,
        )

        torch.manual_seed(123)
        q = torch.randn(100, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        # l=2
        D2 = quaternion_to_wigner_d_l2_einsum(q)
        I5 = torch.eye(5, dtype=dtype, device=device)
        orth_err_2 = (D2 @ D2.transpose(-1, -2) - I5).abs().max().item()
        assert orth_err_2 < 1e-10, f"l=2 orthogonality error: {orth_err_2}"

        # l=3
        D3 = quaternion_to_wigner_d_l3_matmul(q)
        I7 = torch.eye(7, dtype=dtype, device=device)
        orth_err_3 = (D3 @ D3.transpose(-1, -2) - I7).abs().max().item()
        assert orth_err_3 < 1e-9, f"l=3 orthogonality error: {orth_err_3}"

        # l=4
        D4 = quaternion_to_wigner_d_l4_matmul(q)
        I9 = torch.eye(9, dtype=dtype, device=device)
        orth_err_4 = (D4 @ D4.transpose(-1, -2) - I9).abs().max().item()
        assert orth_err_4 < 1e-9, f"l=4 orthogonality error: {orth_err_4}"

    def test_kernels_determinant_one(self, dtype, device):
        """Specialized kernels produce matrices with determinant 1."""
        from fairchem.core.models.uma.common.wigner_d_matexp import (
            quaternion_to_wigner_d_l2_einsum,
            quaternion_to_wigner_d_l3_matmul,
            quaternion_to_wigner_d_l4_matmul,
        )

        torch.manual_seed(456)
        q = torch.randn(100, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        # l=2
        D2 = quaternion_to_wigner_d_l2_einsum(q)
        det_err_2 = (torch.linalg.det(D2) - 1.0).abs().max().item()
        assert det_err_2 < 1e-10, f"l=2 determinant error: {det_err_2}"

        # l=3
        D3 = quaternion_to_wigner_d_l3_matmul(q)
        det_err_3 = (torch.linalg.det(D3) - 1.0).abs().max().item()
        assert det_err_3 < 1e-9, f"l=3 determinant error: {det_err_3}"

        # l=4
        D4 = quaternion_to_wigner_d_l4_matmul(q)
        det_err_4 = (torch.linalg.det(D4) - 1.0).abs().max().item()
        assert det_err_4 < 1e-9, f"l=4 determinant error: {det_err_4}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
