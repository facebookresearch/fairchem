"""
Tests for axis-angle based Wigner D matrix computation.

Tests verify:
1. Mathematical correctness (orthogonality, determinant, edge → +Y)
2. Agreement between different implementations
3. Gradient stability

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
from fairchem.core.models.uma.common.wigner_d_axis_angle import (
    axis_angle_wigner,
    axis_angle_wigner_hybrid,
    axis_angle_wigner_polynomial,
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

        D, D_inv = axis_angle_wigner(edge_t, lmax, gamma=gamma)

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

        # Check inverse equals transpose
        assert torch.allclose(D_inv[0], D[0].T, atol=1e-10)

    @pytest.mark.parametrize("edge,desc", STANDARD_TEST_EDGES)
    def test_edge_to_y(self, lmax, dtype, device, edge, desc):
        """The l=1 block rotates edge → +Y."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        edge_t = torch.nn.functional.normalize(edge_t, dim=-1)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = axis_angle_wigner(edge_t, lmax, gamma=gamma)
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

        D1, _ = axis_angle_wigner(edges1, lmax)
        D2, _ = axis_angle_wigner(edges2, lmax)

        # Compose the Wigner D matrices
        D_product = D1 @ D2

        # The l=1 block of D_product is the composed rotation matrix R1 @ R2
        R_composed = D_product[:, 1:4, 1:4]

        # From R_composed, extract edge (second row, since R @ edge = +Y means edge = R^T @ +Y)
        edge_composed = R_composed[:, 1, :]

        # Compute D for edge_composed with gamma=0 to get the canonical alignment rotation
        D_canonical, _ = axis_angle_wigner(
            edge_composed, lmax, gamma=torch.zeros(n_samples, dtype=dtype, device=device)
        )
        R_canonical = D_canonical[:, 1:4, 1:4]

        # The composed rotation is R_composed = R_gamma @ R_canonical
        # So R_gamma = R_composed @ R_canonical^T
        R_gamma = R_composed @ R_canonical.transpose(-1, -2)

        # R_gamma is rotation around Y by gamma:
        # [[cos γ, 0, sin γ], [0, 1, 0], [-sin γ, 0, cos γ]]
        # So cos(γ) = R_gamma[0, 0] and sin(γ) = R_gamma[0, 2]
        gamma_composed = torch.atan2(R_gamma[:, 0, 2], R_gamma[:, 0, 0])

        # Compute D for the composed rotation
        D_composed, _ = axis_angle_wigner(edge_composed, lmax, gamma=gamma_composed)

        # Check that the product equals the composed Wigner D
        max_err = (D_product - D_composed).abs().max().item()
        assert max_err < 1e-9, f"Composition law failed: max error = {max_err}"


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

        D, _ = axis_angle_wigner(edge_t, lmax, gamma=gamma)
        loss = D.sum()
        loss.backward()

        grad = edge_t.grad
        assert not torch.isnan(grad).any(), f"NaN gradient for {desc}"
        assert not torch.isinf(grad).any(), f"Inf gradient for {desc}"
        assert grad.abs().max() < 1000, f"Gradient too large for {desc}: {grad.abs().max()}"

    @pytest.mark.parametrize("epsilon", [1e-4, 1e-6, 1e-8])
    def test_near_singularity_correctness(self, lmax, dtype, device, epsilon):
        """Edges near ±Y still correctly map to +Y with bounded gradients."""
        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

        for sign in [1.0, -1.0]:
            edge = torch.tensor(
                [[epsilon, sign * 1.0, 0.0]], dtype=dtype, device=device, requires_grad=True
            )
            edge_norm = torch.nn.functional.normalize(edge, dim=-1)

            D, _ = axis_angle_wigner(edge_norm, lmax)
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
# Test Agreement Between Implementations
# =============================================================================


class TestImplementationAgreement:
    """Tests for agreement between different Wigner D implementations."""

    def test_all_methods_match(self, lmax, dtype, device):
        """All axis_angle implementations produce identical results."""
        torch.manual_seed(42)
        edges = torch.randn(50, 3, dtype=dtype, device=device)
        gamma = torch.rand(50, dtype=dtype, device=device) * 6.28

        D_base, _ = axis_angle_wigner(edges, lmax, gamma=gamma)
        D_poly, _ = axis_angle_wigner_polynomial(edges, lmax, gamma=gamma)
        D_hybrid, _ = axis_angle_wigner_hybrid(edges, lmax, gamma=gamma)

        assert (D_base - D_poly).abs().max() < 1e-9, "polynomial differs from base"
        assert (D_base - D_hybrid).abs().max() < 1e-10, "hybrid differs from base"

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
            edge_norm = torch.nn.functional.normalize(edge_t, dim=-1)

            # Compute with axis-angle using Euler gamma
            D_axis, _ = axis_angle_wigner(edge_norm, lmax, use_euler_gamma=True)

            # Compare with Euler
            alpha = torch.atan2(edge_norm[0, 0], edge_norm[0, 2])
            beta = torch.acos(edge_norm[0, 1].clamp(-1, 1))
            gamma_val = torch.zeros(1, dtype=dtype, device=device)

            for ell in range(lmax + 1):
                start = ell * ell
                end = start + 2 * ell + 1
                D_euler = wigner_D(ell, -gamma_val, -beta, -alpha, Jd_matrices)
                D_axis_block = D_axis[0, start:end, start:end]

                assert torch.allclose(D_euler, D_axis_block, atol=1e-10), (
                    f"l={ell} mismatch for edge {edge}"
                )


# =============================================================================
# Test Range Functions (for hybrid lmin support)
# =============================================================================


class TestRangeFunctions:
    """Tests for the lmin-based range functions in wigner_d_quaternion."""

    def test_range_matches_full(self, dtype, device):
        """Range Wigner D computation matches full computation for l >= lmin."""
        from fairchem.core.models.uma.common.wigner_d_quaternion import (
            precompute_wigner_coefficients_symmetric,
            precompute_wigner_coefficients_range,
            precompute_U_blocks,
            precompute_U_blocks_range,
            wigner_d_matrix_complex,
            wigner_d_matrix_complex_range,
            wigner_d_complex_to_real_blockwise,
            wigner_d_complex_to_real_range,
            quaternion_to_ra_rb,
        )

        lmin, lmax = 3, 5
        torch.manual_seed(42)

        # Create quaternions
        q = torch.randn(30, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)
        Ra, Rb = quaternion_to_ra_rb(q)

        # Full computation
        coeffs_full = precompute_wigner_coefficients_symmetric(lmax, dtype, device)
        U_blocks_full = precompute_U_blocks(lmax, dtype, device)
        D_complex_full = wigner_d_matrix_complex(Ra, Rb, coeffs_full)
        D_real_full = wigner_d_complex_to_real_blockwise(D_complex_full, U_blocks_full, lmax)

        # Range computation
        coeffs_range = precompute_wigner_coefficients_range(lmin, lmax, dtype, device)
        U_blocks_range = precompute_U_blocks_range(lmin, lmax, dtype, device)
        D_complex_range = wigner_d_matrix_complex_range(Ra, Rb, coeffs_range)
        D_real_range = wigner_d_complex_to_real_range(D_complex_range, U_blocks_range, lmin, lmax)

        # Extract l >= lmin from full
        block_offset = lmin * lmin
        D_full_subset = D_real_full[:, block_offset:, block_offset:]

        # Compare
        max_err = (D_full_subset - D_real_range).abs().max().item()
        assert max_err < 1e-12, f"Range differs from full by {max_err}"


# =============================================================================
# Test Quaternion to Wigner D l=2 Polynomial Function
# =============================================================================


class TestQuaternionToWignerDL2:
    """Tests for the quaternion_to_wigner_d_l2 polynomial function."""

    def test_matches_cayley_hamilton(self, dtype, device):
        """Polynomial function matches Cayley-Hamilton for random quaternions."""
        from fairchem.core.models.uma.common.wigner_d_axis_angle import (
            quaternion_to_wigner_d_l2,
            _cayley_hamilton_exp_l2,
            quaternion_to_axis_angle,
            get_so3_generators,
        )

        torch.manual_seed(42)
        n_samples = 500
        q = torch.randn(n_samples, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        # Polynomial method
        D_poly = quaternion_to_wigner_d_l2(q)

        # Cayley-Hamilton method
        axis, angle = quaternion_to_axis_angle(q)
        generators = get_so3_generators(2, dtype, device)
        K_x, K_y, K_z = generators['K_x'][2], generators['K_y'][2], generators['K_z'][2]
        K = (
            axis[:, 0:1, None, None] * K_x +
            axis[:, 1:2, None, None] * K_y +
            axis[:, 2:3, None, None] * K_z
        ).squeeze(1)
        D_cayley = _cayley_hamilton_exp_l2(K, angle)

        max_err = (D_poly - D_cayley).abs().max().item()
        assert max_err < 1e-10, f"Polynomial differs from Cayley-Hamilton by {max_err}"

    def test_orthogonality(self, dtype, device):
        """Polynomial function produces orthogonal matrices."""
        from fairchem.core.models.uma.common.wigner_d_axis_angle import (
            quaternion_to_wigner_d_l2,
        )

        torch.manual_seed(123)
        q = torch.randn(100, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        D = quaternion_to_wigner_d_l2(q)
        I = torch.eye(5, dtype=dtype, device=device)

        orth_err = (D @ D.transpose(-1, -2) - I).abs().max().item()
        assert orth_err < 1e-10, f"Orthogonality error: {orth_err}"

    def test_determinant_one(self, dtype, device):
        """Polynomial function produces matrices with determinant 1."""
        from fairchem.core.models.uma.common.wigner_d_axis_angle import (
            quaternion_to_wigner_d_l2,
        )

        torch.manual_seed(456)
        q = torch.randn(100, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        D = quaternion_to_wigner_d_l2(q)
        dets = torch.linalg.det(D)

        det_err = (dets - 1.0).abs().max().item()
        assert det_err < 1e-10, f"Determinant error: {det_err}"

    def test_gradcheck(self, dtype, device):
        """Gradcheck passes for the polynomial function."""
        from fairchem.core.models.uma.common.wigner_d_axis_angle import (
            quaternion_to_wigner_d_l2,
        )

        q = torch.randn(5, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)
        q = q.detach().requires_grad_(True)

        result = torch.autograd.gradcheck(
            quaternion_to_wigner_d_l2, q, eps=1e-6, atol=1e-4
        )
        assert result

    def test_identity_quaternion(self, dtype, device):
        """Identity quaternion produces identity matrix."""
        from fairchem.core.models.uma.common.wigner_d_axis_angle import (
            quaternion_to_wigner_d_l2,
        )

        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)
        D = quaternion_to_wigner_d_l2(q)
        I = torch.eye(5, dtype=dtype, device=device)

        err = (D[0] - I).abs().max().item()
        assert err < 1e-10, f"Identity error: {err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
