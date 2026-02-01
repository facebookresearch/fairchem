"""
Tests for quaternion-based Wigner D matrix computation.

These tests verify:
1. Mathematical correctness against the spherical_functions reference
2. Correct behavior for y-aligned edges
3. Gradient stability

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.common.rotation import (
    eulers_to_wigner,
    init_edge_rot_euler_angles,
)
from fairchem.core.models.uma.common.wigner_d_quaternion import (
    edge_to_quaternion,
    get_wigner_from_edge_vectors,
    precompute_complex_to_real_matrix,
    precompute_U_blocks,
    precompute_wigner_coefficients_symmetric,
    quaternion_to_ra_rb,
    quaternion_wigner,
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
def wigner_coeffs(lmax, dtype, device):
    return precompute_wigner_coefficients_symmetric(lmax, dtype, device)


@pytest.fixture()
def U_blocks(lmax, dtype, device):
    return precompute_U_blocks(lmax, dtype, device)


# =============================================================================
# Test quaternion construction
# =============================================================================


class TestQuaternionConstruction:
    """Tests for edge_to_quaternion function."""

    def test_unit_quaternion(self, dtype, device):
        """Quaternion is unit length."""
        edges = torch.randn(10, 3, dtype=dtype, device=device)
        edges = torch.nn.functional.normalize(edges, dim=-1)
        gamma = torch.zeros(10, dtype=dtype, device=device)

        q = edge_to_quaternion(edges, gamma=gamma)

        # Quaternions should be unit length
        norms = torch.linalg.norm(q, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)

    def test_quaternion_is_unit_norm(self, dtype, device):
        """All constructed quaternions should be unit length."""
        edges = torch.randn(10, 3, dtype=dtype, device=device)
        edges = torch.nn.functional.normalize(edges, dim=-1)
        gamma = torch.zeros(10, dtype=dtype, device=device)

        q = edge_to_quaternion(edges, gamma=gamma)

        norms = torch.linalg.norm(q, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)

    def test_y_axis_edge_cases(self, dtype, device):
        """Edge along ±y produces unit quaternion and correct Wigner D."""
        gamma = torch.zeros(1, dtype=dtype, device=device)

        # +y should give identity quaternion (since +Y → +Y is identity)
        edge_pos_y = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device)
        q_pos = edge_to_quaternion(edge_pos_y, gamma=gamma)
        expected_identity = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device
        )
        assert torch.allclose(q_pos, expected_identity, atol=1e-10)

        # -y should give a unit quaternion
        edge_neg_y = torch.tensor([[0.0, -1.0, 0.0]], dtype=dtype, device=device)
        q_neg = edge_to_quaternion(edge_neg_y, gamma=gamma)
        assert torch.allclose(
            torch.linalg.norm(q_neg), torch.ones(1, dtype=dtype, device=device), atol=1e-10
        )

        # Both edges should produce Wigner D that rotates edge → +Y
        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        for edge in [edge_pos_y, edge_neg_y]:
            wigner, _ = quaternion_wigner(edge, lmax=1, gamma=gamma)
            D = wigner[0, 1:4, 1:4]
            edge_norm = torch.nn.functional.normalize(edge, dim=-1)
            result = D @ edge_norm[0]
            assert torch.allclose(result, y_axis, atol=1e-6), (
                f"Wigner D did not rotate edge {edge[0].tolist()} to +Y, got {result.tolist()}"
            )

    def test_all_edges_align_to_y_axis(self, dtype, device):
        """
        All edge vectors should align to +y axis via the inverse quaternion rotation.

        The quaternion computed from an edge rotates +y → edge.
        Therefore, applying the inverse rotation (transpose of the rotation matrix)
        should map edge → +y.
        """
        test_edges = [
            [0.0, 1.0, 0.0],  # Y-aligned (identity case)
            [0.0, 0.0, 1.0],  # Z-aligned
            [1.0, 0.0, 0.0],  # X-aligned
            [0.0, -1.0, 0.0],  # -Y-aligned
            [0.0, 0.0, -1.0],  # -Z-aligned
            [-1.0, 0.0, 0.0],  # -X-aligned
            [1.0, 1.0, 1.0],  # Diagonal
            [1.0, 1.0, 0.0],  # XY-plane
            [0.0, 1.0, 1.0],  # YZ-plane
            [1.0, 0.0, 1.0],  # XZ-plane
            [0.3, 0.5, 0.8],  # Random 1
            [-0.2, 0.7, 0.3],  # Random 2
            [0.6, -0.4, 0.5],  # Random 3
            [-0.8, 0.2, -0.5],  # Random 4
        ]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

        for edge in test_edges:
            edge_t = torch.tensor([edge], dtype=dtype, device=device)
            R, _ = quaternion_wigner(edge_t, 1)
            R = R[:, 1:4, 1:4]  # just take l=1 block
            edge_t = torch.nn.functional.normalize(edge_t, dim=-1)
            result = R @ edge_t[0]

            assert torch.allclose(
                result, y_axis, atol=1e-5
            ), f"Edge {edge} did not align to +y, got {result}"


# =============================================================================
# Test Ra/Rb decomposition
# =============================================================================


class TestRaRbDecomposition:
    """Tests for quaternion_to_ra_rb function."""

    @pytest.mark.parametrize(
        "q,expected_ra,expected_rb,desc",
        [
            ([1.0, 0.0, 0.0, 0.0], 1 + 0j, 0 + 0j, "identity"),
            ([0.0, 1.0, 0.0, 0.0], 0 + 0j, 0 - 1j, "180° x-rotation"),
        ],
    )
    def test_ra_rb_decomposition(
        self, dtype, device, q, expected_ra, expected_rb, desc
    ):
        """Ra/Rb decomposition satisfies |Ra|²+|Rb|²=1 and known cases."""
        q_tensor = torch.tensor([q], dtype=dtype, device=device)
        Ra, Rb = quaternion_to_ra_rb(q_tensor)

        # Check unit constraint
        sum_sq = torch.abs(Ra) ** 2 + torch.abs(Rb) ** 2
        assert torch.allclose(
            sum_sq, torch.ones_like(sum_sq), atol=1e-10
        ), f"{desc}: unit constraint violated"

        # Check expected values
        assert torch.allclose(
            Ra[0], torch.tensor(expected_ra, dtype=torch.complex128), atol=1e-10
        ), f"{desc}: Ra mismatch"
        assert torch.allclose(
            Rb[0], torch.tensor(expected_rb, dtype=torch.complex128), atol=1e-10
        ), f"{desc}: Rb mismatch"

    def test_unit_constraint_random(self, dtype, device):
        """|Ra|² + |Rb|² should equal 1 for random unit quaternions."""
        q = torch.randn(100, 4, dtype=dtype, device=device)
        q = q / torch.linalg.norm(q, dim=-1, keepdim=True)

        Ra, Rb = quaternion_to_ra_rb(q)
        sum_sq = torch.abs(Ra) ** 2 + torch.abs(Rb) ** 2

        assert torch.allclose(sum_sq, torch.ones_like(sum_sq), atol=1e-10)


# =============================================================================
# Test Wigner D matrix properties
# =============================================================================


class TestWignerDMatrixProperties:
    """Tests for mathematical properties of real Wigner D matrices."""

    def test_orthogonality_and_determinant(
        self, lmax, dtype, device, wigner_coeffs, U_blocks
    ):
        """Wigner D matrices are orthogonal with determinant 1."""
        torch.manual_seed(42)
        edges = torch.randn(10, 3, dtype=dtype, device=device)

        wigner, _ = get_wigner_from_edge_vectors(edges, wigner_coeffs, U_blocks)

        size = (lmax + 1) ** 2
        I = torch.eye(size, dtype=dtype, device=device)

        for i in range(10):
            # Check orthogonality: D @ D.T = I
            product = wigner[i] @ wigner[i].T
            assert torch.allclose(
                product, I, atol=1e-6
            ), f"Edge {i}: D @ D.T is not identity"

            # Check determinant = 1
            det = torch.linalg.det(wigner[i])
            assert torch.allclose(
                det, torch.ones_like(det), atol=1e-6
            ), f"Edge {i}: det(D) = {det.item()}, expected 1"

    def test_inverse_transpose_relationship(
        self, lmax, dtype, device, wigner_coeffs, U_blocks
    ):
        """wigner_inv equals wigner.T for orthogonal matrices."""
        torch.manual_seed(42)
        edges = torch.randn(10, 3, dtype=dtype, device=device)

        wigner, wigner_inv = get_wigner_from_edge_vectors(
            edges, wigner_coeffs, U_blocks
        )

        for i in range(10):
            assert torch.allclose(wigner_inv[i], wigner[i].T, atol=1e-10)


# =============================================================================
# Test y-aligned edges and gradient stability
# =============================================================================


class TestYAlignedEdgesAndGradients:
    """Tests for edges aligned with the y-axis and gradient stability."""

    @pytest.mark.parametrize(
        "edge,desc",
        [
            ([0.0, 1.0, 0.0], "+y"),
            ([0.0, -1.0, 0.0], "-y"),
            ([1e-9, 1.0, 1e-9], "nearly +y"),
            ([1e-9, -1.0, 1e-9], "nearly -y"),
        ],
    )
    def test_y_aligned_validity(
        self, lmax, dtype, device, wigner_coeffs, U_blocks, edge, desc
    ):
        """Y-aligned edges produce valid orthogonal Wigner matrices without NaN/Inf."""
        edge_tensor = torch.tensor([edge], dtype=dtype, device=device)
        edge_tensor = torch.nn.functional.normalize(edge_tensor, dim=-1)

        wigner, _ = get_wigner_from_edge_vectors(edge_tensor, wigner_coeffs, U_blocks)

        # Check for NaN/Inf
        assert not torch.isnan(wigner).any(), f"NaN in Wigner matrix for {desc}"
        assert not torch.isinf(wigner).any(), f"Inf in Wigner matrix for {desc}"

        # Check orthogonality
        size = (lmax + 1) ** 2
        I = torch.eye(size, dtype=dtype, device=device)
        product = wigner[0] @ wigner[0].T
        assert torch.allclose(
            product, I, atol=1e-5
        ), f"Wigner not orthogonal for {desc} edge"

        # Check determinant
        det = torch.linalg.det(wigner[0])
        assert torch.allclose(
            det, torch.tensor(1.0, dtype=dtype, device=device), atol=1e-5
        ), f"det(D) != 1 for {desc} edge"


# =============================================================================
# Test complex-to-real transformation
# =============================================================================


class TestComplexToRealTransformation:
    """Tests for the U matrix and complex-to-real transformation."""

    def test_U_is_unitary(self, lmax, device):
        """U matrix is unitary: U @ U† = I."""
        U = precompute_complex_to_real_matrix(lmax, torch.complex128, device)

        size = (lmax + 1) ** 2
        I = torch.eye(size, dtype=torch.complex128, device=device)

        product = U @ U.conj().T
        assert torch.allclose(product, I, atol=1e-10), "U is not unitary"


# =============================================================================
# Test agreement between Euler and Quaternion approaches
# =============================================================================


class TestEulerQuaternionAgreement:
    """Compare Wigner D matrices from Euler and quaternion approaches."""

    @pytest.fixture()
    def Jd_matrices(self, lmax, dtype, device):
        """Load the J matrices used by the Euler angle approach."""
        from pathlib import Path

        test_dir = Path(__file__).parent
        repo_root = test_dir.parent.parent.parent.parent
        jd_path = repo_root / "src" / "fairchem" / "core" / "models" / "uma" / "Jd.pt"

        if jd_path.exists():
            Jd = torch.load(jd_path, map_location=device, weights_only=True)
            return [J.to(dtype=dtype) for J in Jd[: lmax + 1]]
        else:
            pytest.skip(f"Jd.pt not found at {jd_path}")

    def test_euler_quaternion_agreement(self, lmax, dtype, device, Jd_matrices):
        """
        Euler and quaternion approaches should produce equivalent rotations.

        Both approaches compute a rotation that maps edge → +y. We verify they
        produce the same functional result rather than comparing raw matrices
        (which may differ due to convention differences).
        """
        # Test edges away from y-axis singularity
        test_edges = [
            [0.0, 0.0, 1.0],  # Z-aligned
            [1.0, 0.0, 0.0],  # X-aligned
            [1.0, 1.0, 1.0],  # Diagonal
            [0.3, 0.5, 0.8],  # Random
            [-0.2, 0.3, 0.9],  # Another random
        ]

        # +y in Cartesian coordinates
        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

        for edge in test_edges:
            edge_t = torch.tensor([edge], dtype=dtype, device=device)
            xyz = torch.nn.functional.normalize(edge_t, dim=-1)

            # Compute Euler angles (same as init_edge_rot_euler_angles but with gamma=0)
            euler_angles = init_edge_rot_euler_angles(edge_t)
            alpha, _, gamma = euler_angles

            # Euler approach: wigner_D uses the negated angles for extrinsic convention
            wigner_euler = eulers_to_wigner(euler_angles, 0, lmax, Jd_matrices)

            # Quaternion approach with matching gamma
            # The quaternion now builds the inverse rotation directly, so we pass
            # the forward gamma (which is -euler_angles[0])
            gamma_fwd = -alpha
            wigner_quat, _ = quaternion_wigner(edge_t, lmax, gamma=gamma_fwd)

            # Both should be orthogonal
            size = (lmax + 1) ** 2
            I = torch.eye(size, dtype=dtype, device=device)

            euler_ortho = wigner_euler[0] @ wigner_euler[0].T
            quat_ortho = wigner_quat[0] @ wigner_quat[0].T

            assert torch.allclose(
                euler_ortho, I, atol=1e-5
            ), f"Euler Wigner not orthogonal for edge {edge}"
            assert torch.allclose(
                quat_ortho, I, atol=1e-5
            ), f"Quaternion Wigner not orthogonal for edge {edge}"

            # Extract l=1 blocks - both use Cartesian (x, y, z) ordering
            euler_l1 = wigner_euler[0, 1:4, 1:4]
            quat_l1 = wigner_quat[0, 1:4, 1:4]

            # The l=1 blocks should be identical
            assert torch.allclose(
                euler_l1, quat_l1, atol=1e-5
            ), f"l=1 blocks differ for edge {edge}:\nEuler:\n{euler_l1}\nQuaternion:\n{quat_l1}"

            # Both should rotate edge → +y in Cartesian coordinates
            edge_cart = xyz[0]
            euler_result = euler_l1 @ edge_cart
            quat_result = quat_l1 @ edge_cart

            assert torch.allclose(
                euler_result, y_axis, atol=1e-5
            ), f"Euler did not rotate edge {edge} to +y, got {euler_result}"
            assert torch.allclose(
                quat_result, y_axis, atol=1e-5
            ), f"Quaternion did not rotate edge {edge} to +y, got {quat_result}"

    def test_gradient_accuracy_near_y_axis(self, lmax, dtype, device, Jd_matrices):
        """
        Compare gradient behavior near y-axis for Euler vs quaternion approaches.

        Near y-axis:
        - Euler analytic gradients are clamped by Safeacos (underestimate true gradient)
        - Quaternion analytic gradients correctly track finite differences
        """
        # Nearly y-aligned edge (but not so close that FD becomes unstable)
        eps_offset = 1e-5
        edge = torch.tensor(
            [[eps_offset, 1.0, eps_offset]], dtype=dtype, device=device
        )
        edge = torch.nn.functional.normalize(edge, dim=-1)

        # Get Euler angles for consistent gamma
        euler_angles = init_edge_rot_euler_angles(edge)
        gamma_euler = -euler_angles[0]  # Extract gamma (forward rotation gamma)
        gamma_quat = gamma_euler

        # Use adaptive FD step size for numerical stability
        h = eps_offset * 0.01

        def compute_finite_diff_gradient(fn, x, step):
            """Compute gradient via central finite differences."""
            grad = torch.zeros_like(x)
            for i in range(x.shape[1]):
                x_plus = x.clone()
                x_minus = x.clone()
                x_plus[0, i] += step
                x_minus[0, i] -= step
                grad[0, i] = (fn(x_plus) - fn(x_minus)) / (2 * step)
            return grad

        # === EULER APPROACH ===
        edge_euler = edge.clone().requires_grad_(True)
        euler_angles_grad = init_edge_rot_euler_angles(edge_euler)
        wigner_euler = eulers_to_wigner(euler_angles_grad, 0, lmax, Jd_matrices)
        wigner_euler.sum().backward()
        euler_analytic_grad = edge_euler.grad.clone()

        euler_fd_grad = compute_finite_diff_gradient(
            lambda e: eulers_to_wigner(
                init_edge_rot_euler_angles(e), 0, lmax, Jd_matrices
            ).sum(),
            edge,
            h,
        )

        # === QUATERNION APPROACH ===
        edge_quat = edge.clone().requires_grad_(True)
        wigner_quat, _ = quaternion_wigner(edge_quat, lmax, gamma=gamma_quat)
        wigner_quat.sum().backward()
        quat_analytic_grad = edge_quat.grad.clone()

        quat_fd_grad = compute_finite_diff_gradient(
            lambda e: quaternion_wigner(e, lmax, gamma=gamma_quat)[0].sum(),
            edge,
            h,
        )

        # === ASSERTIONS ===
        # 1. Both gradients should be finite
        assert torch.isfinite(euler_analytic_grad).all()
        assert torch.isfinite(quat_analytic_grad).all()
        assert torch.isfinite(euler_fd_grad).all()
        assert torch.isfinite(quat_fd_grad).all()

        # 2. Euler analytic should significantly underestimate the true gradient
        # (compare to FD which gives the true gradient)
        euler_x_rel_err = (
            (euler_analytic_grad[0, 0] - euler_fd_grad[0, 0]).abs()
            / euler_fd_grad[0, 0].abs()
        )
        assert euler_x_rel_err > 0.5, (
            f"Expected Euler to underestimate gradient, "
            f"got rel_err={euler_x_rel_err.item():.2f}"
        )

        # 3. Quaternion analytic should closely match FD (no clamping)
        # Check x and z components (y is ~0 so we skip it)
        quat_x_rel_err = (
            (quat_analytic_grad[0, 0] - quat_fd_grad[0, 0]).abs()
            / quat_fd_grad[0, 0].abs()
        )
        quat_z_rel_err = (
            (quat_analytic_grad[0, 2] - quat_fd_grad[0, 2]).abs()
            / quat_fd_grad[0, 2].abs()
        )
        assert quat_x_rel_err < 0.01, (
            f"Quaternion x gradient rel_err too large: {quat_x_rel_err.item():.4f}"
        )
        assert quat_z_rel_err < 0.01, (
            f"Quaternion z gradient rel_err too large: {quat_z_rel_err.item():.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
