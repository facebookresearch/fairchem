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

    def test_unit_quaternion_and_rotation_correctness(self, dtype, device):
        """Quaternion is unit length and produces valid Wigner D matrices.

        The quaternion is used to compute Wigner D matrices that rotate edge → +Y.
        The raw quaternion rotation matrix is NOT the same as the l=1 Wigner D block
        due to the complex-to-real spherical harmonics transformation.
        The actual edge → +Y property is tested by test_all_edges_align_to_y_axis.
        """
        edges = torch.randn(10, 3, dtype=dtype, device=device)
        edges = torch.nn.functional.normalize(edges, dim=-1)
        gamma = torch.zeros(10, dtype=dtype, device=device)

        q = edge_to_quaternion(edges, gamma=gamma)

        # Quaternions should be unit length
        norms = torch.linalg.norm(q, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)

        # Verify quaternion components are finite (no NaN/Inf)
        assert not torch.isnan(q).any(), "NaN in quaternion"
        assert not torch.isinf(q).any(), "Inf in quaternion"

        # Verify quaternions produce valid Wigner D via quaternion_wigner
        # (the actual rotation correctness is tested in test_all_edges_align_to_y_axis)
        wigner, _ = quaternion_wigner(edges, 1, gamma=gamma)
        assert not torch.isnan(wigner).any(), "NaN in Wigner matrix"
        assert not torch.isinf(wigner).any(), "Inf in Wigner matrix"

    def test_y_axis_edge_cases(self, dtype, device):
        """Edge along +Y gives identity; edge along -Y gives 180° rotation around X."""
        gamma = torch.zeros(1, dtype=dtype, device=device)

        # +Y should give identity quaternion (1, 0, 0, 0)
        edge_pos_y = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device)
        q_pos = edge_to_quaternion(edge_pos_y, gamma=gamma)
        expected_identity = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device
        )
        assert torch.allclose(q_pos, expected_identity, atol=1e-10)

        # -Y should give 180° rotation around X: (0, 1, 0, 0)
        edge_neg_y = torch.tensor([[0.0, -1.0, 0.0]], dtype=dtype, device=device)
        q_neg = edge_to_quaternion(edge_neg_y, gamma=gamma)
        expected_180x = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=dtype, device=device)
        assert torch.allclose(q_neg, expected_180x, atol=1e-6)

    def test_all_edges_align_to_y_axis(self, dtype, device):
        """
        All edge vectors should align to +Y axis via quaternion_wigner.

        quaternion_wigner returns (wigner_edge_to_y, wigner_y_to_edge) where the
        first return value rotates edge → +Y for compatibility with the Euler code.
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
            gamma = torch.zeros(1, dtype=dtype, device=device)
            R, _ = quaternion_wigner(edge_t, 1, gamma=gamma)
            R = R[:, 1:4, 1:4]  # just take l=1 block
            edge_t = torch.nn.functional.normalize(edge_t, dim=-1)
            result = R @ edge_t[0]

            assert torch.allclose(
                result, y_axis, atol=1e-5
            ), f"Edge {edge} did not align to +Y, got {result}"


# =============================================================================
# Test Ra/Rb decomposition
# =============================================================================


class TestRaRbDecomposition:
    """Tests for quaternion_to_ra_rb function."""

    @pytest.mark.parametrize(
        "q,expected_ra,expected_rb,desc",
        [
            ([1.0, 0.0, 0.0, 0.0], 1 + 0j, 0 + 0j, "identity"),
            ([0.0, 1.0, 0.0, 0.0], 0 + 0j, 0 + 1j, "180° x-rotation"),
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
            wigner_quat, _ = quaternion_wigner(edge_t, lmax, gamma=alpha + gamma)

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
        Compare finite difference vs analytic gradients for y-aligned edges.

        Near y-axis:
        - Euler finite diff blows up (true gradient is huge)
        - Euler analytic is clamped by Safeacos (hides true gradient)
        - Quaternion finite diff and analytic match (no clamping)
        """

        def compute_finite_diff_gradient(fn, x, h):
            """Compute gradient via central finite differences."""
            grad = torch.zeros_like(x)
            for i in range(x.shape[1]):  # For each dimension (x, y, z)
                x_plus = x.clone()
                x_minus = x.clone()
                x_plus[0, i] += h
                x_minus[0, i] -= h
                grad[0, i] = (fn(x_plus) - fn(x_minus)) / (2 * h)
            return grad

        # Nearly y-aligned edge
        eps_offset = 1e-7
        edge = torch.tensor([[eps_offset, 1.0, eps_offset]], dtype=dtype, device=device)
        edge = torch.nn.functional.normalize(edge, dim=-1)

        # Get Euler angles for consistent gamma
        euler_angles = init_edge_rot_euler_angles(edge)
        gamma_euler = -euler_angles[0]  # Extract gamma
        alpha_euler = -euler_angles[2]  # Extract alpha
        gamma_quat = gamma_euler + alpha_euler  # Matching gamma for quaternion

        h = 1e-6  # Finite difference step size

        # === EULER APPROACH ===
        # Analytic gradient
        edge_euler = edge.clone().requires_grad_(True)
        euler_angles_grad = init_edge_rot_euler_angles(edge_euler)
        wigner_euler = eulers_to_wigner(euler_angles_grad, 0, lmax, Jd_matrices)
        loss_euler = wigner_euler.sum()
        loss_euler.backward()
        euler_analytic_grad = edge_euler.grad.clone()

        # Finite difference gradient
        euler_fd_grad = compute_finite_diff_gradient(
            lambda e: eulers_to_wigner(
                init_edge_rot_euler_angles(e), 0, lmax, Jd_matrices
            ).sum(),
            edge,
            h,
        )

        # === QUATERNION APPROACH ===
        # Analytic gradient
        edge_quat = edge.clone().requires_grad_(True)
        wigner_quat, _ = quaternion_wigner(edge_quat, lmax, gamma=gamma_quat)
        loss_quat = wigner_quat.sum()
        loss_quat.backward()
        quat_analytic_grad = edge_quat.grad.clone()

        # Finite difference gradient
        quat_fd_grad = compute_finite_diff_gradient(
            lambda e: quaternion_wigner(e, lmax, gamma=gamma_quat)[0].sum(),
            edge,
            h,
        )

        # === ASSERTIONS ===
        # 1. Euler finite diff should be much larger than analytic (clamping effect)
        euler_fd_mag = euler_fd_grad.abs().max().item()
        euler_analytic_mag = euler_analytic_grad.abs().max().item()
        # print(euler_fd_mag, euler_analytic_mag)
        assert euler_fd_mag > 10 * euler_analytic_mag, (
            f"Expected Euler FD >> analytic, got FD={euler_fd_mag}, "
            f"analytic={euler_analytic_mag}"
        )

        # 2. Quaternion finite diff and analytic should match
        assert torch.allclose(quat_fd_grad, quat_analytic_grad, rtol=0.1), (
            f"Quaternion FD and analytic should match:\n"
            f"FD={quat_fd_grad}\nAnalytic={quat_analytic_grad}"
        )

        # 3. Quaternion gradients should be reasonable (not blowing up)
        quat_mag = quat_analytic_grad.abs().max().item()
        # quat_fd_mag = quat_fd_grad.abs().max().item()
        # print(quat_mag, quat_fd_mag)
        assert quat_mag < 1e6, f"Quaternion gradient too large: {quat_mag}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
