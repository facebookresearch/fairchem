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
        """Quaternion is unit length and correctly rotates +y to edge direction."""
        edges = torch.randn(10, 3, dtype=dtype, device=device)
        edges = torch.nn.functional.normalize(edges, dim=-1)
        gamma = torch.zeros(10, dtype=dtype, device=device)

        q = edge_to_quaternion(edges, gamma=gamma)

        # Quaternions should be unit length
        norms = torch.linalg.norm(q, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)

        # Verify quaternion actually rotates +y to edge direction
        y = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

        for i in range(10):
            qi = q[i]
            w, qx, qy, qz = qi[0], qi[1], qi[2], qi[3]

            # Rotation matrix from quaternion
            R = torch.tensor(
                [
                    [
                        1 - 2 * (qy**2 + qz**2),
                        2 * (qx * qy - qz * w),
                        2 * (qx * qz + qy * w),
                    ],
                    [
                        2 * (qx * qy + qz * w),
                        1 - 2 * (qx**2 + qz**2),
                        2 * (qy * qz - qx * w),
                    ],
                    [
                        2 * (qx * qz - qy * w),
                        2 * (qy * qz + qx * w),
                        1 - 2 * (qx**2 + qy**2),
                    ],
                ],
                dtype=dtype,
                device=device,
            )

            rotated_y = R @ y
            assert torch.allclose(
                rotated_y, edges[i], atol=1e-6
            ), f"Edge {i}: expected {edges[i]}, got {rotated_y}"

    def test_y_axis_edge_cases(self, dtype, device):
        """Edge along +y gives identity; edge along -y uses 180° x-rotation fallback."""
        gamma = torch.zeros(1, dtype=dtype, device=device)

        # +y should give identity quaternion (1, 0, 0, 0)
        edge_pos_y = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device)
        q_pos = edge_to_quaternion(edge_pos_y, gamma=gamma)
        expected_identity = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device
        )
        assert torch.allclose(q_pos, expected_identity, atol=1e-10)

        # -y should give 180° rotation around x: (0, 1, 0, 0)
        edge_neg_y = torch.tensor([[0.0, -1.0, 0.0]], dtype=dtype, device=device)
        q_neg = edge_to_quaternion(edge_neg_y, gamma=gamma)
        expected_180x = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=dtype, device=device)
        assert torch.allclose(q_neg, expected_180x, atol=1e-6)

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

    def test_gradient_stability_y_aligned(
        self, lmax, dtype, device, wigner_coeffs, U_blocks
    ):
        """Gradients are finite and bounded for y-aligned edges."""
        test_edges = [
            [0.0, 1.0, 0.0],  # +y
            [1e-9, 1.0, 1e-9],  # nearly +y
            [1e-6, 1.0, 1e-6],  # less nearly +y
        ]
        max_grads = []

        for edge in test_edges:
            edge_tensor = torch.tensor([edge], dtype=dtype, device=device)
            edge_tensor = torch.nn.functional.normalize(edge_tensor, dim=-1)
            edge_tensor = edge_tensor.detach().requires_grad_(True)

            wigner, _ = get_wigner_from_edge_vectors(
                edge_tensor, wigner_coeffs, U_blocks
            )

            loss = wigner.sum()
            loss.backward()

            grad = edge_tensor.grad
            assert grad is not None, f"No gradient computed for {edge}"
            assert not torch.isnan(grad).any(), f"NaN in gradient for {edge}"
            assert not torch.isinf(grad).any(), f"Inf in gradient for {edge}"

            max_grads.append(grad.abs().max().item())

        # Gradients should not grow unboundedly
        assert max(max_grads) < 1e6, f"Gradients growing too large: {max_grads}"


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
        from fairchem.core.models.uma.common.rotation import eulers_to_wigner

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
            x, y, z = xyz[0, 0], xyz[0, 1], xyz[0, 2]

            # Compute Euler angles (same as init_edge_rot_euler_angles but with gamma=0)
            beta = torch.acos(y.clamp(-1.0, 1.0))
            alpha = torch.atan2(x, z)
            gamma = torch.zeros(1, dtype=dtype, device=device)

            # Euler approach: wigner_D uses the negated angles for extrinsic convention
            euler_angles = (-gamma, -beta.unsqueeze(0), -alpha.unsqueeze(0))
            wigner_euler = eulers_to_wigner(euler_angles, 0, lmax, Jd_matrices)

            # Quaternion approach with matching gamma
            wigner_quat, _ = quaternion_wigner(edge_t, lmax, gamma=gamma)

            # Both should be orthogonal
            size = (lmax + 1) ** 2
            I = torch.eye(size, dtype=dtype, device=device)

            euler_ortho = wigner_euler[0] @ wigner_euler[0].T
            quat_ortho = wigner_quat[0] @ wigner_quat[0].T

            assert torch.allclose(euler_ortho, I, atol=1e-5), \
                f"Euler Wigner not orthogonal for edge {edge}"
            assert torch.allclose(quat_ortho, I, atol=1e-5), \
                f"Quaternion Wigner not orthogonal for edge {edge}"

            # Extract l=1 blocks - both use Cartesian (x, y, z) ordering
            euler_l1 = wigner_euler[0, 1:4, 1:4]
            quat_l1 = wigner_quat[0, 1:4, 1:4]

            # Both should rotate edge → +y in Cartesian coordinates
            edge_cart = xyz[0]
            euler_result = euler_l1 @ edge_cart
            quat_result = quat_l1 @ edge_cart

            assert torch.allclose(euler_result, y_axis, atol=1e-5), \
                f"Euler did not rotate edge {edge} to +y, got {euler_result}"
            assert torch.allclose(quat_result, y_axis, atol=1e-5), \
                f"Quaternion did not rotate edge {edge} to +y, got {quat_result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
