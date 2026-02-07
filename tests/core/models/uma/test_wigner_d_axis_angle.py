"""
Tests for axis-angle based Wigner D matrix computation.

These tests verify:
1. Mathematical correctness (orthogonality, determinant, edge → +Y)
2. Agreement with quaternion-based code
3. Agreement with Euler-based code
4. Gradient stability

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from fairchem.core.models.uma.common.rotation import (
    eulers_to_wigner,
    init_edge_rot_euler_angles,
)
from fairchem.core.models.uma.common.wigner_d_axis_angle import (
    axis_angle_wigner,
    axis_angle_wigner_random_gamma,
    get_so3_generators,
    quaternion_to_axis_angle,
    wigner_d_from_axis_angle_batched,
)
from fairchem.core.models.uma.common.wigner_d_quaternion import quaternion_wigner


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
def generators(lmax, dtype, device):
    return get_so3_generators(lmax, dtype, device)


# =============================================================================
# Test Axis-Angle Conversion
# =============================================================================


class TestAxisAngleConversion:
    """Tests for quaternion_to_axis_angle function."""

    def test_identity_gives_zero_angle(self, dtype, device):
        """Identity quaternion gives zero rotation angle."""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)
        axis, angle = quaternion_to_axis_angle(q)

        assert torch.allclose(angle, torch.zeros_like(angle), atol=1e-10)

    def test_180_rotation(self, dtype, device):
        """180° rotation about X-axis."""
        q = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=dtype, device=device)
        axis, angle = quaternion_to_axis_angle(q)

        expected_axis = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device)
        expected_angle = torch.tensor([math.pi], dtype=dtype, device=device)

        assert torch.allclose(axis, expected_axis, atol=1e-6)
        assert torch.allclose(angle, expected_angle, atol=1e-6)


# =============================================================================
# Test Wigner D Properties
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
            ([0.0, 0.0, -1.0], "-Z-axis"),
            ([1.0, 1.0, 1.0], "diagonal"),
            ([0.6, 0.5, 0.8], "general"),
            ([1e-6, 1.0, 1e-6], "near +Y"),
            ([1e-6, -1.0, 1e-6], "near -Y"),
        ],
    )
    def test_orthogonality(self, lmax, dtype, device, edge, desc):
        """Wigner D matrices are orthogonal: D @ D.T = I."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, D_inv = axis_angle_wigner(edge_t, lmax, gamma=gamma)

        size = (lmax + 1) ** 2
        I = torch.eye(size, dtype=dtype, device=device)

        product = D[0] @ D[0].T
        assert torch.allclose(product, I, atol=1e-5), f"Not orthogonal for {desc}"

    @pytest.mark.parametrize(
        "edge,desc",
        [
            ([1.0, 0.0, 0.0], "X-axis"),
            ([0.0, 1.0, 0.0], "+Y-axis"),
            ([0.0, 0.0, 1.0], "Z-axis"),
            ([1.0, 1.0, 1.0], "diagonal"),
        ],
    )
    def test_determinant_is_one(self, lmax, dtype, device, edge, desc):
        """Wigner D matrices have determinant 1."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = axis_angle_wigner(edge_t, lmax, gamma=gamma)

        det = torch.linalg.det(D[0])
        assert torch.allclose(
            det, torch.tensor(1.0, dtype=dtype, device=device), atol=1e-5
        ), f"det != 1 for {desc}: {det}"

    @pytest.mark.parametrize(
        "edge,desc",
        [
            ([1.0, 0.0, 0.0], "X-axis"),
            ([0.0, 0.0, 1.0], "Z-axis"),
            ([0.0, 0.0, -1.0], "-Z-axis"),
            ([1.0, 1.0, 1.0], "diagonal"),
            ([0.6, 0.5, 0.8], "general"),
            ([-0.3, 0.7, 0.5], "random"),
        ],
    )
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

    def test_inverse_is_transpose(self, lmax, dtype, device):
        """D_inv equals D.T."""
        torch.manual_seed(42)
        edges = torch.randn(10, 3, dtype=dtype, device=device)

        D, D_inv = axis_angle_wigner_random_gamma(edges, lmax)

        for i in range(10):
            assert torch.allclose(D_inv[i], D[i].T, atol=1e-10)


# =============================================================================
# Test Agreement with Quaternion Code
# =============================================================================


class TestQuaternionAgreement:
    """Tests for agreement between axis-angle and quaternion approaches."""

    def test_both_map_edge_to_y(self, lmax, dtype, device):
        """
        Both axis-angle and quaternion correctly map edge → +Y.

        Note: The axis-angle and quaternion approaches use fundamentally different
        rotations (Rodrigues vs ZYZ Euler). Both correctly map edge → +Y, but they
        orient the frame differently. The matrices differ, but this is expected
        and acceptable since:
        1. Both produce valid SO(3) representations (orthogonal, det=1)
        2. Both correctly rotate edge → +Y (verified here)
        3. Training uses random gamma for SO(2) equivariance
        """
        test_edges = [
            [0.0, 0.0, 1.0],  # Z-aligned
            [1.0, 0.0, 0.0],  # X-aligned
            [1.0, 1.0, 1.0],  # Diagonal
            [0.3, 0.5, 0.8],  # Random 1
            [-0.2, 0.3, 0.9],  # Random 2
            [0.8, 0.4, -0.3],  # Random 3
        ]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

        for edge in test_edges:
            edge_t = torch.tensor([edge], dtype=dtype, device=device)
            edge_norm = torch.nn.functional.normalize(edge_t, dim=-1)[0]
            gamma = torch.tensor([0.5], dtype=dtype, device=device)

            wigner_quat, _ = quaternion_wigner(edge_t, lmax, gamma=gamma)
            wigner_axis, _ = axis_angle_wigner(edge_t, lmax, gamma=gamma)

            # Both l=1 blocks should map edge → +Y
            quat_l1 = wigner_quat[0, 1:4, 1:4]
            axis_l1 = wigner_axis[0, 1:4, 1:4]

            quat_result = quat_l1 @ edge_norm
            axis_result = axis_l1 @ edge_norm

            assert torch.allclose(quat_result, y_axis, atol=1e-5), (
                f"Quaternion failed for {edge}: {quat_result}"
            )
            assert torch.allclose(axis_result, y_axis, atol=1e-5), (
                f"Axis-angle failed for {edge}: {axis_result}"
            )


# =============================================================================
# Test Agreement with Euler Code
# =============================================================================


class TestEulerAgreement:
    """Tests for agreement between axis-angle and Euler approaches."""

    @pytest.fixture()
    def Jd_matrices(self, lmax, dtype, device):
        """Load the J matrices used by the Euler angle approach."""
        test_dir = Path(__file__).parent
        repo_root = test_dir.parent.parent.parent.parent
        jd_path = repo_root / "src" / "fairchem" / "core" / "models" / "uma" / "Jd.pt"

        if jd_path.exists():
            Jd = torch.load(jd_path, map_location=device, weights_only=True)
            return [J.to(dtype=dtype) for J in Jd[: lmax + 1]]
        else:
            pytest.skip(f"Jd.pt not found at {jd_path}")

    def test_both_achieve_edge_to_y(self, lmax, dtype, device, Jd_matrices):
        """
        Both axis-angle and Euler correctly achieve edge → +Y mapping.

        The Euler code uses ZYZ decomposition with random gamma for the final roll.
        The axis-angle code uses Rodrigues rotation with gamma roll correction.

        These produce different rotations (both valid, just different frame
        orientations around Y-axis), but both achieve the critical requirement:
        mapping edge → +Y.
        """
        test_edges = [
            [0.0, 0.0, 1.0],  # Z-aligned
            [1.0, 0.0, 0.0],  # X-aligned
            [1.0, 1.0, 1.0],  # Diagonal
            [0.3, 0.5, 0.8],  # Random 1
            [-0.2, 0.3, 0.9],  # Random 2
            [0.8, 0.4, -0.3],  # Random 3
        ]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

        for edge in test_edges:
            edge_t = torch.tensor([edge], dtype=dtype, device=device)

            # Compute Euler angles
            euler_angles = init_edge_rot_euler_angles(edge_t)
            gamma_euler = euler_angles[0]

            # Euler approach
            wigner_euler = eulers_to_wigner(euler_angles, 0, lmax, Jd_matrices)

            # Axis-angle approach
            wigner_axis, _ = axis_angle_wigner(edge_t, lmax, gamma=gamma_euler)

            # Both should rotate edge → +Y
            edge_norm = torch.nn.functional.normalize(edge_t, dim=-1)[0]

            euler_l1 = wigner_euler[0, 1:4, 1:4]
            axis_l1 = wigner_axis[0, 1:4, 1:4]

            euler_result = euler_l1 @ edge_norm
            axis_result = axis_l1 @ edge_norm

            assert torch.allclose(euler_result, y_axis, atol=1e-5), (
                f"Euler failed for {edge}: {euler_result}"
            )
            assert torch.allclose(axis_result, y_axis, atol=1e-5), (
                f"Axis-angle failed for {edge}: {axis_result}"
            )

    def test_match_euler_produces_identical_output(self, lmax, dtype, device, Jd_matrices):
        """
        Axis-angle output exactly matches Euler output.

        This tests that the basis transformation automatically applied for l >= 2
        makes axis-angle a drop-in replacement for the Euler-based implementation.
        """
        from fairchem.core.models.uma.common.rotation import wigner_D

        test_edges = [
            [0.0, 0.0, 1.0],  # Z-aligned
            [1.0, 0.0, 0.0],  # X-aligned
            [1.0, 1.0, 1.0],  # Diagonal
            [0.3, 0.5, 0.8],  # Random 1
            [-0.2, 0.3, 0.9],  # Random 2
        ]

        for edge in test_edges:
            edge_t = torch.tensor([edge], dtype=dtype, device=device)
            edge_norm = torch.nn.functional.normalize(edge_t, dim=-1)

            # Compute Euler angles (matching the Euler code convention)
            alpha = torch.atan2(edge_norm[0, 0], edge_norm[0, 2])
            beta = torch.acos(edge_norm[0, 1].clamp(-1, 1))
            gamma_val = torch.zeros(1, dtype=dtype, device=device)

            # Gamma for axis-angle that matches Euler with alpha_arg=0
            gamma_rod = -torch.atan2(edge_norm[0, 0], edge_norm[0, 2])

            # Compute with axis-angle (now always matches Euler)
            D_axis, _ = axis_angle_wigner(
                edge_norm, lmax, gamma=gamma_rod.unsqueeze(0)
            )

            # Compare each l block with Euler
            for ell in range(lmax + 1):
                start = ell * ell
                end = start + 2 * ell + 1

                D_euler = wigner_D(ell, -gamma_val, -beta, -alpha, Jd_matrices)
                D_axis_block = D_axis[0, start:end, start:end]

                assert torch.allclose(D_euler, D_axis_block, atol=1e-10), (
                    f"l={ell} mismatch for edge {edge}:\n"
                    f"Max error: {(D_euler - D_axis_block).abs().max()}"
                )


# =============================================================================
# Test Gradient Stability
# =============================================================================


class TestGradientStability:
    """Tests for gradient stability."""

    @pytest.mark.parametrize(
        "edge,desc",
        [
            ([1.0, 0.0, 0.0], "X-axis"),
            ([0.6, 0.5, 0.8], "general"),
            ([0.01, 1.0, 0.01], "near +Y"),
        ],
    )
    def test_gradient_flow(self, lmax, dtype, device, edge, desc):
        """Gradients flow without NaN or Inf."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device, requires_grad=True)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = axis_angle_wigner(edge_t, lmax, gamma=gamma)
        loss = D.sum()
        loss.backward()

        grad = edge_t.grad
        assert not torch.isnan(grad).any(), f"NaN gradient for {desc}"
        assert not torch.isinf(grad).any(), f"Inf gradient for {desc}"


# =============================================================================
# Test Random Gamma
# =============================================================================


class TestRandomGamma:
    """Tests for random gamma functionality."""

    def test_random_gamma_produces_valid_matrices(self, lmax, dtype, device):
        """Random gamma still produces valid Wigner D matrices."""
        torch.manual_seed(42)
        edges = torch.randn(10, 3, dtype=dtype, device=device)

        D, D_inv = axis_angle_wigner_random_gamma(edges, lmax)

        size = (lmax + 1) ** 2
        I = torch.eye(size, dtype=dtype, device=device)

        for i in range(10):
            # Check orthogonality
            product = D[i] @ D[i].T
            assert torch.allclose(product, I, atol=1e-5)

            # Check determinant
            det = torch.linalg.det(D[i])
            assert torch.allclose(
                det, torch.tensor(1.0, dtype=dtype, device=device), atol=1e-5
            )

    def test_random_gamma_maps_edge_to_y(self, lmax, dtype, device):
        """Even with random gamma, edge still maps to +Y."""
        torch.manual_seed(42)
        edges = torch.randn(10, 3, dtype=dtype, device=device)
        edges = torch.nn.functional.normalize(edges, dim=-1)

        D, _ = axis_angle_wigner_random_gamma(edges, lmax)

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

        for i in range(10):
            D_l1 = D[i, 1:4, 1:4]
            result = D_l1 @ edges[i]
            assert torch.allclose(result, y_axis, atol=1e-5)


# =============================================================================
# Test SO(3) Generators
# =============================================================================


class TestSO3Generators:
    """Tests for SO(3) generator construction."""

    def test_generators_antisymmetric(self, lmax, dtype, device, generators):
        """K matrices are antisymmetric: K.T = -K."""
        for ell in range(lmax + 1):
            K_x = generators['K_x'][ell]
            K_y = generators['K_y'][ell]
            K_z = generators['K_z'][ell]

            assert torch.allclose(K_x.T, -K_x, atol=1e-10), f"K_x not antisym for l={ell}"
            assert torch.allclose(K_y.T, -K_y, atol=1e-10), f"K_y not antisym for l={ell}"
            assert torch.allclose(K_z.T, -K_z, atol=1e-10), f"K_z not antisym for l={ell}"


# =============================================================================
# Test Pure Axis-Angle Computation
# =============================================================================


class TestPureAxisAngle:
    """Tests for wigner_d_from_axis_angle_batched."""

    def test_z_rotation_90(self, dtype, device):
        """90° Z-rotation produces correct Wigner D."""
        lmax = 2
        generators = get_so3_generators(lmax, dtype, device)

        axis = torch.tensor([[0, 0, 1]], dtype=dtype, device=device)
        angle = torch.tensor([math.pi / 2], dtype=dtype, device=device)

        D = wigner_d_from_axis_angle_batched(axis, angle, generators, lmax)

        # l=0: should be identity (1x1)
        assert torch.allclose(D[0, 0:1, 0:1], torch.ones(1, 1, dtype=dtype), atol=1e-5)

        # l=1: check orthogonality
        D_l1 = D[0, 1:4, 1:4]
        I = torch.eye(3, dtype=dtype)
        assert torch.allclose(D_l1 @ D_l1.T, I, atol=1e-5)

    def test_identity_rotation(self, dtype, device):
        """Zero angle gives identity Wigner D."""
        lmax = 2
        generators = get_so3_generators(lmax, dtype, device)

        axis = torch.tensor([[0, 0, 1]], dtype=dtype, device=device)
        angle = torch.tensor([0.0], dtype=dtype, device=device)

        D = wigner_d_from_axis_angle_batched(axis, angle, generators, lmax)

        size = (lmax + 1) ** 2
        I = torch.eye(size, dtype=dtype)
        assert torch.allclose(D[0], I, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
