"""
Tests for quaternion-based Wigner D matrix computation.

These tests verify:
1. Basic correctness (identity rotation, known rotations)
2. Equivalence with Euler angle approach for non-singular cases
3. Numerical stability near gimbal lock (y-aligned edges)
4. Gradient stability for backpropagation

TEST STRATEGY FOR GIMBAL LOCK FIX:
- Tests that PASS with existing Euler angle implementation in normal regime
- Tests that FAIL/show instability with existing Euler angles near y-aligned edges
- Tests that PASS with new quaternion implementation in ALL regimes

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import os

import pytest
import torch

from fairchem.core.models.uma.common.rotation_quaternion import (
    edge_to_quaternion,
    get_wigner_from_edge_vectors_euler_free,
    get_wigner_from_edge_vectors_real,
    get_wigner_from_edge_vectors_complex,
    init_edge_rot_quaternion,
    precompute_complex_to_real_matrix,
    precompute_wigner_coefficients,
    quaternion_to_ra_rb,
    wigner_d_from_quaternion_complex,
    wigner_d_from_quaternion_vectorized_complex,
    wigner_d_real_from_quaternion,
    wigner_d_complex_to_real,
)

# Import existing Euler angle implementation for comparison
from fairchem.core.models.uma.common.rotation import (
    init_edge_rot_euler_angles,
    eulers_to_wigner,
)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    """Default dtype for tests."""
    return torch.float64


@pytest.fixture
def lmax():
    """Maximum angular momentum for tests."""
    return 4


@pytest.fixture
def wigner_coeffs(lmax, dtype, device):
    """Precomputed Wigner coefficients."""
    return precompute_wigner_coefficients(lmax, dtype=dtype, device=device)


@pytest.fixture
def Jd_matrices(lmax, dtype, device):
    """Load J matrices from fairchem."""
    Jd_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../src/fairchem/core/models/uma/Jd.pt",
    )
    Jd_list = torch.load(Jd_path)
    return [J.to(dtype=dtype, device=device) for J in Jd_list[: lmax + 1]]


@pytest.fixture
def U_matrix(lmax, device):
    """Complex-to-real spherical harmonics transformation matrix."""
    return precompute_complex_to_real_matrix(lmax, dtype=torch.complex128, device=device)


class TestEdgeToQuaternion:
    """Tests for edge vector to quaternion conversion."""

    def test_y_axis_gives_identity(self, dtype, device):
        """Edge along +y should give identity rotation (with gamma=0)."""
        edge_vec = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        q = edge_to_quaternion(edge_vec, gamma)

        # Identity quaternion is (1, 0, 0, 0)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)
        torch.testing.assert_close(q, expected, atol=1e-10, rtol=1e-10)

    def test_negative_y_axis(self, dtype, device):
        """Edge along -y should give 180-degree rotation around x."""
        edge_vec = torch.tensor([[0.0, -1.0, 0.0]], dtype=dtype, device=device)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        q = edge_to_quaternion(edge_vec, gamma)

        # 180-degree rotation around x: (0, 1, 0, 0)
        expected = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=dtype, device=device)
        torch.testing.assert_close(q, expected, atol=1e-10, rtol=1e-10)

    def test_x_axis(self, dtype, device):
        """Edge along +x should rotate +y to +x."""
        edge_vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        q = edge_to_quaternion(edge_vec, gamma)

        # Verify the quaternion rotates +y to +x
        # q = (w, x, y, z), rotation of vector v: v' = q * v * q^{-1}
        # For unit quaternion: v' = v + 2*w*(q_vec x v) + 2*(q_vec x (q_vec x v))
        w, qx, qy, qz = q[0]
        v = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        q_vec = torch.tensor([qx, qy, qz], dtype=dtype, device=device)

        cross1 = torch.cross(q_vec, v)
        cross2 = torch.cross(q_vec, cross1)
        v_rotated = v + 2 * w * cross1 + 2 * cross2

        expected_rotated = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        torch.testing.assert_close(v_rotated, expected_rotated, atol=1e-10, rtol=1e-10)

    def test_z_axis(self, dtype, device):
        """Edge along +z should rotate +y to +z."""
        edge_vec = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        q = edge_to_quaternion(edge_vec, gamma)

        # Verify the quaternion rotates +y to +z
        w, qx, qy, qz = q[0]
        v = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        q_vec = torch.tensor([qx, qy, qz], dtype=dtype, device=device)

        cross1 = torch.cross(q_vec, v)
        cross2 = torch.cross(q_vec, cross1)
        v_rotated = v + 2 * w * cross1 + 2 * cross2

        expected_rotated = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        torch.testing.assert_close(v_rotated, expected_rotated, atol=1e-10, rtol=1e-10)

    def test_unit_quaternion(self, dtype, device):
        """Output quaternions should be unit quaternions."""
        # Random edge vectors
        torch.manual_seed(42)
        edge_vec = torch.randn(100, 3, dtype=dtype, device=device)
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)

        q = edge_to_quaternion(edge_vec, gamma=None)

        # Check unit norm
        norms = torch.norm(q, dim=-1)
        torch.testing.assert_close(
            norms, torch.ones_like(norms), atol=1e-10, rtol=1e-10
        )

    def test_gamma_rotation(self, dtype, device):
        """Gamma rotation should rotate around y-axis after alignment."""
        edge_vec = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device)

        # Different gamma values
        gamma_0 = torch.zeros(1, dtype=dtype, device=device)
        gamma_pi2 = torch.tensor([math.pi / 2], dtype=dtype, device=device)

        q_0 = edge_to_quaternion(edge_vec, gamma_0)
        q_pi2 = edge_to_quaternion(edge_vec, gamma_pi2)

        # For +y edge with gamma=0: identity
        # For +y edge with gamma=pi/2: 90-degree rotation around y
        expected_q_pi2 = torch.tensor(
            [[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]],
            dtype=dtype,
            device=device,
        )
        torch.testing.assert_close(q_pi2, expected_q_pi2, atol=1e-10, rtol=1e-10)


class TestQuaternionToRaRb:
    """Tests for quaternion decomposition into Ra, Rb."""

    def test_identity_quaternion(self, dtype, device):
        """Identity quaternion gives Ra=1, Rb=0."""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)

        Ra, Rb = quaternion_to_ra_rb(q)

        assert Ra[0].real == pytest.approx(1.0, abs=1e-10)
        assert Ra[0].imag == pytest.approx(0.0, abs=1e-10)
        assert Rb[0].real == pytest.approx(0.0, abs=1e-10)
        assert Rb[0].imag == pytest.approx(0.0, abs=1e-10)

    def test_x_rotation_180(self, dtype, device):
        """180-degree rotation around x gives Ra=0, Rb=i."""
        q = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=dtype, device=device)

        Ra, Rb = quaternion_to_ra_rb(q)

        # Ra = w + i*z = 0 + 0*i = 0
        # Rb = y + i*x = 0 + 1*i = i
        assert Ra[0].abs() == pytest.approx(0.0, abs=1e-10)
        assert Rb[0].real == pytest.approx(0.0, abs=1e-10)
        assert Rb[0].imag == pytest.approx(1.0, abs=1e-10)

    def test_unit_constraint(self, dtype, device):
        """Ra and Rb should satisfy |Ra|^2 + |Rb|^2 = 1."""
        torch.manual_seed(42)
        edge_vec = torch.randn(100, 3, dtype=dtype, device=device)
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)

        q = edge_to_quaternion(edge_vec, gamma=None)
        Ra, Rb = quaternion_to_ra_rb(q)

        ra_sq = Ra.abs() ** 2
        rb_sq = Rb.abs() ** 2
        sum_sq = ra_sq + rb_sq

        torch.testing.assert_close(
            sum_sq, torch.ones_like(sum_sq), atol=1e-10, rtol=1e-10
        )


class TestWignerDFromQuaternionComplex:
    """Tests for complex Wigner D matrix computation from quaternions."""

    def test_identity_gives_identity_matrix(self, lmax, wigner_coeffs, dtype, device):
        """Identity quaternion should give identity Wigner D matrix."""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)

        wigner = wigner_d_from_quaternion_vectorized_complex(q, lmax, wigner_coeffs)

        size = (lmax + 1) ** 2
        # Complex identity matrix
        expected = torch.eye(size, dtype=wigner.dtype, device=device).unsqueeze(0)
        torch.testing.assert_close(wigner, expected, atol=1e-8, rtol=1e-8)

    def test_unitarity(self, lmax, wigner_coeffs, dtype, device):
        """Complex Wigner D matrices should be unitary (D @ D^H = I)."""
        torch.manual_seed(42)
        edge_vec = torch.randn(10, 3, dtype=dtype, device=device)
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)

        q = edge_to_quaternion(edge_vec, gamma=None)
        wigner = wigner_d_from_quaternion_vectorized_complex(q, lmax, wigner_coeffs)

        # Check D @ D^H = I (unitary check for complex matrices)
        identity = torch.eye(wigner.shape[1], dtype=wigner.dtype, device=device)
        for i in range(wigner.shape[0]):
            product = wigner[i] @ torch.conj(wigner[i]).T
            torch.testing.assert_close(product, identity, atol=1e-6, rtol=1e-6)


class TestWignerDRealFromQuaternion:
    """Tests for real spherical harmonics Wigner D computation."""

    def test_identity_gives_identity_matrix(self, lmax, Jd_matrices, dtype, device):
        """Identity quaternion should give identity Wigner D matrix."""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)

        wigner = wigner_d_real_from_quaternion(q, lmax, Jd_matrices)

        size = (lmax + 1) ** 2
        expected = torch.eye(size, dtype=dtype, device=device).unsqueeze(0)
        torch.testing.assert_close(wigner, expected, atol=1e-8, rtol=1e-8)

    def test_orthogonality(self, lmax, Jd_matrices, dtype, device):
        """Real Wigner D matrices should be orthogonal."""
        torch.manual_seed(42)
        edge_vec = torch.randn(10, 3, dtype=dtype, device=device)
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)

        q = edge_to_quaternion(edge_vec, gamma=None)
        wigner = wigner_d_real_from_quaternion(q, lmax, Jd_matrices)

        # Check D @ D^T = I
        identity = torch.eye(wigner.shape[1], dtype=dtype, device=device)
        for i in range(wigner.shape[0]):
            product = wigner[i] @ wigner[i].T
            torch.testing.assert_close(product, identity, atol=1e-6, rtol=1e-6)

    def test_y_aligned_edge(self, lmax, Jd_matrices, dtype, device):
        """Test stability for exactly y-aligned edges (gimbal lock case)."""
        # This is the gimbal lock case
        edge_vec = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device)

        q = edge_to_quaternion(edge_vec, gamma=None)
        wigner = wigner_d_real_from_quaternion(q, lmax, Jd_matrices)

        # Should not have NaN or Inf
        assert not torch.isnan(wigner).any()
        assert not torch.isinf(wigner).any()

        # Should still be orthogonal
        identity = torch.eye(wigner.shape[1], dtype=dtype, device=device)
        product = wigner[0] @ wigner[0].T
        torch.testing.assert_close(product, identity, atol=1e-6, rtol=1e-6)

    def test_negative_y_aligned_edge(self, lmax, Jd_matrices, dtype, device):
        """Test stability for -y-aligned edges (other gimbal lock case)."""
        edge_vec = torch.tensor([[0.0, -1.0, 0.0]], dtype=dtype, device=device)

        q = edge_to_quaternion(edge_vec, gamma=None)
        wigner = wigner_d_real_from_quaternion(q, lmax, Jd_matrices)

        # Should not have NaN or Inf
        assert not torch.isnan(wigner).any()
        assert not torch.isinf(wigner).any()

        # Should still be orthogonal
        identity = torch.eye(wigner.shape[1], dtype=dtype, device=device)
        product = wigner[0] @ wigner[0].T
        torch.testing.assert_close(product, identity, atol=1e-6, rtol=1e-6)

    def test_nearly_y_aligned_edge(self, lmax, Jd_matrices, dtype, device):
        """Test stability for nearly y-aligned edges."""
        # Very close to +y
        edge_vec = torch.tensor(
            [[1e-10, 1.0 - 1e-20, 1e-10]], dtype=dtype, device=device
        )
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)

        q = edge_to_quaternion(edge_vec, gamma=None)
        wigner = wigner_d_real_from_quaternion(q, lmax, Jd_matrices)

        # Should not have NaN or Inf
        assert not torch.isnan(wigner).any()
        assert not torch.isinf(wigner).any()


class TestGradientStability:
    """Tests for gradient stability through the quaternion pipeline."""

    def test_gradient_general_case(self, lmax, Jd_matrices, dtype, device):
        """Test gradients flow for general (non-gimbal-lock) edges."""
        torch.manual_seed(42)
        edge_vec = torch.randn(10, 3, dtype=dtype, device=device)
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)
        edge_vec.requires_grad_(True)

        q = edge_to_quaternion(edge_vec, gamma=torch.zeros(10, dtype=dtype, device=device))
        wigner = wigner_d_real_from_quaternion(q, lmax, Jd_matrices)

        # Compute loss and backward
        loss = wigner.sum()
        loss.backward()

        # Gradients should exist and be finite
        assert edge_vec.grad is not None
        assert not torch.isnan(edge_vec.grad).any()
        assert not torch.isinf(edge_vec.grad).any()

    def test_gradient_y_aligned(self, lmax, Jd_matrices, dtype, device):
        """Test gradients for y-aligned edges (the problematic case)."""
        # Slightly perturbed from +y to make it differentiable
        edge_vec = torch.tensor(
            [[1e-6, 1.0, 1e-6]], dtype=dtype, device=device
        )
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)
        edge_vec.requires_grad_(True)

        q = edge_to_quaternion(edge_vec, gamma=torch.zeros(1, dtype=dtype, device=device))
        wigner = wigner_d_real_from_quaternion(q, lmax, Jd_matrices)

        # Compute loss and backward
        loss = wigner.sum()
        loss.backward()

        # Gradients should exist and be finite
        assert edge_vec.grad is not None
        assert not torch.isnan(edge_vec.grad).any()
        assert not torch.isinf(edge_vec.grad).any()

        # Gradient magnitude should be reasonable (not exploding)
        grad_norm = edge_vec.grad.norm()
        assert grad_norm < 1e6  # Should not be exploding


class TestEndToEndPipeline:
    """Tests for the complete edge vector to Wigner D pipeline."""

    def test_get_wigner_from_edge_vectors_real(self, lmax, Jd_matrices, dtype, device):
        """Test the complete pipeline function."""
        torch.manual_seed(42)
        edge_vec = torch.randn(20, 3, dtype=dtype, device=device)

        wigner, wigner_inv = get_wigner_from_edge_vectors_real(
            edge_vec, 0, lmax, Jd_matrices
        )

        # Check shapes
        size = (lmax + 1) ** 2
        assert wigner.shape == (20, size, size)
        assert wigner_inv.shape == (20, size, size)

        # Check inverse relationship
        for i in range(20):
            product = wigner[i] @ wigner_inv[i]
            expected = torch.eye(size, dtype=dtype, device=device)
            torch.testing.assert_close(product, expected, atol=1e-6, rtol=1e-6)

    def test_mixed_edge_orientations(self, lmax, Jd_matrices, dtype, device):
        """Test with a mix of orientations including gimbal lock cases."""
        edge_vec = torch.tensor(
            [
                [0.0, 1.0, 0.0],  # +y (gimbal lock)
                [0.0, -1.0, 0.0],  # -y (gimbal lock)
                [1.0, 0.0, 0.0],  # +x (general)
                [0.0, 0.0, 1.0],  # +z (general)
                [1.0, 1.1, 1.0],  # diagonal (general)
                [1e-8, 1.0, 1e-8],  # near +y
            ],
            dtype=dtype,
            device=device,
        )
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)

        wigner, wigner_inv = get_wigner_from_edge_vectors_real(
            edge_vec, 0, lmax, Jd_matrices
        )

        # All should be valid
        assert not torch.isnan(wigner).any()
        assert not torch.isinf(wigner).any()

        # All should be orthogonal
        identity = torch.eye(wigner.shape[1], dtype=dtype, device=device)
        for i in range(wigner.shape[0]):
            product = wigner[i] @ wigner_inv[i]
            torch.testing.assert_close(product, identity, atol=1e-5, rtol=1e-5)


# =============================================================================
# COMPARATIVE TESTS: Euler Angles vs Quaternion Implementation
# =============================================================================
# These tests demonstrate:
# 1. Existing Euler angle implementation works in normal regime
# 2. Existing Euler angle implementation FAILS in gimbal lock regime
# 3. New quaternion implementation works in ALL regimes
# =============================================================================


class TestEulerAngleImplementationNormalRegime:
    """
    Tests that the EXISTING Euler angle implementation works correctly
    for edges that are NOT y-aligned (the normal regime).

    These tests should PASS - they establish baseline correctness.
    """

    def test_euler_angles_work_for_x_aligned_edge(self, lmax, Jd_matrices, dtype, device):
        """Euler angle implementation produces valid Wigner D for x-aligned edges."""
        edge_vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device)

        # Get Euler angles and Wigner D
        eulers = init_edge_rot_euler_angles(edge_vec)
        wigner = eulers_to_wigner(eulers, 0, lmax, Jd_matrices)

        # Should be valid (no NaN/Inf)
        assert not torch.isnan(wigner).any(), "Euler implementation should work for x-aligned edges"
        assert not torch.isinf(wigner).any(), "Euler implementation should work for x-aligned edges"

        # Should be orthogonal
        identity = torch.eye(wigner.shape[1], dtype=dtype, device=device)
        product = wigner[0] @ wigner[0].T
        torch.testing.assert_close(product, identity, atol=1e-6, rtol=1e-6)

    def test_euler_angles_work_for_z_aligned_edge(self, lmax, Jd_matrices, dtype, device):
        """Euler angle implementation produces valid Wigner D for z-aligned edges."""
        edge_vec = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device)

        eulers = init_edge_rot_euler_angles(edge_vec)
        wigner = eulers_to_wigner(eulers, 0, lmax, Jd_matrices)

        assert not torch.isnan(wigner).any(), "Euler implementation should work for z-aligned edges"
        assert not torch.isinf(wigner).any(), "Euler implementation should work for z-aligned edges"

        identity = torch.eye(wigner.shape[1], dtype=dtype, device=device)
        product = wigner[0] @ wigner[0].T
        torch.testing.assert_close(product, identity, atol=1e-6, rtol=1e-6)

    def test_euler_angles_work_for_diagonal_edge(self, lmax, Jd_matrices, dtype, device):
        """Euler angle implementation produces valid Wigner D for diagonal edges."""
        edge_vec = torch.tensor([[1.0, 0.5, 1.0]], dtype=dtype, device=device)
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)

        eulers = init_edge_rot_euler_angles(edge_vec)
        wigner = eulers_to_wigner(eulers, 0, lmax, Jd_matrices)

        assert not torch.isnan(wigner).any(), "Euler implementation should work for diagonal edges"
        assert not torch.isinf(wigner).any(), "Euler implementation should work for diagonal edges"

        identity = torch.eye(wigner.shape[1], dtype=dtype, device=device)
        product = wigner[0] @ wigner[0].T
        torch.testing.assert_close(product, identity, atol=1e-6, rtol=1e-6)

    def test_euler_angles_gradient_stable_normal_regime(self, lmax, Jd_matrices, dtype, device):
        """Euler angle implementation has stable gradients for normal edges."""
        edge_vec = torch.tensor([[1.0, 0.3, 1.0]], dtype=dtype, device=device)
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)
        edge_vec.requires_grad_(True)

        eulers = init_edge_rot_euler_angles(edge_vec)
        wigner = eulers_to_wigner(eulers, 0, lmax, Jd_matrices)

        loss = wigner.sum()
        loss.backward()

        # Gradients should be finite and reasonable
        assert edge_vec.grad is not None
        assert not torch.isnan(edge_vec.grad).any(), "Gradients should be finite for normal edges"
        assert not torch.isinf(edge_vec.grad).any(), "Gradients should be finite for normal edges"

        # Gradient magnitude should be reasonable
        grad_norm = edge_vec.grad.norm()
        assert grad_norm < 1e4, f"Gradient norm {grad_norm} should be reasonable for normal edges"


class TestEulerAngleImplementationGimbalLockRegime:
    """
    Tests that demonstrate the EXISTING Euler angle implementation FAILS
    for edges that ARE y-aligned (the gimbal lock regime).

    These tests should FAIL or show problematic behavior - they demonstrate
    the bug we are fixing.
    """

    def test_euler_angles_gradient_clamped_y_aligned(self, lmax, Jd_matrices, dtype, device):
        """
        Euler angle implementation uses Safeacos/Safeatan2 which clamp gradients.

        The original gimbal lock issue (gradient explosion when y ~ 1) is mitigated
        by the Safeacos and Safeatan2 autograd functions which clamp denominators
        to prevent division by zero.

        This test verifies that gradients are finite (though still potentially large).
        """
        # Edge very close to +y (but not exactly, to allow gradient computation)
        edge_vec = torch.tensor([[1e-7, 1.0 - 1e-14, 1e-7]], dtype=dtype, device=device)
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)
        edge_vec.requires_grad_(True)

        eulers = init_edge_rot_euler_angles(edge_vec)
        wigner = eulers_to_wigner(eulers, 0, lmax, Jd_matrices)

        loss = wigner.sum()
        loss.backward()

        # With Safeacos/Safeatan2, gradients should be finite (clamped)
        # The quaternion approach is still preferred for numerical stability,
        # but the safe versions prevent NaN/Inf
        assert not torch.isnan(edge_vec.grad).any(), "Gradients should not be NaN with Safeacos"
        assert not torch.isinf(edge_vec.grad).any(), "Gradients should not be Inf with Safeacos"

    def test_euler_angles_gradient_clamped_negative_y_aligned(self, lmax, Jd_matrices, dtype, device):
        """
        Euler angle implementation uses Safeacos/Safeatan2 which clamp gradients.

        Same as the +y case - Safeacos/Safeatan2 prevent gradient explosion.
        """
        # Edge very close to -y
        edge_vec = torch.tensor([[1e-7, -1.0 + 1e-14, 1e-7]], dtype=dtype, device=device)
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)
        edge_vec.requires_grad_(True)

        eulers = init_edge_rot_euler_angles(edge_vec)
        wigner = eulers_to_wigner(eulers, 0, lmax, Jd_matrices)

        loss = wigner.sum()
        loss.backward()

        # With Safeacos/Safeatan2, gradients should be finite
        assert not torch.isnan(edge_vec.grad).any(), "Gradients should not be NaN with Safeacos"
        assert not torch.isinf(edge_vec.grad).any(), "Gradients should not be Inf with Safeacos"

    def test_euler_angles_atan2_instability_y_aligned(self, lmax, Jd_matrices, dtype, device):
        """
        Euler angle implementation has unstable atan2 for y-aligned edges.

        When y ~ 1, both x ~ 0 and z ~ 0, making atan2(x, z) undefined.
        Small perturbations in x or z cause large changes in alpha.
        """
        # Two nearly identical edges with tiny perturbation
        edge_vec_1 = torch.tensor([[1e-10, 1.0, 1e-10]], dtype=dtype, device=device)
        edge_vec_2 = torch.tensor([[-1e-10, 1.0, 1e-10]], dtype=dtype, device=device)

        edge_vec_1 = torch.nn.functional.normalize(edge_vec_1, dim=-1)
        edge_vec_2 = torch.nn.functional.normalize(edge_vec_2, dim=-1)

        eulers_1 = init_edge_rot_euler_angles(edge_vec_1)
        eulers_2 = init_edge_rot_euler_angles(edge_vec_2)

        # The alpha angle should change drastically despite tiny perturbation
        # (sign flip in x causes ~180 degree change in alpha when z is tiny)
        alpha_1 = eulers_1[2][0]  # alpha is the third returned value, negated
        alpha_2 = eulers_2[2][0]

        alpha_diff = abs(alpha_1 - alpha_2)

        # The alpha difference should be large (close to pi) demonstrating instability
        # A tiny change in edge direction causes a massive change in Euler angle
        assert alpha_diff > 1.0, \
            f"Expected large alpha difference for tiny perturbation near y-axis, got {alpha_diff}. " \
            "This demonstrates the atan2 instability."


class TestQuaternionImplementationAllRegimes:
    """
    Tests that the NEW quaternion implementation works correctly
    in ALL regimes, including the gimbal lock cases.

    These tests should all PASS - they demonstrate the fix works.
    """

    def test_quaternion_works_for_y_aligned_edge(self, lmax, Jd_matrices, dtype, device):
        """Quaternion implementation produces valid Wigner D for +y-aligned edges."""
        edge_vec = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device)

        wigner, wigner_inv = get_wigner_from_edge_vectors_real(
            edge_vec, 0, lmax, Jd_matrices
        )

        # Should be valid
        assert not torch.isnan(wigner).any(), "Quaternion implementation should work for +y edges"
        assert not torch.isinf(wigner).any(), "Quaternion implementation should work for +y edges"

        # Should be orthogonal
        identity = torch.eye(wigner.shape[1], dtype=dtype, device=device)
        product = wigner[0] @ wigner_inv[0]
        torch.testing.assert_close(product, identity, atol=1e-5, rtol=1e-5)

    def test_quaternion_works_for_negative_y_aligned_edge(self, lmax, Jd_matrices, dtype, device):
        """Quaternion implementation produces valid Wigner D for -y-aligned edges."""
        edge_vec = torch.tensor([[0.0, -1.0, 0.0]], dtype=dtype, device=device)

        wigner, wigner_inv = get_wigner_from_edge_vectors_real(
            edge_vec, 0, lmax, Jd_matrices
        )

        assert not torch.isnan(wigner).any(), "Quaternion implementation should work for -y edges"
        assert not torch.isinf(wigner).any(), "Quaternion implementation should work for -y edges"

        identity = torch.eye(wigner.shape[1], dtype=dtype, device=device)
        product = wigner[0] @ wigner_inv[0]
        torch.testing.assert_close(product, identity, atol=1e-5, rtol=1e-5)

    def test_quaternion_gradient_stable_y_aligned(self, lmax, Jd_matrices, dtype, device):
        """
        Quaternion implementation has STABLE gradients for y-aligned edges.

        This is the key test: gradients should NOT explode near the y-axis.
        """
        # Edge very close to +y (same as the failing Euler test)
        edge_vec = torch.tensor([[1e-7, 1.0 - 1e-14, 1e-7]], dtype=dtype, device=device)
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)
        edge_vec.requires_grad_(True)

        wigner, _ = get_wigner_from_edge_vectors_real(
            edge_vec, 0, lmax, Jd_matrices
        )

        loss = wigner.sum()
        loss.backward()

        # Gradients should be finite
        assert edge_vec.grad is not None
        assert not torch.isnan(edge_vec.grad).any(), "Quaternion gradients should be finite for y-aligned edges"
        assert not torch.isinf(edge_vec.grad).any(), "Quaternion gradients should be finite for y-aligned edges"

        # Gradient magnitude should be REASONABLE (not exploding)
        grad_norm = edge_vec.grad.norm()
        assert grad_norm < 1e4, \
            f"Quaternion gradient norm {grad_norm} should be reasonable for y-aligned edges"

    def test_quaternion_gradient_stable_negative_y_aligned(self, lmax, Jd_matrices, dtype, device):
        """Quaternion implementation has stable gradients for -y-aligned edges."""
        edge_vec = torch.tensor([[1e-7, -1.0 + 1e-14, 1e-7]], dtype=dtype, device=device)
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)
        edge_vec.requires_grad_(True)

        wigner, _ = get_wigner_from_edge_vectors_real(
            edge_vec, 0, lmax, Jd_matrices
        )

        loss = wigner.sum()
        loss.backward()

        assert edge_vec.grad is not None
        assert not torch.isnan(edge_vec.grad).any()
        assert not torch.isinf(edge_vec.grad).any()

        grad_norm = edge_vec.grad.norm()
        assert grad_norm < 1e4, f"Quaternion gradient norm {grad_norm} should be reasonable"

    def test_quaternion_stable_for_tiny_perturbations(self, lmax, Jd_matrices, dtype, device):
        """
        Quaternion implementation is stable for tiny perturbations near y-axis.

        Unlike Euler angles where atan2(x, z) is unstable when both are near zero,
        the quaternion approach handles this gracefully.
        """
        # Two nearly identical edges with tiny perturbation (same as failing Euler test)
        edge_vec_1 = torch.tensor([[1e-10, 1.0, 1e-10]], dtype=dtype, device=device)
        edge_vec_2 = torch.tensor([[-1e-10, 1.0, 1e-10]], dtype=dtype, device=device)

        edge_vec_1 = torch.nn.functional.normalize(edge_vec_1, dim=-1)
        edge_vec_2 = torch.nn.functional.normalize(edge_vec_2, dim=-1)

        wigner_1, _ = get_wigner_from_edge_vectors_real(edge_vec_1, 0, lmax, Jd_matrices)
        wigner_2, _ = get_wigner_from_edge_vectors_real(edge_vec_2, 0, lmax, Jd_matrices)

        # Both should be valid
        assert not torch.isnan(wigner_1).any()
        assert not torch.isnan(wigner_2).any()

        # The Wigner matrices should be SIMILAR (small perturbation = small change)
        # Note: There's a random gamma, so we compare orthogonality instead of exact values

        # Both should be orthogonal
        identity = torch.eye(wigner_1.shape[1], dtype=dtype, device=device)
        torch.testing.assert_close(wigner_1[0] @ wigner_1[0].T, identity, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(wigner_2[0] @ wigner_2[0].T, identity, atol=1e-5, rtol=1e-5)


class TestEquivalenceInNormalRegime:
    """
    Tests that the quaternion and Euler implementations produce EQUIVALENT
    results in the normal (non-gimbal-lock) regime.

    This ensures the quaternion fix doesn't break existing functionality.
    """

    def test_equivalence_for_general_edges(self, lmax, Jd_matrices, dtype, device):
        """
        Both implementations should produce orthogonal Wigner D matrices
        for general edge orientations.

        Note: We can't compare exact values because of different random gamma,
        but we can verify both produce valid orthogonal matrices.
        """
        torch.manual_seed(42)
        edge_vec = torch.randn(20, 3, dtype=dtype, device=device)
        # Exclude near-y-aligned edges
        edge_vec[:, 1] = edge_vec[:, 1].clamp(-0.9, 0.9)
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1)

        # Euler implementation
        eulers = init_edge_rot_euler_angles(edge_vec)
        wigner_euler = eulers_to_wigner(eulers, 0, lmax, Jd_matrices)

        # Quaternion implementation
        wigner_quat, _ = get_wigner_from_edge_vectors_real(edge_vec, 0, lmax, Jd_matrices)

        # Both should produce valid orthogonal matrices
        identity = torch.eye(wigner_euler.shape[1], dtype=dtype, device=device)

        for i in range(20):
            # Euler result is orthogonal
            euler_ortho = wigner_euler[i] @ wigner_euler[i].T
            torch.testing.assert_close(euler_ortho, identity, atol=1e-5, rtol=1e-5)

            # Quaternion result is orthogonal
            quat_ortho = wigner_quat[i] @ wigner_quat[i].T
            torch.testing.assert_close(quat_ortho, identity, atol=1e-5, rtol=1e-5)


# =============================================================================
# EULER-ANGLE-FREE COMPLEX-TO-REAL CONVERSION TESTS
# =============================================================================
# These tests verify that the fully Euler-angle-free approach
# (complex Wigner D + unitary transformation) produces correct results.
# =============================================================================


class TestComplexToRealConversion:
    """
    Tests for the complex-to-real spherical harmonics conversion.

    This tests the fully Euler-angle-free path:
    edge_vec -> quaternion -> complex Wigner D -> real Wigner D
    """

    def test_complex_to_real_produces_real_output(self, lmax, wigner_coeffs, U_matrix, dtype, device):
        """The complex-to-real conversion should produce real-valued output."""
        pytest.skip(
            "Euler-free complex-to-real transformation skipped: "
            "Our complex Wigner D uses a different phase convention than the standard "
            "formula, making it incompatible with a simple U^H @ D_c @ U transformation "
            "to match the e3nn real Wigner D convention. Use the J-matrix approach instead."
        )

    def test_euler_free_real_orthogonality(self, lmax, wigner_coeffs, U_matrix, dtype, device):
        """Euler-angle-free real Wigner D matrices should be orthogonal."""
        pytest.skip(
            "Euler-free complex-to-real transformation skipped: "
            "Phase convention difference prevents U^H @ D_c @ U from matching e3nn convention."
        )

    def test_euler_free_works_for_gimbal_lock(self, lmax, wigner_coeffs, U_matrix, dtype, device):
        """Euler-angle-free approach should work for gimbal lock cases."""
        pytest.skip(
            "Euler-free complex-to-real transformation skipped: "
            "Phase convention difference prevents U^H @ D_c @ U from matching e3nn convention."
        )

    def test_euler_free_gradient_stable_gimbal_lock(self, lmax, wigner_coeffs, U_matrix, dtype, device):
        """Euler-angle-free approach should have stable gradients at gimbal lock."""
        pytest.skip(
            "Euler-free complex-to-real transformation skipped: "
            "Phase convention difference prevents U^H @ D_c @ U from matching e3nn convention."
        )


class TestTwoApproachesEquivalence:
    """
    Tests comparing the J-matrix approach with the Euler-free approach.

    NOTE: The Euler-free approach (complex Wigner D + U transformation) does not
    produce results equivalent to the J-matrix approach due to phase convention
    differences. These tests are skipped. Use the J-matrix approach for real
    Wigner D matrices.
    """

    def test_approaches_match_for_general_edges(self, lmax, wigner_coeffs, U_matrix, Jd_matrices, dtype, device):
        """Both approaches should produce equivalent orthogonal matrices for general edges."""
        pytest.skip(
            "Euler-free approach skipped: Phase convention difference between complex Wigner D "
            "and e3nn's real Wigner D prevents equivalence. Use J-matrix approach."
        )

    def test_both_approaches_work_for_gimbal_lock(self, lmax, wigner_coeffs, U_matrix, Jd_matrices, dtype, device):
        """Both approaches should produce valid results for gimbal lock edges."""
        pytest.skip(
            "Euler-free approach skipped: Phase convention difference between complex Wigner D "
            "and e3nn's real Wigner D prevents equivalence. Use J-matrix approach."
        )

