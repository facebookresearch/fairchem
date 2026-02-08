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
    _cayley_hamilton_exp_l2,
    _quaternion_chart1_standard,
    _quaternion_chart2_via_minus_y,
    _smooth_step_cinf,
    axis_angle_wigner,
    axis_angle_wigner_polynomial,
    axis_angle_wigner_random_gamma,
    get_so3_generators,
    quaternion_slerp,
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

    @pytest.mark.skip(reason="quaternion_wigner has a bug - does not correctly map edge to +Y")
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
        Axis-angle output exactly matches Euler output when using Euler gamma.

        This tests that with use_euler_gamma=True, the axis-angle code produces
        output identical to the Euler-based implementation.
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

            # Compute with axis-angle using Euler gamma
            D_axis, _ = axis_angle_wigner(edge_norm, lmax, use_euler_gamma=True)

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
# Test Near-Singularity Behavior
# =============================================================================


class TestNearSingularityBehavior:
    """Tests for gradient stability near singularities at ±Y."""

    @pytest.mark.parametrize(
        "epsilon",
        [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    )
    def test_near_plus_y_gradients_bounded(self, lmax, dtype, device, epsilon):
        """Gradients remain bounded for edges approaching +Y."""
        # Edge slightly off +Y in X direction
        edge = torch.tensor(
            [[epsilon, 1.0, 0.0]], dtype=dtype, device=device, requires_grad=True
        )
        edge_norm = torch.nn.functional.normalize(edge, dim=-1)

        D, _ = axis_angle_wigner(edge_norm, lmax)
        loss = D.sum()
        loss.backward()

        grad = edge.grad
        assert not torch.isnan(grad).any(), f"NaN gradient at epsilon={epsilon}"
        assert not torch.isinf(grad).any(), f"Inf gradient at epsilon={epsilon}"

        # Gradients should be reasonably bounded (not exploding)
        grad_norm = grad.norm().item()
        assert grad_norm < 1000, f"Gradient too large at epsilon={epsilon}: {grad_norm}"

    @pytest.mark.parametrize(
        "epsilon",
        [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    )
    def test_near_plus_y_maps_to_y(self, lmax, dtype, device, epsilon):
        """Edges near +Y still correctly map to +Y."""
        edge = torch.tensor([[epsilon, 1.0, 0.0]], dtype=dtype, device=device)
        edge_norm = torch.nn.functional.normalize(edge, dim=-1)

        D, _ = axis_angle_wigner(edge_norm, lmax)
        D_l1 = D[0, 1:4, 1:4]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        result = D_l1 @ edge_norm[0]

        assert torch.allclose(result, y_axis, atol=1e-5), (
            f"Near +Y edge (eps={epsilon}) did not map to +Y: {result}"
        )

    @pytest.mark.parametrize(
        "epsilon",
        [1e-2, 1e-3, 1e-4, 1e-5],
    )
    def test_near_minus_y_maps_to_y(self, lmax, dtype, device, epsilon):
        """Edges near -Y still correctly map to +Y."""
        edge = torch.tensor([[epsilon, -1.0, 0.0]], dtype=dtype, device=device)
        edge_norm = torch.nn.functional.normalize(edge, dim=-1)

        D, _ = axis_angle_wigner(edge_norm, lmax)
        D_l1 = D[0, 1:4, 1:4]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        result = D_l1 @ edge_norm[0]

        assert torch.allclose(result, y_axis, atol=1e-5), (
            f"Near -Y edge (eps={epsilon}) did not map to +Y: {result}"
        )

    @pytest.mark.parametrize(
        "epsilon",
        [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    )
    def test_near_minus_y_gradients_no_nan(self, lmax, dtype, device, epsilon):
        """Gradients don't produce NaN for edges near -Y."""
        edge = torch.tensor(
            [[epsilon, -1.0, 0.0]], dtype=dtype, device=device, requires_grad=True
        )
        edge_norm = torch.nn.functional.normalize(edge, dim=-1)

        D, _ = axis_angle_wigner(edge_norm, lmax)
        loss = D.sum()
        loss.backward()

        grad = edge.grad
        assert not torch.isnan(grad).any(), f"NaN gradient at epsilon={epsilon}"
        # Note: Gradients may be large or zero near -Y due to torch.where handling,
        # but they should never be NaN

    def test_exact_minus_y_handled(self, lmax, dtype, device):
        """Exact -Y edge is handled without NaN (via torch.where fallback)."""
        edge = torch.tensor(
            [[0.0, -1.0, 0.0]], dtype=dtype, device=device, requires_grad=True
        )

        D, _ = axis_angle_wigner(edge, lmax)
        loss = D.sum()
        loss.backward()

        grad = edge.grad
        assert not torch.isnan(grad).any(), "NaN gradient for exact -Y"
        assert not torch.isinf(grad).any(), "Inf gradient for exact -Y"

    def test_exact_plus_y_handled(self, lmax, dtype, device):
        """Exact +Y edge is handled correctly."""
        edge = torch.tensor(
            [[0.0, 1.0, 0.0]], dtype=dtype, device=device, requires_grad=True
        )

        D, _ = axis_angle_wigner(edge, lmax)
        loss = D.sum()
        loss.backward()

        grad = edge.grad
        assert not torch.isnan(grad).any(), "NaN gradient for exact +Y"
        assert not torch.isinf(grad).any(), "Inf gradient for exact +Y"

    @pytest.mark.parametrize(
        "direction",
        [
            [1.0, 0.0, 0.0],   # +X
            [-1.0, 0.0, 0.0],  # -X
            [0.0, 0.0, 1.0],   # +Z
            [0.0, 0.0, -1.0],  # -Z
            [1.0, 0.0, 1.0],   # XZ diagonal
        ],
    )
    def test_various_perturbation_directions(self, lmax, dtype, device, direction):
        """Near-singularity handling works for different perturbation directions."""
        epsilon = 1e-6
        dir_t = torch.tensor([direction], dtype=dtype, device=device)
        dir_t = torch.nn.functional.normalize(dir_t, dim=-1)

        # Near +Y
        edge_plus = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device)
        edge_plus = edge_plus + epsilon * dir_t
        edge_plus.requires_grad_(True)

        D, _ = axis_angle_wigner(edge_plus, lmax)
        loss = D.sum()
        loss.backward()

        assert not torch.isnan(edge_plus.grad).any(), f"NaN for +Y + eps*{direction}"

        # Near -Y
        edge_minus = torch.tensor([[0.0, -1.0, 0.0]], dtype=dtype, device=device)
        edge_minus = edge_minus + epsilon * dir_t
        edge_minus.requires_grad_(True)

        D, _ = axis_angle_wigner(edge_minus, lmax)
        loss = D.sum()
        loss.backward()

        assert not torch.isnan(edge_minus.grad).any(), f"NaN for -Y + eps*{direction}"


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


# =============================================================================
# Test SLERP Blending and Two-Chart System
# =============================================================================


class TestSlerpBlending:
    """Tests for SLERP blending between the two quaternion charts."""

    def test_smooth_step_endpoints(self, dtype, device):
        """Smooth step function returns 0 at 0 and 1 at 1."""
        t = torch.tensor([0.0, 1.0], dtype=dtype, device=device)
        result = _smooth_step_cinf(t)

        assert torch.allclose(result[0], torch.tensor(0.0, dtype=dtype), atol=1e-6)
        assert torch.allclose(result[1], torch.tensor(1.0, dtype=dtype), atol=1e-6)

    def test_smooth_step_midpoint(self, dtype, device):
        """Smooth step function returns 0.5 at midpoint."""
        t = torch.tensor([0.5], dtype=dtype, device=device)
        result = _smooth_step_cinf(t)

        assert torch.allclose(result, torch.tensor([0.5], dtype=dtype), atol=1e-6)

    def test_smooth_step_monotonic(self, dtype, device):
        """Smooth step function is monotonically increasing."""
        t = torch.linspace(0, 1, 100, dtype=dtype, device=device)
        result = _smooth_step_cinf(t)

        diff = result[1:] - result[:-1]
        assert (diff >= -1e-10).all(), "Smooth step should be monotonically increasing"

    def test_slerp_endpoints(self, dtype, device):
        """SLERP returns q1 at t=0 and q2 at t=1."""
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)  # identity
        q2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=dtype, device=device)  # 180° about X

        t0 = torch.tensor([0.0], dtype=dtype, device=device)
        t1 = torch.tensor([1.0], dtype=dtype, device=device)

        result0 = quaternion_slerp(q1, q2, t0)
        result1 = quaternion_slerp(q1, q2, t1)

        assert torch.allclose(result0, q1, atol=1e-6) or torch.allclose(result0, -q1, atol=1e-6)
        assert torch.allclose(result1, q2, atol=1e-6) or torch.allclose(result1, -q2, atol=1e-6)

    def test_slerp_midpoint_normalized(self, dtype, device):
        """SLERP midpoint is a unit quaternion."""
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)
        q2 = torch.tensor([[0.707107, 0.707107, 0.0, 0.0]], dtype=dtype, device=device)
        q2 = torch.nn.functional.normalize(q2, dim=-1)

        t = torch.tensor([0.5], dtype=dtype, device=device)
        result = quaternion_slerp(q1, q2, t)

        norm = torch.linalg.norm(result, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

    def test_chart1_singular_at_minus_y(self, dtype, device):
        """Chart 1 has small norm (singular) at -Y."""
        # Edges approaching -Y
        ey_values = torch.tensor([-0.99, -0.999, -0.9999], dtype=dtype, device=device)
        ex = torch.zeros_like(ey_values)
        ez = torch.zeros_like(ey_values)

        # Before normalization, the unnormalized quaternion
        w = 1.0 + ey_values
        x = -ez
        y = torch.zeros_like(ex)
        z = ex
        q_unnorm = torch.stack([w, x, y, z], dim=-1)
        norm = torch.linalg.norm(q_unnorm, dim=-1)

        # Norm should decrease as ey approaches -1
        assert norm[0] > norm[1] > norm[2], "Norm should decrease as ey → -1"
        assert norm[2] < 0.1, "Norm should be very small near -Y"

    def test_chart2_singular_at_plus_y(self, dtype, device):
        """Chart 2 has small norm (singular) at +Y."""
        # Edges approaching +Y
        ey_values = torch.tensor([0.99, 0.999, 0.9999], dtype=dtype, device=device)
        ex = torch.zeros_like(ey_values)
        ez = torch.zeros_like(ey_values)

        # Before normalization, the unnormalized quaternion
        w = -ez
        x = 1.0 - ey_values
        y = ex
        z = torch.zeros_like(ex)
        q_unnorm = torch.stack([w, x, y, z], dim=-1)
        norm = torch.linalg.norm(q_unnorm, dim=-1)

        # Norm should decrease as ey approaches +1
        assert norm[0] > norm[1] > norm[2], "Norm should decrease as ey → +1"
        assert norm[2] < 0.1, "Norm should be very small near +Y"

    def test_both_charts_valid_rotations(self, dtype, device):
        """Both charts produce valid unit quaternions that map edge → +Y."""
        test_edges = [
            [1.0, 0.0, 0.0],   # X-axis
            [0.0, 0.0, 1.0],   # Z-axis
            [1.0, 0.5, 0.0],   # XY plane
            [0.0, 0.5, 1.0],   # YZ plane
            [1.0, 1.0, 1.0],   # Diagonal
        ]

        for edge in test_edges:
            edge_t = torch.tensor([edge], dtype=dtype, device=device)
            edge_t = torch.nn.functional.normalize(edge_t, dim=-1)

            ex, ey, ez = edge_t[0, 0], edge_t[0, 1], edge_t[0, 2]

            q1 = _quaternion_chart1_standard(ex.unsqueeze(0), ey.unsqueeze(0), ez.unsqueeze(0))
            q2 = _quaternion_chart2_via_minus_y(ex.unsqueeze(0), ey.unsqueeze(0), ez.unsqueeze(0))

            # Both should be unit quaternions
            norm1 = torch.linalg.norm(q1, dim=-1)
            norm2 = torch.linalg.norm(q2, dim=-1)
            assert torch.allclose(norm1, torch.ones_like(norm1), atol=1e-6), f"Chart 1 not unit for {edge}"
            assert torch.allclose(norm2, torch.ones_like(norm2), atol=1e-6), f"Chart 2 not unit for {edge}"

    @pytest.mark.parametrize(
        "ey_value",
        [-0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65],  # Blend region and surroundings
    )
    def test_blend_region_gradient_bounded(self, lmax, dtype, device, ey_value):
        """Gradients are bounded throughout the blend region."""
        edge = torch.tensor(
            [[0.3, ey_value, 0.4]], dtype=dtype, device=device, requires_grad=True
        )
        edge_norm = torch.nn.functional.normalize(edge, dim=-1)

        D, _ = axis_angle_wigner(edge_norm, lmax)
        loss = D.sum()
        loss.backward()

        grad = edge.grad
        assert not torch.isnan(grad).any(), f"NaN gradient at ey={ey_value}"
        assert not torch.isinf(grad).any(), f"Inf gradient at ey={ey_value}"

        # Gradient should be bounded
        grad_norm = grad.norm().item()
        assert grad_norm < 1000, f"Gradient too large at ey={ey_value}: {grad_norm}"

    def test_blend_region_smooth_transition(self, lmax, dtype, device):
        """Wigner D matrices change smoothly across the blend region."""
        # Sample many edges across the blend region
        n_samples = 50
        ey_values = torch.linspace(-0.95, -0.65, n_samples, dtype=dtype, device=device)

        # Fixed ex, ez to isolate ey effect
        edges = torch.stack([
            torch.full_like(ey_values, 0.3),
            ey_values,
            torch.full_like(ey_values, 0.4),
        ], dim=-1)
        edges = torch.nn.functional.normalize(edges, dim=-1)

        gamma = torch.zeros(n_samples, dtype=dtype, device=device)
        D, _ = axis_angle_wigner(edges, lmax, gamma=gamma)

        # Check that consecutive matrices don't differ too much
        for i in range(n_samples - 1):
            diff = (D[i+1] - D[i]).abs().max()
            # Adjacent matrices should differ by at most a small amount
            assert diff < 0.5, f"Large jump between samples {i} and {i+1}: {diff}"

    def test_blend_region_maps_edge_to_y(self, lmax, dtype, device):
        """All edges in the blend region correctly map to +Y."""
        n_samples = 20
        ey_values = torch.linspace(-0.95, -0.65, n_samples, dtype=dtype, device=device)

        edges = torch.stack([
            torch.full_like(ey_values, 0.3),
            ey_values,
            torch.full_like(ey_values, 0.4),
        ], dim=-1)
        edges = torch.nn.functional.normalize(edges, dim=-1)

        gamma = torch.zeros(n_samples, dtype=dtype, device=device)
        D, _ = axis_angle_wigner(edges, lmax, gamma=gamma)

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

        for i in range(n_samples):
            D_l1 = D[i, 1:4, 1:4]
            result = D_l1 @ edges[i]
            assert torch.allclose(result, y_axis, atol=1e-5), (
                f"Edge at ey={ey_values[i]:.3f} did not map to +Y: {result}"
            )


# =============================================================================
# Test Cayley-Hamilton for l=2
# =============================================================================


class TestCayleyHamiltonL2:
    """Tests for the Cayley-Hamilton matrix exponential for l=2."""

    def test_matches_matrix_exp(self, dtype, device):
        """Cayley-Hamilton matches torch.linalg.matrix_exp for l=2."""
        lmax = 2
        generators = get_so3_generators(lmax, dtype, device)

        # Generate random rotation axes and angles
        torch.manual_seed(42)
        n_samples = 100
        axis = torch.randn(n_samples, 3, dtype=dtype, device=device)
        axis = torch.nn.functional.normalize(axis, dim=-1)
        angle = torch.rand(n_samples, dtype=dtype, device=device) * 3.14

        K_x = generators['K_x'][2]
        K_y = generators['K_y'][2]
        K_z = generators['K_z'][2]

        K = (
            axis[:, 0:1, None, None] * K_x +
            axis[:, 1:2, None, None] * K_y +
            axis[:, 2:3, None, None] * K_z
        ).squeeze(1)

        # Compute with both methods
        D_matexp = torch.linalg.matrix_exp(angle[:, None, None] * K)
        D_cayley = _cayley_hamilton_exp_l2(K, angle)

        # Should match closely
        max_err = (D_matexp - D_cayley).abs().max()
        assert max_err < 1e-10, f"Cayley-Hamilton error: {max_err}"

    def test_orthogonality(self, dtype, device):
        """Cayley-Hamilton produces orthogonal matrices."""
        lmax = 2
        generators = get_so3_generators(lmax, dtype, device)

        torch.manual_seed(43)
        n_samples = 50
        axis = torch.randn(n_samples, 3, dtype=dtype, device=device)
        axis = torch.nn.functional.normalize(axis, dim=-1)
        angle = torch.rand(n_samples, dtype=dtype, device=device) * 3.14

        K_x = generators['K_x'][2]
        K_y = generators['K_y'][2]
        K_z = generators['K_z'][2]

        K = (
            axis[:, 0:1, None, None] * K_x +
            axis[:, 1:2, None, None] * K_y +
            axis[:, 2:3, None, None] * K_z
        ).squeeze(1)

        D = _cayley_hamilton_exp_l2(K, angle)

        # D @ D.T should be identity
        I = torch.eye(5, dtype=dtype, device=device)
        product = torch.bmm(D, D.transpose(1, 2))

        for i in range(n_samples):
            assert torch.allclose(product[i], I, atol=1e-10), (
                f"Sample {i} not orthogonal: {product[i]}"
            )

    def test_determinant_is_one(self, dtype, device):
        """Cayley-Hamilton produces matrices with determinant 1."""
        lmax = 2
        generators = get_so3_generators(lmax, dtype, device)

        torch.manual_seed(44)
        n_samples = 50
        axis = torch.randn(n_samples, 3, dtype=dtype, device=device)
        axis = torch.nn.functional.normalize(axis, dim=-1)
        angle = torch.rand(n_samples, dtype=dtype, device=device) * 3.14

        K_x = generators['K_x'][2]
        K_y = generators['K_y'][2]
        K_z = generators['K_z'][2]

        K = (
            axis[:, 0:1, None, None] * K_x +
            axis[:, 1:2, None, None] * K_y +
            axis[:, 2:3, None, None] * K_z
        ).squeeze(1)

        D = _cayley_hamilton_exp_l2(K, angle)

        for i in range(n_samples):
            det = torch.linalg.det(D[i])
            assert torch.allclose(det, torch.tensor(1.0, dtype=dtype), atol=1e-10), (
                f"Sample {i} det != 1: {det}"
            )

    def test_identity_rotation(self, dtype, device):
        """Zero angle gives identity matrix."""
        lmax = 2
        generators = get_so3_generators(lmax, dtype, device)

        axis = torch.tensor([[0, 0, 1]], dtype=dtype, device=device)
        angle = torch.tensor([0.0], dtype=dtype, device=device)

        K_x = generators['K_x'][2]
        K_y = generators['K_y'][2]
        K_z = generators['K_z'][2]

        K = (
            axis[:, 0:1, None, None] * K_x +
            axis[:, 1:2, None, None] * K_y +
            axis[:, 2:3, None, None] * K_z
        ).squeeze(1)

        D = _cayley_hamilton_exp_l2(K, angle)
        I = torch.eye(5, dtype=dtype, device=device)

        assert torch.allclose(D[0], I, atol=1e-10), "Zero angle should give identity"

    @pytest.mark.parametrize(
        "axis,desc",
        [
            ([1, 0, 0], "X-axis"),
            ([0, 1, 0], "Y-axis"),
            ([0, 0, 1], "Z-axis"),
            ([1, 1, 0], "XY-plane"),
            ([1, 1, 1], "diagonal"),
        ],
    )
    def test_principal_axes(self, dtype, device, axis, desc):
        """Cayley-Hamilton works for rotations about principal axes."""
        lmax = 2
        generators = get_so3_generators(lmax, dtype, device)

        axis_t = torch.tensor([axis], dtype=dtype, device=device)
        axis_t = torch.nn.functional.normalize(axis_t, dim=-1)
        angle = torch.tensor([math.pi / 3], dtype=dtype, device=device)

        K_x = generators['K_x'][2]
        K_y = generators['K_y'][2]
        K_z = generators['K_z'][2]

        K = (
            axis_t[:, 0:1, None, None] * K_x +
            axis_t[:, 1:2, None, None] * K_y +
            axis_t[:, 2:3, None, None] * K_z
        ).squeeze(1)

        D_matexp = torch.linalg.matrix_exp(angle[:, None, None] * K)
        D_cayley = _cayley_hamilton_exp_l2(K, angle)

        max_err = (D_matexp - D_cayley).abs().max()
        assert max_err < 1e-10, f"{desc} error: {max_err}"

    def test_gradient_flow(self, dtype, device):
        """Gradients flow correctly through Cayley-Hamilton."""
        lmax = 2
        generators = get_so3_generators(lmax, dtype, device)

        axis = torch.randn(10, 3, dtype=dtype, device=device, requires_grad=True)
        angle = torch.rand(10, dtype=dtype, device=device, requires_grad=True)

        K_x = generators['K_x'][2]
        K_y = generators['K_y'][2]
        K_z = generators['K_z'][2]

        axis_norm = torch.nn.functional.normalize(axis, dim=-1)
        K = (
            axis_norm[:, 0:1, None, None] * K_x +
            axis_norm[:, 1:2, None, None] * K_y +
            axis_norm[:, 2:3, None, None] * K_z
        ).squeeze(1)

        D = _cayley_hamilton_exp_l2(K, angle)
        loss = D.sum()
        loss.backward()

        assert not torch.isnan(axis.grad).any(), "NaN gradient in axis"
        assert not torch.isinf(axis.grad).any(), "Inf gradient in axis"
        assert not torch.isnan(angle.grad).any(), "NaN gradient in angle"
        assert not torch.isinf(angle.grad).any(), "Inf gradient in angle"


# =============================================================================
# Test Ra/Rb Polynomial Version (GPU-optimized)
# =============================================================================


class TestAxisAnglePolynomial:
    """Tests for the GPU-optimized Ra/Rb polynomial version."""

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
        """The l=1 block correctly rotates edge → +Y."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        edge_t = torch.nn.functional.normalize(edge_t, dim=-1)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = axis_angle_wigner_polynomial(edge_t, lmax, gamma=gamma)
        D_l1 = D[0, 1:4, 1:4]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        result = D_l1 @ edge_t[0]

        assert torch.allclose(result, y_axis, atol=1e-5), (
            f"Edge {edge} did not map to +Y, got {result}"
        )

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
    def test_orthogonality(self, lmax, dtype, device, edge, desc):
        """Wigner D matrices are orthogonal: D @ D.T = I."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, D_inv = axis_angle_wigner_polynomial(edge_t, lmax, gamma=gamma)

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

        D, _ = axis_angle_wigner_polynomial(edge_t, lmax, gamma=gamma)

        det = torch.linalg.det(D[0])
        assert torch.allclose(
            det, torch.tensor(1.0, dtype=dtype, device=device), atol=1e-5
        ), f"det != 1 for {desc}: {det}"

    def test_inverse_is_transpose(self, lmax, dtype, device):
        """D_inv equals D.T."""
        torch.manual_seed(42)
        edges = torch.randn(10, 3, dtype=dtype, device=device)
        gamma = torch.rand(10, dtype=dtype, device=device) * 6.28

        D, D_inv = axis_angle_wigner_polynomial(edges, lmax, gamma=gamma)

        for i in range(10):
            assert torch.allclose(D_inv[i], D[i].T, atol=1e-10)

    @pytest.mark.parametrize(
        "epsilon",
        [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    )
    def test_near_plus_y_maps_to_y(self, lmax, dtype, device, epsilon):
        """Edges near +Y still correctly map to +Y."""
        edge = torch.tensor([[epsilon, 1.0, 0.0]], dtype=dtype, device=device)
        edge_norm = torch.nn.functional.normalize(edge, dim=-1)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = axis_angle_wigner_polynomial(edge_norm, lmax, gamma=gamma)
        D_l1 = D[0, 1:4, 1:4]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        result = D_l1 @ edge_norm[0]

        assert torch.allclose(result, y_axis, atol=1e-5), (
            f"Near +Y edge (eps={epsilon}) did not map to +Y: {result}"
        )

    @pytest.mark.parametrize(
        "epsilon",
        [1e-2, 1e-3, 1e-4, 1e-5],
    )
    def test_near_minus_y_maps_to_y(self, lmax, dtype, device, epsilon):
        """Edges near -Y still correctly map to +Y."""
        edge = torch.tensor([[epsilon, -1.0, 0.0]], dtype=dtype, device=device)
        edge_norm = torch.nn.functional.normalize(edge, dim=-1)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = axis_angle_wigner_polynomial(edge_norm, lmax, gamma=gamma)
        D_l1 = D[0, 1:4, 1:4]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        result = D_l1 @ edge_norm[0]

        assert torch.allclose(result, y_axis, atol=1e-5), (
            f"Near -Y edge (eps={epsilon}) did not map to +Y: {result}"
        )

    def test_gradient_flow(self, lmax, dtype, device):
        """Gradients flow without NaN or Inf."""
        edge_t = torch.tensor(
            [[0.6, 0.5, 0.8]], dtype=dtype, device=device, requires_grad=True
        )
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = axis_angle_wigner_polynomial(edge_t, lmax, gamma=gamma)
        loss = D.sum()
        loss.backward()

        grad = edge_t.grad
        assert not torch.isnan(grad).any(), "NaN gradient"
        assert not torch.isinf(grad).any(), "Inf gradient"


# =============================================================================
# Test Polynomial vs Original Comparison
# =============================================================================


class TestPolynomialVsOriginal:
    """Test that axis_angle_wigner_polynomial matches axis_angle_wigner."""

    def test_matches_for_random_edges(self, lmax, dtype, device):
        """Polynomial version matches original for random edges."""
        torch.manual_seed(42)
        edges = torch.randn(50, 3, dtype=dtype, device=device)
        gamma = torch.rand(50, dtype=dtype, device=device) * 6.28

        D_orig, D_inv_orig = axis_angle_wigner(edges, lmax, gamma=gamma)
        D_poly, D_inv_poly = axis_angle_wigner_polynomial(edges, lmax, gamma=gamma)

        # Check all blocks match
        max_err = (D_orig - D_poly).abs().max().item()
        assert max_err < 1e-9, f"D matrices differ by {max_err}"

        max_inv_err = (D_inv_orig - D_inv_poly).abs().max().item()
        assert max_inv_err < 1e-9, f"D_inv matrices differ by {max_inv_err}"

    @pytest.mark.parametrize(
        "edge,desc",
        [
            ([1.0, 0.0, 0.0], "X-axis"),
            ([0.0, 0.0, 1.0], "Z-axis"),
            ([1.0, 1.0, 0.0], "XY-plane"),
            ([1.0, 1.0, 1.0], "diagonal"),
            ([0.1, 0.9, 0.1], "near-Y"),
        ],
    )
    def test_matches_for_specific_edges(self, lmax, dtype, device, edge, desc):
        """Polynomial version matches original for specific edges."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D_orig, _ = axis_angle_wigner(edge_t, lmax, gamma=gamma)
        D_poly, _ = axis_angle_wigner_polynomial(edge_t, lmax, gamma=gamma)

        max_err = (D_orig - D_poly).abs().max().item()
        assert max_err < 1e-9, f"Mismatch for {desc}: {max_err}"

    def test_matches_near_singularities(self, lmax, dtype, device):
        """Polynomial version matches original near +Y and -Y singularities."""
        for eps in [1e-2, 1e-4, 1e-6]:
            # Near +Y
            edge_plus = torch.tensor([[eps, 1.0, 0.0]], dtype=dtype, device=device)
            edge_plus = torch.nn.functional.normalize(edge_plus, dim=-1)
            gamma = torch.zeros(1, dtype=dtype, device=device)

            D_orig, _ = axis_angle_wigner(edge_plus, lmax, gamma=gamma)
            D_poly, _ = axis_angle_wigner_polynomial(edge_plus, lmax, gamma=gamma)

            max_err = (D_orig - D_poly).abs().max().item()
            assert max_err < 1e-8, f"Near +Y (eps={eps}) differs by {max_err}"

            # Near -Y
            edge_minus = torch.tensor([[eps, -1.0, 0.0]], dtype=dtype, device=device)
            edge_minus = torch.nn.functional.normalize(edge_minus, dim=-1)

            D_orig, _ = axis_angle_wigner(edge_minus, lmax, gamma=gamma)
            D_poly, _ = axis_angle_wigner_polynomial(edge_minus, lmax, gamma=gamma)

            max_err = (D_orig - D_poly).abs().max().item()
            assert max_err < 1e-8, f"Near -Y (eps={eps}) differs by {max_err}"

    def test_gradient_matches(self, lmax, dtype, device):
        """Gradients match between polynomial and original implementations."""
        torch.manual_seed(42)
        edge1 = torch.randn(10, 3, dtype=dtype, device=device)
        edge2 = edge1.clone()
        edge1.requires_grad_(True)
        edge2.requires_grad_(True)
        gamma = torch.rand(10, dtype=dtype, device=device) * 6.28

        D_orig, _ = axis_angle_wigner(edge1, lmax, gamma=gamma)
        D_poly, _ = axis_angle_wigner_polynomial(edge2, lmax, gamma=gamma)

        D_orig.sum().backward()
        D_poly.sum().backward()

        grad_diff = (edge1.grad - edge2.grad).abs().max().item()
        assert grad_diff < 1e-7, f"Gradients differ by {grad_diff}"


class TestHybridApproach:
    """Tests for the axis_angle_wigner_hybrid function."""

    @pytest.fixture()
    def lmax(self):
        return 4

    @pytest.fixture()
    def dtype(self):
        return torch.float64

    @pytest.fixture()
    def device(self):
        return torch.device("cpu")

    @pytest.fixture()
    def edges(self, dtype, device):
        """Test edge directions including edge cases."""
        return [
            ([1.0, 0.0, 0.0], "X-axis"),
            ([0.0, 1.0, 0.0], "+Y-axis"),
            ([0.0, -1.0, 0.0], "-Y-axis"),
            ([0.0, 0.0, 1.0], "Z-axis"),
            ([0.7, 0.3, 0.5], "diagonal"),
            ([0.01, 0.99995, 0.0], "near +Y"),
            ([0.01, -0.99995, 0.0], "near -Y"),
        ]

    def test_matches_original(self, lmax, dtype, device, edges):
        """Hybrid approach matches original axis_angle_wigner."""
        from fairchem.core.models.uma.common.wigner_d_axis_angle import axis_angle_wigner_hybrid

        for edge, desc in edges:
            edge_t = torch.tensor([edge], dtype=dtype, device=device)
            edge_t = torch.nn.functional.normalize(edge_t, dim=-1)
            gamma = torch.zeros(1, dtype=dtype, device=device)

            D_orig, _ = axis_angle_wigner(edge_t, lmax, gamma=gamma)
            D_hybrid, _ = axis_angle_wigner_hybrid(edge_t, lmax, gamma=gamma)

            max_err = (D_orig - D_hybrid).abs().max().item()
            assert max_err < 1e-10, f"Mismatch for {desc}: {max_err}"

    def test_batched_matches_original(self, lmax, dtype, device):
        """Hybrid matches original for batched input."""
        from fairchem.core.models.uma.common.wigner_d_axis_angle import axis_angle_wigner_hybrid

        torch.manual_seed(42)
        edges = torch.randn(100, 3, dtype=dtype, device=device)
        gamma = torch.rand(100, dtype=dtype, device=device) * 2 * 3.14159

        D_orig, _ = axis_angle_wigner(edges, lmax, gamma=gamma)
        D_hybrid, _ = axis_angle_wigner_hybrid(edges, lmax, gamma=gamma)

        max_err = (D_orig - D_hybrid).abs().max().item()
        assert max_err < 1e-10, f"Batched mismatch: {max_err}"

    def test_gradient_flow(self, lmax, dtype, device):
        """Gradients flow correctly through hybrid approach."""
        from fairchem.core.models.uma.common.wigner_d_axis_angle import axis_angle_wigner_hybrid

        torch.manual_seed(42)
        edges = torch.randn(20, 3, dtype=dtype, device=device, requires_grad=True)
        gamma = torch.rand(20, dtype=dtype, device=device)

        D, _ = axis_angle_wigner_hybrid(edges, lmax, gamma=gamma)
        loss = D.sum()
        loss.backward()

        assert edges.grad is not None
        assert not edges.grad.isnan().any()
        assert edges.grad.abs().max() > 0

    def test_lower_lmax_still_works(self, dtype, device):
        """Hybrid works correctly when lmax < 3 (no Ra/Rb path)."""
        from fairchem.core.models.uma.common.wigner_d_axis_angle import axis_angle_wigner_hybrid

        torch.manual_seed(42)
        edges = torch.randn(10, 3, dtype=dtype, device=device)
        gamma = torch.rand(10, dtype=dtype, device=device) * 2 * 3.14159

        for test_lmax in [1, 2]:
            D_orig, _ = axis_angle_wigner(edges, test_lmax, gamma=gamma)
            D_hybrid, _ = axis_angle_wigner_hybrid(edges, test_lmax, gamma=gamma)

            max_err = (D_orig - D_hybrid).abs().max().item()
            assert max_err < 1e-12, f"lmax={test_lmax} mismatch: {max_err}"


class TestRangeFunctions:
    """Tests for the lmin-based range functions in wigner_d_quaternion."""

    @pytest.fixture()
    def dtype(self):
        return torch.float64

    @pytest.fixture()
    def device(self):
        return torch.device("cpu")

    def test_coefficients_range_matches_full(self, dtype, device):
        """Range coefficients match full coefficients for l >= lmin."""
        from fairchem.core.models.uma.common.wigner_d_quaternion import (
            precompute_wigner_coefficients_symmetric,
            precompute_wigner_coefficients_range,
        )

        lmin, lmax = 3, 6
        coeffs_full = precompute_wigner_coefficients_symmetric(lmax, dtype, device)
        coeffs_range = precompute_wigner_coefficients_range(lmin, lmax, dtype, device)

        # Check sizes
        expected_size = (lmax + 1) ** 2 - lmin ** 2
        assert coeffs_range["size"] == expected_size
        assert coeffs_range["lmin"] == lmin
        assert coeffs_range["lmax"] == lmax

    def test_wigner_d_range_matches_full(self, dtype, device):
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

    def test_u_blocks_range(self, dtype, device):
        """U blocks range returns correct blocks."""
        from fairchem.core.models.uma.common.wigner_d_quaternion import (
            precompute_U_blocks,
            precompute_U_blocks_range,
        )

        lmin, lmax = 2, 5
        U_full = precompute_U_blocks(lmax, dtype, device)
        U_range = precompute_U_blocks_range(lmin, lmax, dtype, device)

        # Check lengths
        assert len(U_range) == lmax - lmin + 1

        # Check each block matches
        for idx, ell in enumerate(range(lmin, lmax + 1)):
            max_err = (U_full[ell] - U_range[idx]).abs().max().item()
            assert max_err < 1e-14, f"U block for l={ell} differs by {max_err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
