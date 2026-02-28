"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for Wigner D matrix computation.

Tests verify:
1. Mathematical correctness (orthogonality, determinant, edge -> +Y)
2. Agreement between all entry point functions
3. Agreement with Euler-based rotation.py
4. Gradient stability
5. torch.compile compatibility
6. Range functions (lmin support)
7. Specialized kernels (l=2 polynomial, l=3/4 matmul)
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from fairchem.core.models.uma.common.quaternion_utils import (
    BLEND_START,
    BLEND_WIDTH,
    _smooth_step_cinf,
)
from fairchem.core.models.uma.common.quaternion_wigner_utils import (
    _build_euler_transform,
    _build_u_matrix,
    create_wigner_data_module,
    precompute_U_blocks_euler_aligned_real,
    precompute_wigner_coefficients,
    quaternion_to_ra_rb_real,
    wigner_d_matrix_real,
    wigner_d_pair_to_real,
)
from fairchem.core.models.uma.common.rotation import (
    init_edge_rot_euler_angles,
    wigner_D,
)
from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
    CustomKernelModule,
    quaternion_to_wigner_d_l2_einsum,
    quaternion_to_wigner_d_matmul,
)
from fairchem.core.models.uma.common.wigner_d_hybrid import (
    axis_angle_wigner_hybrid,
)

# =============================================================================
# Reference Implementation for Testing (Not Used at Runtime)
#
# These functions provide a mathematically principled reference implementation
# for computing Wigner D matrices via matrix exponential of SO(3) Lie algebra
# generators. They validate that the optimized polynomial kernels in
# wigner_d_custom_kernels.py produce correct results.
# =============================================================================


def quaternion_to_axis_angle(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert quaternion to axis-angle representation.

    Uses the stable formula:
        angle = 2 * atan2(|xyz|, w)
        axis = xyz / |xyz|

    For small angles (|xyz| ~ 0), axis is undefined but angle ~ 0.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        (axis, angle) where:
        - axis has shape (N, 3), unit vectors
        - angle has shape (N,), in radians
    """
    w = q[..., 0]
    xyz = q[..., 1:4]

    xyz_norm = torch.linalg.norm(xyz, dim=-1)
    angle = 2.0 * torch.atan2(xyz_norm, w)

    safe_xyz_norm = xyz_norm.clamp(min=1e-12)
    axis = xyz / safe_xyz_norm.unsqueeze(-1)

    small_angle = xyz_norm < 1e-8
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=q.dtype, device=q.device)
    z_axis = z_axis.expand_as(axis)
    axis = torch.where(small_angle.unsqueeze(-1), z_axis, axis)

    return axis, angle


def _build_so3_generators(
    ell: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build SO(3) Lie algebra generators K_x, K_y, K_z for representation ell.

    These are real antisymmetric (2*ell+1) x (2*ell+1) matrices satisfying:
        D^ell(n, theta) = exp(theta * (n_x K_x + n_y K_y + n_z K_z))

    Args:
        ell: Angular momentum quantum number

    Returns:
        (K_x, K_y, K_z) tuple of generator matrices in float64
    """
    size = 2 * ell + 1

    if ell == 0:
        z = torch.zeros(1, 1, dtype=torch.float64)
        return z, z.clone(), z.clone()

    m_values = torch.arange(-ell, ell + 1, dtype=torch.float64)
    J_z = torch.diag(m_values.to(torch.complex128))

    J_plus = torch.zeros(size, size, dtype=torch.complex128)
    J_minus = torch.zeros(size, size, dtype=torch.complex128)

    for m in range(-ell, ell):
        coeff = math.sqrt(ell * (ell + 1) - m * (m + 1))
        J_plus[m + 1 + ell, m + ell] = coeff

    for m in range(-ell + 1, ell + 1):
        coeff = math.sqrt(ell * (ell + 1) - m * (m - 1))
        J_minus[m - 1 + ell, m + ell] = coeff

    J_x = (J_plus + J_minus) / 2
    J_y = (J_plus - J_minus) / 2j

    U = _build_u_matrix(ell)
    U_dag = U.conj().T

    K_x = (U @ (1j * J_x) @ U_dag).real
    K_y = -(U @ (1j * J_y) @ U_dag).real
    K_z = (U @ (1j * J_z) @ U_dag).real

    return K_x, K_y, K_z


def get_so3_generators(
    lmax: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, list[torch.Tensor]]:
    """
    Compute K_x, K_y, K_z lists for l=0..lmax.

    For l >= 2, the generators include the Euler-matching transformation folded in,
    so the matrix exponential produces output directly in the Euler basis.

    For l=1, a permutation matrix P is also returned to convert to Cartesian basis.

    Args:
        lmax: Maximum angular momentum
        dtype: Data type for the generators
        device: Device for the generators

    Returns:
        Dictionary with 'K_x', 'K_y', 'K_z' lists and 'P' for l=1 permutation
    """
    jd_path = (
        Path(__file__).parents[4]
        / "src"
        / "fairchem"
        / "core"
        / "models"
        / "uma"
        / "Jd.pt"
    )
    Jd_list = torch.load(jd_path, map_location=device, weights_only=True)

    K_x_list = []
    K_y_list = []
    K_z_list = []

    for ell in range(lmax + 1):
        K_x, K_y, K_z = _build_so3_generators(ell)
        K_x = K_x.to(device=device, dtype=dtype)
        K_y = K_y.to(device=device, dtype=dtype)
        K_z = K_z.to(device=device, dtype=dtype)

        if ell >= 2:
            Jd = Jd_list[ell].to(dtype=dtype, device=device)
            U = _build_euler_transform(ell, Jd)
            K_x = U @ K_x @ U.T
            K_y = U @ K_y @ U.T
            K_z = U @ K_z @ U.T

        K_x_list.append(K_x)
        K_y_list.append(K_y)
        K_z_list.append(K_z)

    P = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    )

    return {
        "K_x": K_x_list,
        "K_y": K_y_list,
        "K_z": K_z_list,
        "P": P,
    }


def compute_euler_matching_gamma(edge_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute gamma to match the Euler convention.

    Uses a two-chart approach matching the quaternion_edge_to_y_stable function:
    - Chart 1 (ey >= 0.9): gamma = -atan2(ex, ez)
    - Chart 2 (ey <= -0.9): gamma = +atan2(ex, ez)
    - Blend region (-0.9 < ey < 0.9): smooth interpolation

    For edges on Y-axis (ex = ez ~ 0): gamma = 0 (degenerate case).

    Note: In the blend region, there is inherent approximation error
    due to the NLERP quaternion blending. This is acceptable for the intended
    use case of matching Euler output for testing/validation. Properly determined
    gamma values are used in the test.

    Args:
        edge_vec: Edge vectors of shape (N, 3), assumed normalized

    Returns:
        Gamma angles of shape (N,)
    """
    ex = edge_vec[..., 0]
    ey = edge_vec[..., 1]
    ez = edge_vec[..., 2]

    # Chart 1 gamma (used for ey >= 0.9)
    gamma_chart1 = -torch.atan2(ex, ez)

    # Chart 2 gamma (used for ey <= -0.9)
    gamma_chart2 = torch.atan2(ex, ez)

    # Blend factor (same as quaternion_edge_to_y_stable)
    t = (ey - BLEND_START) / BLEND_WIDTH
    t_smooth = _smooth_step_cinf(t)

    # Interpolate: t_smooth=0 -> chart2, t_smooth=1 -> chart1
    gamma = t_smooth * gamma_chart1 + (1 - t_smooth) * gamma_chart2

    return gamma


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


@pytest.fixture()
def wigner_data(lmax):
    """
    Create a WignerDataModule with all precomputed data for the given lmax.
    """
    return create_wigner_data_module(lmax=max(lmax, 5), lmin=5)


@pytest.fixture()
def custom_kernels():
    """
    Create a CustomKernelModule with l=2,3,4 coefficient buffers.
    """
    return CustomKernelModule()


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
    def test_orthogonality_and_determinant(
        self, lmax, dtype, device, wigner_data, edge, desc
    ):
        """Wigner D matrices are orthogonal with determinant 1."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, D_inv = axis_angle_wigner_hybrid(
            edge_t,
            lmax,
            gamma=gamma,
            coeffs=wigner_data.coeffs,
            U_blocks=wigner_data.U_blocks,
            custom_kernels=wigner_data.custom_kernels,
        )

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
    def test_edge_to_y(self, lmax, dtype, device, wigner_data, edge, desc):
        """The l=1 block rotates edge -> +Y."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        edge_t = torch.nn.functional.normalize(edge_t, dim=-1)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = axis_angle_wigner_hybrid(
            edge_t,
            lmax,
            gamma=gamma,
            coeffs=wigner_data.coeffs,
            U_blocks=wigner_data.U_blocks,
            custom_kernels=wigner_data.custom_kernels,
        )
        D_l1 = D[0, 1:4, 1:4]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        result = D_l1 @ edge_t[0]

        assert torch.allclose(
            result, y_axis, atol=1e-5
        ), f"Edge {edge} did not map to +Y, got {result}"

    @pytest.mark.parametrize("edge,desc", STANDARD_TEST_EDGES)
    def test_y_to_edge(self, lmax, dtype, device, wigner_data, edge, desc):
        """D_inv l=1 block rotates +Y -> edge (inverse of edge -> +Y)."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        edge_t = torch.nn.functional.normalize(edge_t, dim=-1)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        _, D_inv = axis_angle_wigner_hybrid(
            edge_t,
            lmax,
            gamma=gamma,
            coeffs=wigner_data.coeffs,
            U_blocks=wigner_data.U_blocks,
            custom_kernels=wigner_data.custom_kernels,
        )
        D_inv_l1 = D_inv[0, 1:4, 1:4]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        result = D_inv_l1 @ y_axis

        assert torch.allclose(
            result, edge_t[0], atol=1e-5
        ), f"+Y did not map to edge {edge}, got {result}"

    def test_composition_law(self, lmax, dtype, device, wigner_data):
        """D(R1) @ D(R2) = D(R1 @ R2) - the fundamental group composition property."""
        torch.manual_seed(123)
        n_samples = 10

        # Generate two random rotations (gamma is randomized internally when not specified)
        edges1 = torch.randn(n_samples, 3, dtype=dtype, device=device)
        edges2 = torch.randn(n_samples, 3, dtype=dtype, device=device)

        D1, _ = axis_angle_wigner_hybrid(
            edges1,
            lmax,
            coeffs=wigner_data.coeffs,
            U_blocks=wigner_data.U_blocks,
            custom_kernels=wigner_data.custom_kernels,
        )
        D2, _ = axis_angle_wigner_hybrid(
            edges2,
            lmax,
            coeffs=wigner_data.coeffs,
            U_blocks=wigner_data.U_blocks,
            custom_kernels=wigner_data.custom_kernels,
        )

        # Compose the Wigner D matrices
        D_product = D1 @ D2

        # The l=1 block of D_product is the composed rotation matrix R1 @ R2
        R_composed = D_product[:, 1:4, 1:4]

        # From R_composed, extract edge (second row, since R @ edge = +Y means edge = R^T @ +Y)
        edge_composed = R_composed[:, 1, :]

        # Compute D for edge_composed with gamma=0 to get the canonical alignment rotation
        D_canonical, _ = axis_angle_wigner_hybrid(
            edge_composed,
            lmax,
            gamma=torch.zeros(n_samples, dtype=dtype, device=device),
            coeffs=wigner_data.coeffs,
            U_blocks=wigner_data.U_blocks,
            custom_kernels=wigner_data.custom_kernels,
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
        D_composed, _ = axis_angle_wigner_hybrid(
            edge_composed,
            lmax,
            gamma=gamma_composed,
            coeffs=wigner_data.coeffs,
            U_blocks=wigner_data.U_blocks,
            custom_kernels=wigner_data.custom_kernels,
        )

        # Check that the product equals the composed Wigner D
        max_err = (D_product - D_composed).abs().max().item()
        assert max_err < 1e-9, f"Composition law failed: max error = {max_err}"


# =============================================================================
# Test Entry Point Agreement
# =============================================================================


class TestEntryPointAgreement:
    """Tests for agreement between all Wigner D entry point functions."""


# =============================================================================
# Test Euler Agreement
# =============================================================================


class TestEulerAgreement:
    """Tests for agreement with Euler-based rotation.py."""

    def test_matches_euler_code(self, lmax, dtype, device, wigner_data, Jd_matrices):
        """Axis-angle with use_euler_gamma matches Euler implementation exactly.

        Note: Only tests edges outside the blend region (|ey| > 0.9) where
        exact Euler matching is possible.
        """
        # Edges outside blend region (|ey| > 0.9)
        test_edges = [
            [0.0, 1.0, 0.0],  # +Y (ey=1, Chart1)
            [0.0, -1.0, 0.0],  # -Y (ey=-1, Chart2)
            [0.1, 0.99, 0.0],  # near +Y
            [0.0, 0.95, 0.3],  # near +Y
        ]

        for edge in test_edges:
            edge_t = torch.tensor([edge], dtype=dtype, device=device)

            # Compute with axis-angle using Euler gamma
            edge_normalized = torch.nn.functional.normalize(edge_t, dim=-1)
            gamma = compute_euler_matching_gamma(edge_normalized)
            D_axis, _ = axis_angle_wigner_hybrid(
                edge_t,
                lmax,
                gamma=gamma,
                coeffs=wigner_data.coeffs,
                U_blocks=wigner_data.U_blocks,
                custom_kernels=wigner_data.custom_kernels,
            )

            # Get Euler angles from production code, zero out random gamma
            gamma, beta, alpha = init_edge_rot_euler_angles(edge_t)
            gamma_zero = torch.zeros_like(gamma)

            for ell in range(lmax + 1):
                start = ell * ell
                end = start + 2 * ell + 1
                D_euler = wigner_D(ell, gamma_zero, beta, alpha, Jd_matrices)
                D_axis_block = D_axis[0, start:end, start:end]

                assert torch.allclose(
                    D_euler, D_axis_block, atol=1e-10
                ), f"l={ell} mismatch for edge {edge}"

    def test_blend_region_matches_euler(
        self, lmax, dtype, device, wigner_data, Jd_matrices
    ):
        """Blend region edges (ey in [-0.9, 0.9]) match Euler with correct gamma."""
        blend_region_edges = [
            # yz-plane (ex=0), ey=-0.8
            ([0.0, -0.8, 0.6], 0.0),
            # xz-plane (ez=0), ey=-0.8
            ([0.6, -0.8, 0.0], 1.5707964146104496),
            # yz-plane, ey=-0.85
            ([0.0, -0.8499922481060458, 0.5267951956497234], 0.0),
            # xy-plane, at blend boundary (ey=-0.9)
            ([0.43589807987318724, -0.8999960355261952, 0.0], 1.5707963267950893),
            # general edge, ey=-0.75
            (
                [0.1000022780778422, -0.7500170855838165, 0.6538148940729324],
                0.1517701818416542,
            ),
            # diagonal (1,1,1)
            ([1.0, 1.0, 1.0], -0.7674963309777119),
            # +X axis
            ([1.0, 0.0, 0.0], -3.1415926535897931),
            # +Z axis
            ([0.0, 0.0, 1.0], 0.0),
        ]

        for edge, gamma in blend_region_edges:
            edge_t = torch.tensor([edge], dtype=dtype, device=device)
            gamma_t = torch.tensor([gamma], dtype=dtype, device=device)

            # Compute with axis-angle hybrid using pre-computed gamma
            D_hybrid, _ = axis_angle_wigner_hybrid(
                edge_t,
                lmax,
                gamma=gamma_t,
                coeffs=wigner_data.coeffs,
                U_blocks=wigner_data.U_blocks,
                custom_kernels=wigner_data.custom_kernels,
            )

            # Get Euler angles from production code, zero out random gamma
            _, beta, alpha = init_edge_rot_euler_angles(edge_t)
            gamma_zero = torch.zeros(1, dtype=dtype, device=device)

            for ell in range(lmax + 1):
                start = ell * ell
                end = start + 2 * ell + 1
                D_euler = wigner_D(ell, gamma_zero, beta, alpha, Jd_matrices)
                D_hybrid_block = D_hybrid[0, start:end, start:end]

                assert torch.allclose(
                    D_euler[0], D_hybrid_block, atol=1e-10
                ), f"l={ell} mismatch for blend region edge {edge}"


# =============================================================================
# Test Gradient Stability
# =============================================================================


class TestGradientStability:
    """Tests for gradient stability including near singularities."""

    @pytest.mark.parametrize("edge,desc", STANDARD_TEST_EDGES)
    def test_gradient_flow(self, lmax, dtype, device, wigner_data, edge, desc):
        """Gradients flow without NaN/Inf and are reasonably bounded."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device, requires_grad=True)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        D, _ = axis_angle_wigner_hybrid(
            edge_t,
            lmax,
            gamma=gamma,
            coeffs=wigner_data.coeffs,
            U_blocks=wigner_data.U_blocks,
            custom_kernels=wigner_data.custom_kernels,
        )
        loss = D.sum()
        loss.backward()

        grad = edge_t.grad
        assert not torch.isnan(grad).any(), f"NaN gradient for {desc}"
        assert not torch.isinf(grad).any(), f"Inf gradient for {desc}"
        assert (
            grad.abs().max() < 1000
        ), f"Gradient too large for {desc}: {grad.abs().max()}"

    @pytest.mark.parametrize("epsilon", [1e-4, 1e-6, 1e-8])
    def test_near_singularity_correctness(
        self, lmax, dtype, device, wigner_data, epsilon
    ):
        """Edges near +/-Y still correctly map to +Y with bounded gradients."""
        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

        for sign in [1.0, -1.0]:
            edge = torch.tensor(
                [[epsilon, sign * 1.0, 0.0]],
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            edge_norm = torch.nn.functional.normalize(edge, dim=-1)

            D, _ = axis_angle_wigner_hybrid(
                edge_norm,
                lmax,
                coeffs=wigner_data.coeffs,
                U_blocks=wigner_data.U_blocks,
                custom_kernels=wigner_data.custom_kernels,
            )
            D_l1 = D[0, 1:4, 1:4]
            result = D_l1 @ edge_norm[0]

            # Check maps to Y
            assert torch.allclose(
                result, y_axis, atol=1e-5
            ), f"Near {'+'if sign>0 else '-'}Y edge (eps={epsilon}) did not map to +Y"

            # Check gradients are valid and bounded
            D.sum().backward()
            assert not torch.isnan(edge.grad).any()
            assert (
                edge.grad.abs().max() < 1000
            ), f"Gradient too large near {'+'if sign>0 else '-'}Y (eps={epsilon}): {edge.grad.abs().max()}"


# =============================================================================
# Test torch.compile Compatibility
# =============================================================================


class TestTorchCompileCompatibility:
    """Tests for torch.compile compatibility of real-arithmetic functions."""

    @pytest.mark.skipif(
        not hasattr(torch, "_dynamo"), reason="torch.compile not available"
    )
    def test_hybrid_compiles(
        self, lmax, dtype, device, wigner_data, compile_reset_state
    ):
        """axis_angle_wigner_hybrid should compile without graph breaks."""
        import torch._dynamo as dynamo

        edges = torch.randn(10, 3, dtype=dtype, device=device)
        gamma = torch.rand(10, dtype=dtype, device=device) * 6.28

        coeffs = wigner_data.coeffs
        U_blocks = wigner_data.U_blocks
        ck = wigner_data.custom_kernels

        # Define function to compile
        def fn(edge_vec, lmax_val, g):
            return axis_angle_wigner_hybrid(
                edge_vec,
                lmax_val,
                gamma=g,
                coeffs=coeffs,
                U_blocks=U_blocks,
                custom_kernels=ck,
            )

        # Try to compile and run
        try:
            compiled_fn = torch.compile(fn, fullgraph=True)
            D, D_inv = compiled_fn(edges, lmax, gamma)

            # Verify output matches uncompiled version
            D_ref, D_inv_ref = axis_angle_wigner_hybrid(
                edges,
                lmax,
                gamma=gamma,
                coeffs=wigner_data.coeffs,
                U_blocks=wigner_data.U_blocks,
                custom_kernels=wigner_data.custom_kernels,
            )
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
        not hasattr(torch, "_dynamo"), reason="torch.compile not available"
    )
    def test_wigner_d_matrix_real_compiles(
        self, lmax, dtype, device, compile_reset_state
    ):
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
        ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)

        # Full computation
        coeffs_full = precompute_wigner_coefficients(lmax, dtype, device)
        U_blocks_full = precompute_U_blocks_euler_aligned_real(lmax, dtype, device)
        D_re_full, D_im_full = wigner_d_matrix_real(
            ra_re, ra_im, rb_re, rb_im, coeffs_full
        )
        D_real_full = wigner_d_pair_to_real(D_re_full, D_im_full, U_blocks_full, lmax)

        # Range computation
        coeffs_range = precompute_wigner_coefficients(lmax, dtype, device, lmin=lmin)
        full_U_blocks_real = precompute_U_blocks_euler_aligned_real(
            lmax, dtype=dtype, device=device
        )
        U_blocks_range = full_U_blocks_real[lmin:]
        D_re_range, D_im_range = wigner_d_matrix_real(
            ra_re, ra_im, rb_re, rb_im, coeffs_range
        )
        D_real_range = wigner_d_pair_to_real(
            D_re_range, D_im_range, U_blocks_range, lmax, lmin=lmin
        )

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

    @pytest.mark.parametrize(
        "ell,n_samples",
        [
            (2, 500),
            (3, 100),
            (4, 100),
        ],
    )
    def test_kernel_matches_matexp(self, dtype, device, custom_kernels, ell, n_samples):
        """Specialized kernels match matrix exponential method."""
        torch.manual_seed(42)
        q = torch.randn(n_samples, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        # Kernel method
        if ell == 2:
            D_kernel = quaternion_to_wigner_d_l2_einsum(q, custom_kernels.C_l2)
        elif ell == 3:
            D_kernel = quaternion_to_wigner_d_matmul(
                q, 3, custom_kernels.C_l3, custom_kernels.monomials_l3
            )
        else:
            D_kernel = quaternion_to_wigner_d_matmul(
                q, 4, custom_kernels.C_l4, custom_kernels.monomials_l4
            )

        # Matrix exponential method
        axis, angle = quaternion_to_axis_angle(q)
        generators = get_so3_generators(ell, dtype, device)
        K_x, K_y, K_z = (
            generators["K_x"][ell],
            generators["K_y"][ell],
            generators["K_z"][ell],
        )
        K = (
            axis[:, 0:1, None, None] * K_x
            + axis[:, 1:2, None, None] * K_y
            + axis[:, 2:3, None, None] * K_z
        ).squeeze(1)
        D_matexp = torch.linalg.matrix_exp(angle[:, None, None] * K)

        max_err = (D_kernel - D_matexp).abs().max().item()
        assert (
            max_err < _KERNEL_THRESHOLD
        ), f"l={ell} kernel differs from matexp by {max_err}"

    @pytest.mark.parametrize("ell", [2, 3, 4])
    def test_kernel_orthogonality(self, dtype, device, custom_kernels, ell):
        """Specialized kernels produce orthogonal matrices."""
        torch.manual_seed(123)
        q = torch.randn(100, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        if ell == 2:
            D = quaternion_to_wigner_d_l2_einsum(q, custom_kernels.C_l2)
        elif ell == 3:
            D = quaternion_to_wigner_d_matmul(
                q, 3, custom_kernels.C_l3, custom_kernels.monomials_l3
            )
        else:
            D = quaternion_to_wigner_d_matmul(
                q, 4, custom_kernels.C_l4, custom_kernels.monomials_l4
            )

        size = 2 * ell + 1
        I = torch.eye(size, dtype=dtype, device=device)
        orth_err = (D @ D.transpose(-1, -2) - I).abs().max().item()
        assert orth_err < _KERNEL_THRESHOLD, f"l={ell} orthogonality error: {orth_err}"

    @pytest.mark.parametrize("ell", [2, 3, 4])
    def test_kernel_determinant_one(self, dtype, device, custom_kernels, ell):
        """Specialized kernels produce matrices with determinant 1."""
        torch.manual_seed(456)
        q = torch.randn(100, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        if ell == 2:
            D = quaternion_to_wigner_d_l2_einsum(q, custom_kernels.C_l2)
        elif ell == 3:
            D = quaternion_to_wigner_d_matmul(
                q, 3, custom_kernels.C_l3, custom_kernels.monomials_l3
            )
        else:
            D = quaternion_to_wigner_d_matmul(
                q, 4, custom_kernels.C_l4, custom_kernels.monomials_l4
            )

        det_err = (torch.linalg.det(D) - 1.0).abs().max().item()
        assert det_err < _KERNEL_THRESHOLD, f"l={ell} determinant error: {det_err}"


# Threshold for all kernel tests (all kernels achieve ~1e-15 accuracy)
_KERNEL_THRESHOLD = 5e-14


# =============================================================================
# Test Ra/Rb Path (l >= 5)
# =============================================================================

# Edges that exercise the Ra/Rb special cases: near +Y (Rb ≈ 0),
# near -Y (Ra ≈ 0), on coordinate planes, and axis-aligned.
_RARB_TEST_EDGES = [
    ([1.0, 0.0, 0.0], "+X"),
    ([0.0, 1.0, 0.0], "+Y (Rb=0)"),
    ([0.0, -1.0, 0.0], "-Y (Ra=0)"),
    ([0.0, 0.0, 1.0], "+Z"),
    ([0.01, 0.9999, 0.01], "near +Y"),
    ([0.01, -0.9999, 0.01], "near -Y"),
    ([0.866, 0.5, 0.0], "XY plane (ez=0)"),
    ([0.0, 0.5, 0.866], "YZ plane (ex=0)"),
    ([0.866, 0.0, 0.5], "XZ plane (ey=0)"),
    ([0.3, 0.5, 0.8], "off-axis"),
]


class TestRaRbPath:
    """
    Tests for the Ra/Rb polynomial Wigner D path (l >= 5).

    This path uses wigner_d_matrix_real with degree-2l polynomials,
    log/exp of magnitudes, and Horner evaluation. It is exercised
    only when lmax >= 5 in the hybrid pipeline.
    """

    @pytest.fixture()
    def wigner_data_lmax6(self):
        """Create WignerDataModule for lmax=6 tests."""
        return create_wigner_data_module(lmax=6, lmin=5)

    def test_orthogonality(self, dtype, device, wigner_data_lmax6):
        """D matrices from the Ra/Rb path are orthogonal with det=1."""
        edges = torch.tensor(
            [e for e, _ in _RARB_TEST_EDGES], dtype=dtype, device=device
        )
        edges = edges / edges.norm(dim=-1, keepdim=True)
        gamma = torch.zeros(edges.shape[0], dtype=dtype, device=device)
        D, _ = axis_angle_wigner_hybrid(
            edges,
            6,
            gamma=gamma,
            coeffs=wigner_data_lmax6.coeffs,
            U_blocks=wigner_data_lmax6.U_blocks,
            custom_kernels=wigner_data_lmax6.custom_kernels,
        )

        DtD = D @ D.transpose(1, 2)
        eye = torch.eye(D.shape[1], dtype=dtype, device=device)
        ortho_err = (DtD - eye).abs().max().item()
        assert ortho_err < 1e-12, f"Orthogonality error {ortho_err}"

        dets = torch.linalg.det(D)
        det_err = (dets - 1.0).abs().max().item()
        assert det_err < 1e-10, f"Determinant error {det_err}"

    def test_gradient_no_nan(self, dtype, device, wigner_data_lmax6):
        """Gradients through the Ra/Rb path are finite and bounded."""
        edges = torch.tensor(
            [e for e, _ in _RARB_TEST_EDGES], dtype=dtype, device=device
        )
        edges = edges / edges.norm(dim=-1, keepdim=True)
        e = edges.clone().requires_grad_(True)
        gamma = torch.zeros(e.shape[0], dtype=dtype, device=device)
        D, _ = axis_angle_wigner_hybrid(
            e,
            6,
            gamma=gamma,
            coeffs=wigner_data_lmax6.coeffs,
            U_blocks=wigner_data_lmax6.U_blocks,
            custom_kernels=wigner_data_lmax6.custom_kernels,
        )
        D.sum().backward()
        assert not torch.isnan(e.grad).any(), "NaN in gradient"
        assert not torch.isinf(e.grad).any(), "Inf in gradient"
        assert e.grad.abs().max() < 1000, f"Gradient too large: {e.grad.abs().max()}"

    def test_gradient_no_nan_fp32(self, device, wigner_data_lmax6):
        """
        fp32 gradients through the Ra/Rb path are NaN-free.

        The Ra/Rb path internally upcasts to fp64 to avoid overflow in
        degree-2l polynomials. This test verifies the upcast works and
        no NaN leaks through torch.where backward.
        """
        edges = torch.tensor(
            [e for e, _ in _RARB_TEST_EDGES],
            dtype=torch.float32,
            device=device,
        )
        edges = edges / edges.norm(dim=-1, keepdim=True)
        for test_lmax in [5, 6]:
            wd = create_wigner_data_module(lmax=test_lmax, lmin=5)
            e = edges.clone().requires_grad_(True)
            gamma = torch.zeros(e.shape[0], dtype=torch.float32, device=device)
            D, _ = axis_angle_wigner_hybrid(
                e,
                test_lmax,
                gamma=gamma,
                coeffs=wd.coeffs,
                U_blocks=wd.U_blocks,
                custom_kernels=wd.custom_kernels,
            )
            D.sum().backward()
            assert not torch.isnan(
                e.grad
            ).any(), f"NaN in fp32 gradient at lmax={test_lmax}"
            assert not torch.isinf(
                e.grad
            ).any(), f"Inf in fp32 gradient at lmax={test_lmax}"

    def test_matches_matexp(self, dtype, device):
        """Ra/Rb Wigner D matches matrix exponential for l=5 and l=6."""
        torch.manual_seed(42)
        q = torch.randn(100, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)

        for test_lmax in [5, 6]:
            coeffs = precompute_wigner_coefficients(test_lmax, dtype, device)
            U_blocks = precompute_U_blocks_euler_aligned_real(test_lmax, dtype, device)
            D_re, D_im = wigner_d_matrix_real(ra_re, ra_im, rb_re, rb_im, coeffs)
            D_real = wigner_d_pair_to_real(D_re, D_im, U_blocks, test_lmax)

            # Compare against matrix exponential for each l >= 5
            axis, angle = quaternion_to_axis_angle(q)
            generators = get_so3_generators(test_lmax, dtype, device)
            for ell in range(5, test_lmax + 1):
                K_x = generators["K_x"][ell]
                K_y = generators["K_y"][ell]
                K_z = generators["K_z"][ell]
                K = (
                    axis[:, 0:1, None, None] * K_x
                    + axis[:, 1:2, None, None] * K_y
                    + axis[:, 2:3, None, None] * K_z
                ).squeeze(1)
                D_matexp = torch.linalg.matrix_exp(angle[:, None, None] * K)

                offset = ell * ell
                size = 2 * ell + 1
                D_block = D_real[:, offset : offset + size, offset : offset + size]
                max_err = (D_block - D_matexp).abs().max().item()
                assert (
                    max_err < 1e-12
                ), f"l={ell} Ra/Rb differs from matexp by {max_err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
