"""
Tests for Wigner D matrix computation.

Tests verify:
1. Mathematical correctness (orthogonality, determinant, edge -> +Y)
2. Agreement between all entry point functions
3. Agreement with Euler-based rotation.py
4. Gradient stability
5. torch.compile compatibility
6. Range functions (lmin support)
7. Specialized kernels (l=2 polynomial, l=3/4 matmul)

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fairchem.core.models.uma.common.quaternion_wigner_utils import (
    get_ra_rb_coefficients_real,
    get_so3_generators,
    precompute_U_blocks_euler_aligned_real,
    precompute_wigner_coefficients,
    quaternion_to_axis_angle,
    quaternion_to_ra_rb_real,
    wigner_d_matrix_real,
    wigner_d_pair_to_real,
)
from fairchem.core.models.uma.common.rotation import (
    init_edge_rot_euler_angles,
    wigner_D,
)
from fairchem.core.models.uma.common.wigner_d_custom_kernels import (
    preload_kernel_caches,
    quaternion_to_wigner_d_l2_einsum,
    quaternion_to_wigner_d_matmul,
)
from fairchem.core.models.uma.common.wigner_d_hybrid import (
    axis_angle_wigner_hybrid,
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


@pytest.fixture()
def preload_kernel_coefficients(dtype, device):
    """Pre-load kernel coefficients to avoid torch.load during torch.compile.

    torch.compile cannot trace through torch.load, so we must ensure all
    coefficient caches are populated before running compiled functions.
    """
    preload_kernel_caches(dtype, device)


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

        assert torch.allclose(
            result, y_axis, atol=1e-5
        ), f"Edge {edge} did not map to +Y, got {result}"

    @pytest.mark.parametrize("edge,desc", STANDARD_TEST_EDGES)
    def test_y_to_edge(self, lmax, dtype, device, edge, desc):
        """D_inv l=1 block rotates +Y -> edge (inverse of edge -> +Y)."""
        edge_t = torch.tensor([edge], dtype=dtype, device=device)
        edge_t = torch.nn.functional.normalize(edge_t, dim=-1)
        gamma = torch.zeros(1, dtype=dtype, device=device)

        _, D_inv = axis_angle_wigner_hybrid(edge_t, lmax, gamma=gamma)
        D_inv_l1 = D_inv[0, 1:4, 1:4]

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        result = D_inv_l1 @ y_axis

        assert torch.allclose(
            result, edge_t[0], atol=1e-5
        ), f"+Y did not map to edge {edge}, got {result}"

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
            edge_composed,
            lmax,
            gamma=torch.zeros(n_samples, dtype=dtype, device=device),
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
            edge_composed, lmax, gamma=gamma_composed
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

    def test_matches_euler_code(self, lmax, dtype, device, Jd_matrices):
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
            D_axis, _ = axis_angle_wigner_hybrid(edge_t, lmax, use_euler_gamma=True)

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

    def test_blend_region_matches_euler(self, lmax, dtype, device, Jd_matrices):
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
            D_hybrid, _ = axis_angle_wigner_hybrid(edge_t, lmax, gamma=gamma_t)

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
        assert (
            grad.abs().max() < 1000
        ), f"Gradient too large for {desc}: {grad.abs().max()}"

    @pytest.mark.parametrize("epsilon", [1e-4, 1e-6, 1e-8])
    def test_near_singularity_correctness(self, lmax, dtype, device, epsilon):
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

            D, _ = axis_angle_wigner_hybrid(edge_norm, lmax)
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
    def test_hybrid_compiles(self, lmax, dtype, device, preload_kernel_coefficients):
        """axis_angle_wigner_hybrid should compile without graph breaks."""
        import torch._dynamo as dynamo

        edges = torch.randn(10, 3, dtype=dtype, device=device)
        gamma = torch.rand(10, dtype=dtype, device=device) * 6.28

        # Define function to compile
        def fn(edge_vec, lmax_val, g):
            return axis_angle_wigner_hybrid(edge_vec, lmax_val, gamma=g)

        # Try to compile and run
        try:
            compiled_fn = torch.compile(fn, fullgraph=True)
            D, D_inv = compiled_fn(edges, lmax, gamma)

            # Verify output matches uncompiled version
            D_ref, D_inv_ref = axis_angle_wigner_hybrid(edges, lmax, gamma=gamma)
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
        ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)

        # Full computation
        coeffs_full = precompute_wigner_coefficients(lmax, dtype, device)
        U_blocks_full = precompute_U_blocks_euler_aligned_real(lmax, dtype, device)
        D_re_full, D_im_full = wigner_d_matrix_real(
            ra_re, ra_im, rb_re, rb_im, coeffs_full
        )
        D_real_full = wigner_d_pair_to_real(D_re_full, D_im_full, U_blocks_full, lmax)

        # Range computation
        coeffs_range, U_blocks_range = get_ra_rb_coefficients_real(
            lmax, dtype, device, lmin=lmin
        )
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

# Kernel test configuration: (ell, kernel_fn)
_KERNEL_TEST_PARAMS = [
    (2, quaternion_to_wigner_d_l2_einsum),
    (3, lambda q: quaternion_to_wigner_d_matmul(q, 3)),
    (4, lambda q: quaternion_to_wigner_d_matmul(q, 4)),
]

# Threshold for all kernel tests (all kernels achieve ~1e-15 accuracy)
_KERNEL_THRESHOLD = 5e-14


class TestSpecializedKernels:
    """Tests for the specialized l=2 polynomial and l=3/4 matmul kernels."""

    @pytest.mark.parametrize(
        "ell,kernel_fn,n_samples",
        [
            (2, quaternion_to_wigner_d_l2_einsum, 500),
            (3, lambda q: quaternion_to_wigner_d_matmul(q, 3), 100),
            (4, lambda q: quaternion_to_wigner_d_matmul(q, 4), 100),
        ],
    )
    def test_kernel_matches_matexp(self, dtype, device, ell, kernel_fn, n_samples):
        """Specialized kernels match matrix exponential method."""
        torch.manual_seed(42)
        q = torch.randn(n_samples, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        # Kernel method
        D_kernel = kernel_fn(q)

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

    @pytest.mark.parametrize("ell,kernel_fn", _KERNEL_TEST_PARAMS)
    def test_kernel_orthogonality(self, dtype, device, ell, kernel_fn):
        """Specialized kernels produce orthogonal matrices."""
        torch.manual_seed(123)
        q = torch.randn(100, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        D = kernel_fn(q)
        size = 2 * ell + 1
        I = torch.eye(size, dtype=dtype, device=device)
        orth_err = (D @ D.transpose(-1, -2) - I).abs().max().item()
        assert orth_err < _KERNEL_THRESHOLD, f"l={ell} orthogonality error: {orth_err}"

    @pytest.mark.parametrize("ell,kernel_fn", _KERNEL_TEST_PARAMS)
    def test_kernel_determinant_one(self, dtype, device, ell, kernel_fn):
        """Specialized kernels produce matrices with determinant 1."""
        torch.manual_seed(456)
        q = torch.randn(100, 4, dtype=dtype, device=device)
        q = q / q.norm(dim=-1, keepdim=True)

        D = kernel_fn(q)
        det_err = (torch.linalg.det(D) - 1.0).abs().max().item()
        assert det_err < _KERNEL_THRESHOLD, f"l={ell} determinant error: {det_err}"


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

    def test_orthogonality(self, dtype, device):
        """
        D matrices from the Ra/Rb path are orthogonal with det=1.
        """
        edges = torch.tensor(
            [e for e, _ in _RARB_TEST_EDGES], dtype=dtype, device=device
        )
        edges = edges / edges.norm(dim=-1, keepdim=True)
        gamma = torch.zeros(edges.shape[0], dtype=dtype, device=device)
        D, _ = axis_angle_wigner_hybrid(edges, 6, gamma=gamma)

        DtD = D @ D.transpose(1, 2)
        eye = torch.eye(D.shape[1], dtype=dtype, device=device)
        ortho_err = (DtD - eye).abs().max().item()
        assert ortho_err < 1e-12, f"Orthogonality error {ortho_err}"

        dets = torch.linalg.det(D)
        det_err = (dets - 1.0).abs().max().item()
        assert det_err < 1e-10, f"Determinant error {det_err}"

    def test_gradient_no_nan(self, dtype, device):
        """
        Gradients through the Ra/Rb path are finite and bounded.
        """
        edges = torch.tensor(
            [e for e, _ in _RARB_TEST_EDGES], dtype=dtype, device=device
        )
        edges = edges / edges.norm(dim=-1, keepdim=True)
        e = edges.clone().requires_grad_(True)
        gamma = torch.zeros(e.shape[0], dtype=dtype, device=device)
        D, _ = axis_angle_wigner_hybrid(e, 6, gamma=gamma)
        D.sum().backward()
        assert not torch.isnan(e.grad).any(), "NaN in gradient"
        assert not torch.isinf(e.grad).any(), "Inf in gradient"
        assert e.grad.abs().max() < 1000, f"Gradient too large: {e.grad.abs().max()}"

    def test_gradient_no_nan_fp32(self, device):
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
            e = edges.clone().requires_grad_(True)
            gamma = torch.zeros(e.shape[0], dtype=torch.float32, device=device)
            D, _ = axis_angle_wigner_hybrid(e, test_lmax, gamma=gamma)
            D.sum().backward()
            assert not torch.isnan(
                e.grad
            ).any(), f"NaN in fp32 gradient at lmax={test_lmax}"
            assert not torch.isinf(
                e.grad
            ).any(), f"Inf in fp32 gradient at lmax={test_lmax}"

    def test_matches_matexp(self, dtype, device):
        """
        Ra/Rb Wigner D matches matrix exponential for l=5 and l=6.
        """
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
