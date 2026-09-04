"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  Validation that the UMA-S fast-GPU execution backend correctly
        accepts/rejects model configurations, plus unit tests for the
        triton kernels (M_TO_L_GATHER_IDX, node-to-edge / edge-to-node
        Wigner permutations). E2E accuracy tests live in
        run_benchmarks.sh + compare_forces.py, not here.
Models: uma-s-1p1, uma-s-1p2 on the @pretrained-locked load test
        (resolves a registered name to a checkpoint path).
CI:     test_gpu_sweep (units shard).
"""

from __future__ import annotations

import os

import pytest
import torch
from ase.build import bulk

from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name
from fairchem.core.common import device_utils
from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.models.uma.nn.activation import GateActivation
from fairchem.core.models.uma.nn.execution_backends import UMASFastGPUBackend
from fairchem.core.models.uma.triton.constants import M_TO_L_GATHER_IDX
from fairchem.core.models.uma.triton.node_to_edge_wigner_permute import (
    NodeToEdgeWignerPermuteFunction,
)
from fairchem.core.models.uma.triton.permute_wigner_inv_edge_to_node import (
    PermuteWignerInvEdgeToNodeFunction,
)
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from tests.core.models.uma.uma_fast.triton_test_utils import (
    node_to_edge_wigner_permute_launcher,
    permute_wigner_inv_edge_to_node_launcher,
)

# Accelerator these Triton tests run on. Triton itself is portable -- Intel
# ships triton-xpu and the kernels compile and execute there -- so the
# device follows the hardware rather than being pinned to NVIDIA. Numerical
# agreement with the PyTorch reference is what these tests assert.
ACCELERATOR = device_utils.get_available_accelerator() or "cpu"

# L_TO_M_GATHER_IDX is the inverse of M_TO_L_GATHER_IDX - used only in test reference implementations
L_TO_M_GATHER_IDX = [0] * 9
for i, val in enumerate(M_TO_L_GATHER_IDX):
    L_TO_M_GATHER_IDX[val] = i


def _compact_l2_wigner(wigner: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            wigner[:, :1, :1].flatten(1),
            wigner[:, 1:4, 1:4].flatten(1),
            wigner[:, 4:9, 4:9].flatten(1),
        ),
        dim=1,
    )


# =============================================================================
# Tests: Validation Errors
# =============================================================================


def _mock_settings(
    merge_mole: bool = True, activation_checkpointing: bool = False
) -> InferenceSettings:
    """Create mock inference settings for validation tests."""
    return InferenceSettings(
        merge_mole=merge_mole,
        activation_checkpointing=activation_checkpointing,
        external_graph_gen=False,
    )


@pytest.mark.gpu()
def test_umas_fast_gpu_validation_requires_correct_lmax():
    """
    Verify that umas_fast_gpu raises ValueError for incorrect lmax.
    """
    settings = _mock_settings()

    with pytest.raises(ValueError, match="lmax==2 and mmax==2"):
        UMASFastGPUBackend.validate(lmax=3, mmax=2, settings=settings)


@pytest.mark.gpu()
def test_umas_fast_gpu_validation_requires_correct_mmax():
    """
    Verify that umas_fast_gpu raises ValueError for incorrect mmax.
    """
    settings = _mock_settings()

    with pytest.raises(ValueError, match="lmax==2 and mmax==2"):
        UMASFastGPUBackend.validate(lmax=2, mmax=1, settings=settings)


@pytest.mark.gpu()
def test_umas_fast_gpu_validation_accepts_correct_config():
    """
    Verify that umas_fast_gpu validation passes for correct lmax/mmax.
    """
    settings = _mock_settings()

    # Should not raise
    UMASFastGPUBackend.validate(lmax=2, mmax=2, settings=settings)


@pytest.mark.gpu()
def test_umas_fast_gpu_validation_requires_merge_mole():
    """
    Verify that umas_fast_gpu raises ValueError when merge_mole=False.
    """
    settings = _mock_settings(merge_mole=False)  # Wrong - should be True

    with pytest.raises(ValueError, match="merge_mole=True"):
        UMASFastGPUBackend.validate(lmax=2, mmax=2, settings=settings)


@pytest.mark.gpu()
def test_umas_fast_gpu_validation_rejects_hessian_vmap():
    """
    Verify that umas_fast_gpu rejects vectorized Hessian computation.
    """
    settings = _mock_settings()
    settings.predict_untrained_hessian = {"omol"}

    with pytest.raises(ValueError, match="set hessian_vmap=False"):
        UMASFastGPUBackend.validate(lmax=2, mmax=2, settings=settings)


@pytest.mark.gpu()
def test_umas_fast_gpu_validation_accepts_hessian_loop():
    """
    Verify that umas_fast_gpu accepts sequential Hessian computation.
    """
    settings = _mock_settings()
    settings.predict_untrained_hessian = {"omol"}
    settings.hessian_vmap = False

    UMASFastGPUBackend.validate(lmax=2, mmax=2, settings=settings)


@pytest.mark.gpu()
@pytest.mark.parametrize(
    ("channels", "dtype"),
    [(96, torch.float32), (128, torch.bfloat16)],
)
def test_umas_fast_gpu_gate_activation_fallback(channels, dtype):
    torch.manual_seed(42)
    num_edges = 16
    inputs = (
        torch.randn(num_edges, 5 * channels, device=ACCELERATOR, dtype=dtype),
        torch.randn(num_edges, 4 * channels, device=ACCELERATOR, dtype=dtype),
        torch.randn(num_edges, 2 * channels, device=ACCELERATOR, dtype=dtype),
    )
    activation = GateActivation(2, 2, channels, m_prime=True).to(ACCELERATOR)
    gating, x0 = inputs[0].split((2 * channels, 3 * channels), dim=-1)
    expected = activation.forward_m_blocks(gating, (x0, inputs[1], inputs[2]))
    actual = UMASFastGPUBackend.gate_activation(*inputs, channels, activation)

    for actual_block, expected_block in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_block, expected_block, rtol=0, atol=0)


@pytest.mark.gpu()
def test_umas_fast_gpu_gate_activation_fallback_dynamic_compile(
    compile_reset_state,
):
    torch.manual_seed(42)
    channels = 96
    activation = GateActivation(2, 2, channels, m_prime=True).to(ACCELERATOR)

    def fn(x0_full, x1, x2):
        return UMASFastGPUBackend.gate_activation(x0_full, x1, x2, channels, activation)

    compiled = torch.compile(fn, fullgraph=True, dynamic=True)
    for num_edges in (17, 31):
        inputs = (
            torch.randn(
                num_edges, 5 * channels, device=ACCELERATOR, requires_grad=True
            ),
            torch.randn(
                num_edges, 4 * channels, device=ACCELERATOR, requires_grad=True
            ),
            torch.randn(
                num_edges, 2 * channels, device=ACCELERATOR, requires_grad=True
            ),
        )
        reference_inputs = tuple(
            value.detach().clone().requires_grad_() for value in inputs
        )
        actual = compiled(*inputs)
        expected = fn(*reference_inputs)
        grad_outputs = tuple(torch.randn_like(value) for value in actual)
        actual_grads = torch.autograd.grad(actual, inputs, grad_outputs)
        expected_grads = torch.autograd.grad(expected, reference_inputs, grad_outputs)

        for actual_block, expected_block in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_block, expected_block, rtol=1e-6, atol=1e-6
            )
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-6)


@pytest.mark.gpu()
def test_compact_edge_degree_matches_dense():
    torch.manual_seed(42)
    num_edges, channels = 16, 8
    scatter_target = torch.arange(num_edges, device=ACCELERATOR)
    x = torch.randn(
        num_edges,
        9,
        channels,
        device=ACCELERATOR,
        dtype=torch.float64,
        requires_grad=True,
    )
    radial = torch.randn(
        num_edges,
        3 * channels,
        device=ACCELERATOR,
        dtype=torch.float64,
        requires_grad=True,
    )
    dense = torch.zeros(
        num_edges, 9, 9, device=ACCELERATOR, dtype=torch.float64, requires_grad=True
    )
    with torch.no_grad():
        dense[:, 0, 0] = torch.randn(num_edges, device=ACCELERATOR, dtype=torch.float64)
        dense[:, 1:4, 1:4] = torch.randn(
            num_edges, 3, 3, device=ACCELERATOR, dtype=torch.float64
        )
        dense[:, 4:9, 4:9] = torch.randn(
            num_edges, 5, 5, device=ACCELERATOR, dtype=torch.float64
        )
    compact = _compact_l2_wigner(dense.detach()).requires_grad_()
    dense_inputs = (x, radial, dense)
    compact_inputs = tuple(
        value.detach().clone().requires_grad_() for value in (x, radial)
    ) + (compact,)

    def run(inputs):
        return UMASFastGPUBackend.edge_degree_scatter(
            inputs[0], inputs[1], inputs[2], scatter_target, 3, channels, 4.0
        )

    dense_out = run(dense_inputs)
    compact_out = run(compact_inputs)
    grad_out = torch.randn_like(dense_out)
    dense_grads = torch.autograd.grad(dense_out, dense_inputs, grad_out)
    compact_grads = torch.autograd.grad(compact_out, compact_inputs, grad_out)

    torch.testing.assert_close(compact_out, dense_out, rtol=0, atol=0)
    torch.testing.assert_close(compact_grads[0], dense_grads[0], rtol=0, atol=0)
    torch.testing.assert_close(compact_grads[1], dense_grads[1], rtol=2e-15, atol=3e-16)
    torch.testing.assert_close(
        compact_grads[2],
        _compact_l2_wigner(dense_grads[2]),
        rtol=1e-14,
        atol=5e-16,
    )


# =============================================================================
# Tests: E2E Force Correctness
# =============================================================================


@pytest.mark.gpu()
def test_umas_fast_pytorch_forces_match_baseline_pbc(
    conserving_mole_checkpoint, fake_uma_dataset
):
    """
    E2E test: verify umas_fast_pytorch produces forces matching general backend.

    Uses PBC system from fake_uma_dataset (oc20, 5-20 atoms).
    """
    checkpoint_pt, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})
    atoms = db.get_atoms(0)  # PBC system

    # Build batch
    sample = AtomicData.from_ase(
        atoms,
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    sample["dataset"] = "oc20"
    batch = data_list_collater([sample], otf_graph=True)

    # Baseline (general backend)
    baseline_settings = InferenceSettings(
        activation_checkpointing=False,
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="general",
    )
    baseline_predictor = MLIPPredictUnit(
        checkpoint_pt, ACCELERATOR, inference_settings=baseline_settings
    )

    # Test (umas_fast_pytorch backend)
    test_settings = InferenceSettings(
        activation_checkpointing=False,
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="umas_fast_pytorch",
    )
    test_predictor = MLIPPredictUnit(
        checkpoint_pt, ACCELERATOR, inference_settings=test_settings
    )

    # Compare
    baseline_out = baseline_predictor.predict(batch.clone())
    test_out = test_predictor.predict(batch.clone())

    # Forces should match within tolerance (backend precision difference)
    assert torch.allclose(
        baseline_out["forces"], test_out["forces"], rtol=5e-4, atol=5e-5
    ), f"Force mismatch: max diff = {(baseline_out['forces'] - test_out['forces']).abs().max()}"
    assert torch.allclose(
        baseline_out["energy"], test_out["energy"], rtol=5e-4, atol=5e-5
    ), f"Energy mismatch: {baseline_out['energy']} vs {test_out['energy']}"


@pytest.mark.gpu()
def test_umas_fast_pytorch_forces_match_baseline_no_pbc(
    conserving_mole_checkpoint, fake_uma_dataset
):
    """
    E2E test: verify umas_fast_pytorch produces forces matching general backend.

    Uses non-PBC system from fake_uma_dataset (omol, 2-5 atoms).
    """
    checkpoint_pt, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "omol")})
    atoms = db.get_atoms(0)  # Non-PBC molecule
    atoms.pbc = [False, False, False]

    # Build batch
    sample = AtomicData.from_ase(
        atoms,
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    sample["dataset"] = "omol"
    batch = data_list_collater([sample], otf_graph=True)

    # Baseline (general backend)
    baseline_settings = InferenceSettings(
        activation_checkpointing=False,
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="general",
    )
    baseline_predictor = MLIPPredictUnit(
        checkpoint_pt, ACCELERATOR, inference_settings=baseline_settings
    )

    # Test (umas_fast_pytorch backend)
    test_settings = InferenceSettings(
        activation_checkpointing=False,
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="umas_fast_pytorch",
    )
    test_predictor = MLIPPredictUnit(
        checkpoint_pt, ACCELERATOR, inference_settings=test_settings
    )

    # Compare
    baseline_out = baseline_predictor.predict(batch.clone())
    test_out = test_predictor.predict(batch.clone())

    # Forces should match within tolerance (backend precision difference)
    assert torch.allclose(
        baseline_out["forces"], test_out["forces"], rtol=5e-4, atol=5e-5
    ), f"Force mismatch: max diff = {(baseline_out['forces'] - test_out['forces']).abs().max()}"
    assert torch.allclose(
        baseline_out["energy"], test_out["energy"], rtol=5e-4, atol=5e-5
    ), f"Energy mismatch: {baseline_out['energy']} vs {test_out['energy']}"


# =============================================================================
# Tests: Triton Autograd Gradcheck
# =============================================================================


@pytest.mark.gpu()
@pytest.mark.parametrize("sphere_channels", [128, 512])
def test_node_to_edge_wigner_permute_gradcheck(sphere_channels):
    """
    Verify NodeToEdgeWignerPermuteFunction backward pass is correct via gradcheck.

    Uses fast_mode=True for statistical gradient validation (random projections)
    instead of full Jacobian computation to avoid OOM.
    """
    torch.manual_seed(42)
    device = ACCELERATOR
    num_nodes = 8
    num_edges = 16

    # Create test inputs
    x = torch.randn(
        num_nodes, 9, sphere_channels, device=device, dtype=torch.float64
    ).requires_grad_(True)
    wigner = torch.randn(
        num_edges, 9, 9, device=device, dtype=torch.float64
    ).requires_grad_(True)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_tgt = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_index = torch.stack([edge_src, edge_tgt], dim=0)

    # Gradcheck with fast_mode to avoid full Jacobian OOM
    assert torch.autograd.gradcheck(
        lambda x_in, w_in: NodeToEdgeWignerPermuteFunction.apply(
            x_in, edge_index, w_in
        ),
        (x, wigner),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        fast_mode=True,
    )


@pytest.mark.gpu()
@pytest.mark.parametrize("sphere_channels", [128, 512])
def test_permute_wigner_inv_edge_to_node_gradcheck(sphere_channels):
    """
    Verify PermuteWignerInvEdgeToNodeFunction backward pass is correct via gradcheck.

    Uses fast_mode=True for statistical gradient validation (random projections)
    instead of full Jacobian computation to avoid OOM.
    """
    torch.manual_seed(42)
    device = ACCELERATOR
    num_edges = 16

    # Create test inputs
    x = torch.randn(
        num_edges, 9, sphere_channels, device=device, dtype=torch.float64
    ).requires_grad_(True)
    wigner = torch.randn(
        num_edges, 9, 9, device=device, dtype=torch.float64
    ).requires_grad_(True)

    # Gradcheck with fast_mode to avoid full Jacobian OOM
    assert torch.autograd.gradcheck(
        lambda x_in, w_in: PermuteWignerInvEdgeToNodeFunction.apply(x_in, w_in),
        (x, wigner),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        fast_mode=True,
    )


# =============================================================================
# Tests: Triton Kernel vs PyTorch Reference
# =============================================================================


def _ref_node_to_edge_wigner_permute(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch reference for node_to_edge_wigner_permute.

    Args:
        x: Node features [N, 9, C] in L-major order
        edge_index: [2, E]
        wigner: [E, 9, 9]

    Returns:
        out: [E, 9, 2C] in M-major order (rotated src||tgt)
    """
    # Gather
    x_src = x[edge_index[0]]  # [E, 9, C]
    x_tgt = x[edge_index[1]]  # [E, 9, C]

    # Wigner rotation (on L-order data): [E, 9, 9] @ [E, 9, C] -> [E, 9, C]
    rot_src = torch.bmm(wigner, x_src)
    rot_tgt = torch.bmm(wigner, x_tgt)

    # L->M permutation on output
    rot_src_m = rot_src[:, L_TO_M_GATHER_IDX, :]
    rot_tgt_m = rot_tgt[:, L_TO_M_GATHER_IDX, :]

    # Concat along channel dim
    return torch.cat([rot_src_m, rot_tgt_m], dim=-1)


def _ref_permute_wigner_inv(
    x: torch.Tensor,
    wigner_inv: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch reference for permute_wigner_inv_edge_to_node.

    Args:
        x: Edge features [E, 9, C] in M-major order
        wigner_inv: [E, 9, 9]

    Returns:
        out: [E, 9, C] in L-major order
    """
    # M->L permutation first (inverse of the L->M gather in forward)
    x_l = x[:, M_TO_L_GATHER_IDX, :]

    # Wigner inverse rotation
    return torch.bmm(wigner_inv, x_l)


def _create_block_diagonal_wigner(num_edges: int, device: str) -> torch.Tensor:
    """
    Create block-diagonal Wigner matrix [E, 9, 9].

    Structure: L=0 (1x1), L=1 (3x3), L=2 (5x5)
    """
    wigner = torch.zeros(num_edges, 9, 9, device=device)
    # L=0 block: [0, 0]
    wigner[:, 0, 0] = torch.randn(num_edges, device=device)
    # L=1 block: [1:4, 1:4]
    wigner[:, 1:4, 1:4] = torch.randn(num_edges, 3, 3, device=device)
    # L=2 block: [4:9, 4:9]
    wigner[:, 4:9, 4:9] = torch.randn(num_edges, 5, 5, device=device)
    return wigner


@pytest.mark.gpu()
@pytest.mark.parametrize("sphere_channels", [128, 512])
def test_node_to_edge_wigner_permute_matches_pytorch(sphere_channels):
    """
    Verify Triton kernel output matches PyTorch reference.
    """
    torch.manual_seed(42)
    device = ACCELERATOR
    num_nodes = 16
    num_edges = 32

    # Create inputs
    x = torch.randn(num_nodes, 9, sphere_channels, device=device)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_tgt = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_index = torch.stack([edge_src, edge_tgt], dim=0)
    wigner = _create_block_diagonal_wigner(num_edges, device)

    # PyTorch reference
    ref_out = _ref_node_to_edge_wigner_permute(x, edge_index, wigner)

    # Triton kernel
    triton_out, _ = node_to_edge_wigner_permute_launcher(x, edge_index, wigner)

    # Compare
    assert torch.allclose(
        ref_out, triton_out, rtol=1e-4, atol=1e-4
    ), f"Max diff: {(ref_out - triton_out).abs().max()}"


@pytest.mark.gpu()
@pytest.mark.parametrize("sphere_channels", [128, 512])
def test_permute_wigner_inv_matches_pytorch(sphere_channels):
    """
    Verify Triton kernel output matches PyTorch reference.
    """
    torch.manual_seed(42)
    device = ACCELERATOR
    num_edges = 32

    # Create inputs
    x = torch.randn(num_edges, 9, sphere_channels, device=device)
    wigner_inv = _create_block_diagonal_wigner(num_edges, device)

    # PyTorch reference
    ref_out = _ref_permute_wigner_inv(x, wigner_inv)

    # Triton kernel
    triton_out, _ = permute_wigner_inv_edge_to_node_launcher(x, wigner_inv)

    # Compare
    assert torch.allclose(
        ref_out, triton_out, rtol=1e-4, atol=1e-4
    ), f"Max diff: {(ref_out - triton_out).abs().max()}"


@pytest.mark.gpu()
def test_legacy_backend_rotations_accept_compact_wigner():
    torch.manual_seed(42)
    num_nodes, num_edges, channels = 16, 32, 128
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=ACCELERATOR)
    dense = _create_block_diagonal_wigner(num_edges, ACCELERATOR)
    compact = _compact_l2_wigner(dense)
    nodes = torch.randn(num_nodes, 9, channels, device=ACCELERATOR)
    edges = torch.randn(num_edges, 9, channels, device=ACCELERATOR)
    scatter_target = torch.arange(num_edges, device=ACCELERATOR)

    dense_node_to_edge = UMASFastGPUBackend.node_to_edge_wigner_permute(
        nodes, edge_index, dense
    )
    compact_node_to_edge = UMASFastGPUBackend.node_to_edge_wigner_permute(
        nodes, edge_index, compact
    )
    dense_edge_to_node = UMASFastGPUBackend.permute_wigner_inv_edge_to_node(
        edges, dense, scatter_target, num_edges
    )
    compact_edge_to_node = UMASFastGPUBackend.permute_wigner_inv_edge_to_node(
        edges, compact, scatter_target, num_edges
    )

    torch.testing.assert_close(compact_node_to_edge, dense_node_to_edge)
    torch.testing.assert_close(compact_edge_to_node, dense_edge_to_node)


@pytest.mark.gpu()
@pytest.mark.parametrize("sphere_channels", [128, 256, 512])
def test_permute_wigner_inv_bwd_dw_matches_pytorch(sphere_channels):
    """
    Verify permute_wigner_inv backward dW kernel matches PyTorch reference.

    Tests that dW = grad_out @ x_l^T is computed correctly over ALL channels.
    Regression test for a bug where channels > 128 were silently dropped.
    """
    torch.manual_seed(42)
    device = ACCELERATOR
    num_edges = 32

    # Create inputs (L-major for grad_out, L-major for x_l)
    grad_out = torch.randn(num_edges, 9, sphere_channels, device=device)
    x_l = torch.randn(num_edges, 9, sphere_channels, device=device)

    # PyTorch reference: block-diagonal outer product
    ref_dw = torch.zeros(num_edges, 9, 9, device=device)
    # L=0 block (1x1)
    ref_dw[:, 0, 0] = (grad_out[:, 0, :] * x_l[:, 0, :]).sum(dim=-1)
    # L=1 block (3x3)
    ref_dw[:, 1:4, 1:4] = torch.bmm(grad_out[:, 1:4, :], x_l[:, 1:4, :].transpose(1, 2))
    # L=2 block (5x5)
    ref_dw[:, 4:9, 4:9] = torch.bmm(grad_out[:, 4:9, :], x_l[:, 4:9, :].transpose(1, 2))

    # Triton kernel via custom op
    import fairchem.core.models.uma.triton.custom_ops  # noqa: F401

    grad_wigner_flat = torch.zeros(num_edges, 81, device=device)
    torch.ops.fairchem._kernel_permute_wigner_inv_edge_to_node_bwd_dw(
        grad_out, x_l, grad_wigner_flat
    )
    triton_dw = grad_wigner_flat.reshape(num_edges, 9, 9)

    # Compare — tolerance should be tight (numerical precision only)
    assert torch.allclose(ref_dw, triton_dw, rtol=1e-4, atol=1e-4), (
        f"permute_wigner_inv bwd_dw mismatch at sphere_channels={sphere_channels}: "
        f"max abs diff={( ref_dw - triton_dw).abs().max().item():.6e}, "
        f"ref norm={ref_dw.norm().item():.4f}, "
        f"triton norm={triton_dw.norm().item():.4f}"
    )


# =============================================================================
# Tests: E2E umas_fast_gpu Backend
# =============================================================================


@pytest.mark.gpu()
def test_umas_fast_gpu_forces_match_baseline_pbc(
    conserving_mole_checkpoint, fake_uma_dataset
):
    """
    E2E test: verify umas_fast_gpu produces forces matching general backend.

    Uses PBC system from fake_uma_dataset (oc20, 5-20 atoms).
    """
    checkpoint_pt, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})
    atoms = db.get_atoms(0)  # PBC system

    # Build batch
    sample = AtomicData.from_ase(
        atoms,
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    sample["dataset"] = "oc20"
    batch = data_list_collater([sample], otf_graph=True)

    # Baseline (general backend)
    baseline_settings = InferenceSettings(
        activation_checkpointing=False,
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="general",
    )
    baseline_predictor = MLIPPredictUnit(
        checkpoint_pt, ACCELERATOR, inference_settings=baseline_settings
    )

    # Test (umas_fast_gpu backend)
    test_settings = InferenceSettings(
        activation_checkpointing=False,
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="umas_fast_gpu",
    )
    test_predictor = MLIPPredictUnit(
        checkpoint_pt, ACCELERATOR, inference_settings=test_settings
    )

    # Compare
    baseline_out = baseline_predictor.predict(batch.clone())
    test_out = test_predictor.predict(batch.clone())

    # Forces should match within tolerance (backend precision difference)
    assert torch.allclose(
        baseline_out["forces"], test_out["forces"], rtol=5e-4, atol=5e-5
    ), f"Force mismatch: max diff = {(baseline_out['forces'] - test_out['forces']).abs().max()}"
    assert torch.allclose(
        baseline_out["energy"], test_out["energy"], rtol=5e-4, atol=5e-5
    ), f"Energy mismatch: {baseline_out['energy']} vs {test_out['energy']}"
    assert torch.allclose(
        baseline_out["stress"], test_out["stress"], rtol=5e-4, atol=5e-5
    ), f"Stress mismatch: max diff = {(baseline_out['stress'] - test_out['stress']).abs().max()}"


@pytest.mark.gpu()
def test_umas_fast_gpu_forces_match_baseline_no_pbc(
    conserving_mole_checkpoint, fake_uma_dataset
):
    """
    E2E test: verify umas_fast_gpu produces forces matching general backend.

    Uses non-PBC system from fake_uma_dataset (omol, 2-5 atoms).
    """
    checkpoint_pt, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "omol")})
    atoms = db.get_atoms(0)  # Non-PBC molecule
    atoms.pbc = [False, False, False]

    # Build batch
    sample = AtomicData.from_ase(
        atoms,
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    sample["dataset"] = "omol"
    batch = data_list_collater([sample], otf_graph=True)

    # Baseline (general backend)
    baseline_settings = InferenceSettings(
        activation_checkpointing=False,
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="general",
    )
    baseline_predictor = MLIPPredictUnit(
        checkpoint_pt, ACCELERATOR, inference_settings=baseline_settings
    )

    # Test (umas_fast_gpu backend)
    test_settings = InferenceSettings(
        activation_checkpointing=False,
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="umas_fast_gpu",
    )
    test_predictor = MLIPPredictUnit(
        checkpoint_pt, ACCELERATOR, inference_settings=test_settings
    )

    # Compare
    baseline_out = baseline_predictor.predict(batch.clone())
    test_out = test_predictor.predict(batch.clone())

    # Forces should match within tolerance (backend precision difference)
    assert torch.allclose(
        baseline_out["forces"], test_out["forces"], rtol=5e-4, atol=5e-5
    ), f"Force mismatch: max diff = {(baseline_out['forces'] - test_out['forces']).abs().max()}"
    assert torch.allclose(
        baseline_out["energy"], test_out["energy"], rtol=5e-4, atol=5e-5
    ), f"Energy mismatch: {baseline_out['energy']} vs {test_out['energy']}"


# =============================================================================
# Tests: Compiled Backend E2E with Pretrained Models
# =============================================================================


@pytest.mark.gpu()
@pytest.mark.compile_gpu()
@pytest.mark.pretrained("uma-s-1p1", "uma-s-1p2")
def test_compiled_backends_match_baseline(pretrained_model_name, compile_reset_state):
    """
    Test compiled execution modes produce same results as non-compiled baseline.

    Tests:
    - general compiled vs general non-compiled
    - umas_fast_gpu compiled vs general non-compiled

    Uses pretrained checkpoints (cached by HuggingFace Hub) — or a direct
    filesystem path if --sweep-model is set to one.
    """
    # Resolve to a checkpoint file: accept either a registered model name
    # or an already-on-disk path.
    if os.path.exists(pretrained_model_name):
        checkpoint_pt = pretrained_model_name
    else:
        checkpoint_pt = pretrained_checkpoint_path_from_name(pretrained_model_name)

    # Create test system (32-atom Cu FCC)
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    sample = AtomicData.from_ase(atoms, task_name="omat")
    batch = data_list_collater([sample], otf_graph=True)

    # Compute baseline ONCE (general, non-compiled)
    baseline_settings = InferenceSettings(
        activation_checkpointing=False,
        merge_mole=True,
        external_graph_gen=False,
        execution_mode="general",
        compile=False,
    )
    baseline_predictor = MLIPPredictUnit(
        checkpoint_pt, ACCELERATOR, inference_settings=baseline_settings
    )
    baseline_out = baseline_predictor.predict(batch.clone())

    # Test configurations: (execution_mode, compile)
    test_configs = [
        ("general", True),
        ("umas_fast_gpu", True),
    ]

    for test_mode, test_compile in test_configs:
        test_settings = InferenceSettings(
            activation_checkpointing=False,
            merge_mole=True,
            external_graph_gen=False,
            execution_mode=test_mode,
            compile=test_compile,
        )
        test_predictor = MLIPPredictUnit(
            checkpoint_pt, ACCELERATOR, inference_settings=test_settings
        )
        test_out = test_predictor.predict(batch.clone())

        # Force comparison
        assert torch.allclose(
            baseline_out["forces"], test_out["forces"], rtol=5e-4, atol=5e-5
        ), (
            f"{pretrained_model_name} {test_mode} compile={test_compile}: "
            f"force mismatch max diff = {(baseline_out['forces'] - test_out['forces']).abs().max()}"
        )
        # Energy comparison
        assert torch.allclose(
            baseline_out["energy"], test_out["energy"], rtol=5e-4, atol=5e-5
        ), (
            f"{pretrained_model_name} {test_mode} compile={test_compile}: "
            f"energy mismatch {baseline_out['energy']} vs {test_out['energy']}"
        )
