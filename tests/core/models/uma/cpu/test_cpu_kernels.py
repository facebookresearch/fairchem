"""
Tests for umas_fast_cpu C++ kernels.

Verifies forward correctness against PyTorch reference and
backward correctness via torch.autograd.gradcheck.
"""

import pytest
import torch

# C++ kernels are float32 only; use float32 with relaxed tolerance
DTYPE = torch.float32


def _make_block_diag_wigner(E, dtype=DTYPE):
    """Create a proper block-diagonal Wigner matrix [E, 9, 9]."""
    W = torch.zeros(E, 9, 9, dtype=dtype)
    W[:, 0, 0] = torch.randn(E, dtype=dtype)
    W[:, 1:4, 1:4] = torch.randn(E, 3, 3, dtype=dtype)
    W[:, 4:9, 4:9] = torch.randn(E, 5, 5, dtype=dtype)
    return W


def _pytorch_ref_n2e(x, edge_index, wigner):
    """PyTorch reference for node_to_edge_wigner_permute."""
    L_TO_M = [0, 2, 6, 3, 7, 1, 5, 8, 4]
    E = edge_index.shape[1]
    xs = x[edge_index[0]]
    xt = x[edge_index[1]]
    xcat = torch.cat((xs, xt), dim=2)
    W = wigner.reshape(E, 9, 9)
    rot = torch.empty_like(xcat)
    rot[:, 0:1, :] = W[:, 0:1, 0:1] * xcat[:, 0:1, :]
    rot[:, 1:4, :] = torch.bmm(W[:, 1:4, 1:4], xcat[:, 1:4, :])
    rot[:, 4:9, :] = torch.bmm(W[:, 4:9, 4:9], xcat[:, 4:9, :])
    return rot[:, L_TO_M, :]


def _pytorch_ref_pwi(x, wigner_inv):
    """PyTorch reference for permute_wigner_inv."""
    M_TO_L = [0, 5, 1, 3, 8, 6, 2, 4, 7]
    E = x.shape[0]
    xl = x[:, M_TO_L, :]
    W = wigner_inv.reshape(E, 9, 9)
    out = torch.empty_like(xl)
    out[:, 0:1, :] = W[:, 0:1, 0:1] * xl[:, 0:1, :]
    out[:, 1:4, :] = torch.bmm(W[:, 1:4, 1:4], xl[:, 1:4, :])
    out[:, 4:9, :] = torch.bmm(W[:, 4:9, 4:9], xl[:, 4:9, :])
    return out


class TestNodeToEdgeWignerPermute:
    """Test CPUNodeToEdgeWignerPermuteFunction."""

    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        N, E, C = 10, 30, 16
        x = torch.randn(N, 9, C, dtype=DTYPE, requires_grad=True)
        edge_index = torch.stack([
            torch.randint(0, N, (E,)),
            torch.randint(0, N, (E,)),
        ]).to(torch.int64)
        wigner = _make_block_diag_wigner(E).requires_grad_(True)
        return x, edge_index, wigner

    def test_forward_matches_reference(self, setup):
        from fairchem.core.models.uma.cpu.ops import CPUNodeToEdgeWignerPermuteFunction
        x, edge_index, wigner = setup

        ref = _pytorch_ref_n2e(x, edge_index, wigner)
        cpp = CPUNodeToEdgeWignerPermuteFunction.apply(x, edge_index, wigner)
        assert torch.allclose(ref, cpp, atol=1e-5), \
            f"Forward mismatch: max diff = {(ref - cpp).abs().max()}"

    def test_backward_grad_x(self, setup):
        """Verify grad_x matches numerical gradient."""
        from fairchem.core.models.uma.cpu.ops import CPUNodeToEdgeWignerPermuteFunction
        x, edge_index, wigner = setup

        out = CPUNodeToEdgeWignerPermuteFunction.apply(x, edge_index, wigner)
        loss = out.sum()
        loss.backward()

        # Numerical gradient for x
        eps = 1e-4
        grad_x_num = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x_plus = x.clone(); x_plus[i, j, k] += eps
                    x_minus = x.clone(); x_minus[i, j, k] -= eps
                    f_plus = CPUNodeToEdgeWignerPermuteFunction.apply(x_plus, edge_index, wigner).sum()
                    f_minus = CPUNodeToEdgeWignerPermuteFunction.apply(x_minus, edge_index, wigner).sum()
                    grad_x_num[i, j, k] = (f_plus - f_minus) / (2 * eps)

        assert torch.allclose(x.grad, grad_x_num, atol=1e-3, rtol=1e-3), \
            f"grad_x mismatch: max diff = {(x.grad - grad_x_num).abs().max()}"

    def test_backward_grad_wigner(self, setup):
        """Verify grad_wigner matches numerical gradient (critical for forces)."""
        from fairchem.core.models.uma.cpu.ops import CPUNodeToEdgeWignerPermuteFunction
        x, edge_index, wigner = setup

        out = CPUNodeToEdgeWignerPermuteFunction.apply(x, edge_index, wigner)
        loss = out.sum()
        loss.backward()

        # Numerical gradient for a few wigner elements
        eps = 1e-4
        # Check first edge, block-diagonal elements
        for (i, j) in [(0, 0), (1, 1), (4, 4), (4, 5)]:
            w_plus = wigner.clone(); w_plus[0, i, j] += eps
            w_minus = wigner.clone(); w_minus[0, i, j] -= eps
            f_plus = CPUNodeToEdgeWignerPermuteFunction.apply(x, edge_index, w_plus).sum()
            f_minus = CPUNodeToEdgeWignerPermuteFunction.apply(x, edge_index, w_minus).sum()
            num_grad = (f_plus - f_minus) / (2 * eps)
            ana_grad = wigner.grad[0, i, j]
            assert abs(ana_grad - num_grad) < 1e-2, \
                f"Wigner grad mismatch at [{i},{j}]: analytic={ana_grad:.6f} numerical={num_grad:.6f}"

    def test_wigner_grad_is_nonzero(self, setup):
        """Verify wigner gradient is computed (needed for forces)."""
        from fairchem.core.models.uma.cpu.ops import CPUNodeToEdgeWignerPermuteFunction
        x, edge_index, wigner = setup

        out = CPUNodeToEdgeWignerPermuteFunction.apply(x, edge_index, wigner)
        loss = out.sum()
        loss.backward()

        assert wigner.grad is not None, "Wigner gradient is None!"
        assert wigner.grad.abs().sum() > 0, "Wigner gradient is all zeros!"


class TestPermuteWignerInv:
    """Test CPUPermuteWignerInvEdgeToNodeFunction."""

    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        E, C = 30, 16
        x = torch.randn(E, 9, C, dtype=DTYPE, requires_grad=True)
        wigner_inv = _make_block_diag_wigner(E).requires_grad_(True)
        return x, wigner_inv

    def test_forward_matches_reference(self, setup):
        from fairchem.core.models.uma.cpu.ops import CPUPermuteWignerInvEdgeToNodeFunction
        x, wigner_inv = setup

        ref = _pytorch_ref_pwi(x, wigner_inv)
        cpp = CPUPermuteWignerInvEdgeToNodeFunction.apply(x, wigner_inv)
        assert torch.allclose(ref, cpp, atol=1e-5), \
            f"Forward mismatch: max diff = {(ref - cpp).abs().max()}"

    def test_backward_grad_wigner_inv(self, setup):
        """Verify wigner_inv gradient matches numerical gradient (critical for forces)."""
        from fairchem.core.models.uma.cpu.ops import CPUPermuteWignerInvEdgeToNodeFunction
        x, wigner_inv = setup

        out = CPUPermuteWignerInvEdgeToNodeFunction.apply(x, wigner_inv)
        loss = out.sum()
        loss.backward()

        # Numerical gradient for a few wigner_inv elements
        eps = 1e-4
        for (i, j) in [(0, 0), (1, 2), (4, 5), (7, 8)]:
            w_plus = wigner_inv.clone(); w_plus[0, i, j] += eps
            w_minus = wigner_inv.clone(); w_minus[0, i, j] -= eps
            f_plus = CPUPermuteWignerInvEdgeToNodeFunction.apply(x, w_plus).sum()
            f_minus = CPUPermuteWignerInvEdgeToNodeFunction.apply(x, w_minus).sum()
            num_grad = (f_plus - f_minus) / (2 * eps)
            ana_grad = wigner_inv.grad[0, i, j]
            assert abs(ana_grad - num_grad) < 1e-2, \
                f"Wigner_inv grad mismatch at [{i},{j}]: analytic={ana_grad:.6f} numerical={num_grad:.6f}"

    def test_wigner_grad_is_nonzero(self, setup):
        """Verify wigner inverse gradient is computed (needed for forces)."""
        from fairchem.core.models.uma.cpu.ops import CPUPermuteWignerInvEdgeToNodeFunction
        x, wigner_inv = setup

        out = CPUPermuteWignerInvEdgeToNodeFunction.apply(x, wigner_inv)
        loss = out.sum()
        loss.backward()

        assert wigner_inv.grad is not None, "Wigner inv gradient is None!"
        assert wigner_inv.grad.abs().sum() > 0, "Wigner inv gradient is all zeros!"


class TestEndToEnd:
    """End-to-end test: forces match gold standard."""

    def test_forces_match_gold(self):
        """Run umas_fast_cpu and verify forces match general backend."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                        '..', '..', '..', '..', '..', 'configs', 'uma', 'speed'))
        try:
            from bench_common import (
                load_predictor, make_system, attach_calculator,
                load_gold, compare, GOLD_PKL,
            )
        except ImportError:
            pytest.skip("bench_common not available")

        if not GOLD_PKL.exists():
            pytest.skip("Gold standard PKL not found")

        predictor = load_predictor(backend='umas_fast_cpu', compile=False, device='cpu')
        atoms = make_system()
        attach_calculator(atoms, predictor)

        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        gold = load_gold()
        passed, details = compare(energy, forces, gold)

        assert passed, (
            f"Forces mismatch: energy_err={details['energy_abs_err']:.2e}, "
            f"forces_max_err={details['forces_abs_max_err']:.2e}"
        )
