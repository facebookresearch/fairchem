"""Tests for restricting frozen radial input gradients to distance features."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from fairchem.core.models.uma.nn.execution_backends import (
    _configure_radial_first_linear_prefix_grad,
)
from fairchem.core.models.uma.nn.radial import RadialMLP
from fairchem.core.models.uma.nn.unified_radial import UnifiedRadialMLP


def test_radial_first_linear_prefix_forward_and_vjp():
    torch.manual_seed(0)
    expected_model = RadialMLP([7, 5]).double()
    actual_model = copy.deepcopy(expected_model)
    actual_model.configure_first_linear_grad_prefix(3, 7)
    reference_input = torch.randn(11, 7, dtype=torch.float64, requires_grad=True)
    actual_input = reference_input.detach().clone().requires_grad_()
    grad_output = torch.randn(11, 5, dtype=torch.float64)

    expected = expected_model(reference_input)
    actual = actual_model(actual_input)
    expected_grad = torch.autograd.grad(expected, reference_input, grad_output)[0]
    actual_grad, weight_grad, bias_grad = torch.autograd.grad(
        actual,
        (actual_input, actual_model.net[0].weight, actual_model.net[0].bias),
        grad_output,
        allow_unused=True,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad[:, :3], expected_grad[:, :3])
    torch.testing.assert_close(actual_grad[:, 3:], torch.zeros_like(actual_grad[:, 3:]))
    assert weight_grad is None
    assert bias_grad is None


def test_radial_first_linear_prefix_double_backward():
    torch.manual_seed(1)
    expected_model = RadialMLP([7, 5]).double()
    actual_model = copy.deepcopy(expected_model)
    actual_model.configure_first_linear_grad_prefix(3, 7)
    tail = torch.randn(4, dtype=torch.float64)

    def expected(prefix):
        inputs = torch.cat((prefix, tail)).unsqueeze(0)
        return expected_model(inputs).sin().square().sum()

    def actual(prefix):
        inputs = torch.cat((prefix, tail)).unsqueeze(0)
        return actual_model(inputs).sin().square().sum()

    prefix = torch.randn(3, dtype=torch.float64)
    expected_hessian = torch.autograd.functional.hessian(expected, prefix)
    actual_hessian = torch.autograd.functional.hessian(actual, prefix)
    torch.testing.assert_close(actual_hessian, expected_hessian, rtol=0, atol=0)


def test_radial_first_linear_prefix_validation():
    for prefix in (0, 7, 8):
        with pytest.raises(ValueError, match="between zero and the input width"):
            RadialMLP([7, 5]).configure_first_linear_grad_prefix(prefix, 7)


def test_radial_first_linear_prefix_preserves_state_dict():
    model = RadialMLP([7, 5, 5, 6])
    state_dict_keys = tuple(model.state_dict())
    weight = model.net[0].weight
    model.configure_first_linear_grad_prefix(3, 7)
    assert tuple(model.state_dict()) == state_dict_keys
    assert model.net[0].weight is weight


def test_unified_radial_prefix_grad_matches_full_input_grad():
    torch.manual_seed(2)
    radial_mlps = [RadialMLP([7, 5, 5, 6]), RadialMLP([7, 5, 5, 6])]
    expected_model = UnifiedRadialMLP(radial_mlps).double()
    actual_model = copy.deepcopy(expected_model)
    actual_model.configure_first_linear_grad_prefix(3, 7)
    expected_input = torch.randn(13, 7, dtype=torch.float64, requires_grad=True)
    actual_input = expected_input.detach().clone().requires_grad_()

    expected_outputs = expected_model(expected_input)
    actual_outputs = actual_model(actual_input)
    grads = [torch.randn_like(output) for output in expected_outputs]
    expected_grad = torch.autograd.grad(expected_outputs, expected_input, grads)[0]
    actual_grad = torch.autograd.grad(actual_outputs, actual_input, grads)[0]

    for actual, expected in zip(actual_outputs, expected_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad[:, :3], expected_grad[:, :3])
    torch.testing.assert_close(actual_grad[:, 3:], torch.zeros_like(actual_grad[:, 3:]))


def _make_model(edge_input_features=7):
    model = torch.nn.Module()
    model.num_distance_basis = 3
    model.edge_channels = 2
    model.regress_config = SimpleNamespace(hessian=False)
    model.edge_degree_embedding = torch.nn.Module()
    model.edge_degree_embedding.rad_func = RadialMLP([edge_input_features, 5, 5, 6])
    return model


def test_configure_radial_first_linear_prefix_grad():
    model = _make_model()
    unified = UnifiedRadialMLP([RadialMLP([7, 5, 5, 6])])
    _configure_radial_first_linear_prefix_grad(model, unified)
    assert model.edge_degree_embedding.rad_func.first_linear_grad_prefix == 3
    assert unified.first_linear_grad_prefix == 3


def test_configure_radial_first_linear_prefix_grad_validates_widths():
    with pytest.raises(ValueError, match="radial.*width"):
        _configure_radial_first_linear_prefix_grad(
            _make_model(edge_input_features=8),
            UnifiedRadialMLP([RadialMLP([7, 5, 5, 6])]),
        )

    with pytest.raises(ValueError, match="UnifiedRadialMLP.*width"):
        _configure_radial_first_linear_prefix_grad(
            _make_model(), UnifiedRadialMLP([RadialMLP([8, 5, 5, 6])])
        )

    model = _make_model()
    model.regress_config.hessian = True
    with pytest.raises(ValueError, match="does not support Hessians"):
        _configure_radial_first_linear_prefix_grad(
            model, UnifiedRadialMLP([RadialMLP([7, 5, 5, 6])])
        )


@pytest.mark.gpu()
@pytest.mark.compile_gpu()
def test_radial_first_linear_prefix_compile_cuda(compile_reset_state):
    torch.manual_seed(3)
    model = RadialMLP([288, 128]).cuda()
    model.configure_first_linear_grad_prefix(32, 288)
    compiled = torch.compile(model, dynamic=True, fullgraph=True)
    inputs = torch.randn(257, 288, device="cuda", requires_grad=True)
    grad_output = torch.randn(257, 128, device="cuda")

    actual = compiled(inputs)
    actual_grad = torch.autograd.grad(actual, inputs, grad_output)[0]
    expected = model.net[0](inputs)
    expected_grad = torch.autograd.grad(expected, inputs, grad_output)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad[:, :32], expected_grad[:, :32])
    torch.testing.assert_close(
        actual_grad[:, 32:], torch.zeros_like(actual_grad[:, 32:])
    )
