"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.nn.fused_so2_block import (
    _eager_so2_block_forward,
    fused_so2_block,
)

# Production model dimensions (UMA-S 1p2): lmax=mmax=2, sphere=128,
# hidden=128. m_split for lmax=mmax=2: m=0 gets 3 SH coefs, m=1 gets 4
# (real+imag for l=1,2), m=2 gets 2 (real+imag for l=2).
LMAX = 2
SPHERE = 128
HIDDEN = 128
M_SPLIT = (3, 4, 2)
EXTRA_M0 = LMAX * HIDDEN  # 256, the GateActivation gating channel count
M_OUT_1 = HIDDEN  # 128
M_OUT_2 = SPHERE  # 128

# Edge-radial widths per-m, matching SO2_Conv1_WithRadialBlock layout:
#   m=0: m_split[0] * sphere_channels       = 3 * 128 = 384
#   m=1: lmax * sphere_channels             = 2 * 128 = 256
#   m=2: (lmax-1) * sphere_channels         = 1 * 128 = 128
EDGE_SPLIT = (
    M_SPLIT[0] * SPHERE,
    LMAX * SPHERE,
    (LMAX - 1) * SPHERE,
)
TOTAL_RADIAL = sum(EDGE_SPLIT)


def _make_inputs(E: int, requires_grad: bool = True, seed: int = 42):
    """Build a fully-shaped set of inputs for the fused block."""
    torch.manual_seed(seed)

    # m=0 conv1 Linear: in = m_split[0]*S = 384,
    #                   out = m_out_1*(lmax+1) + extra_m0 = 384 + 256 = 640
    w_m0_1_in = M_SPLIT[0] * SPHERE
    w_m0_1_out = M_OUT_1 * (LMAX + 1) + EXTRA_M0

    # Block-diag weights:
    #   conv1 m=1: (2*lmax*m_out_1, 2*lmax*sphere)         = (512, 512)
    #   conv1 m=2: (2*(lmax-1)*m_out_1, 2*(lmax-1)*sphere) = (256, 256)
    W_b1_m1 = torch.randn(2 * LMAX * M_OUT_1, 2 * LMAX * SPHERE)
    W_b1_m2 = torch.randn(2 * (LMAX - 1) * M_OUT_1, 2 * (LMAX - 1) * SPHERE)

    # conv2 m=0 Linear: in = m_split[0]*hidden = 384, out = m_out_2*(lmax+1) = 384
    w_m0_2_in = M_SPLIT[0] * HIDDEN
    w_m0_2_out = M_OUT_2 * (LMAX + 1)
    W_b2_m1 = torch.randn(2 * LMAX * M_OUT_2, 2 * LMAX * HIDDEN)
    W_b2_m2 = torch.randn(2 * (LMAX - 1) * M_OUT_2, 2 * (LMAX - 1) * HIDDEN)

    x = torch.randn(E, (LMAX + 1) ** 2, SPHERE, requires_grad=requires_grad)
    x_edge = torch.randn(E, TOTAL_RADIAL, requires_grad=requires_grad)

    w_m0_1 = torch.randn(w_m0_1_out, w_m0_1_in, requires_grad=requires_grad)
    b_m0_1 = torch.randn(w_m0_1_out, requires_grad=requires_grad)
    w_m0_2 = torch.randn(w_m0_2_out, w_m0_2_in, requires_grad=requires_grad)
    b_m0_2 = torch.randn(w_m0_2_out, requires_grad=requires_grad)

    W_b1_m1.requires_grad_(requires_grad)
    W_b1_m2.requires_grad_(requires_grad)
    W_b2_m1.requires_grad_(requires_grad)
    W_b2_m2.requires_grad_(requires_grad)

    return {
        "x": x,
        "x_edge": x_edge,
        "w_m0_1": w_m0_1,
        "b_m0_1": b_m0_1,
        "W_b1_m1": W_b1_m1,
        "W_b1_m2": W_b1_m2,
        "w_m0_2": w_m0_2,
        "b_m0_2": b_m0_2,
        "W_b2_m1": W_b2_m1,
        "W_b2_m2": W_b2_m2,
        "m_split_sizes": M_SPLIT,
        "edge_split_sizes": EDGE_SPLIT,
        "extra_m0": EXTRA_M0,
        "lmax": LMAX,
        "m_out_1": M_OUT_1,
        "m_out_2": M_OUT_2,
    }


# === Phase 0 sanity tests (still apply) ===


def test_fused_so2_block_api_loads():
    inputs = _make_inputs(E=64, requires_grad=False)
    out = fused_so2_block(**inputs)
    assert out.shape == (64, (LMAX + 1) ** 2, M_OUT_2)
    assert out.dtype == torch.float32


def test_fused_so2_block_backward_propagates():
    """Backward reaches every Tensor input that has requires_grad."""
    inputs = _make_inputs(E=64, requires_grad=True)
    out = fused_so2_block(**inputs)
    out.sum().backward()

    tensor_keys = [
        "x",
        "x_edge",
        "w_m0_1",
        "b_m0_1",
        "W_b1_m1",
        "W_b1_m2",
        "w_m0_2",
        "b_m0_2",
        "W_b2_m1",
        "W_b2_m2",
    ]
    for k in tensor_keys:
        t = inputs[k]
        assert t.grad is not None, f"{k} grad missing"
        assert t.grad.shape == t.shape, f"{k} grad shape mismatch"


# === Phase 1 correctness tests ===


@pytest.mark.parametrize("E", [1, 16, 17, 64, 1024])
def test_phase1_cpp_forward_matches_eager_reference(E: int):
    """T1: C++ forward bit-exact vs Python eager reference."""
    inputs = _make_inputs(E=E, requires_grad=False)
    out_cpp = fused_so2_block(**inputs)
    out_ref = _eager_so2_block_forward(**inputs)

    diff = (out_cpp - out_ref).abs().max().item()
    assert diff < 1e-6, f"E={E}: forward max abs diff {diff:.3e} > 1e-6"


def test_phase1_eager_reference_matches_module_pipeline():
    """
    T2: independent verification that the Python eager reference produces
    the same output as the live module pipeline (SO2_Conv1_WithRadialBlock
    + GateActivation(m_prime=True) + SO2_Conv2_InternalBlock).
    """
    from fairchem.core.models.uma.nn.activation import GateActivation
    from fairchem.core.models.uma.nn.so2_layers import (
        SO2_Conv1_WithRadialBlock,
        SO2_Conv2_InternalBlock,
    )

    # Bypass the CoefficientMapping dependency by building a thin stub.
    class _MapStub:
        def __init__(self):
            self.m_size = [3, 2, 1]  # m_size[0]=3, m_size[1]=2, m_size[2]=1
            #   m_split = [3, 4, 2]  (m_size[0], 2*m_size[1], 2*m_size[2])

    map_stub = _MapStub()

    torch.manual_seed(0)
    E = 32
    conv1 = SO2_Conv1_WithRadialBlock(
        sphere_channels=SPHERE,
        m_output_channels=M_OUT_1,
        lmax=LMAX,
        mmax=LMAX,
        mappingReduced=map_stub,
        extra_m0_output_channels=EXTRA_M0,
        edge_channels_list=[1, 1],  # radial MLP shape doesn't matter here
    )
    # Build the cached _w_block on the SO2_m_Conv_Block sub-modules
    for sub in conv1.so2_m_conv:
        sub._maybe_build_w_block()

    conv2 = SO2_Conv2_InternalBlock(
        sphere_channels=M_OUT_1,
        m_output_channels=M_OUT_2,
        lmax=LMAX,
        mmax=LMAX,
        mappingReduced=map_stub,
    )
    for sub in conv2.so2_m_conv:
        sub._maybe_build_w_block()

    act = GateActivation(lmax=LMAX, mmax=LMAX, num_channels=M_OUT_1, m_prime=True)

    # Use the modules' own weights/buffers
    inputs = {
        "x": torch.randn(E, 9, SPHERE),
        "x_edge": torch.randn(E, sum(conv1.edge_split_sizes)),
        "w_m0_1": conv1.fc_m0.weight.data,
        "b_m0_1": conv1.fc_m0.bias.data,
        "W_b1_m1": conv1.so2_m_conv[0]._w_block,
        "W_b1_m2": conv1.so2_m_conv[1]._w_block,
        "w_m0_2": conv2.fc_m0.weight.data,
        "b_m0_2": conv2.fc_m0.bias.data,
        "W_b2_m1": conv2.so2_m_conv[0]._w_block,
        "W_b2_m2": conv2.so2_m_conv[1]._w_block,
        "m_split_sizes": tuple(conv1.m_split_sizes),
        "edge_split_sizes": tuple(conv1.edge_split_sizes),
        "extra_m0": EXTRA_M0,
        "lmax": LMAX,
        "m_out_1": M_OUT_1,
        "m_out_2": M_OUT_2,
    }

    # Module pipeline:
    out_msg, out_gating = conv1(inputs["x"], inputs["x_edge"])
    out_act = act(out_gating, out_msg)
    out_module = conv2(out_act)

    out_ref = _eager_so2_block_forward(**inputs)

    diff = (out_module - out_ref).abs().max().item()
    assert diff < 1e-6, f"module-vs-reference forward max abs diff {diff:.3e}"


@pytest.mark.parametrize("E", [16, 64])
def test_phase1_cpp_backward_matches_module_grads(E: int):
    """
    T3: backward gradients from the C++/Python autograd.Function match the
    gradients produced by autograd through the live module pipeline.
    """
    from fairchem.core.models.uma.nn.activation import GateActivation
    from fairchem.core.models.uma.nn.so2_layers import (
        SO2_Conv1_WithRadialBlock,
        SO2_Conv2_InternalBlock,
    )

    class _MapStub:
        def __init__(self):
            self.m_size = [3, 2, 1]

    map_stub = _MapStub()

    torch.manual_seed(0)
    conv1 = SO2_Conv1_WithRadialBlock(
        sphere_channels=SPHERE,
        m_output_channels=M_OUT_1,
        lmax=LMAX,
        mmax=LMAX,
        mappingReduced=map_stub,
        extra_m0_output_channels=EXTRA_M0,
        edge_channels_list=[1, 1],
    )
    for sub in conv1.so2_m_conv:
        sub._maybe_build_w_block()
    conv2 = SO2_Conv2_InternalBlock(
        sphere_channels=M_OUT_1,
        m_output_channels=M_OUT_2,
        lmax=LMAX,
        mmax=LMAX,
        mappingReduced=map_stub,
    )
    for sub in conv2.so2_m_conv:
        sub._maybe_build_w_block()
    act = GateActivation(lmax=LMAX, mmax=LMAX, num_channels=M_OUT_1, m_prime=True)

    x_data = torch.randn(E, 9, SPHERE)
    x_edge_data = torch.randn(E, sum(conv1.edge_split_sizes))

    # === Module pipeline grads (reference) ===
    x_mod = x_data.clone().requires_grad_(True)
    xe_mod = x_edge_data.clone().requires_grad_(True)
    msg, gating = conv1(x_mod, xe_mod)
    a_out = act(gating, msg)
    out_mod = conv2(a_out)
    grad_out = torch.randn_like(out_mod)
    grad_x_mod, grad_xe_mod = torch.autograd.grad(
        out_mod, [x_mod, xe_mod], grad_outputs=grad_out
    )

    # === Fused-block grads ===
    x_f = x_data.clone().requires_grad_(True)
    xe_f = x_edge_data.clone().requires_grad_(True)

    out_f = fused_so2_block(
        x_f,
        xe_f,
        conv1.fc_m0.weight.data,
        conv1.fc_m0.bias.data,
        conv1.so2_m_conv[0]._w_block,
        conv1.so2_m_conv[1]._w_block,
        conv2.fc_m0.weight.data,
        conv2.fc_m0.bias.data,
        conv2.so2_m_conv[0]._w_block,
        conv2.so2_m_conv[1]._w_block,
        tuple(conv1.m_split_sizes),
        tuple(conv1.edge_split_sizes),
        EXTRA_M0,
        LMAX,
        M_OUT_1,
        M_OUT_2,
    )
    grad_x_f, grad_xe_f = torch.autograd.grad(out_f, [x_f, xe_f], grad_outputs=grad_out)

    # Compare
    fwd_diff = (out_f - out_mod).abs().max().item()
    gx_diff = (grad_x_f - grad_x_mod).abs().max().item()
    gxe_diff = (grad_xe_f - grad_xe_mod).abs().max().item()
    assert fwd_diff < 1e-6, f"E={E}: fwd diff {fwd_diff:.3e}"
    assert gx_diff < 1e-5, f"E={E}: grad_x diff {gx_diff:.3e}"
    assert gxe_diff < 1e-5, f"E={E}: grad_x_edge diff {gxe_diff:.3e}"
