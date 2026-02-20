"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# =============================================================================
# Triton Kernels
# =============================================================================


@triton.jit
def gate_activation_lmax2_fwd_kernel(
    gating_ptr,  # [E, 2, C] gating scalars (reshaped from [E, 2*C])
    input_ptr,  # [E, 9, C] input tensors
    output_ptr,  # [E, 9, C] output tensors
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused forward pass for GateActivation (lmax=2, m_prime=True).

    Computes:
        output[0] = SiLU(input[0])
        output[i] = input[i] * sigmoid(gating[expand_index[i-1]]) for i=1..8

    where expand_index = [0, 1, 0, 1, 0, 1, 1, 1]

    Note: Does not save sigmoid(gating) - backward recomputes it for better
    memory bandwidth utilization.
    """
    edge_id = tl.program_id(0)
    c_block = tl.program_id(1)

    c_start = c_block * BLOCK_C
    c_range = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_range < C

    # Base offsets for this edge
    g_base = edge_id * 2 * C
    x_base = edge_id * 9 * C

    # Load and compute sigmoid of gating scalars
    g0 = tl.load(gating_ptr + g_base + 0 * C + c_range, mask=c_mask, other=0.0)
    g1 = tl.load(gating_ptr + g_base + 1 * C + c_range, mask=c_mask, other=0.0)
    sig_g0 = tl.sigmoid(g0)
    sig_g1 = tl.sigmoid(g1)

    # Load all 9 input components
    x0 = tl.load(input_ptr + x_base + 0 * C + c_range, mask=c_mask, other=0.0)
    x1 = tl.load(input_ptr + x_base + 1 * C + c_range, mask=c_mask, other=0.0)
    x2 = tl.load(input_ptr + x_base + 2 * C + c_range, mask=c_mask, other=0.0)
    x3 = tl.load(input_ptr + x_base + 3 * C + c_range, mask=c_mask, other=0.0)
    x4 = tl.load(input_ptr + x_base + 4 * C + c_range, mask=c_mask, other=0.0)
    x5 = tl.load(input_ptr + x_base + 5 * C + c_range, mask=c_mask, other=0.0)
    x6 = tl.load(input_ptr + x_base + 6 * C + c_range, mask=c_mask, other=0.0)
    x7 = tl.load(input_ptr + x_base + 7 * C + c_range, mask=c_mask, other=0.0)
    x8 = tl.load(input_ptr + x_base + 8 * C + c_range, mask=c_mask, other=0.0)

    # Compute outputs
    # y0 = SiLU(x0) = x0 * sigmoid(x0)
    sig_x0 = tl.sigmoid(x0)
    y0 = x0 * sig_x0

    # y1-y8: gating with expand_index = [0, 1, 0, 1, 0, 1, 1, 1]
    y1 = x1 * sig_g0  # expand_index[0] = 0
    y2 = x2 * sig_g1  # expand_index[1] = 1
    y3 = x3 * sig_g0  # expand_index[2] = 0
    y4 = x4 * sig_g1  # expand_index[3] = 1
    y5 = x5 * sig_g0  # expand_index[4] = 0
    y6 = x6 * sig_g1  # expand_index[5] = 1
    y7 = x7 * sig_g1  # expand_index[6] = 1
    y8 = x8 * sig_g1  # expand_index[7] = 1

    # Store outputs
    tl.store(output_ptr + x_base + 0 * C + c_range, y0, mask=c_mask)
    tl.store(output_ptr + x_base + 1 * C + c_range, y1, mask=c_mask)
    tl.store(output_ptr + x_base + 2 * C + c_range, y2, mask=c_mask)
    tl.store(output_ptr + x_base + 3 * C + c_range, y3, mask=c_mask)
    tl.store(output_ptr + x_base + 4 * C + c_range, y4, mask=c_mask)
    tl.store(output_ptr + x_base + 5 * C + c_range, y5, mask=c_mask)
    tl.store(output_ptr + x_base + 6 * C + c_range, y6, mask=c_mask)
    tl.store(output_ptr + x_base + 7 * C + c_range, y7, mask=c_mask)
    tl.store(output_ptr + x_base + 8 * C + c_range, y8, mask=c_mask)


@triton.jit
def gate_activation_lmax2_bwd_kernel(
    grad_out_ptr,  # [E, 9, C] upstream gradient
    input_ptr,  # [E, 9, C] saved input tensors
    gating_ptr,  # [E, 2, C] saved gating scalars (recompute sigmoid)
    grad_input_ptr,  # [E, 9, C] gradient wrt input
    grad_gating_ptr,  # [E, 2, C] gradient wrt gating
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Backward pass for GateActivation.

    Gradient wrt input:
        grad_x[0] = grad_out[0] * SiLU'(x[0])
        grad_x[i] = grad_out[i] * sigmoid(g[expand_index[i-1]]) for i=1..8

    Gradient wrt gating:
        grad_g[j] = sum over i where expand_index[i-1]==j of:
                    grad_out[i] * x[i] * sigmoid(g[j]) * (1 - sigmoid(g[j]))

    Note: Recomputes sigmoid(gating) instead of loading saved values.
    This trades cheap compute for expensive memory bandwidth.
    """
    edge_id = tl.program_id(0)
    c_block = tl.program_id(1)

    c_start = c_block * BLOCK_C
    c_range = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_range < C

    g_base = edge_id * 2 * C
    x_base = edge_id * 9 * C

    # Load gating and RECOMPUTE sigmoid (faster than loading saved sigmoid)
    g0 = tl.load(gating_ptr + g_base + 0 * C + c_range, mask=c_mask, other=0.0)
    g1 = tl.load(gating_ptr + g_base + 1 * C + c_range, mask=c_mask, other=0.0)
    sig_g0 = tl.sigmoid(g0)
    sig_g1 = tl.sigmoid(g1)

    # Load upstream gradients
    dy0 = tl.load(grad_out_ptr + x_base + 0 * C + c_range, mask=c_mask, other=0.0)
    dy1 = tl.load(grad_out_ptr + x_base + 1 * C + c_range, mask=c_mask, other=0.0)
    dy2 = tl.load(grad_out_ptr + x_base + 2 * C + c_range, mask=c_mask, other=0.0)
    dy3 = tl.load(grad_out_ptr + x_base + 3 * C + c_range, mask=c_mask, other=0.0)
    dy4 = tl.load(grad_out_ptr + x_base + 4 * C + c_range, mask=c_mask, other=0.0)
    dy5 = tl.load(grad_out_ptr + x_base + 5 * C + c_range, mask=c_mask, other=0.0)
    dy6 = tl.load(grad_out_ptr + x_base + 6 * C + c_range, mask=c_mask, other=0.0)
    dy7 = tl.load(grad_out_ptr + x_base + 7 * C + c_range, mask=c_mask, other=0.0)
    dy8 = tl.load(grad_out_ptr + x_base + 8 * C + c_range, mask=c_mask, other=0.0)

    # Load saved inputs
    x0 = tl.load(input_ptr + x_base + 0 * C + c_range, mask=c_mask, other=0.0)
    x1 = tl.load(input_ptr + x_base + 1 * C + c_range, mask=c_mask, other=0.0)
    x2 = tl.load(input_ptr + x_base + 2 * C + c_range, mask=c_mask, other=0.0)
    x3 = tl.load(input_ptr + x_base + 3 * C + c_range, mask=c_mask, other=0.0)
    x4 = tl.load(input_ptr + x_base + 4 * C + c_range, mask=c_mask, other=0.0)
    x5 = tl.load(input_ptr + x_base + 5 * C + c_range, mask=c_mask, other=0.0)
    x6 = tl.load(input_ptr + x_base + 6 * C + c_range, mask=c_mask, other=0.0)
    x7 = tl.load(input_ptr + x_base + 7 * C + c_range, mask=c_mask, other=0.0)
    x8 = tl.load(input_ptr + x_base + 8 * C + c_range, mask=c_mask, other=0.0)

    # --- Gradient wrt input ---
    # SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    sig_x0 = tl.sigmoid(x0)
    dx0 = dy0 * sig_x0 * (1.0 + x0 * (1.0 - sig_x0))

    # dx[i] = dy[i] * sigmoid(g[expand_index[i-1]])
    dx1 = dy1 * sig_g0
    dx2 = dy2 * sig_g1
    dx3 = dy3 * sig_g0
    dx4 = dy4 * sig_g1
    dx5 = dy5 * sig_g0
    dx6 = dy6 * sig_g1
    dx7 = dy7 * sig_g1
    dx8 = dy8 * sig_g1

    # Store grad_input
    tl.store(grad_input_ptr + x_base + 0 * C + c_range, dx0, mask=c_mask)
    tl.store(grad_input_ptr + x_base + 1 * C + c_range, dx1, mask=c_mask)
    tl.store(grad_input_ptr + x_base + 2 * C + c_range, dx2, mask=c_mask)
    tl.store(grad_input_ptr + x_base + 3 * C + c_range, dx3, mask=c_mask)
    tl.store(grad_input_ptr + x_base + 4 * C + c_range, dx4, mask=c_mask)
    tl.store(grad_input_ptr + x_base + 5 * C + c_range, dx5, mask=c_mask)
    tl.store(grad_input_ptr + x_base + 6 * C + c_range, dx6, mask=c_mask)
    tl.store(grad_input_ptr + x_base + 7 * C + c_range, dx7, mask=c_mask)
    tl.store(grad_input_ptr + x_base + 8 * C + c_range, dx8, mask=c_mask)

    # --- Gradient wrt gating ---
    # sigmoid'(g) = sigmoid(g) * (1 - sigmoid(g))
    sig_prime_g0 = sig_g0 * (1.0 - sig_g0)
    sig_prime_g1 = sig_g1 * (1.0 - sig_g1)

    # expand_index = [0, 1, 0, 1, 0, 1, 1, 1]
    # g0 controls positions 1, 3, 5; g1 controls positions 2, 4, 6, 7, 8
    sum_for_g0 = dy1 * x1 + dy3 * x3 + dy5 * x5
    sum_for_g1 = dy2 * x2 + dy4 * x4 + dy6 * x6 + dy7 * x7 + dy8 * x8

    dg0 = sum_for_g0 * sig_prime_g0
    dg1 = sum_for_g1 * sig_prime_g1

    # Store grad_gating
    tl.store(grad_gating_ptr + g_base + 0 * C + c_range, dg0, mask=c_mask)
    tl.store(grad_gating_ptr + g_base + 1 * C + c_range, dg1, mask=c_mask)


# =============================================================================
# Autograd Function
# =============================================================================


class GateActivationTritonFunction(torch.autograd.Function):
    """Triton-accelerated GateActivation for lmax=2, m_prime=True."""

    @staticmethod
    def forward(
        ctx,
        gating_scalars: torch.Tensor,  # [E, 2*C]
        input_tensors: torch.Tensor,  # [E, 9, C]
    ) -> torch.Tensor:
        # Ensure contiguous for Triton kernel
        gating_scalars = gating_scalars.contiguous()
        input_tensors = input_tensors.contiguous()

        E, _, C = input_tensors.shape

        output = torch.empty_like(input_tensors)

        BLOCK_C = 128
        num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C

        gate_activation_lmax2_fwd_kernel[(E, num_c_blocks)](
            gating_scalars, input_tensors, output, C, BLOCK_C
        )

        # Save gating (not sigmoid_g) - backward will recompute sigmoid
        ctx.save_for_backward(gating_scalars, input_tensors)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gating_scalars, input_tensors = ctx.saved_tensors
        E, _, C = input_tensors.shape

        # Ensure contiguous for kernel
        grad_output = grad_output.contiguous()

        grad_input = torch.empty_like(input_tensors)
        grad_gating = torch.empty(
            E, 2, C, device=input_tensors.device, dtype=input_tensors.dtype
        )

        BLOCK_C = 128
        num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C

        gate_activation_lmax2_bwd_kernel[(E, num_c_blocks)](
            grad_output,
            input_tensors,
            gating_scalars,
            grad_input,
            grad_gating,
            C,
            BLOCK_C,
        )

        # Reshape [E, 2, C] -> [E, 2*C] to match input format
        grad_gating = grad_gating.view(E, -1)

        return grad_gating, grad_input


def gate_activation_triton(
    gating_scalars: torch.Tensor,
    input_tensors: torch.Tensor,
) -> torch.Tensor:
    """Triton-accelerated gate activation for UMA-S.

    Drop-in replacement for GateActivation with lmax=2, m_prime=True.

    Args:
        gating_scalars: [E, 2*C] gating scalars (lmax * hidden_channels)
        input_tensors: [E, 9, C] input tensors ((lmax+1)^2 components)

    Returns:
        [E, 9, C] gated output tensor

    Example:
        >>> # Replace:
        >>> # output = self.act(gating_scalars, input_tensors)
        >>> # With:
        >>> output = gate_activation_triton(gating_scalars, input_tensors)
    """
    return GateActivationTritonFunction.apply(gating_scalars, input_tensors)
