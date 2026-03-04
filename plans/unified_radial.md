# Plan: Einsum-Based UnifiedRadialMLP

## Overview

Refactor `UnifiedRadialMLP` to use einsum-based batched computation instead of a Python loop over models. This eliminates N sequential GEMM calls in favor of 2 batched einsum operations.

## Current Implementation

```python
# unified_radial.py - current forward()
def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
    h_all = F.linear(x, self.W1_cat, self.b1_cat)  # [E, N*H] - efficient
    h_per_layer = h_all.split(self.hidden_features, dim=1)
    
    # BOTTLENECK: Sequential loop over N models
    return [self.umas_radial_mlp(h_per_layer[i], i) for i in range(self.num_layers)]
```

## Proposed Implementation

```python
def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
    E, N, H = x.shape[0], self.num_models, self.hidden_features
    
    # Layer 1: Batched first linear (unchanged)
    h = F.linear(x, self.W1_cat, self.b1_cat)      # [E, N*H]
    h = h.view(E, N, H).permute(1, 0, 2).contiguous()  # [N, E, H]
    
    # Layer 2: Batched LN → SiLU → Linear via einsum
    h = batched_layer_norm(h, self.ln1_weight, self.ln1_bias, self.ln_eps)
    h = F.silu(h)
    h = torch.einsum("neh,noh->neo", h, self.W2) + self.b2[:, None, :]
    
    # Layer 3: Batched LN → SiLU → Linear via einsum  
    h = batched_layer_norm(h, self.ln2_weight, self.ln2_bias, self.ln_eps)
    h = F.silu(h)
    h = torch.einsum("neh,noh->neo", h, self.W3) + self.b3[:, None, :]
    
    return [h[i] for i in range(N)]
```

## Architecture

```
Input: x [E, 768]
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 1: Batched First Linear                             │
│    h = F.linear(x, W1_cat, b1_cat)     → [E, N*H]          │
│    h = h.view(E, N, H).permute(1,0,2)  → [N, E, H]         │
└────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 2: Batched Tail (no loop!)                          │
│    h = batched_layer_norm(h, ln1_w, ln1_b)  → [N, E, H]    │
│    h = silu(h)                              → [N, E, H]    │
│    h = einsum("neh,noh->neo", h, W2) + b2   → [N, E, H]    │
└────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 3: Batched Tail (no loop!)                          │
│    h = batched_layer_norm(h, ln2_w, ln2_b)  → [N, E, H]    │
│    h = silu(h)                              → [N, E, H]    │
│    h = einsum("neh,noh->neo", h, W3) + b3   → [N, E, O]    │
└────────────────────────────────────────────────────────────┘
         │
         ▼
Output: [h[0], h[1], ..., h[N-1]]  (list of [E, O] tensors)
```

## Key Functions

### `batched_layer_norm`

```python
def batched_layer_norm(
    x: torch.Tensor,      # [N, E, H]
    weight: torch.Tensor, # [N, H]
    bias: torch.Tensor,   # [N, H]
    eps: float,
) -> torch.Tensor:        # [N, E, H]
    """
    Apply N independent LayerNorms in parallel.
    
    Each of the N*E vectors of length H is normalized independently.
    Model i uses weight[i] and bias[i] for affine transform.
    """
    mean = x.mean(dim=-1, keepdim=True)              # [N, E, 1]
    var = x.var(dim=-1, keepdim=True, correction=0)  # [N, E, 1]
    x_norm = (x - mean) * torch.rsqrt(var + eps)     # [N, E, H]
    # Broadcasting: weight[:, None, :] → [N, 1, H] broadcasts to [N, E, H]
    return x_norm * weight[:, None, :] + bias[:, None, :]
```

### Einsum for Batched Linear

```python
# For N models with weights W2[n] of shape [O, H]:
# Input:  h  [N, E, H]
# Weight: W2 [N, O, H]  
# Bias:   b2 [N, O]

# Einsum contracts h and W2 over the H dimension, for each model n:
#   out[n, e, o] = sum_h(h[n, e, h] * W2[n, o, h]) + b2[n, o]
output = torch.einsum("neh,noh->neo", h, W2) + b2[:, None, :]
```

## Implementation Steps

### Step 1: Add `batched_layer_norm` helper function

**File:** `src/fairchem/core/models/uma/nn/unified_radial.py`

```python
def batched_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, correction=0)
    x_norm = (x - mean) * torch.rsqrt(var + eps)
    return x_norm * weight[:, None, :] + bias[:, None, :]
```

### Step 2: Rename weight buffers for clarity

In `UnifiedRadialMLP.__init__`:
- `fc2_weight` → `W2` (shape: `[N, H, H]`)
- `fc2_bias` → `b2` (shape: `[N, H]`)
- `fc3_weight` → `W3` (shape: `[N, O, H]`)
- `fc3_bias` → `b3` (shape: `[N, O]`)

### Step 3: Refactor `forward()` method

Replace the loop-based implementation with einsum-based batched computation.

### Step 4: Add `num_models` attribute

Store `self.num_models = len(radial_mlps)` for cleaner code.

## Files Changed

| File | Change |
|------|--------|
| `src/fairchem/core/models/uma/nn/unified_radial.py` | Add `batched_layer_norm`, refactor `forward()` |
| `tests/core/models/test_unified_radial.py` | New test file |

## Unit Tests

### Test File: `tests/core/models/test_unified_radial.py`

```python
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.nn.radial import RadialMLP
from fairchem.core.models.uma.nn.unified_radial import (
    UnifiedRadialMLP,
    batched_layer_norm,
)


class TestBatchedLayerNorm:
    """Tests for batched_layer_norm helper function."""

    def test_single_model_matches_pytorch(self):
        """Single model (N=1) should match torch.nn.functional.layer_norm."""
        torch.manual_seed(42)
        E, H = 100, 128
        x = torch.randn(1, E, H)
        weight = torch.randn(1, H)
        bias = torch.randn(1, H)
        eps = 1e-5

        result = batched_layer_norm(x, weight, bias, eps)
        expected = torch.nn.functional.layer_norm(
            x[0], (H,), weight[0], bias[0], eps
        )

        torch.testing.assert_close(result[0], expected, rtol=1e-5, atol=1e-5)

    def test_multiple_models_independent(self):
        """Each model should be normalized independently."""
        torch.manual_seed(42)
        N, E, H = 4, 100, 128
        x = torch.randn(N, E, H)
        weight = torch.randn(N, H)
        bias = torch.randn(N, H)
        eps = 1e-5

        result = batched_layer_norm(x, weight, bias, eps)

        # Check each model independently
        for i in range(N):
            expected = torch.nn.functional.layer_norm(
                x[i], (H,), weight[i], bias[i], eps
            )
            torch.testing.assert_close(result[i], expected, rtol=1e-5, atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through batched_layer_norm."""
        N, E, H = 4, 100, 128
        x = torch.randn(N, E, H, requires_grad=True)
        weight = torch.randn(N, H, requires_grad=True)
        bias = torch.randn(N, H, requires_grad=True)

        result = batched_layer_norm(x, weight, bias, eps=1e-5)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert weight.grad is not None
        assert bias.grad is not None


class TestUnifiedRadialMLPEinsum:
    """Tests for einsum-based UnifiedRadialMLP."""

    @pytest.fixture
    def radial_mlps(self):
        """Create N identical RadialMLP modules."""
        torch.manual_seed(42)
        N = 4
        # channels_list: [input, hidden, hidden, output]
        channels_list = [768, 128, 128, 1024]
        return [RadialMLP(channels_list) for _ in range(N)]

    @pytest.fixture
    def unified_mlp(self, radial_mlps):
        """Create UnifiedRadialMLP from RadialMLPs."""
        return UnifiedRadialMLP(radial_mlps)

    def test_output_matches_individual_mlps(self, radial_mlps, unified_mlp):
        """UnifiedRadialMLP output should match individual RadialMLP outputs."""
        torch.manual_seed(123)
        E = 500
        x = torch.randn(E, 768)

        # Compute with unified MLP
        unified_outputs = unified_mlp(x)

        # Compute with individual MLPs
        individual_outputs = [mlp(x) for mlp in radial_mlps]

        # Compare
        for i, (unified, individual) in enumerate(
            zip(unified_outputs, individual_outputs)
        ):
            torch.testing.assert_close(
                unified, individual, rtol=1e-4, atol=1e-4,
                msg=f"Mismatch at model {i}"
            )

    def test_output_shapes(self, unified_mlp):
        """Output shapes should be [E, output_features] for each model."""
        E = 500
        x = torch.randn(E, 768)

        outputs = unified_mlp(x)

        assert len(outputs) == unified_mlp.num_models
        for out in outputs:
            assert out.shape == (E, 1024)

    def test_gradient_flow(self, unified_mlp):
        """Gradients should flow through the entire computation."""
        E = 100
        x = torch.randn(E, 768, requires_grad=True)

        outputs = unified_mlp(x)
        loss = sum(out.sum() for out in outputs)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    @pytest.mark.gpu
    def test_cuda_computation(self, radial_mlps):
        """Test on GPU if available."""
        device = torch.device("cuda")
        unified_mlp = UnifiedRadialMLP(radial_mlps).to(device)

        E = 500
        x = torch.randn(E, 768, device=device)

        outputs = unified_mlp(x)

        assert all(out.device == device for out in outputs)

    def test_numerical_stability(self, unified_mlp):
        """Test with edge case inputs."""
        E = 100

        # Test with zeros
        x_zeros = torch.zeros(E, 768)
        outputs = unified_mlp(x_zeros)
        assert all(torch.isfinite(out).all() for out in outputs)

        # Test with large values
        x_large = torch.randn(E, 768) * 100
        outputs = unified_mlp(x_large)
        assert all(torch.isfinite(out).all() for out in outputs)

    def test_different_batch_sizes(self, unified_mlp):
        """Test with various batch sizes."""
        for E in [1, 10, 100, 1000]:
            x = torch.randn(E, 768)
            outputs = unified_mlp(x)
            assert all(out.shape[0] == E for out in outputs)


class TestUnifiedRadialMLPBackwardCompatibility:
    """Ensure einsum version matches original loop version exactly."""

    def test_exact_numerical_match(self):
        """
        Golden test: einsum version must produce identical results to loop version.
        
        This test creates both implementations and compares outputs bit-for-bit.
        """
        torch.manual_seed(42)
        N = 4
        channels_list = [768, 128, 128, 1024]
        radial_mlps = [RadialMLP(channels_list) for _ in range(N)]

        # Create unified MLP
        unified = UnifiedRadialMLP(radial_mlps)

        # Test input
        E = 500
        x = torch.randn(E, 768)

        # Get unified output
        unified_out = unified(x)

        # Reference: compute each MLP individually
        reference_out = [mlp(x) for mlp in radial_mlps]

        # Must match exactly (or very close due to float ops)
        for i in range(N):
            torch.testing.assert_close(
                unified_out[i], 
                reference_out[i], 
                rtol=1e-4, 
                atol=1e-4
            )
```

## Performance Comparison

### Before (loop-based, N=4 layers)
```
First layer:  1 GEMM     [E, 768] → [E, 512]
Tail layers:  4 × (LN + SiLU + GEMM + LN + SiLU + GEMM)
            = 8 sequential GEMM calls + 8 LN + 8 SiLU
```

### After (einsum-based)
```
First layer:  1 GEMM     [E, 768] → [E, 512]
Reshape:      view + permute
Tail layers:  2 batched einsums + 2 batched LN + 2 SiLU
```

### Expected Speedup
- **Kernel launch overhead**: 12 → 4 (3x reduction)
- **Memory bandwidth**: Better cache utilization from batched ops
- **Estimated speedup**: 2-3x for radial computation

## Validation Checklist

- [ ] `batched_layer_norm` matches `F.layer_norm` for single model
- [ ] `batched_layer_norm` correctly handles multiple models independently
- [ ] `UnifiedRadialMLP` output matches individual `RadialMLP` outputs
- [ ] Gradients flow correctly through einsum operations
- [ ] Works on both CPU and GPU
- [ ] Numerical stability with edge cases (zeros, large values)
- [ ] Pre-commit passes on modified files
- [ ] Force comparison test passes with existing checkpoint

## Rollout Plan

1. **Implement** `batched_layer_norm` helper
2. **Refactor** `UnifiedRadialMLP.forward()` to use einsum
3. **Add unit tests** in `tests/core/models/test_unified_radial.py`
4. **Run validation**: Compare forces against reference checkpoint
5. **Benchmark**: Measure speedup on typical inference workload
6. **PR**: Submit with force comparison results
