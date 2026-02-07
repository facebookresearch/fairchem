# Axis-Angle Wigner D Profiling Results

## Summary

Profiling and optimization of `wigner_d_axis_angle.py` for computing Wigner D matrices
using axis-angle representation with matrix exponentials.

## Baseline Performance (before optimization)

| Component | Time (1k edges, lmax=6) | % of Total |
|-----------|-------------------------|------------|
| wigner_d (matrix_exp for all l) | 64 ms | 31% |
| y_rotation (matrix_exp) | 68 ms | 33% |
| euler_transform | 6 ms | 3% |
| bmm | 5 ms | 2.5% |
| quaternion computation | 0.2 ms | 0.1% |
| axis_angle extraction | 0.04 ms | 0% |
| **Total** | **~205 ms** | **100%** |

### Per-l breakdown of matrix_exp (10k edges)

| l | Block size | Time |
|---|------------|------|
| 0 | 1×1 | 0.3 ms |
| 1 | 3×3 | 6.8 ms |
| 2 | 5×5 | 14.6 ms |
| 3 | 7×7 | 36.2 ms |
| 4 | 9×9 | 321.6 ms |
| 5 | 11×11 | 197.2 ms |
| 6 | 13×13 | 210.8 ms |

## Optimizations Applied

### 1. Y-rotation eigendecomposition (commit `980453ba`)

**Problem**: Y-rotation used `torch.linalg.matrix_exp` which is O(n³) per sample.

**Solution**: Since K_y is a fixed matrix for each l, precompute its eigendecomposition:
```
exp(γK_y) = V @ diag(exp(i*γ*λ)) @ V^H
```

This reduces Y-rotation to O(n²) per sample.

**Impact**: Y-rotation dropped from 33% to ~8% of total time.

### 2. Rodrigues formula for l=1 (commit `bd53b056`)

**Problem**: The 3×3 rotation at l=1 used matrix_exp.

**Solution**: Use the closed-form Rodrigues formula:
```
exp(θK) = I + sin(θ)K + (1-cos(θ))K²
```

This is ~5x faster than matrix_exp for 3×3 matrices.

**Impact**: l=1 computation reduced from ~7ms to ~1ms (for 10k edges).

## Final Performance

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| 1k edges, lmax=6 | ~205 ms | ~90-120 ms | 1.7-2.2x |
| Per-edge (median) | ~205 μs | ~90 μs | ~2x |

Note: Timing is variable due to system load and CPU throttling.

## Remaining Bottleneck

The `matrix_exp` calls for l≥2 blocks now dominate (~55% of time):

| l | Block size | Time (10k edges) | % of wigner_d |
|---|------------|------------------|---------------|
| 2 | 5×5 | 12 ms | 1.4% |
| 3 | 7×7 | 17 ms | 2% |
| 4 | 9×9 | 147 ms | 17% |
| 5 | 11×11 | 191 ms | 22% |
| 6 | 13×13 | 471 ms | 54% |

## Future Optimization Opportunities

### High Impact

1. **Batch all l blocks together**
   - Pad matrices to max size (13×13 for lmax=6)
   - Single batched matrix_exp call
   - Trade-off: Wastes computation on padding for small l

2. **Precompute eigendecomposition for general rotation**
   - Challenge: The axis n varies per sample, so (n·K) has different eigenvectors
   - Possible: Use SO(3) structure - eigenvalues of (n·K) are fixed (±i*m, 0)
   - Could potentially derive eigenvectors analytically

3. **Use Wigner D recurrence relations**
   - Compute D^l from D^(l-1) using known recurrence
   - Avoids matrix_exp entirely
   - Reference: Gimbutas & Greengard (2009)

### Medium Impact

4. **Generalized Rodrigues for l=2**
   - Extend closed-form solution to 5×5 matrices
   - More complex than l=1 but still tractable

5. **GPU optimization**
   - Current code is CPU-focused
   - GPU batching could provide significant speedups

### Low Impact (already fast)

6. Quaternion computation: Already <1% of time
7. Axis-angle extraction: Already negligible

## Test Status

- 81/82 tests pass
- 1 skipped: `TestQuaternionAgreement::test_both_map_edge_to_y`
  - Issue is in `quaternion_wigner` module, not axis-angle
  - The quaternion module does not correctly map edge → +Y

## Files Modified

- `src/fairchem/core/models/uma/common/wigner_d_axis_angle.py`
  - Added `_eigendecompose_antisymmetric()`
  - Modified `get_so3_generators()` to cache K_y eigendecomposition
  - Modified `wigner_d_y_rotation_batched()` to use eigendecomposition
  - Modified `wigner_d_from_axis_angle_batched()` to use Rodrigues for l=1
