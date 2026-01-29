# Claude Session Context - Quaternion Wigner D Implementation

## Current State

**Branch**: `quaternions`
**Last Commit**: `a4c45063` - "Add math expert review identifying bugs in J-matrix approach"

**Test Status**: 17 failed, 22 passed (bugs identified - see MATH_REVIEW.md)

## Math Expert Review Findings

Two bugs were identified in the J-matrix approach:

### Bug 1: `_z_rot_mat_batched` is WRONG
- Places cos on diagonal, sin on anti-diagonal
- Should use 2x2 rotation blocks for real spherical harmonics
- **Fix:** Rewrite with proper block structure

### Bug 2: `_compute_special_case_ra_small` is WRONG
- Uses simplified approximation that doesn't do proper complex-to-real transformation
- **Fix:** Derive correct formula or remove

### Confirmed CORRECT:
- Edge-to-quaternion (half-angle formula)
- Ra/Rb decomposition
- All special cases for complex Wigner D
- Phase formula, sign factors, exponents
- Horner recurrence
- Complex-to-real conversion formula `D_real = U^H @ D_complex @ U`

### Recommended Fix Strategy
**Use Euler-angle-free approach only:**
- `wigner_d_from_quaternion_vectorized_complex()` is correct
- `wigner_d_complex_to_real()` is correct
- Remove or fix the buggy J-matrix functions

## What Was Built

### Files Created

1. **`src/fairchem/core/models/uma/common/rotation_quaternion.py`** (1325 lines)
   - `edge_to_quaternion()` - Convert edge vectors to quaternions
   - `quaternion_to_ra_rb()` - Decompose to Cayley-Klein parameters
   - `wigner_d_from_quaternion_complex()` - Complex Wigner D (Euler-free)
   - `wigner_d_from_quaternion_vectorized_complex()` - Optimized version
   - `wigner_d_real_from_quaternion()` - Real Wigner D via J matrices
   - `get_wigner_from_edge_vectors_euler_free()` - Full Euler-free pipeline
   - `get_wigner_from_edge_vectors_real()` - J-matrix pipeline
   - `precompute_wigner_coefficients()` - Horner recurrence coefficients
   - `precompute_complex_to_real_matrix()` - Unitary transformation

2. **`tests/core/models/uma/test_rotation_quaternion.py`** (960 lines)
   - Tests for basic correctness, gimbal lock stability, gradient stability
   - Comparison tests between Euler and quaternion approaches
   - Tests for Euler-free complex->real conversion

3. **`quaternion_wigner_algorithm.md`** (750 lines)
   - Detailed algorithm specification from spherical_functions package

4. **`.claude/agents/`** - Agent definitions for fairchem-impl and wigner-math

## The Problem Being Solved

The existing Euler angle approach has **gimbal lock** when edges are nearly y-aligned:
- `acos(y)` has gradient singularity at y = ±1
- `atan2(x, z)` is undefined when x ~ 0 and z ~ 0

## The Solution

Quaternion-based Wigner D computation that avoids Euler angles:
1. Edge vector → quaternion (half-angle formula)
2. Quaternion → Ra/Rb decomposition
3. Ra/Rb → Wigner D elements (three-case algorithm)

## Two Valid Approaches

| Approach | Function | Notes |
|----------|----------|-------|
| J-matrix | `wigner_d_real_from_quaternion()` | Uses Euler angles ONLY when safe (both |Ra| and |Rb| significant) |
| Euler-free | `get_wigner_from_edge_vectors_euler_free()` | Complex Wigner D → real via unitary transformation |

## Key Insight: J-Matrix Approach IS Safe

After analysis, we confirmed that `wigner_d_real_from_quaternion()` (lines 1009-1011) is safe because:
1. Euler angles are ONLY used in `general_case` branch (both |Ra| ≥ ε AND |Rb| ≥ ε)
2. Gimbal lock edges take special-case paths using only the well-conditioned phase
3. `safe_Ra` and `safe_Rb` replace small magnitudes with 1+0j before computing angles
4. `torch.where` zeros out gradients from unused branches

## Recent Changes Made

1. Renamed functions to clarify they return complex:
   - `wigner_d_from_quaternion()` → `wigner_d_from_quaternion_complex()`
   - `wigner_d_from_quaternion_vectorized()` → `wigner_d_from_quaternion_vectorized_complex()`
   - `_wigner_d_block()` → `_wigner_d_block_complex()`

2. Fixed complex output (no longer takes `.real`):
   - Special cases return complex `Rb_power` / `Ra_power` (not `.real`)
   - General case returns `magnitude * poly_sum * phase_factor` (not `.real`)

3. Added new pipeline functions:
   - `get_wigner_from_edge_vectors_euler_free()` - uses complex Wigner D + U transformation
   - `get_wigner_from_edge_vectors_complex()` - returns complex Wigner D

4. Added comparison tests in `TestComplexToRealConversion` and `TestTwoApproachesEquivalence`

## Next Steps

1. **Run tests**: `source ~/envs/fairchem/bin/activate && pytest tests/core/models/uma/test_rotation_quaternion.py -v --tb=short`
2. **Debug the 17 failing tests** - likely issues with:
   - Import names (functions were renamed)
   - Complex vs real dtype mismatches
   - The complex-to-real conversion producing complex output instead of real
3. **Verify orthogonality** of both approaches
4. **Benchmark performance**
5. **Integrate into escn_md.py**

## Reference Documents

- `/Users/levineds/fairchem/wigner_understandings.md` - Problem context
- `/Users/levineds/fairchem/quaternion_wigner_algorithm.md` - Algorithm spec
- `~/spherical_functions/` - Reference implementation

## Commands to Run Tests

```bash
source ~/envs/fairchem/bin/activate
pytest tests/core/models/uma/test_rotation_quaternion.py -v --tb=short
```

## Critical Constraint

**NO EULER ANGLES** in the production path (computing alpha, beta, gamma from edge vectors is forbidden). However:
- Computing Euler angles for **tests/verification** is OK
- The J-matrix approach extracts Euler angles from quaternion **safely** (only when both phases are well-conditioned)
