# Math Expert Review - Quaternion Wigner D Implementation

## Executive Summary

The core algorithm is **largely correct**, but there are **two bugs** in the J-matrix approach that explain some test failures.

---

## Confirmed Correct

1. **Edge-to-quaternion** - Half-angle formula `q = normalize(1+y, z, 0, -x)` ✓
2. **Y = -1 fallback** - 180° rotation around x ✓
3. **Gamma rotation** - Quaternion multiplication ✓
4. **Ra/Rb decomposition** - `Ra = w + iz`, `Rb = y + ix` ✓
5. **Special case |Ra| ~ 0** - `D^l_{-m,m} = (-1)^{l-m} * Rb^{2m}` ✓
6. **Special case |Rb| ~ 0** - `D^l_{m,m} = Ra^{2m}` ✓
7. **Phase formula** - `(m'+m)*phia + (m-m')*phib` ✓
8. **Sign factor** - `(-1)^{l-m}` in Case 2 ✓
9. **Coefficient exponents** - Both cases correct ✓
10. **Horner recurrence** - Direction and formula correct ✓
11. **Complex-to-real conversion** - `D_real = U^H @ D_complex @ U` ✓
12. **Wigner coefficient computation** - Binomial and sqrt factors correct ✓

---

## Bugs Found

### Bug 1: `_z_rot_mat_batched` is incorrect

**Location:** Lines 1250-1284

**Issue:** The function places `cos(f*angle)` on diagonal and `sin(f*angle)` on anti-diagonal. This is WRONG for real spherical harmonics z-rotation.

**Correct form:** Should have 2x2 rotation blocks:
```
For each |m|:
[cos(m*θ)   -sin(m*θ)]
[sin(m*θ)    cos(m*θ)]
```

**Impact:** J-matrix approach produces incorrect results.

**Fix:** Rewrite to use proper block-diagonal structure with 2x2 rotation matrices.

### Bug 2: `_compute_special_case_ra_small` uses wrong approximation

**Location:** Lines 1148-1247

**Issue:** The formula `sign * magnitude * cos(2*m*phib)` is a simplified approximation that doesn't properly handle the complex-to-real transformation at β = π.

**Impact:** J-matrix approach at β ≈ π (near -y edges) is incorrect.

**Fix:** Derive correct real spherical harmonics formula at β = π, or remove and use the Euler-free approach instead.

---

## Recommendations

1. **Use the Euler-angle-free approach** as the primary path:
   - `wigner_d_from_quaternion_vectorized_complex()` is fully correct
   - `wigner_d_complex_to_real()` for real spherical harmonics
   - This avoids both bugs above

2. **Fix or remove J-matrix approach functions:**
   - `_z_rot_mat_batched` needs complete rewrite
   - `_compute_special_case_ra_small` needs correct derivation

3. **Add tests:**
   - Composition test: `D(q1 * q2) = D(q1) @ D(q2)`
   - Known-value tests: 90° rotations around each axis
   - Higher lmax: Test up to lmax = 8 or 10
   - Compare against e3nn's Wigner D directly

---

## Test Failures Likely Cause

The 17 failing tests are likely due to:
1. **Incorrect `_z_rot_mat_batched`** - breaks J-matrix approach
2. **Wrong `_compute_special_case_ra_small`** - breaks β ≈ π handling
3. **Possible import errors** - functions were renamed

---

## Priority Fix Strategy

**Option A (Recommended):** Remove J-matrix approach entirely, use only Euler-free path
- Simpler, fewer places for bugs
- Truly Euler-angle-free

**Option B:** Fix both bugs in J-matrix approach
- More complex
- Need to derive correct real spherical harmonics formulas
