# Quaternion-Based Wigner D Matrix Algorithm

## Technical Reference for PyTorch Implementation

This document provides a complete, accurate specification of the quaternion-based algorithm for computing Wigner D matrices, derived from the `spherical_functions` Python package by Mike Boyle.

---

## CRITICAL CONSTRAINT: NO EULER ANGLES

**The implementation MUST NOT compute Euler angles (alpha, beta, gamma) at any point.**

This is not a stylistic preference - it is a fundamental requirement. The entire purpose of this refactoring is to avoid the gimbal lock singularity inherent to Euler angle representations:

1. **arccos(y) for beta** has gradient singularity at y = +/-1
2. **atan2(x, z) for alpha** is undefined when x ~ 0 and z ~ 0 (which happens exactly when beta ~ 0 or pi)

The quaternion algorithm works by:
- Going directly from **edge vector -> quaternion -> (Ra, Rb) -> Wigner D elements**
- The (Ra, Rb) decomposition encodes the same information as Euler angles but without singularities
- When |Ra| ~ 0 or |Rb| ~ 0 (the "gimbal lock" regions), special formulas apply that DON'T require computing phases

**DO NOT** be tempted to:
- Compute Euler angles "just for verification"
- Use the geometric interpretation formulas like `beta = 2 * arccos(|Ra|)`
- Convert quaternion to Euler angles as an intermediate step

If you find yourself writing `arccos`, `acos`, or computing angles from the quaternion components, **STOP** - you are reintroducing the singularity.

---

## 1. Quaternion Conventions

### 1.1 Quaternion Components

The `spherical_functions` package uses the standard quaternion convention:
```
q = (w, x, y, z) = w + x*i + y*j + z*k
```

Where:
- `w` = scalar (real) part
- `x, y, z` = vector (imaginary) parts

For unit quaternions: `w^2 + x^2 + y^2 + z^2 = 1`

### 1.2 Euler Angle Convention (ZYZ)

The package uses the ZYZ intrinsic Euler angle convention:
```
R(alpha, beta, gamma) = R_z(alpha) * R_y(beta) * R_z(gamma)
```

The corresponding quaternion is:
```
q = q_z(alpha) * q_y(beta) * q_z(gamma)
```

Where:
- `q_z(theta) = (cos(theta/2), 0, 0, sin(theta/2))`
- `q_y(theta) = (cos(theta/2), 0, sin(theta/2), 0)`

### 1.3 Quaternion for ZYZ Euler Angles

For ZYZ angles (alpha, beta, gamma), the quaternion components are:
```
w = cos(alpha/2)*cos(beta/2)*cos(gamma/2) - sin(alpha/2)*cos(beta/2)*sin(gamma/2)
x = cos(alpha/2)*sin(beta/2)*sin(gamma/2) - sin(alpha/2)*sin(beta/2)*cos(gamma/2)
y = cos(alpha/2)*sin(beta/2)*cos(gamma/2) + sin(alpha/2)*sin(beta/2)*sin(gamma/2)
z = sin(alpha/2)*cos(beta/2)*cos(gamma/2) + cos(alpha/2)*cos(beta/2)*sin(gamma/2)
```

---

## 2. Quaternion to Ra/Rb Decomposition

### 2.1 The Decomposition

Given a unit quaternion `q = (w, x, y, z)`, decompose into two complex numbers:

```
Ra = w + i*z    (complex number: real=w, imag=z)
Rb = y + i*x    (complex number: real=y, imag=x)
```

**CRITICAL**: Note the component mapping:
- Ra uses the scalar `w` and z-component `z`
- Rb uses the y-component `y` and x-component `x`

For a unit quaternion: `|Ra|^2 + |Rb|^2 = w^2 + z^2 + y^2 + x^2 = 1`

### 2.2 Geometric Interpretation

The magnitudes and phases of Ra and Rb encode the ZYZ Euler angles:

```
|Ra| = cos(beta/2)
|Rb| = sin(beta/2)

arg(Ra) = (alpha + gamma) / 2
arg(Rb) = (gamma - alpha) / 2 = -(alpha - gamma) / 2
```

Derivation: For a ZYZ rotation q = q_z(alpha) * q_y(beta) * q_z(gamma):
```
Ra = cos(beta/2) * exp(i * (alpha + gamma) / 2)
Rb = sin(beta/2) * exp(i * (gamma - alpha) / 2)
```

This means:
- The **magnitude** of Ra/Rb encodes the polar angle beta
- The **sum of phases** (arg(Ra) + arg(Rb)) = gamma (the final z-rotation)
- The **difference of phases** (arg(Ra) - arg(Rb)) = alpha (the initial z-rotation)

### 2.3 Why This Decomposition Works

This decomposition is related to the **Cayley-Klein parameters** or **spinor representation** of rotations. The key insight is that:

1. The pair (Ra, Rb) uniquely determines the rotation (up to a global sign)
2. The decomposition naturally handles the poles (beta=0 and beta=pi) without singularity
3. Complex arithmetic makes the Wigner D formula elegant

---

## 3. Wigner D Matrix Element Formula

### 3.1 Definition and Convention

The Wigner D matrix element D^l_{m',m}(R) describes how spherical harmonics transform under rotation R:
```
Y_l^{m'}(R^{-1} * r) = sum_m D^l_{m',m}(R) * Y_l^m(r)
```

**Important convention note**: The spherical_functions package uses a specific sign convention that
may differ from textbook conventions. For integration with existing code (like fairchem), verify that
the conventions match by testing against known values or the existing implementation.

### 3.2 General Formula

The exact formula using Ra and Rb is:

```
D^l_{m',m}(R) = sqrt[(l+m)!(l-m)! / ((l+m')!(l-m')!)] *
               sum_{rho} C(l+m', rho) * C(l-m', l-rho-m) * (-1)^rho *
               Ra^{l+m'-rho} * conj(Ra)^{l-rho-m} * Rb^{rho-m'+m} * conj(Rb)^rho
```

Where:
- `C(n, k)` = binomial coefficient "n choose k"
- The sum is over all valid rho values where binomial coefficients are non-zero
- `conj()` denotes complex conjugation

### 3.3 Summation Bounds

The rho summation has bounds determined by when the binomial coefficients are valid:
```
rho_min = max(0, m' - m)
rho_max = min(l + m', l - m)
```

---

## 4. Optimized Two-Branch Algorithm

### 4.1 Why Two Branches?

Direct evaluation can cause overflow/underflow when:
- `|Ra|` is much larger than `|Rb|` (near beta=0)
- `|Rb|` is much larger than `|Ra|` (near beta=pi)

The solution: factor out the dominant term and compute a convergent series.

### 4.2 Case 1: |Ra| >= |Rb| (beta closer to 0)

When `|Ra| >= |Rb|`, the formula from the documentation is:

```
D^l_{m',m}(R) = sqrt[(l+m)!(l-m)!/((l+m')!(l-m')!)] * |Ra|^{2l-2m} * Ra^{m'+m} * Rb^{m-m'}
              * sum_rho C(l+m', rho) * C(l-m', l-rho-m) * (-(rb/ra)^2)^rho
```

**Prefactor decomposition:**

To evaluate `|Ra|^{2l-2m} * Ra^{m'+m} * Rb^{m-m'}` stably, we express it in magnitude-phase form:

1. `Ra^{m'+m}` has magnitude `ra^{m'+m}` and phase `(m'+m)*phia`
2. `Rb^{m-m'}` has magnitude `rb^{m-m'}` and phase `(m-m')*phib`
3. Combined with `|Ra|^{2l-2m}`:
   - Total magnitude: `ra^{2l-2m} * ra^{m'+m} * rb^{m-m'} = ra^{2l-m+m'-m} * rb^{m-m'}`
   - But wait: `ra^{m'+m}` for negative exponent means `ra` raised to a negative power

More carefully, using polar form throughout:
```
Ra = ra * exp(i * phia)
Ra^{m'+m} = ra^{m'+m} * exp(i * (m'+m) * phia)  # works for negative exponents too

Rb = rb * exp(i * phib)
Rb^{m-m'} = rb^{m-m'} * exp(i * (m-m') * phib)
```

So the total prefactor (before polynomial sum) is:
```
prefactor = coeff * ra^{2l-2m+m'+m} * rb^{m-m'} * exp(i * ((m'+m)*phia + (m-m')*phib))
          = coeff * ra^{2l+m'-m} * rb^{m-m'} * exp(i * ((m'+m)*phia + (m-m')*phib))
```

Now, we factor out `(-(rb/ra)^2)^{rho_min}` from the polynomial to avoid computing large powers:
```
(rb/ra)^{2*rho_min} = rb^{2*rho_min} / ra^{2*rho_min}
```

After factoring, the magnitude exponents become:
```
ra_exponent = 2l + m' - m - 2*rho_min
rb_exponent = m - m' + 2*rho_min
```

Where `rho_min = max(0, m' - m)`.

**Phase:**
```
phase = (m' + m) * phia + (m - m') * phib
```

**Polynomial (evaluated via Horner's method):**
```
ratio = -(rb/ra)^2

# Start from rho_max, work down to rho_min
Sum = 1.0
for rho in range(rho_max, rho_min, -1):
    N1 = l + m' - rho + 1
    N2 = l - m' - (l - m - rho) = rho + m - m'  # Actually: l - m' - (l - rho - m - 1) = rho - 1 + m - m' + 1

    # The recurrence is:
    # Sum = 1 + ratio * Sum * (N1_term * N2_term) / (rho * M_term)

Sum *= ratio * ((l+m'-rho+1) * (l-m'-rho+1)) / (rho * (rho + m - m'))
Sum += 1

Result = prefactor * Sum
```

**Corrected Horner recurrence** (from the documentation):

For `|Ra| >= |Rb|`, the parameters are:
- `N1 = l + m'`
- `N2 = l - m`
- `M = m - m'`
- `x = -(rb/ra)^2`

The recurrence (iterating from `rho = rho_max` down to `rho = rho_min + 1`):
```
tot = 1 + x * tot * ((N1 - rho + 1) * (N2 - rho + 1)) / (rho * (M + rho))
    = 1 + x * tot * ((l + m' - rho + 1) * (l - m - rho + 1)) / (rho * (m - m' + rho))
```

### 4.3 Case 2: |Ra| < |Rb| (beta closer to pi)

When `|Ra| < |Rb|`, use the alternative form (from documentation):

```
D^l_{m',m}(R) = (-1)^{l-m} * sqrt[(l+m)!(l-m)!/((l+m')!(l-m')!)] * Ra^{m'+m} * Rb^{m-m'} * |Rb|^{2l-2m}
              * sum_rho C(l+m', l-m-rho) * C(l-m', rho) * (-(ra/rb)^2)^rho
```

**Prefactor decomposition:**

Total magnitude (before factoring):
```
ra^{m'+m} * rb^{m-m'} * rb^{2l-2m} = ra^{m'+m} * rb^{2l - m - m'}
```

After factoring out `(ra/rb)^{2*rho_min_alt}`:
```
ra_exponent = m' + m + 2*rho_min_alt
rb_exponent = 2l - m' - m - 2*rho_min_alt
```

Where `rho_min_alt = max(0, -(m' + m))`.

**Phase:**
```
phase = (m' + m) * phia + (m - m') * phib
```

**Sign factor:**
```
sign = (-1)^{l-m}
```

**Polynomial (evaluated via Horner's method):**
```
ratio = -(ra/rb)^2

Sum = 1.0
for rho in range(rho_max_alt, rho_min_alt, -1):
    # Horner recurrence with N1 = l-m, N2 = l-m', M = m'+m
    Sum = 1 + Sum * ratio * ((l-m-rho+1) * (l-m'-rho+1)) / (rho * (m'+m+rho))

Result = sign * coeff_alt * ra^{ra_exp} * rb^{rb_exp} * exp(i*phase) * Sum
```

The rho bounds for the alternative form:
```
rho_min_alt = max(0, -(m' + m))
rho_max_alt = min(l - m, l - m')
```

### 4.4 Sign Convention

**Important**: There's a factor of `(-1)^{l-m}` in Case 2 that accounts for the variable substitution.

---

## 5. Special Cases (Edge Cases)

### 5.1 When |Ra| ~ 0 (beta ~ pi)

When `|Ra| <= epsilon` (typically epsilon ~ 1e-15):
- The rotation is approximately 180 degrees around some axis in the xy-plane
- Only anti-diagonal elements are non-zero

```
if ra <= epsilon:
    if m' == -m:
        D^l_{m',m} = (-1)^{l-m} * Rb^{2m}
    else:
        D^l_{m',m} = 0
```

**Key point**: This formula does NOT use `arg(Ra)`, avoiding the singularity.

### 5.2 When |Rb| ~ 0 (beta ~ 0)

When `|Rb| <= epsilon`:
- The rotation is approximately around the z-axis only (alpha + gamma rotation)
- Only diagonal elements are non-zero

```
if rb <= epsilon:
    if m' == m:
        D^l_{m',m} = Ra^{2m}  # equivalently Ra^{2m'} since m = m'
    else:
        D^l_{m',m} = 0
```

**Note on sign convention**: With Ra = exp(i*(alpha+gamma)/2), this gives Ra^{2m} = exp(i*m*(alpha+gamma)).
This matches the spherical_functions convention. If your application expects exp(-i*m*(alpha+gamma)),
you may need to conjugate the result or adjust the quaternion convention.

**Key point**: This formula does NOT use `arg(Rb)`, avoiding the singularity.

### 5.3 Why This Avoids Gimbal Lock

The traditional Euler angle approach requires:
- `alpha = atan2(x, z)` - problematic when x ~ 0 and z ~ 0
- `beta = acos(y)` - has gradient singularity at y ~ +-1

The quaternion approach:
- When beta ~ 0: Uses `Ra^{2m}` which is smooth in quaternion components
- When beta ~ pi: Uses `Rb^{2m}` which is smooth in quaternion components
- The `atan2` for phase is only used when the magnitude is large (well-conditioned)

---

## 6. Precomputed Coefficients

### 6.1 Wigner Coefficient for Case 1 (ra >= rb)

For Case 1, the coefficient at rho_min is:

```
rho_min = max(0, m' - m)

Wigner_coeff(l, m', m) = sqrt[(l+m)!(l-m)! / ((l+m')!(l-m')!)] *
                         C(l+m', rho_min) * C(l-m', l-m-rho_min)
```

### 6.2 Wigner Coefficient for Case 2 (ra < rb)

For Case 2, the coefficient at rho_min_alt is:

```
rho_min_alt = max(0, -(m' + m))

Wigner_coeff_alt(l, m', m) = sqrt[(l+m)!(l-m)! / ((l+m')!(l-m')!)] *
                              C(l+m', l-m-rho_min_alt) * C(l-m', rho_min_alt)
```

Note: The binomial coefficient arguments are different from Case 1!

### 6.3 Relationship Between Coefficients

Both coefficients are related by the variable substitution. In practice, you can precompute a single coefficient table and derive the other, or just precompute both.

### 6.4 Storage

The coefficients can be stored in a 1D array with indexing:
```
index(l, m', m) = l*(2l+1)^2/... + (l+m')*(2l+1) + (l+m)
```

Or more simply as a 3D tensor of shape `(l_max+1, 2*l_max+1, 2*l_max+1)`.

For the implementation, it's often cleaner to precompute BOTH Case 1 and Case 2 coefficient tables.

### 6.5 Binomial Coefficients

For the Horner recurrence, you don't need to precompute all binomials - they're computed implicitly through the ratio:
```
C(n, k) / C(n, k-1) = (n - k + 1) / k
```

---

## 7. Complete Pseudocode

```python
def wigner_D_element(l, mp, m, Ra, Rb, epsilon=1e-15):
    """
    Compute D^l_{m',m}(R) from quaternion components Ra, Rb.

    Args:
        l: angular momentum quantum number (non-negative integer)
        mp: m' index (-l <= mp <= l)
        m: m index (-l <= m <= l)
        Ra: complex number = w + i*z from quaternion
        Rb: complex number = y + i*x from quaternion
        epsilon: threshold for near-zero detection

    Returns:
        Complex value D^l_{m',m}
    """
    # Get magnitudes and phases
    ra = abs(Ra)
    rb = abs(Rb)
    phia = cmath.phase(Ra)  # = atan2(z, w)
    phib = cmath.phase(Rb)  # = atan2(x, y)

    # Special case: Ra ~ 0 (beta ~ pi)
    if ra <= epsilon:
        if mp == -m:
            return ((-1) ** (l - m)) * (Rb ** (2 * m))
        else:
            return 0.0 + 0.0j

    # Special case: Rb ~ 0 (beta ~ 0)
    if rb <= epsilon:
        if mp == m:
            return Ra ** (2 * m)
        else:
            return 0.0 + 0.0j

    # General case: both Ra and Rb significant
    rho_min = max(0, mp - m)
    rho_max = min(l + mp, l - m)

    # Get precomputed coefficient
    coeff = wigner_coefficient(l, mp, m)  # Lookup table

    if ra >= rb:
        # Case 1: |Ra| >= |Rb|
        # Parameters for Horner recurrence
        # N1 = l + m', N2 = l - m, M = m - m'
        ratio = -(rb * rb) / (ra * ra)  # Negative ratio!

        # Horner evaluation from rho_max down to rho_min+1
        Sum = 1.0
        for rho in range(rho_max, rho_min, -1):
            # factor = x * (N1 - rho + 1) * (N2 - rho + 1) / (rho * (M + rho))
            N1_term = l + mp - rho + 1
            N2_term = l - m - rho + 1
            M_term = m - mp + rho
            factor = ratio * N1_term * N2_term / (rho * M_term)
            Sum = 1.0 + Sum * factor

        # Compute prefactor magnitude exponents
        # From: ra^{2l-2m} * ra^{m'+m} * rb^{m-m'} factoring out (rb/ra)^{2*rho_min}
        # ra exponent: 2l - 2m + m' + m - 2*rho_min = 2l + m' - m - 2*rho_min
        # rb exponent: m - m' + 2*rho_min
        prefactor_exp_ra = 2 * l + mp - m - 2 * rho_min
        prefactor_exp_rb = m - mp + 2 * rho_min

        magnitude = coeff * (ra ** prefactor_exp_ra) * (rb ** prefactor_exp_rb)
        phase = (mp + m) * phia + (m - mp) * phib

        prefactor = magnitude * cmath.exp(1j * phase)

        return prefactor * Sum

    else:
        # Case 2: |Ra| < |Rb|
        # Use alternative form with variable substitution rho -> l - m - rho
        # Parameters for Horner recurrence (different from Case 1!)
        # N1 = l - m, N2 = l - m', M = m' + m
        ratio = -(ra * ra) / (rb * rb)

        rho_min_alt = max(0, -(mp + m))
        rho_max_alt = min(l - m, l - mp)

        coeff_alt = wigner_coefficient_alt(l, mp, m)  # Different coefficient!

        Sum = 1.0
        for rho in range(rho_max_alt, rho_min_alt, -1):
            # factor = x * (N1 - rho + 1) * (N2 - rho + 1) / (rho * (M + rho))
            N1_term = l - m - rho + 1
            N2_term = l - mp - rho + 1
            M_term = mp + m + rho
            factor = ratio * N1_term * N2_term / (rho * M_term)
            Sum = 1.0 + Sum * factor

        sign = ((-1) ** (l - m))

        # Compute prefactor magnitude exponents
        # From: ra^{m'+m} * rb^{m-m'} * rb^{2l-2m} factoring out (ra/rb)^{2*rho_min_alt}
        # ra exponent: m' + m + 2*rho_min_alt
        # rb exponent: 2l - 2m + m - m' - 2*rho_min_alt = 2l - m - m' - 2*rho_min_alt
        prefactor_exp_ra = mp + m + 2 * rho_min_alt
        prefactor_exp_rb = 2 * l - mp - m - 2 * rho_min_alt

        magnitude = sign * coeff_alt * (ra ** prefactor_exp_ra) * (rb ** prefactor_exp_rb)
        phase = (mp + m) * phia + (m - mp) * phib

        prefactor = magnitude * cmath.exp(1j * phase)

        return prefactor * Sum
```

---

## 8. PyTorch Implementation Considerations

### 8.1 Complex Tensor Operations

```python
# Create Ra, Rb from quaternion
Ra = torch.complex(q[..., 0], q[..., 3])  # w + i*z
Rb = torch.complex(q[..., 2], q[..., 1])  # y + i*x

# Polar decomposition
ra = torch.abs(Ra)
rb = torch.abs(Rb)
phia = torch.angle(Ra)
phib = torch.angle(Rb)
```

### 8.2 Differentiable Branching

Use `torch.where` for smooth branching:
```python
# Instead of if-else
result = torch.where(
    ra <= epsilon,
    special_case_ra_zero(...),
    torch.where(
        rb <= epsilon,
        special_case_rb_zero(...),
        general_case(...)
    )
)
```

### 8.3 Vectorization Strategy

- Batch dimension: quaternions (N edges)
- Output shape: (N, 2*l+1, 2*l+1) for each l
- Or concatenated for all l: (N, sum of (2*l+1)^2 for l in 0..l_max)

### 8.4 Coefficient Storage

```python
# Precompute as buffer
self.register_buffer('wigner_coeffs', compute_wigner_coefficients(l_max))
```

### 8.5 Numerical Stability

1. The ratio `-(rb/ra)^2` or `-(ra/rb)^2` is always in [-1, 0], ensuring convergence
2. Use `torch.clamp` on ra, rb before division to avoid division by zero
3. The phase calculation `torch.angle` is well-conditioned when magnitude is not tiny

---

## 9. Symmetry Relations

For efficiency, compute only a subset of elements and derive the rest:

```
D^l_{-m',-m}(R) = (-1)^{m'+m} * conj(D^l_{m',m}(R))
D^l_{m,m'}(R) = conj(D^l_{m',m}(conj(R)))
D^l_{-m,-m'}(R) = (-1)^{m'+m} * D^l_{m',m}(conj(R))
```

This allows computing only ~1/4 of the elements.

---

## 10. Verification Tests

### 10.1 Known Values

For identity rotation q = (1, 0, 0, 0):
- Ra = 1, Rb = 0
- D^l_{m',m} = delta_{m',m} (identity matrix)

For 180-degree rotation around x-axis q = (0, 1, 0, 0):
- Ra = 0, Rb = i
- D^l_{m',m} = (-1)^{l-m} * delta_{m',-m} * i^{2m}

### 10.2 Euler Angle Comparison

For any quaternion, compute Wigner D via both:
1. Quaternion algorithm (this document)
2. Traditional Euler angle ZYZ decomposition + standard formula

They should match to numerical precision.

### 10.3 Rotation Composition

D(R1 * R2) should equal D(R1) @ D(R2) for matrix multiplication.

---

## 11. References

- [spherical_functions documentation](https://moble.github.io/spherical_functions/)
- [Wigner D Matrices page](https://moble.github.io/spherical_functions/WignerDMatrices.html)
- [Polynomial evaluation with factorials](https://moble.github.io/spherical_functions/PolynomialsWithFactorials.html)
- [spherical_functions GitHub](https://github.com/moble/spherical_functions)
- [quaternion package](https://github.com/moble/quaternion)

---

## 12. Summary: Key Points for Implementation

1. **Ra = w + i*z, Rb = y + i*x** - Get the component mapping right!

2. **Three cases**:
   - ra <= epsilon: Only anti-diagonal, use Rb^{2m}
   - rb <= epsilon: Only diagonal, use Ra^{2m}
   - Both significant: Two-branch algorithm based on ra vs rb

3. **The ratio is NEGATIVE**: `ratio = -(smaller/larger)^2`

4. **Horner evaluation goes BACKWARDS**: from rho_max down to rho_min+1

5. **Phase accumulates**: `(mp + m) * phia + (m - mp) * phib`

6. **The Wigner coefficient includes rho_min**: It's not just the sqrt of factorials

7. **Sign factor in Case 2**: `(-1)^{l-m}` from variable substitution

8. **Gradient safety**: The atan2 (for phases) is only used when magnitude is large

---

## 13. Common Implementation Pitfalls

### 13.1 Component Mapping Errors

**WRONG**: `Ra = w + i*x` (using x instead of z)
**CORRECT**: `Ra = w + i*z`

**WRONG**: `Rb = x + i*y` (wrong order)
**CORRECT**: `Rb = y + i*x`

### 13.2 Horner Recurrence Direction

**WRONG**: Iterating from `rho_min` to `rho_max`
**CORRECT**: Iterating from `rho_max` down to `rho_min + 1`

The recurrence is:
```
tot = 1  # Initialize
for rho in range(rho_max, rho_min, -1):  # DESCENDING
    tot = 1 + ratio * tot * (terms)
```

### 13.3 Missing Negative Sign in Ratio

**WRONG**: `ratio = (rb/ra)^2`
**CORRECT**: `ratio = -(rb/ra)^2`

The negative sign comes from the `(-1)^rho` factor in the original sum.

### 13.4 Exponent Errors

Case 1 (ra >= rb):
- ra exponent: `2*l + m' - m - 2*rho_min` (NOT `2*l - m' - m - ...`)
- rb exponent: `m - m' + 2*rho_min`

Case 2 (ra < rb):
- ra exponent: `m' + m + 2*rho_min_alt`
- rb exponent: `2*l - m' - m - 2*rho_min_alt`

### 13.5 Sign Factor in Case 2

Don't forget `(-1)^{l-m}` in Case 2. This factor is essential for correctness.

### 13.6 Coefficient Indexing

The Wigner coefficient depends on `rho_min` (or `rho_min_alt`), which varies with (l, m', m). Make sure the coefficient lookup uses the correct rho_min for each case.

### 13.7 Division by Zero in Horner Recurrence

When `rho = 0`, there's a division by zero. This is avoided because:
- Case 1: `rho_min = max(0, m'-m)`, so if `m' <= m`, the loop starts at `rho > 0`
- When `rho_min = 0`, the loop is `range(rho_max, 0, -1)`, never reaching `rho = 0`

Similarly, `M + rho` can be zero. Check that `M = m - m'` for Case 1 and `M = m' + m` for Case 2.

### 13.8 Edge Cases in Special Formulas

When `ra <= epsilon`:
- The formula `(-1)^{l-m} * Rb^{2m}` only applies when `m' = -m`
- For `m' != -m`, the element is exactly 0

When `rb <= epsilon`:
- The formula `Ra^{2m}` only applies when `m' = m`
- For `m' != m`, the element is exactly 0

---

## 14. Implementation Verification Checklist

Before trusting your implementation, verify:

1. [ ] **Identity rotation**: q = (1,0,0,0) gives D = identity matrix
2. [ ] **90-degree rotations**: Compare with known analytical values
3. [ ] **Symmetry**: D^l_{-m',-m} = (-1)^{m'+m} * conj(D^l_{m',m})
4. [ ] **Unitarity**: D @ D^H = I (D matrices are unitary)
5. [ ] **Composition**: D(q1 * q2) = D(q1) @ D(q2)
6. [ ] **Euler angle equivalence**: Compare with ZYZ Euler angle formula
7. [ ] **Gradient stability**: No NaN/Inf gradients for any rotation
8. [ ] **Edge cases**: Test with beta ~ 0 and beta ~ pi

---

## 15. Simplified Implementation Strategy

For a first implementation, consider this simplified approach:

1. **Always use the general formula** (don't switch on ra vs rb)
2. **Use double precision** to avoid overflow/underflow for small l
3. **Add the two-branch algorithm later** if needed for large l

For l <= 6 (typical in molecular ML), numerical stability is usually fine without the two-branch algorithm. You can start simple and optimize later.

However, you MUST handle the special cases (ra ~ 0, rb ~ 0) to avoid computing phases of near-zero complex numbers.
