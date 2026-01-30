# Wigner D Matrix Computation: Understanding and Implementation Plan

## The Problem

SO2 convolutions in fairchem require rotating edges to align with the y-axis. This rotation is encoded as a Wigner D matrix in the spherical harmonics basis. The current implementation uses Euler angles, which have singularities (gimbal lock) when edges are nearly y-aligned.

### Current Implementation (rotation.py)

```python
def init_edge_rot_euler_angles(edge_distance_vec):
    xyz = normalize(edge_distance_vec)
    x, y, z = split(xyz)

    beta = arccos(y)           # SINGULARITY: gradient → ∞ as y → ±1
    alpha = atan2(x, z)        # SINGULARITY: undefined when x ≈ z ≈ 0
    gamma = random(0, 2π)

    return -gamma, -beta, -alpha  # intrinsic → extrinsic swap
```

The `Safeacos` class clamps gradients to prevent NaN/Inf, but this introduces **gradient bias** rather than correct gradients. This is problematic for training models with gradient-based force/stress prediction.

## The Solution: Quaternion-Based Wigner D

Based on the spherical_functions package by Mike Boyle, we can compute Wigner D matrices directly from quaternions without ever computing Euler angles.

### Key Insight: Ra/Rb Decomposition

A unit quaternion q = (w, x, y, z) can be decomposed into two complex numbers:
- **Ra = w + iz** (encodes cos(β/2) and (α+γ)/2)
- **Rb = y + ix** (encodes sin(β/2) and (γ-α)/2)

These satisfy |Ra|² + |Rb|² = 1.

### Relationship to Euler Angles (for understanding, NOT for computation)

For a rotation with Euler angles (α, β, γ) in the ZYZ convention:
```
Ra = cos(β/2) · exp(i·(α+γ)/2)
Rb = sin(β/2) · exp(i·(γ-α)/2)
```

Therefore:
- |Ra| = cos(β/2), |Rb| = sin(β/2)
- arg(Ra) = (α+γ)/2, arg(Rb) = (γ-α)/2
- β = 2·atan2(|Rb|, |Ra|) — but we DON'T need to compute this!

### Why Re-extracting Euler Angles Doesn't Help

If we try:
1. edge_vec → quaternion (singularity-free via half-angle formula)
2. quaternion → Euler angles → Wigner D

Step 2 still requires atan2/arccos which have the same singularities. The solution is to compute Wigner D directly from Ra/Rb.

## The Four Cases for Wigner D Computation

### Case 1: |Ra| ≈ 0 (β ≈ π, 180° rotation)

Only the anti-diagonal elements are non-zero:
```
D^ℓ_{-m,m}(R) = (-1)^{ℓ-m} · Rb^{2m}
```

This is a 180° flip combined with z-rotation.

### Case 2: |Rb| ≈ 0 (β ≈ 0, near-identity)

Only the diagonal elements are non-zero:
```
D^ℓ_{m,m}(R) = Ra^{2m}
```

This is essentially a pure z-rotation.

### Case 3: |Ra| ≥ |Rb| (general case, branch 1)

Use the full formula with ratio r = -(|Rb|/|Ra|)²:
```
D^ℓ_{m',m}(R) = coeff · |Ra|^{2ℓ-2m} · Ra^{m'+m} · Rb^{m-m'} · Σ_ρ [binomials · r^ρ]
```

The phase is: (m'+m)·arg(Ra) + (m-m')·arg(Rb)

### Case 4: |Ra| < |Rb| (general case, branch 2)

Use alternative formula with ratio r = -(|Ra|/|Rb|)²:
```
D^ℓ_{m',m}(R) = (-1)^{ℓ-m} · coeff · Ra^{m'+m} · Rb^{m-m'} · |Rb|^{2ℓ-2m} · Σ_ρ [binomials · r^ρ]
```

### Why This Works

- **No arccos**: We never compute arccos(y) directly
- **No problematic atan2**: The phases arg(Ra) and arg(Rb) are computed via torch.angle(), but crucially:
  - When |Ra| ≈ 0, we use Case 1 which doesn't need arg(Ra)
  - When |Rb| ≈ 0, we use Case 2 which doesn't need arg(Rb)
  - In Cases 3 & 4, both magnitudes are significant, so both phases are well-conditioned

## Complex vs Real Spherical Harmonics

The spherical_functions formulas produce **complex** Wigner D matrices. fairchem uses **real** spherical harmonics (e3nn convention).

### Transformation

The relationship is:
```
D_real = U† · D_complex · U
```

where U is the unitary matrix transforming complex → real spherical harmonics.

### Real Spherical Harmonics Convention

For each ℓ, real spherical harmonics are defined as:
- m > 0: Y^m_ℓ(real) = (1/√2) · [Y^{-m}_ℓ(complex) + (-1)^m · Y^m_ℓ(complex)]
- m = 0: Y^0_ℓ(real) = Y^0_ℓ(complex)
- m < 0: Y^m_ℓ(real) = (i/√2) · [Y^m_ℓ(complex) - (-1)^{|m|} · Y^{-m}_ℓ(complex)]

The U matrix encodes this transformation.

## Implementation Plan

### Step 1: Edge Vector to Quaternion

Use the half-angle formula (singularity-free for edge ≠ -y):
```python
def edge_to_quaternion(edge_vec, gamma=None):
    # Rotate +y to edge direction
    # q = normalize(1 + dot(+y, edge), cross(+y, edge))
    # For +y = (0,1,0): dot = y, cross = (z, 0, -x)
    # So q = normalize(1+y, z, 0, -x)

    # Handle singularity at edge = -y with fallback
```

### Step 2: Quaternion to Ra/Rb

```python
def quaternion_to_ra_rb(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    Ra = torch.complex(w, z)
    Rb = torch.complex(y, x)
    return Ra, Rb
```

### Step 3: Complex Wigner D from Ra/Rb

Implement the four cases directly from spherical_functions.

### Step 4: Complex to Real Transformation

Precompute U matrix, apply D_real = U† @ D_complex @ U.

### Step 5: Integration

Provide drop-in replacement for `get_wigner_from_edge_vectors_real()`.

## Testing Strategy

1. **Correctness for general rotations**: Compare quaternion approach to Euler approach for random non-singular rotations. They should match within numerical precision.

2. **Correctness for y-aligned edges**: Verify the Wigner D matrices are valid (orthogonal, det=1) for y-aligned edges where Euler approach fails.

3. **Gradient stability**: Compute gradients through both approaches for y-aligned edges. Euler approach should have large/biased gradients; quaternion approach should have bounded, correct gradients.

4. **Integration test**: Run a forward pass through the full model with y-aligned edges using both approaches. Quaternion should not produce NaN/Inf.

## References

- spherical_functions package: https://github.com/moble/spherical_functions
- Math documentation: ~/spherical_functions/math.html
- Conventions: ~/spherical_functions/conventions.html
- Paper: arxiv:1302.2919 (Boyle's derivation)
