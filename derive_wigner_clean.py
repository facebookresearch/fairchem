#!/usr/bin/env python3
"""
Improved symbolic derivation of Wigner D matrix formulas for l=3 and l=4.

This generates clean PyTorch code with proper symbolic coefficients like
sqrt(3), sqrt(5), sqrt(6), etc.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import torch
import math
import sympy as sp
from itertools import product
from pathlib import Path
from fractions import Fraction


def derive_coefficients_numerical(ell, apply_euler_transform=True):
    """
    Derive Wigner D polynomial coefficients numerically with high precision.
    """
    size = 2 * ell + 1
    degree = 2 * ell

    # Generate all monomials of degree exactly 2l
    monomials = []
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            for c in range(degree + 1 - a - b):
                d = degree - a - b - c
                monomials.append((a, b, c, d))

    n_monomials = len(monomials)
    print(f"l={ell}: {n_monomials} monomials of degree {degree}")

    n_samples = n_monomials + 50

    torch.manual_seed(42)
    q_raw = torch.randn(n_samples, 4, dtype=torch.float64)
    q = q_raw / q_raw.norm(dim=1, keepdim=True)

    # Build Vandermonde matrix
    X = torch.zeros(n_samples, n_monomials, dtype=torch.float64)
    for i, (a, b, c, d) in enumerate(monomials):
        X[:, i] = (q[:, 0] ** a) * (q[:, 1] ** b) * (q[:, 2] ** c) * (q[:, 3] ** d)

    from fairchem.core.models.uma.common.wigner_d_axis_angle import (
        get_so3_generators,
        quaternion_to_axis_angle,
    )

    gens = get_so3_generators(ell, torch.float64, torch.device('cpu'))
    K_x = gens['K_x'][ell]
    K_y = gens['K_y'][ell]
    K_z = gens['K_z'][ell]

    axis, angle = quaternion_to_axis_angle(q)

    D_ref = torch.zeros(n_samples, size, size, dtype=torch.float64)
    for i in range(n_samples):
        n = axis[i]
        K = n[0] * K_x + n[1] * K_y + n[2] * K_z
        D_ref[i] = torch.linalg.matrix_exp(angle[i] * K)

    coefficients = torch.zeros(size, size, n_monomials, dtype=torch.float64)

    for i in range(size):
        for j in range(size):
            y = D_ref[:, i, j]
            c, _, _, _ = torch.linalg.lstsq(X, y.unsqueeze(1))
            coefficients[i, j, :] = c.squeeze()

            y_pred = X @ c.squeeze()
            max_err = (y - y_pred).abs().max().item()
            if max_err > 1e-10:
                print(f"  D[{i},{j}]: residual = {max_err:.2e}")

    return coefficients.numpy(), monomials


def identify_coefficient(c_val, threshold=1e-9):
    """
    Identify a numerical coefficient as a symbolic expression.

    Returns a tuple (numerator, sqrt_part, denominator) where the value is
    numerator * sqrt(sqrt_part) / denominator.
    """
    if abs(c_val) < threshold:
        return None

    sign = 1 if c_val > 0 else -1
    c_abs = abs(c_val)

    # Try integer
    rounded = round(c_abs)
    if abs(c_abs - rounded) < threshold:
        return (sign * rounded, 1, 1)

    # Try sqrt(n) for small n
    for n in [2, 3, 5, 6, 7, 10, 14, 15, 21, 35, 42, 70, 105]:
        sqrt_n = math.sqrt(n)
        for mult in range(1, 300):
            for denom in [1, 2, 4]:
                val = mult * sqrt_n / denom
                if abs(c_abs - val) < threshold:
                    return (sign * mult, n, denom)

    # Try rational
    for denom in range(1, 10):
        for numer in range(1, 500):
            if abs(c_abs - numer/denom) < threshold:
                return (sign * numer, 1, denom)

    # Give up, return float
    return (c_val, 0, 1)  # 0 signals "use float"


def format_coefficient_symbolic(coeff_tuple):
    """Format a coefficient tuple as a code string."""
    if coeff_tuple is None:
        return None

    numer, sqrt_part, denom = coeff_tuple

    if sqrt_part == 0:
        # Use float
        return f"{numer:.10f}"

    abs_numer = abs(numer)
    sign = "-" if numer < 0 else ""

    if sqrt_part == 1:
        # Integer or rational
        if denom == 1:
            return f"{numer}"
        else:
            return f"{numer}/{denom}"
    else:
        # Has sqrt
        sqrt_str = f"sqrt{sqrt_part}" if sqrt_part in [3, 5, 6, 7, 10, 14, 15, 21, 35, 42, 70, 105] else f"math.sqrt({sqrt_part})"

        if denom == 1:
            if abs_numer == 1:
                return f"{sign}{sqrt_str}"
            else:
                return f"{sign}{abs_numer}*{sqrt_str}"
        else:
            if abs_numer == 1:
                return f"{sign}{sqrt_str}/{denom}"
            else:
                return f"{sign}{abs_numer}*{sqrt_str}/{denom}"


def generate_pytorch_code_clean(ell, coefficients, monomials, threshold=1e-9):
    """Generate clean PyTorch code with symbolic coefficients."""
    size = 2 * ell + 1
    degree = 2 * ell

    lines = []
    lines.append(f"def quaternion_to_wigner_d_l{ell}(q: torch.Tensor) -> torch.Tensor:")
    lines.append(f'    """')
    lines.append(f'    Convert quaternion directly to {size}x{size} l={ell} Wigner D matrix.')
    lines.append(f'    ')
    lines.append(f'    Uses degree-{degree} polynomial formulas in quaternion components (w,x,y,z).')
    lines.append(f'    The matrix is built using torch.stack to minimize autograd graph nodes,')
    lines.append(f'    which significantly speeds up the backward pass compared to indexed')
    lines.append(f'    assignment (D[:, i, j] = ...).')
    lines.append(f'    ')
    lines.append(f'    Output matches the Euler-aligned basis of the axis-angle implementation.')
    lines.append(f'    ')
    lines.append(f'    Args:')
    lines.append(f'        q: Quaternions of shape (N, 4) in (w, x, y, z) convention')
    lines.append(f'    ')
    lines.append(f'    Returns:')
    lines.append(f'        Wigner D matrices of shape (N, {size}, {size}) for l={ell}')
    lines.append(f'    """')
    lines.append(f'    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]')
    lines.append(f'    ')

    # Identify which sqrt values are needed
    sqrt_needed = set()
    for i in range(size):
        for j in range(size):
            for k, (a, b, c, d) in enumerate(monomials):
                c_val = coefficients[i, j, k]
                if abs(c_val) > threshold:
                    result = identify_coefficient(c_val, threshold)
                    if result and result[1] > 1:
                        sqrt_needed.add(result[1])

    # Generate sqrt constants
    if sqrt_needed:
        lines.append(f"    # Precompute square root constants")
        for n in sorted(sqrt_needed):
            lines.append(f"    sqrt{n} = {math.sqrt(n):.16f}  # math.sqrt({n})")
        lines.append(f"    ")

    # Generate powers
    for var, name in [(0, 'w'), (1, 'x'), (2, 'y'), (3, 'z')]:
        max_pow = max(m[var] for m in monomials if any(abs(coefficients[i, j, k]) > threshold
                      for i in range(size) for j in range(size) for k, mm in enumerate(monomials) if mm == m))
        if max_pow >= 2:
            lines.append(f"    # Powers of {name}")
            for p in range(2, max_pow + 1):
                if p == 2:
                    lines.append(f"    {name}{p} = {name}*{name}")
                else:
                    lines.append(f"    {name}{p} = {name}{p-1}*{name}")
            lines.append(f"    ")

    # Generate matrix elements
    lines.append(f"    # Build all {size*size} elements")

    for i in range(size):
        lines.append(f"    # Row {i}")
        for j in range(size):
            terms = []
            for k, (a, b, c, d) in enumerate(monomials):
                c_val = coefficients[i, j, k]
                if abs(c_val) < threshold:
                    continue

                coeff_tuple = identify_coefficient(c_val, threshold)
                c_str = format_coefficient_symbolic(coeff_tuple)

                # Format monomial
                mon_parts = []
                for power, name in [(a, 'w'), (b, 'x'), (c, 'y'), (d, 'z')]:
                    if power == 0:
                        continue
                    elif power == 1:
                        mon_parts.append(name)
                    else:
                        mon_parts.append(f"{name}{power}")

                mon_str = "*".join(mon_parts) if mon_parts else "1"

                if c_str == "1":
                    terms.append(mon_str)
                elif c_str == "-1":
                    terms.append(f"-{mon_str}")
                else:
                    terms.append(f"{c_str}*{mon_str}")

            if not terms:
                expr = "torch.zeros_like(w)"
            else:
                expr = " + ".join(terms)
                expr = expr.replace("+ -", "- ")

            lines.append(f"    d{i}{j} = {expr}")
        lines.append(f"    ")

    # Stack into matrix
    lines.append(f"    # Stack into matrix - builds the whole matrix in one operation")
    lines.append(f"    # This minimizes autograd graph nodes compared to indexed assignment")
    lines.append(f"    D = torch.stack([")
    for i in range(size):
        row = ", ".join([f"d{i}{j}" for j in range(size)])
        lines.append(f"        torch.stack([{row}], dim=-1),")
    lines.append(f"    ], dim=-2)")
    lines.append(f"    ")
    lines.append(f"    return D")

    return "\n".join(lines)


def verify_formula(ell, func):
    """Verify the generated formula matches reference implementation."""
    from fairchem.core.models.uma.common.wigner_d_axis_angle import (
        get_so3_generators,
        quaternion_to_axis_angle,
    )

    torch.manual_seed(123)
    n_test = 100
    q_raw = torch.randn(n_test, 4, dtype=torch.float64)
    q = q_raw / q_raw.norm(dim=1, keepdim=True)

    # Reference
    size = 2 * ell + 1
    gens = get_so3_generators(ell, torch.float64, torch.device('cpu'))
    K_x = gens['K_x'][ell]
    K_y = gens['K_y'][ell]
    K_z = gens['K_z'][ell]

    axis, angle = quaternion_to_axis_angle(q)
    D_ref = torch.zeros(n_test, size, size, dtype=torch.float64)
    for i in range(n_test):
        n = axis[i]
        K = n[0] * K_x + n[1] * K_y + n[2] * K_z
        D_ref[i] = torch.linalg.matrix_exp(angle[i] * K)

    # Generated formula
    D_test = func(q)

    max_err = (D_ref - D_test).abs().max().item()
    print(f"l={ell}: max error = {max_err:.2e}")
    return max_err < 1e-10


if __name__ == "__main__":
    print("="*60)
    print("Deriving Wigner D polynomial formulas for l=3 and l=4")
    print("="*60)

    for ell in [3, 4]:
        print(f"\n{'='*60}")
        print(f"l = {ell}")
        print(f"{'='*60}")

        coefficients, monomials = derive_coefficients_numerical(ell)
        code = generate_pytorch_code_clean(ell, coefficients, monomials)

        print("\nGenerated PyTorch code:")
        print("-"*40)
        # Print first 100 lines for preview
        code_lines = code.split('\n')
        for line in code_lines[:100]:
            print(line)
        if len(code_lines) > 100:
            print(f"... ({len(code_lines) - 100} more lines)")
        print("-"*40)

        # Save to file
        output_file = Path(__file__).parent / f"wigner_l{ell}_clean.py"
        with open(output_file, 'w') as f:
            f.write(f"# Auto-generated Wigner D formula for l={ell}\n")
            f.write("# Copyright (c) Meta Platforms, Inc. and affiliates.\n\n")
            f.write("import torch\n")
            f.write("import math\n\n")
            f.write(code)
            f.write("\n")
        print(f"\nSaved to {output_file}")

        # Verify
        print("\nVerifying...")
        exec(f"import torch; import math\n{code}")
        exec(f"verify_formula({ell}, quaternion_to_wigner_d_l{ell})")
