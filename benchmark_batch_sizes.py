"""
Performance comparison: Euler vs Quaternion for varying batch sizes.
"""

import torch
import time
from pathlib import Path

from fairchem.core.models.uma.common.rotation import init_edge_rot_euler_angles, eulers_to_wigner
from fairchem.core.models.uma.common.wigner_d_quaternion import precompute_all_wigner_tables, get_wigner_from_edge_vectors

jd_path = Path(__file__).parent.resolve() / "src" / "fairchem" / "core" / "models" / "uma" / "Jd.pt"
dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lmax = 6

Jd = torch.load(jd_path, map_location=device, weights_only=True)
Jd = [J.to(dtype=dtype) for J in Jd]
coeffs, U_blocks = precompute_all_wigner_tables(lmax, dtype=dtype, device=device)

batch_sizes = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000]
batch_sizes = list(range(10000,100000,10000))
n_warmup = 5
n_runs = 20

print("=" * 80)
print("PERFORMANCE: Euler vs Quaternion by Batch Size")
print("=" * 80)
print(f"lmax = {lmax}, dtype = {dtype}, device = {device}")
print(f"Warmup runs: {n_warmup}, Timed runs: {n_runs}")
print()

# Correctness checks
print("CORRECTNESS CHECKS")
print("-" * 80)

def check_wigner_correctness(W, method_name, n_edges):
    """Check Wigner D matrix correctness."""
    errors = []

    # Check shape: should be (n_edges, (lmax+1)^2, (lmax+1)^2)
    expected_size = (lmax + 1) ** 2
    if W.shape != (n_edges, expected_size, expected_size):
        errors.append(f"wrong shape {W.shape}, expected ({n_edges}, {expected_size}, {expected_size})")

    # Check unitarity of each l-block (D @ D.H should be identity)
    # Wigner D matrices are unitary, so each block should satisfy this
    max_unitarity_err = 0.0
    offset = 0
    for l in range(lmax + 1):
        block_size = 2 * l + 1
        D_l = W[:, offset:offset+block_size, offset:offset+block_size]
        I_expected = torch.eye(block_size, dtype=W.dtype, device=W.device).unsqueeze(0)
        DDH = torch.bmm(D_l, D_l.conj().transpose(-2, -1))
        unitarity_err = (DDH - I_expected).abs().max().item()
        max_unitarity_err = max(max_unitarity_err, unitarity_err)
        offset += block_size

    if max_unitarity_err > 1e-6:
        errors.append(f"unitarity error = {max_unitarity_err:.2e}")

    if errors:
        print(f"  {method_name}: FAIL - {', '.join(errors)}")
        return False
    else:
        print(f"  {method_name}: OK (unitarity err = {max_unitarity_err:.2e})")
        return True

# Test with a moderate batch size
test_batch = 100
torch.manual_seed(42)
test_edges = torch.randn(test_batch, 3, dtype=dtype, device=device)
test_edges = torch.nn.functional.normalize(test_edges, dim=-1)

# Euler
angles = init_edge_rot_euler_angles(test_edges)
W_euler = eulers_to_wigner(angles, 0, lmax, Jd)
euler_ok = check_wigner_correctness(W_euler, "Euler", test_batch)

# Quaternion
W_quat,_ = get_wigner_from_edge_vectors(test_edges, coeffs, U_blocks)
quat_ok = check_wigner_correctness(W_quat, "Quaternion", test_batch)

# Compare l=0 block (should be identical since it's always 1)
l0_diff = (W_euler[:, 0, 0] - W_quat[:, 0, 0]).abs().max().item()
print(f"  l=0 block difference: {l0_diff:.2e} (should be ~0)")

# Compare Frobenius norms per edge (should be same since unitary matrices have fixed norm)
euler_norms = torch.linalg.norm(W_euler, dim=(1, 2))
quat_norms = torch.linalg.norm(W_quat, dim=(1, 2))
norm_diff = (euler_norms - quat_norms).abs().max().item()
print(f"  Frobenius norm difference: {norm_diff:.2e} (should be ~0)")

# Check that l=1 block correctly aligns random edges to +y axis
print("\n  l=1 alignment check (edges -> +y axis):")
n_random_checks = 5
random_indices = torch.randperm(test_batch)[:n_random_checks]
y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)  # +y axis

alignment_ok = True
for method_name, W in [("Euler", W_euler), ("Quaternion", W_quat)]:
    max_err = 0.0
    for idx in random_indices:
        edge = test_edges[idx]
        # Apply l=1 block (indices 1:4 in the full Wigner matrix)
        D_l1 = W[idx, 1:4, 1:4]
        rotated = D_l1 @ edge
        err = (rotated - y_axis).abs().max().item()
        max_err = max(max_err, err)
    if max_err > 1e-6:
        print(f"    {method_name}: FAIL (max error = {max_err:.2e})")
        alignment_ok = False
    else:
        print(f"    {method_name}: OK (max error = {max_err:.2e})")

if not (euler_ok and quat_ok and alignment_ok):
    print("\nWARNING: Correctness checks failed!")
print()

# Forward only
print("FORWARD PASS ONLY")
print("-" * 80)
print(f"{'Batch':<8} {'Euler (ms)':<15} {'Quat (ms)':<15} {'Ratio':<10} {'Euler/edge (µs)':<18} {'Quat/edge (µs)':<18}")
print("-" * 80)

forward_results = []

for n_edges in batch_sizes:
    torch.manual_seed(42)
    edges_base = torch.randn(n_edges, 3, dtype=dtype, device=device)
    edges_base = torch.nn.functional.normalize(edges_base, dim=-1)

    # Euler warmup
    for _ in range(n_warmup):
        angles = init_edge_rot_euler_angles(edges_base)
        W = eulers_to_wigner(angles, 0, lmax, Jd)

    # Euler timed
    start = time.perf_counter()
    for _ in range(n_runs):
        angles = init_edge_rot_euler_angles(edges_base)
        W = eulers_to_wigner(angles, 0, lmax, Jd)
    t_euler = (time.perf_counter() - start) / n_runs * 1000

    # Quaternion warmup
    for _ in range(n_warmup):
        _, W = get_wigner_from_edge_vectors(edges_base, coeffs, U_blocks)

    # Quaternion timed
    start = time.perf_counter()
    for _ in range(n_runs):
        _, W = get_wigner_from_edge_vectors(edges_base, coeffs, U_blocks)
    t_quat = (time.perf_counter() - start) / n_runs * 1000

    ratio = t_quat / t_euler
    euler_per_edge = t_euler / n_edges * 1000  # microseconds
    quat_per_edge = t_quat / n_edges * 1000

    forward_results.append((n_edges, t_euler, t_quat, ratio))

    print(f"{n_edges:<8} {t_euler:<15.3f} {t_quat:<15.3f} {ratio:<10.2f} {euler_per_edge:<18.2f} {quat_per_edge:<18.2f}")

print()
print("FORWARD + BACKWARD PASS")
print("-" * 80)
print(f"{'Batch':<8} {'Euler (ms)':<15} {'Quat (ms)':<15} {'Ratio':<10} {'Euler/edge (µs)':<18} {'Quat/edge (µs)':<18}")
print("-" * 80)

backward_results = []

for n_edges in batch_sizes:
    torch.manual_seed(42)
    edges_base = torch.randn(n_edges, 3, dtype=dtype, device=device)
    edges_base = torch.nn.functional.normalize(edges_base, dim=-1)

    # Euler warmup
    for _ in range(n_warmup):
        edges = edges_base.clone().requires_grad_(True)
        angles = init_edge_rot_euler_angles(edges)
        W = eulers_to_wigner(angles, 0, lmax, Jd)
        torch.autograd.grad(W.sum(), edges)

    # Euler timed
    start = time.perf_counter()
    for _ in range(n_runs):
        edges = edges_base.clone().requires_grad_(True)
        angles = init_edge_rot_euler_angles(edges)
        W = eulers_to_wigner(angles, 0, lmax, Jd)
        torch.autograd.grad(W.sum(), edges)
    t_euler = (time.perf_counter() - start) / n_runs * 1000

    # Quaternion warmup
    for _ in range(n_warmup):
        edges = edges_base.clone().requires_grad_(True)
        _, W = get_wigner_from_edge_vectors(edges, coeffs, U_blocks)
        torch.autograd.grad(W.sum(), edges)

    # Quaternion timed
    start = time.perf_counter()
    for _ in range(n_runs):
        edges = edges_base.clone().requires_grad_(True)
        _, W = get_wigner_from_edge_vectors(edges, coeffs, U_blocks)
        torch.autograd.grad(W.sum(), edges)
    t_quat = (time.perf_counter() - start) / n_runs * 1000

    ratio = t_quat / t_euler
    euler_per_edge = t_euler / n_edges * 1000  # microseconds
    quat_per_edge = t_quat / n_edges * 1000

    backward_results.append((n_edges, t_euler, t_quat, ratio))

    print(f"{n_edges:<8} {t_euler:<15.3f} {t_quat:<15.3f} {ratio:<10.2f} {euler_per_edge:<18.2f} {quat_per_edge:<18.2f}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nForward pass ratio range: {min(r[3] for r in forward_results):.2f}x - {max(r[3] for r in forward_results):.2f}x")
print(f"Forward+backward ratio range: {min(r[3] for r in backward_results):.2f}x - {max(r[3] for r in backward_results):.2f}x")
