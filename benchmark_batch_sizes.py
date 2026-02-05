"""
Performance comparison: Euler vs Quaternion for varying batch sizes.
"""

import torch
import time
from pathlib import Path

from fairchem.core.models.uma.common.rotation import init_edge_rot_euler_angles, eulers_to_wigner
from fairchem.core.models.uma.common.wigner_d_quaternion import precompute_all_wigner_tables, get_wigner_from_edge_vectors

jd_path = Path("/Users/levineds/fairchem/src/fairchem/core/models/uma/Jd.pt")
dtype = torch.float64
device = torch.device("cpu")
lmax = 6

Jd = torch.load(jd_path, map_location=device, weights_only=True)
Jd = [J.to(dtype=dtype) for J in Jd]
coeffs, U_blocks = precompute_all_wigner_tables(lmax, dtype=dtype, device=device)

batch_sizes = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
n_warmup = 5
n_runs = 20

print("=" * 80)
print("PERFORMANCE: Euler vs Quaternion by Batch Size")
print("=" * 80)
print(f"lmax = {lmax}, dtype = {dtype}, device = {device}")
print(f"Warmup runs: {n_warmup}, Timed runs: {n_runs}")
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
        W.sum().backward()

    # Euler timed
    start = time.perf_counter()
    for _ in range(n_runs):
        edges = edges_base.clone().requires_grad_(True)
        angles = init_edge_rot_euler_angles(edges)
        W = eulers_to_wigner(angles, 0, lmax, Jd)
        W.sum().backward()
    t_euler = (time.perf_counter() - start) / n_runs * 1000

    # Quaternion warmup
    for _ in range(n_warmup):
        edges = edges_base.clone().requires_grad_(True)
        _, W = get_wigner_from_edge_vectors(edges, coeffs, U_blocks)
        W.sum().backward()

    # Quaternion timed
    start = time.perf_counter()
    for _ in range(n_runs):
        edges = edges_base.clone().requires_grad_(True)
        _, W = get_wigner_from_edge_vectors(edges, coeffs, U_blocks)
        W.sum().backward()
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
