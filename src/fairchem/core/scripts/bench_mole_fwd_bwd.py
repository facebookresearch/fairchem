"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.models.uma.nn.mole import MOLE, MOLEFairchemCpp, MOLEGlobals


def segment_mm_ref(A, B, seglen):
    """
    Reference segment_mm: per-segment matmul via a simple loop.
    """
    C_parts = []
    off = 0
    for i in range(B.shape[0]):
        n = int(seglen[i])
        C_parts.append(A[off : off + n] @ B[i])
        off += n
    return torch.cat(C_parts)


def _make_mole_pair(
    num_experts, in_features, out_features, num_systems, atoms_per_system, device
):
    """
    Create matched MOLE (ref) and MOLEFairchemCpp (fairchem_cpp) layers with shared weights.
    """
    dtype = torch.float32
    mole_sizes = torch.full((num_systems,), atoms_per_system, dtype=torch.int32)
    coefficients = torch.randn(
        num_systems, num_experts, device=device, dtype=dtype
    ).softmax(dim=1)

    global_tensors = MOLEGlobals(
        expert_mixing_coefficients=coefficients,
        mole_sizes=mole_sizes,
    )

    ref = MOLE(
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        global_mole_tensors=global_tensors,
        bias=True,
    ).to(device=device, dtype=dtype)

    cpp = MOLEFairchemCpp(
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        global_mole_tensors=global_tensors,
        bias=True,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        cpp.weights.copy_(ref.weights)
        cpp.bias.copy_(ref.bias)

    return ref, cpp


def bench_fwd_bwd(layer, x, warmup=10, repeats=100):
    """
    Time forward + backward pass using CUDA events.
    """
    for _ in range(warmup):
        xi = x.detach().requires_grad_(True)
        out = layer(xi)
        out.sum().backward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        xi = x.detach().requires_grad_(True)
        out = layer(xi)
        out.sum().backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats  # ms


def bench_fwd_bwd_bwd(layer, x, warmup=10, repeats=100):
    """
    Time forward + backward + double backward pass using CUDA events.
    """
    for _ in range(warmup):
        xi = x.detach().requires_grad_(True)
        out = layer(xi)
        grads = torch.autograd.grad(out.sum(), [xi, layer.weights], create_graph=True)
        sum(g.sum() for g in grads).backward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        xi = x.detach().requires_grad_(True)
        out = layer(xi)
        grads = torch.autograd.grad(out.sum(), [xi, layer.weights], create_graph=True)
        sum(g.sum() for g in grads).backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats  # ms


def check_accuracy(ref, cpp, x, atol=1e-5):
    """
    Check MOLE vs MOLEFairchemCpp accuracy for forward, backward, and double backward.
    Returns True if all checks pass, prints details on failure.
    """
    all_ok = True

    # --- Forward ---
    x_ref = x.clone()
    x_cpp = x.clone()
    y_ref = ref(x_ref)
    y_cpp = cpp(x_cpp)
    fwd_diff = (y_ref - y_cpp).abs().max().item()
    fwd_ok = fwd_diff < atol
    if not fwd_ok:
        all_ok = False
    print(
        f"    forward:         max diff = {fwd_diff:.2e}  {'OK' if fwd_ok else 'FAIL'}"
    )

    # --- Backward ---
    ref.zero_grad()
    cpp.zero_grad()
    x_ref = x.clone().requires_grad_(True)
    x_cpp = x.clone().requires_grad_(True)
    y_ref = ref(x_ref)
    y_cpp = cpp(x_cpp)
    y_ref.sum().backward()
    y_cpp.sum().backward()

    xg_diff = (x_ref.grad - x_cpp.grad).abs().max().item()
    wg_diff = (ref.weights.grad - cpp.weights.grad).abs().max().item()
    bg_diff = (ref.bias.grad - cpp.bias.grad).abs().max().item()
    bwd_ok = xg_diff < atol and wg_diff < atol and bg_diff < atol
    if not bwd_ok:
        all_ok = False
    print(
        f"    backward x.grad: max diff = {xg_diff:.2e}  {'OK' if xg_diff < atol else 'FAIL'}"
    )
    print(
        f"    backward w.grad: max diff = {wg_diff:.2e}  {'OK' if wg_diff < atol else 'FAIL'}"
    )
    print(
        f"    backward b.grad: max diff = {bg_diff:.2e}  {'OK' if bg_diff < atol else 'FAIL'}"
    )

    # --- Double backward ---
    ref.zero_grad()
    cpp.zero_grad()
    x_ref = x.clone().requires_grad_(True)
    x_cpp = x.clone().requires_grad_(True)

    y_ref = ref(x_ref)
    grads_ref = torch.autograd.grad(
        y_ref.sum(), [x_ref, ref.weights], create_graph=True
    )
    sum(g.sum() for g in grads_ref).backward()
    x_ref_g2 = x_ref.grad.clone()
    w_ref_g2 = ref.weights.grad.clone()

    y_cpp = cpp(x_cpp)
    grads_cpp = torch.autograd.grad(
        y_cpp.sum(), [x_cpp, cpp.weights], create_graph=True
    )
    sum(g.sum() for g in grads_cpp).backward()

    xg2_diff = (x_ref_g2 - x_cpp.grad).abs().max().item()
    wg2_diff = (w_ref_g2 - cpp.weights.grad).abs().max().item()
    dbwd_ok = xg2_diff < atol and wg2_diff < atol
    if not dbwd_ok:
        all_ok = False
    print(
        f"    dbl bwd  x.grad: max diff = {xg2_diff:.2e}  {'OK' if xg2_diff < atol else 'FAIL'}"
    )
    print(
        f"    dbl bwd  w.grad: max diff = {wg2_diff:.2e}  {'OK' if wg2_diff < atol else 'FAIL'}"
    )

    return all_ok


def main():
    device = "cuda"
    num_experts = 64
    in_features = 128
    out_features = 128
    atoms_per_system = 100

    system_counts = [5, 10, 15]

    print(f"{'='*72}")
    print("MOLE Benchmark: PyTorch ref (MOLE) vs fairchem_cpp (MOLEFairchemCpp)")
    print(f"{'='*72}")
    print(f"  device:           {device}")
    print(f"  num_experts:      {num_experts}")
    print(f"  in_features:      {in_features}")
    print(f"  out_features:     {out_features}")
    print(f"  atoms_per_system: {atoms_per_system}")
    print(f"{'='*72}")
    print()

    # --- Accuracy check ---
    print("--- Accuracy Check (MOLE vs MOLEFairchemCpp) ---")
    print()
    atol = 1e-4
    all_ok = True
    for ns in system_counts:
        ref, cpp = _make_mole_pair(
            num_experts, in_features, out_features, ns, atoms_per_system, device
        )
        total_atoms = ns * atoms_per_system
        x = torch.randn(total_atoms, in_features, device=device, dtype=torch.float32)
        print(f"  num_systems={ns}, total_atoms={total_atoms}, atol={atol}")
        ok = check_accuracy(ref, cpp, x, atol=atol)
        if not ok:
            all_ok = False
        print()

    if all_ok:
        print("  All accuracy checks PASSED")
    else:
        print("  *** Some accuracy checks FAILED ***")
    print()

    # --- Forward + Backward ---
    print("--- Forward + Backward (timing) ---")
    print()
    hdr = f"  {'num_sys':>8} {'total_atoms':>12} {'ref (ms)':>12} {'cpp (ms)':>12} {'speedup':>10}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for ns in system_counts:
        ref, cpp = _make_mole_pair(
            num_experts, in_features, out_features, ns, atoms_per_system, device
        )
        total_atoms = ns * atoms_per_system
        x = torch.randn(total_atoms, in_features, device=device, dtype=torch.float32)

        t_ref = bench_fwd_bwd(ref, x)
        t_cpp = bench_fwd_bwd(cpp, x)
        speedup = t_ref / t_cpp if t_cpp > 0 else float("nan")
        print(
            f"  {ns:>8} {total_atoms:>12} {t_ref:>11.3f} {t_cpp:>11.3f} {speedup:>9.2f}x"
        )

    print()

    # --- Forward + Backward + Double Backward ---
    print("--- Forward + Backward + Double Backward (timing) ---")
    print()
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for ns in system_counts:
        ref, cpp = _make_mole_pair(
            num_experts, in_features, out_features, ns, atoms_per_system, device
        )
        total_atoms = ns * atoms_per_system
        x = torch.randn(total_atoms, in_features, device=device, dtype=torch.float32)

        t_ref = bench_fwd_bwd_bwd(ref, x)

        try:
            t_cpp = bench_fwd_bwd_bwd(cpp, x)
            speedup = t_ref / t_cpp if t_cpp > 0 else float("nan")
            print(
                f"  {ns:>8} {total_atoms:>12} {t_ref:>11.3f} {t_cpp:>11.3f} {speedup:>9.2f}x"
            )
        except RuntimeError as e:
            print(
                f"  {ns:>8} {total_atoms:>12} {t_ref:>11.3f} {'FAILED':>11} {'N/A':>10}"
            )
            if ns == system_counts[-1]:
                print(f"\n  fairchem_cpp double backward error: {e}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
