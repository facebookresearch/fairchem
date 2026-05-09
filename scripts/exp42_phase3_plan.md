# exp42 — Phase 3: C++ Fused SO2 Block

## Goal

Replace the eager Python sequence

```python
# eSCNMD_Edgewise.forward_chunk
x_message, x_0_gating = so2_conv_1(x_message, x_edge)   # SO2_Conv1_WithRadialBlock (eager)
x_message = act(x_0_gating, x_message)                  # GateActivation (m_prime=True)
x_message = so2_conv_2(x_message)                       # SO2_Conv2_InternalBlock (eager)
```

with a single C++ op exposed via `torch.autograd.Function`, doing the *same math* as the eager path with fewer ATen op dispatches.

**Target deployment:** 80–170 atom systems, eager-path always (`FAIRCHEM_FUSED_E_THRESHOLD=999999`). The C++ kernel must not regress the existing eager-path bit-exactness.

## Design decisions (with rationale)

1. **`torch.autograd.Function`, not `torch.library.custom_op`.**
   - exp28 + exp30 both died at the `custom_op` API. Existing `cpu_kernels.py` uses `autograd.Function` cleanly. Stick with what works.
   - Cost: ~10–50 us / call Python frame; the win comes from eliminating ~140 *internal* ATen dispatches, not from eliminating the wrapper.

2. **No buffer caching. Allocate output fresh per call.**
   - Reviewer 1 issue 7 + exp28 lesson: mutating cached buffers + autograd is a footgun.
   - Under tcmalloc, alloc cost is negligible.

3. **Save the COMPLETE backward state.**
   - All 8 weights: `fc_m0_w_1, fc_m0_b_1, W_block_1_m1, W_block_1_m2, fc_m0_w_2, fc_m0_b_2, W_block_2_m1, W_block_2_m2`.
   - Forward intermediates required for backward: `x` (input), `x_edge` (radial), `x_0_radial_1` (post-radial-mul m=0 conv1), `x_1_cat_1, x_2_cat_1` (post-radial-mul m=1,2 conv1), `gating_post_sigmoid` (post-sigmoid pre-multiply), `conv1_main_post_act` (post-activation, pre-conv2), `x_0_flat_2, x_1_cat_2, x_2_cat_2` (pre-conv2 per-m).
   - This matches what eager autograd would save — no fewer.

4. **Backward replicates eager autograd's matmul count exactly.**
   - exp36 regression root cause: hand-rolled backward added +160 mm calls (per-m grad), causing +160 OMP barriers. Wall went up despite CPU going down.
   - Mitigation: backward kernel doesn't split per-m for weight grads — combine where shapes allow, or fall back to mm with same count as eager.

5. **Hardcoded for `lmax = mmax = 2` and `m_prime = True` (production config only).**
   - Assert at first-call wrap. Fall through to eager for any other shape/config (no-op kernel).
   - Hardcode the GateActivation `expand_index = [0, 1, 0, 1, 0, 1, 1, 1]` pattern with a runtime assert against the actual buffer.

6. **Gated by `_FUSED_E_THRESHOLD` semantics:** the fused-block path runs whenever the eager SO2 path would. Not adding a separate threshold. The existing `_FUSED_E_THRESHOLD=999999` already routes 80-170 systems through eager, so the C++ kernel kicks in there.

7. **Use `at::matmul` from C++ for the GEMMs** (oneDNN brg_matmul). Hand-rolling SGEMM would lose to oneDNN; not worth the engineering risk.

8. **Test against eager Python path** (not against existing `_FusedConv1Func`, which has its own ULP drift from a different summation order).

## Files

| file | change |
|---|---|
| `src/fairchem/core/models/uma/nn/cpu_kernels.py` | add 5 new C++ functions to `_CPP_SRC` and 1 new `autograd.Function` (~600 lines) |
| `src/fairchem/core/models/uma/escn_md_block.py` | wire the fused-block call into `eSCNMD_Edgewise.forward_chunk` behind a `_USE_FUSED_BLOCK` env-var-controlled flag (default off for safety; on for the deployment) |
| `tests/core/models/uma/test_fused_so2_block.py` | new unit tests (~300 lines) |

Estimated total: **~900 lines new code, ~3-4 days end-to-end**.

## Phased implementation

Each phase ends in a pre-commit-ready, bit-exact-validated state. **No phase commits without passing its tests.**

| phase | scope | estimated time | tests |
|---|---|---:|---|
| **0** | API + binding spike | 1 h | T0 |
| **1** | C++ forward-only kernel + Python wrapper (Python eager backward via re-running forward in `torch.no_grad`) | 1 day | T1, T2, T3 fwd |
| **2** | C++ backward kernel | 1 day | T1, T2, T3 bwd |
| **3** | Integration into `eSCNMD_Edgewise.forward_chunk` behind env-var flag | ½ day | T5, T6 |
| **4** | perf_check 4-system + 80-170 size sweep | ½ day | T7, T8 |

### Phase 0 — API spike (1 h)

Write a stub C++ extension exposing `fused_so2_block_forward` that takes all the input tensors and ints, returns `out_buf` of correct shape. Body just memsets to zero. Wrap in Python `autograd.Function` with stub backward returning zeros. Verify:
- It links + loads.
- Runs end-to-end without crashing.
- Backward grads reach all 8 weight inputs (no "needs grad but no grad_fn" errors).

**T0:** `test_aoti_spike_e2e()` — random inputs, fused returns zeros, backward propagates without errors.

### Phase 1 — C++ forward kernel (1 day)

Implement `fused_so2_block_forward(x, x_edge, w_m0_1, b_m0_1, W_b1_m1, W_b1_m2, w_m0_2, b_m0_2, W_b2_m1, W_b2_m2, m_split_sizes, edge_split_sizes, lmax, m_out_1, m_out_2)`:

```cpp
// Inputs:
//   x: [E, 9, S]  — sphere features post-Wigner
//   x_edge: [E, R_total] — radial features
//   weights as listed above
// Returns:
//   y: [E, 9, m_out_2] — block output
//   plus tuple of intermediates for backward

// Body:
// 1. Conv1 m=0:
//    x_0_flat = x[:, 0:m0_size, :].reshape(E, m0_size*S)
//    x_0_radial = x_0_flat * x_edge[:, 0:r0]
//    z_0 = at::addmm(b_m0_1, x_0_radial, w_m0_1.t())
//    gating_pre_sigmoid = z_0[:, :extra_m0]
//    conv1_out_m0 = z_0[:, extra_m0:].view(E, m0_size, m_out_1)
//
// 2. Conv1 m=1, m=2 (per-m block GEMM):
//    similar, writing into conv1_out[:, offset:offset+...]
//
// 3. Apply Wigner GateActivation(m_prime=True):
//    gating = at::sigmoid(gating_pre_sigmoid).view(E, lmax, m_out_1)
//    scalars = at::silu(conv1_out[:, 0:1, :])
//    For m_prime=True interleave [0,1,0,1,0,1,1,1]:
//      v_first = conv1_out[:, 1:7, :].reshape(E, 3, 2, m_out_1)
//      v_first = v_first * gating.unsqueeze(1)
//      v_last = conv1_out[:, 7:9, :] * gating[:, 1:2, :]
//      act_out: cat scalars + v_first + v_last
//
// 4. Conv2 m=0,1,2 (no radial mul, internal_weights):
//    write into final y buffer
//
// 5. Save intermediates for backward, return (y, intermediates_tuple)
```

The matmuls (`at::addmm`, the two block-GEMMs in conv1, the three matmuls in conv2) call into oneDNN. The slicing/reshapes/copies are cheap C++ ops between them.

The point: ~18 ATen-dispatches per (conv1+act+conv2) collapsed to 1 C++ op call → save ~17 dispatches × ~100 us × 8 calls/iter ≈ 14 ms/iter on N=80.

For Phase 1, **backward is implemented in Python by retracing forward in `torch.no_grad()` inside backward + manually computing grads.** Slower backward (we'll fix in Phase 2), but lets us validate the forward separately.

### Phase 2 — C++ backward kernel (1 day)

Implement backward in C++. The backward graph is fixed (manual), no autograd retrace. Computes:

```cpp
fused_so2_block_backward(
    grad_y: [E, 9, m_out_2],
    saved_intermediates_tuple
) -> (grad_x, grad_x_edge, grad_w_m0_1, grad_b_m0_1, grad_W_b1_m1, grad_W_b1_m2,
      grad_w_m0_2, grad_b_m0_2, grad_W_b2_m1, grad_W_b2_m2)
```

Implementation walks back: conv2-bwd → act-bwd → conv1-bwd. Each step uses oneDNN `at::matmul` for the GEMMs.

**Key constraint from exp36 diagnosis:** the backward must NOT introduce extra mm calls vs eager. Specifically:
- Eager autograd computes weight grads via `grad.T @ x` style matmuls — same count.
- Don't split per-m where eager autograd uses a single combined matmul.

### Phase 3 — Integration (½ day)

Modify `eSCNMD_Edgewise.forward_chunk`:

```python
USE_FUSED_BLOCK = os.environ.get("FAIRCHEM_FUSED_SO2_BLOCK", "1") == "1"

# In forward_chunk:
if USE_FUSED_BLOCK and x.shape[0] < FUSED_BLOCK_E_LIMIT and self._eligible():
    x_message = fused_so2_block(x_message, x_edge, ...)  # one call
else:
    x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
    x_message = self.act(x_0_gating, x_message)
    x_message = self.so2_conv_2(x_message)
```

`self._eligible()` checks: `lmax==mmax==2`, weights present, MOLE merged, etc. Default-on for the deployment (env var=1), default-off elsewhere.

### Phase 4 — Validation (½ day)

Run perf_check (4-system) + size sweep. Verify gate passes, measure wall delta.

## Test plan

Tests in `tests/core/models/uma/test_fused_so2_block.py`. All tests deterministic (seeded), tolerances chosen to allow fp32-reduction-order ULPs but catch real bugs.

### Tolerance contract

| quantity | absolute | relative |
|---|---:|---:|
| forward output | 1e-6 | — |
| input gradient (`grad_x`) | 1e-5 | — |
| radial gradient (`grad_x_edge`) | 1e-5 | — |
| weight gradients | 1e-5 | — |
| force_mae on perf_check | (existing 1e-2 cap) | — |

### Test list

**T0. API spike — `test_fused_so2_block_api()`**
Stub kernel returns zeros; backward propagates gradients through all 8 weight inputs without "needs grad but no grad_fn" errors. Just verifies binding works.

**T1. GateActivation in isolation — `test_fused_gate_act_forward()` + `test_fused_gate_act_backward()`**
- m_prime=True, lmax=mmax=2, num_channels=128, E ∈ {1, 17, 1024}.
- Compare standalone C++ kernel for the activation against the eager Python `GateActivation.forward`.
- Forward: `max abs diff < 1e-6`.
- Backward: `max abs diff < 1e-5` for both `grad_x_0_gating` and `grad_input_tensors`.

**T2. SO2_Conv1 eager equivalent — `test_fused_conv1_forward()` + `test_fused_conv1_backward()`**
Build a `SO2_Conv1_WithRadialBlock(lmax=2, mmax=2, sphere_channels=128, ...)` with random weights. Run `_w_block` build. Run forward of fused C++ kernel (forward portion only); compare against eager forward. Same for backward, computing grads w.r.t. all inputs.

**T3. SO2_Conv2 eager equivalent — `test_fused_conv2_forward()` + `test_fused_conv2_backward()`**
Same as T2 for conv2 (no x_edge / radial mul).

**T4. Full block — `test_fused_block_forward()` + `test_fused_block_backward()`**
Build a synthetic mini-Edgewise that runs `conv1 → act → conv2`. Compare full-block C++ kernel against the same Python sequence. E ∈ {1, 32, 1024, 5000}, S=128, hidden=128.

**T5. Force-MAE gate — `test_fused_block_perf_check_gate()`**
Slow integration test (marked `pytest.mark.slow`). Loads UMA-S 1p2, builds 4-system perf_check inputs, runs with FAIRCHEM_FUSED_SO2_BLOCK=1, asserts force_mae stays under existing per-system caps.

**T6. Size sweep timing — `test_fused_block_size_sweep()`**
Slow benchmark. Times N=80, 120, 170 with fused vs eager-baseline (env var off). Asserts at least neutral on small_mol-like, ≥ 0% gain at N≥120 (median over 3 trials).

**T7. Existing tests pass — `pytest tests/core/models/test_uma.py`**
Plain regression check. Run with FAIRCHEM_FUSED_SO2_BLOCK=0 (default off) to confirm no behavior change when disabled, and again with =1 to confirm no functional regression.

**T8. Edge cases — `test_fused_block_edge_cases()`**
- E=1 (single edge).
- E=17 (AVX-512 tail).
- `requires_grad=False` on weights — verify no-grad inference works.
- `dtype=fp64` — verify clear error message (not silent garbage).
- GPU tensor — verify clear error message or eager fallback.

**T9. Optional: torch.compile compat — `test_fused_block_compile_compat()`**
Wrap fused block with `torch.compile`, run forward + backward, verify no compile error (graph break is fine).

## Risk mitigations

- **Numerical drift:** fused-vs-eager ULP differences are real where summation order differs (e.g., the m=1+m=2 block-GEMM vs eager's per-m matmuls). Tolerance contract above accepts 1e-5 fp32 ULP. Forward tighter at 1e-6 because the matmuls themselves are shape-identical.
- **Backward correctness:** the most error-prone part. Implementation strategy: hand-derive each grad with paper math, write Python reference backward, compare to autograd-generated grads (via `torch.autograd.grad`) on small inputs to double-check the math, *then* port to C++.
- **`m_prime=True` interleave bug:** add a runtime assert at first call: load `expand_index` from the `GateActivation` module, compare to expected `[0,1,0,1,0,1,1,1]`, raise if mismatch. Future config change won't silently corrupt outputs.
- **Buffer-aliasing safety:** allocate every output and saved-tensor fresh. No buffer reuse. Trade allocator cost (cheap under tcmalloc) for autograd correctness.
- **Lint/test coverage of the C++ string:** add a `tests/core/models/uma/test_cpu_kernels_compile.py` smoke test that imports `cpu_kernels._build()` and forces compilation at CI time. Catches typos before they hit a developer machine.
- **Rollback:** the fused-block path is gated on `FAIRCHEM_FUSED_SO2_BLOCK=1` (default-off). Setting `=0` (or unsetting) restores the eager Python path with no other code changes. Single-line revert.
- **Compile time:** existing `_CPP_SRC` is ~700 lines / ~15 s build. Adding ~600 more lines pushes to ~25 s. Folded into existing `_build()` lazy init. Document in commit message.

## Effort estimate

| phase | optimistic | realistic | pessimistic |
|---|---:|---:|---:|
| 0 spike | 1 h | 1 h | 2 h |
| 1 forward | 6 h | 1 day | 1.5 days |
| 2 backward | 6 h | 1 day | 2 days |
| 3 integration | 3 h | ½ day | 1 day |
| 4 validation | 3 h | ½ day | 1 day |
| **total** | **~2 days** | **~3-4 days** | **~5-6 days** |

Realistic: **3-4 days**.

## Expected outcome

Based on the size-target profile: ~14 ms / iter saved on N=80 (~7 % wall) by eliminating ~140 ATen dispatches per iter.

| natoms | wall now | est savings | est new wall | est gain |
|---|---:|---:|---:|---:|
| 80 | 190 ms | 14 ms | ~176 ms | **+8 %** |
| 100 | 273 ms | 14 ms | ~259 ms | +5 % |
| 120 | 412 ms | 14 ms | ~398 ms | +4 % |
| 150 | 375 ms | 14 ms | ~361 ms | +4 % |
| 170 | 470 ms | 14 ms | ~456 ms | +3 % |

Weighted toward your average (N=80): **expected average gain ~5-8 %** on top of the current `+35 %` baseline. Brings cumulative to **~+42-45 %** over main on the 80-170 deployment.

**Risk-adjusted expectation:** ~30 % chance the gain is < +2 % (similar to exp36/37/38). In that case we revert with the env-var flag and the only cost is the engineering time.

## What this plan does NOT include

- Multi-system batching (out of scope per deployment constraint).
- Energy-only mode (out of scope, forces required).
- Graph caching (separate plan, ~2 % gain when applicable).
- C++ backward of the conv1/conv2 individually (Phase 2 covers it as part of the fused block, not as standalone primitives).
- bf16 / TF32 precision changes (vetoed).
