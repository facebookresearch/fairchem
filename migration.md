# Migration Plan: umas_fast_gpu_unified_radial → umas_fast_gpu_unified_radial_clean

## Objective
Identify which code change causes the ~1.4% performance regression (16.19 QPS → 15.96 QPS).

## Branches
- **Source (fast)**: `umas_fast_gpu_unified_radial` @ 065f9f31e - **16.19 QPS**
- **Target (slow)**: `umas_fast_gpu_unified_radial_clean` @ aa2494a8b - **15.96 QPS**

## Strategy
Start from the fast original branch and incrementally apply changes from the clean branch.
Benchmark after each step. When QPS drops, we've found the culprit.

---

## Progress Log

| Step | Change | QPS | Delta | Status |
|------|--------|-----|-------|--------|
| 0 | Baseline (original) | 16.23 | - | ✅ |
| 1 | unified_radial.py changes | 16.23 | 0% | ✅ |
| 2 | Move scatter inside record_function block | 16.27 | +0.2% | ✅ |
| 3 | Inline forward kernel wrapper | 16.27 | +0.2% | ✅ |
| 4 | Method renaming + escn_md changes (SAME classes) | 16.17 | -0.4% | ✅ |
| 5 | All triton reorg (new ops.py, _kernels/) | 16.02 | **-1.3%** | ❌ |
| 6 | **SOLUTION**: Keep original classes, add aliases | 16.14 | -0.6% | ✅ |

## Root Cause Analysis

### NOT the cause:
- Method renaming (`gather_rotate` → `node_to_edge_wigner_permute`)
- Moving scatter inside `with record_function` block
- Removing unused `edge_distance` parameter
- `unified_radial.py` changes
- Inlining kernel wrapper vs calling wrapper function

### IS the cause:
The new `ops.py` autograd classes (`UMASFastGPUNodeToEdgeWignerPermute`, 
`UMASFastGPUPermuteWignerInvEdgeToNode`) that INLINE kernel calls vs the 
original classes that call WRAPPER FUNCTIONS.

When:
- Using original `FusedEdgeGatherWignerL2MTritonBwdEmitFunction` → **16.17 QPS**
- Using new `UMASFastGPUNodeToEdgeWignerPermute` from ops.py → **16.02 QPS**

The classes are functionally identical but organized differently!

---

## Solution Implemented

Keep the original autograd classes but add aliases with the new clean names:

### Changes made:

1. **triton/__init__.py**: Add aliases that point to original classes
   ```python
   UMASFastGPUNodeToEdgeWignerPermute = FusedEdgeGatherWignerL2MTritonBwdEmitFunction
   UMASFastGPUPermuteWignerInvEdgeToNode = FusedMToLThenWignerLmax2Function
   ```

2. **execution_backends.py**: Add new method names as aliases
   - `node_to_edge_wigner_permute = gather_rotate` (alias in both base and GPU backend)
   - `permute_wigner_inv_edge_to_node()` - new method combining rotate_back + scatter

3. **escn_md_block.py**: Update to use new method names
   - `gather_rotate` → `node_to_edge_wigner_permute`
   - `rotate_back` + manual scatter → `permute_wigner_inv_edge_to_node`

### Result: **16.14 QPS** ✅

This maintains performance within acceptable variance (~0.6% from baseline) while 
getting the cleaner API from the clean branch.
