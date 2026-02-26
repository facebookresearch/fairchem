# Triton Cleanup Plan

Date: February 26, 2026
Current Branch: `umas_fast_gpu_unified_radial_clean`

## Summary

Removed test-only code from the triton folder, keeping only production code used by the `umas_fast_gpu` execution backend.

---

## 1. Changes Made

### Files Modified in `src/fairchem/core/models/uma/triton/`

| File | Before | After | Change |
|------|--------|-------|--------|
| `wigner_ops.py` | 41KB (1029 lines) | 23KB (~560 lines) | Removed 4 test-only kernels, 4 wrappers, 1 class |
| `edge_gather_wigner_fwd.py` | 30KB (1030 lines) | 17KB (~550 lines) | Removed non-emit kernel and wrapper |
| `edge_gather_wigner_bwd.py` | 20KB (~700 lines) | 18KB (~640 lines) | Removed V2BwdFunction class |
| `__init__.py` | 1.1KB | 0.9KB | Removed 2 unused exports |

### Code Removed from `wigner_ops.py`
- `wigner_lmax2_fwd_kernel` (test-only)
- `wigner_lmax2_bwd_dx_kernel` (test-only)
- `l_to_m_kernel` (test-only)
- `m_to_l_kernel` (test-only)
- `_wigner_lmax2_fwd` wrapper (test-only)
- `_wigner_lmax2_bwd_dx` wrapper (test-only)
- `_l_to_m_lmax2_fwd` wrapper (test-only)
- `_m_to_l_lmax2_fwd` wrapper (test-only)
- `MToLThenWignerLmax2Function` class (test-only, non-fused version)

### Code Removed from `edge_gather_wigner_fwd.py`
- `fused_edge_gather_wigner_l2m_kernel` (test-only, non-emit version)
- `fused_edge_gather_wigner_l2m_lmax2` wrapper (test-only)

### Code Removed from `edge_gather_wigner_bwd.py`
- `FusedEdgeGatherWignerL2MTritonV2BwdFunction` class (test-only)

### Exports Removed from `__init__.py`
- `FusedEdgeGatherWignerL2MTritonV2BwdFunction`
- `MToLThenWignerLmax2Function`

### Test File Deleted
- `tests/core/models/uma/uma_fast/triton/test_triton_kernels.py` (833 lines)
  - Had broken imports from non-existent `_kernels/` submodule
  - Tests were for the removed test-only code

### Production Code Retained
- `FusedEdgeGatherWignerL2MTritonBwdEmitFunction` → aliased as `UMASFastGPUNodeToEdgeWignerPermute`
- `FusedMToLThenWignerLmax2Function` → aliased as `UMASFastGPUPermuteWignerInvEdgeToNode`
- All supporting kernels and wrappers for these two autograd functions

---

## 2. Benchmark Results

### GPU Benchmark (compile=True, 2000 atoms)
```
umas_fast_gpu: 16.36 qps, 1.41 ns/day
```

Benchmark command:
```bash
bash run_benchmarks.sh --compile --speed-only
```

### Verification
- All lint checks passed (`pre-commit run --files ...`)
- Benchmark runs successfully with cleaned code
- No import errors

---

## 3. Files to Delete Before Commit

Backup files created during cleanup (untracked):
```
src/fairchem/core/models/uma/triton/wigner_ops_old.py      (41KB)
src/fairchem/core/models/uma/triton/edge_gather_wigner_fwd_old.py  (30KB)
src/fairchem/core/models/uma/triton/edge_gather_wigner_bwd_old.py  (20KB)
```

---

## 4. Git Workflow Plan

### Constraint
**`umas_fast_gpu_backend_clean` must NEVER receive a force push.**

### Current State
```
umas_fast_gpu_backend_clean (39d5896a7)
         |
         v
umas_fast_gpu_unified_radial_clean (20a7d664b) <- HEAD with uncommitted changes
```

### Step-by-Step Procedure

#### Step 1: Create patch from current changes
```bash
cd /home/misko/env/feb17_speed_refactor/fairchem_feb22_gpu_speed

# Delete backup files first
rm src/fairchem/core/models/uma/triton/*_old.py

# Create a patch of src/ changes only
git diff src/ > /tmp/triton_cleanup.patch
git diff --staged src/ >> /tmp/triton_cleanup.patch

# Also capture the deleted test file
git diff tests/core/models/uma/uma_fast/triton/ >> /tmp/triton_cleanup.patch
```

#### Step 2: Stash current work on unified_radial_clean
```bash
# Stash everything including untracked
git stash push -m "triton cleanup WIP on unified_radial_clean"
```

#### Step 3: Switch to backend_clean and apply changes
```bash
git checkout umas_fast_gpu_backend_clean

# Apply the patch
git apply /tmp/triton_cleanup.patch

# Delete the backup files if they exist
rm -f src/fairchem/core/models/uma/triton/*_old.py

# Verify with pre-commit
pre-commit run --files \
  src/fairchem/core/models/uma/triton/__init__.py \
  src/fairchem/core/models/uma/triton/wigner_ops.py \
  src/fairchem/core/models/uma/triton/edge_gather_wigner_fwd.py \
  src/fairchem/core/models/uma/triton/edge_gather_wigner_bwd.py

# Run benchmark to verify
bash run_benchmarks.sh --compile --speed-only
```

#### Step 4: Commit to backend_clean
```bash
git add src/fairchem/core/models/uma/triton/
git add tests/core/models/uma/uma_fast/triton/test_triton_kernels.py

git commit -m "Clean triton: remove test-only code, keep production kernels

- Remove test-only kernels and wrappers from wigner_ops.py
- Remove non-emit forward kernel from edge_gather_wigner_fwd.py
- Remove V2BwdFunction from edge_gather_wigner_bwd.py
- Remove unused exports from __init__.py
- Delete broken test_triton_kernels.py (imported from non-existent _kernels/)

Production code retained:
- FusedEdgeGatherWignerL2MTritonBwdEmitFunction (UMASFastGPUNodeToEdgeWignerPermute)
- FusedMToLThenWignerLmax2Function (UMASFastGPUPermuteWignerInvEdgeToNode)

Benchmark verified: 16.36 qps @ 2000 atoms (compile=True)"
```

#### Step 5: Push backend_clean (normal push, no force)
```bash
git push origin umas_fast_gpu_backend_clean
```

#### Step 6: Rebase unified_radial_clean onto updated backend_clean
```bash
git checkout umas_fast_gpu_unified_radial_clean

# Drop the stash since we're rebasing onto the committed changes
git stash drop

# Rebase onto the updated backend_clean
git rebase umas_fast_gpu_backend_clean

# If conflicts occur in triton files, take ours (backend_clean version)
# since unified_radial_clean doesn't modify triton files
```

#### Step 7: Force push unified_radial_clean (OK since it's a feature branch)
```bash
git push --force-with-lease origin umas_fast_gpu_unified_radial_clean
```

---

## 5. Expected Final State

```
main
  |
  v
umas_fast_gpu_backend_clean (NEW_COMMIT) <- triton cleanup here
  |
  v
umas_fast_gpu_unified_radial_clean (rebased) <- UnifiedRadialMLP feature
```

### Branch Contents After Workflow

**umas_fast_gpu_backend_clean:**
- Clean triton infrastructure
- Production-only code
- No test-only kernels

**umas_fast_gpu_unified_radial_clean:**
- Everything from backend_clean
- Plus UnifiedRadialMLP feature (+6 files)

---

## 6. Rollback Plan

If something goes wrong after pushing to backend_clean:

```bash
# Revert the commit (creates new commit, no force push needed)
git checkout umas_fast_gpu_backend_clean
git revert HEAD
git push origin umas_fast_gpu_backend_clean
```

---

## 7. Verification Checklist

- [ ] Delete `*_old.py` backup files
- [ ] Pre-commit passes on all modified files
- [ ] Benchmark runs successfully on backend_clean
- [ ] Commit to backend_clean with descriptive message
- [ ] Push backend_clean (normal push)
- [ ] Rebase unified_radial_clean
- [ ] Resolve any conflicts
- [ ] Force push unified_radial_clean
- [ ] Final benchmark verification
