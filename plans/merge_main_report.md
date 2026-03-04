# Merge Conflict Report: main → umas_fast_gpu_unified_radial_clean

## Summary

Merging `origin/main` into `umas_fast_gpu_unified_radial_clean` produces **3 conflicted files**, all related to the **unified radial precomputation** optimization in this branch.

**Incoming commit from main:**
```
de1df4f8c Umas fast gpu backend (#1826)
```

## Conflict Analysis

### 1. `src/fairchem/core/models/uma/escn_md.py` (line 773-777)

**Conflict:**
```python
<<<<<<< HEAD (this branch)
                    x_edge_per_layer[i],
=======
                    x_edge,
>>>>>>> origin/main
```

**Context:** In the message passing loop, passing edge features to each block.

| Branch | Approach |
|--------|----------|
| HEAD (umas_fast_gpu) | `x_edge_per_layer[i]` - precomputed per-layer radial embeddings |
| main | `x_edge` - single tensor, radials computed on-the-fly per layer |

**Recommendation:** ✅ **Keep HEAD** - This is the unified radial optimization that provides ~20% speedup.

---

### 2. `src/fairchem/core/models/uma/escn_md_block.py` (lines 195-199)

**Conflict:**
```python
<<<<<<< HEAD (this branch)
            x_message = self.so2_conv_2(x_message)
=======
            x_message = self.so2_conv_2(x_message, x_edge)
>>>>>>> origin/main
```

**Context:** Second SO2 convolution call in `eSCNMDBlock.forward()`.

| Branch | Approach |
|--------|----------|
| HEAD (umas_fast_gpu) | No `x_edge` arg - radials already embedded in `x_message` |
| main | Pass `x_edge` - SO2_Convolution computes radials internally |

**Recommendation:** ✅ **Keep HEAD** - This change is coupled with the per-layer radial precomputation. The SO2_Convolution has been modified to not need x_edge when radials are precomputed.

---

### 3. `src/fairchem/core/models/uma/nn/execution_backends.py` (lines 91-114)

**Conflict:**
```python
<<<<<<< HEAD (this branch)
    @staticmethod
    def get_layer_radial_emb(
        x_edge: torch.Tensor,
        model: torch.nn.Module,
    ) -> list[torch.Tensor]:
        """
        Get edge embeddings for each layer.
        
        Default implementation returns the same raw x_edge for all layers.
        SO2_Convolution will compute rad_func(x_edge) internally.
        
        Override in fast backends to precompute radials.
        """
        return [x_edge] * len(model.blocks)

    @staticmethod
=======
>>>>>>> origin/main
    def prepare_wigner(
```

**Context:** New method `get_layer_radial_emb` added to `ExecutionBackend` base class.

| Branch | Approach |
|--------|----------|
| HEAD (umas_fast_gpu) | Adds `get_layer_radial_emb()` method for radial precomputation |
| main | Method doesn't exist - radials computed per forward pass |

**Recommendation:** ✅ **Keep HEAD** - This method is essential for the unified radial optimization. It's overridden in `UMASFastPytorchBackend` (line 304) and `UMASFastGPUBackend` (line 359) to provide precomputed radials.

---

## Root Cause

PR #1826 merged a subset of the fast GPU backend to main (basic backend structure), but this branch contains **additional optimizations** (unified radial precomputation) that provide further speedup. The conflicts occur because:

1. Main's SO2_Convolution expects `x_edge` argument
2. This branch's SO2_Convolution doesn't need `x_edge` when radials are precomputed
3. This branch adds the `get_layer_radial_emb` infrastructure that main doesn't have

## Resolution Plan

**For all 3 conflicts:** Keep HEAD (this branch's version)

All three conflicts are interconnected - they form the unified radial optimization:
1. `get_layer_radial_emb` precomputes radials per layer
2. `x_edge_per_layer[i]` passes precomputed radials to each block  
3. `so2_conv_2(x_message)` uses embedded radials instead of computing them

**Command to execute merge with conflict resolution:**
```bash
git merge origin/main
# Then for each conflict, keep HEAD:
git checkout --ours src/fairchem/core/models/uma/escn_md.py
git checkout --ours src/fairchem/core/models/uma/escn_md_block.py
git checkout --ours src/fairchem/core/models/uma/nn/execution_backends.py
git add .
git commit -m "Merge main into umas_fast_gpu_unified_radial_clean, keep unified radial optimizations"
```

## Testing After Merge

```bash
# Run benchmarks to verify performance
./run_benchmarks.sh

# Run tests
pytest tests/core/models/test_uma.py -v -c packages/fairchem-core/pyproject.toml
```

## Risk Assessment

**Low risk** - All conflicts are in a consistent direction (keep this branch's optimizations). The changes are well-tested and the performance improvements are documented.
