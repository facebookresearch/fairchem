# Plan: Fix Head MOLE Merging for UMA-S 1.2

## Problem

When `merge_mole=True`, the backbone correctly merges MOLE layers → Linear, but `DatasetSpecificMoEWrapper` (used by UMA-S 1.2) does not. The class has **no `merge_MOLE_model` method at all**, causing ~4% slowdown vs UMA-S 1.1.

## Root Cause

```python
# DatasetSpecificMoEWrapper - MISSING merge_MOLE_model entirely!
# The head still runs expensive 8-expert MOLE computation instead of single Linear.
#
# Current class structure:
#   - eSCNMDMoeBackbone.merge_MOLE_model (line 82) - EXISTS, works
#   - DatasetSpecificSingleHeadWrapper.merge_MOLE_model (line 364) - EXISTS, but only sets flag
#   - DatasetSpecificMoEWrapper.merge_MOLE_model - MISSING!
```

## Solution

**ADD** `merge_MOLE_model` and `prepare_for_inference` methods to `DatasetSpecificMoEWrapper`, plus a fast-path in `forward()`:

### 0. Add import at top of escn_moe.py

```python
from fairchem.core.models.uma.nn.mole import MOLE
```

### 1. Add `merge_MOLE_model` method

```python
def merge_MOLE_model(self, data):
    """
    Merge MOLE layers into single Linear for single-dataset inference.
    
    Sets one-hot expert coefficients and replaces all MOLE→Linear.
    """
    self.merged_on_dataset = data.dataset[0]
    expert_idx = self.dataset_name_to_exp[self.merged_on_dataset]
    self.global_mole_tensors.expert_mixing_coefficients = (
        torch.zeros(1, len(self.dataset_name_to_exp), dtype=data.pos.dtype)
        .scatter_(1, torch.tensor([[expert_idx]]), 1.0)
        .to(data.pos.device)
    )
    
    # Replace MOLE → Linear (4-line inline helper)
    def replace_mole(module):
        for name, child in list(module.named_children()):
            if isinstance(child, MOLE):
                setattr(module, name, child.merged_linear_layer())
            else:
                replace_mole(child)
    replace_mole(self.head)
    
    self.non_merged_dataset_names = [
        n for n in self.dataset_names if n != self.merged_on_dataset
    ]
    return self
```

### 2. Add `prepare_for_inference` method (for consistent API)

```python
def prepare_for_inference(self, data, settings):
    """
    Prepare head for inference. Handles MOLE merging if needed.
    """
    if settings.merge_mole:
        return self.merge_MOLE_model(data)
    return self
```

### 3. Add merged fast-path to `forward()` 

Add at start of `forward()` method:

```python
@conditional_grad(torch.enable_grad())
def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Fast path for merged model - skip MOLE routing overhead
    if self.merged_on_dataset is not None:
        head_output = self.head(data, emb)
        full_output = {}
        for key in head_output:
            full_output[f"{self.merged_on_dataset}_{key}"] = (
                {key: head_output[key]} if self.wrap_property else head_output[key]
            )
            nan_tensor = head_output[key].new_full(head_output[key].shape, float("nan"))
            for dataset in self.non_merged_dataset_names:
                full_output[f"{dataset}_{key}"] = (
                    {key: nan_tensor} if self.wrap_property else nan_tensor
                )
        return full_output

    # Original MOLE routing logic follows...
    self.global_mole_tensors.mole_sizes = torch.zeros(...)
    # ... rest of existing forward()
```

### 4. Add instance variable initialization

In `__init__`, add:
```python
# Track merge state
self.merged_on_dataset = None
self.non_merged_dataset_names = None
```

## Files to Modify

- `src/fairchem/core/models/uma/escn_moe.py`: 
  - **ADD** import `MOLE` at top of file
  - **ADD** `merge_MOLE_model` method to `DatasetSpecificMoEWrapper`
  - **ADD** `prepare_for_inference` method to `DatasetSpecificMoEWrapper`  
  - **MODIFY** `forward()` to add merged fast-path
  - **MODIFY** `__init__` to initialize `merged_on_dataset = None`

## Testing

### 1. Unit Test: MOLE Layers Replaced
```python
def test_moe_wrapper_merge_replaces_mole():
    """Verify MOLE layers are converted to Linear after merge."""
    from fairchem.core.models.uma.nn.mole import MOLE
    
    # Load UMA-S 1.2 model
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
    head = predictor.model.heads["energy"]
    
    # Count MOLE layers before merge
    mole_count_before = sum(1 for m in head.modules() if isinstance(m, MOLE))
    assert mole_count_before > 0, "Should have MOLE layers before merge"
    
    # Merge on oc20 dataset
    data = make_test_data(dataset="oc20")
    head.merge_MOLE_model(data)
    
    # Count MOLE layers after merge
    mole_count_after = sum(1 for m in head.modules() if isinstance(m, MOLE))
    assert mole_count_after == 0, "All MOLE layers should be replaced"
    assert head.merged_on_dataset == "oc20"
```

### 2. Integration Test: Output Equivalence
```python
def test_moe_wrapper_merge_preserves_output():
    """Verify merged model produces same output as unmerged for target dataset."""
    predictor_unmerged = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
    predictor_merged = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
    
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    # Use oc20 dataset for both - merged model only works for merged dataset
    data = AtomicData.from_ase(atoms, dataset="oc20")
    
    # Get unmerged output (with oc20 dataset routing)
    output_unmerged = predictor_unmerged.predict(data)
    
    # Merge on oc20 and get output
    predictor_merged.model.heads["energy"].merge_MOLE_model(data)
    output_merged = predictor_merged.predict(data)
    
    # Compare oc20-specific outputs
    assert torch.allclose(
        output_unmerged["oc20_energy"], 
        output_merged["oc20_energy"], 
        atol=1e-5
    )
    assert torch.allclose(
        output_unmerged["oc20_forces"], 
        output_merged["oc20_forces"], 
        atol=1e-5
    )
```

### 3. Benchmark Test: Performance Parity
```bash
# Run benchmarks - UMA-S 1.2 should match UMA-S 1.1 after fix
./run_benchmarks.sh --checkpoint /path/to/uma-s-1p2.pt --compile

# Expected: UMA-S 1.2 ≈ 2.45ms (was 2.55ms before fix)
```

## Design Decisions

1. **Returns `self` not new model**: Unlike backbone's `merge_MOLE_model` which returns a new `eSCNMDBackbone`, the head version returns `self`. This is intentional - heads don't need to change class type, just replace internal layers.

2. **Fast-path in forward()**: After merge, skip all MOLE routing (expert coefficient calculation, etc). This matches `DatasetSpecificSingleHeadWrapper` behavior.

3. **NaN for non-merged datasets**: Other dataset outputs are filled with NaN to catch accidental usage.

4. **dtype from input**: Use `data.pos.dtype` instead of hardcoded `float32` to support mixed precision.

## Success Criteria

1. No MOLE layers remain in head after `merge_MOLE_model`
2. Output numerically equivalent (within fp32 tolerance) for merged dataset
3. UMA-S 1.2 benchmark matches UMA-S 1.1 (~2.45ms with compile)
4. `prepare_for_inference` API consistent with `DatasetSpecificSingleHeadWrapper`
