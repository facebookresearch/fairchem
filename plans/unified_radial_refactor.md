# Plan: Move Unified Radial to UMASFastPytorchBackend

## Current Structure

```
ExecutionBackend (base)
├── get_layer_radial_emb: returns [x_edge] * num_layers (no computation)
│
└── UMASFastPytorchBackend
    ├── prepare_model_for_inference: converts SO2 layers to block-diagonal
    ├── get_layer_radial_emb: calls rad_func(x_edge) per layer sequentially
    │
    └── UMASFastGPUBackend
        ├── prepare_model_for_inference: parent + creates UnifiedRadialMLP
        ├── get_layer_radial_emb: uses _unified_radial_mlp (batched)
        └── Triton kernels for wigner ops
```

## Problem

UnifiedRadialMLP is a pure PyTorch optimization (batched computation vs sequential), not a GPU-specific optimization. It should benefit `umas_fast_pytorch` too.

## Target Structure

```
ExecutionBackend (base)
├── get_layer_radial_emb: returns [x_edge] * num_layers
│
└── UMASFastPytorchBackend
    ├── prepare_model_for_inference: SO2 conversion + UnifiedRadialMLP
    ├── get_layer_radial_emb: uses _unified_radial_mlp (batched)
    │
    └── UMASFastGPUBackend
        ├── prepare_model_for_inference: calls parent (inherits unified radial)
        ├── get_layer_radial_emb: (inherits from parent)
        └── Triton kernels for wigner ops
```

## Changes

### 1. Move unified radial creation to `UMASFastPytorchBackend.prepare_model_for_inference`

```python
@staticmethod
def prepare_model_for_inference(model: torch.nn.Module) -> None:
    from fairchem.core.models.uma.nn.so2_layers import (
        convert_so2_conv1,
        convert_so2_conv2,
    )

    for block in model.blocks:
        block.edge_wise.so2_conv_1 = convert_so2_conv1(block.edge_wise.so2_conv_1)
        block.edge_wise.so2_conv_2 = convert_so2_conv2(block.edge_wise.so2_conv_2)

    # NEW: Create unified radial MLP for batched computation
    rad_funcs = [block.edge_wise.so2_conv_1.rad_func for block in model.blocks]
    model._unified_radial_mlp = UnifiedRadialMLP(rad_funcs)
```

### 2. Move batched `get_layer_radial_emb` to `UMASFastPytorchBackend`

```python
@staticmethod
def get_layer_radial_emb(
    x_edge: torch.Tensor,
    model: torch.nn.Module,
) -> list[torch.Tensor]:
    """
    Compute radial embeddings for all layers using batched UnifiedRadialMLP.
    """
    return model._unified_radial_mlp(x_edge)
```

### 3. Simplify `UMASFastGPUBackend` - remove duplicate code

Delete from `UMASFastGPUBackend`:
- `prepare_model_for_inference` override (just calls parent now)
- `get_layer_radial_emb` override (inherited from parent)

```python
class UMASFastGPUBackend(UMASFastPytorchBackend):
    """
    GPU-optimized backend: inherits SO2 + unified radial, adds Triton kernels.
    """

    @staticmethod
    def validate(...):
        # unchanged

    # DELETED: prepare_model_for_inference (inherits from parent)
    # DELETED: get_layer_radial_emb (inherits from parent)

    @staticmethod
    def prepare_wigner(...):
        # unchanged - Triton passthrough

    @staticmethod
    def node_to_edge_wigner_permute(...):
        # unchanged - Triton kernel

    @staticmethod
    def permute_wigner_inv_edge_to_node(...):
        # unchanged - Triton kernel

    @staticmethod
    def edge_degree_scatter(...):
        # unchanged - Triton kernel
```

## Files to Modify

- `src/fairchem/core/models/uma/nn/execution_backends.py`:
  - **MOVE** UnifiedRadialMLP creation from `UMASFastGPUBackend.prepare_model_for_inference` to `UMASFastPytorchBackend.prepare_model_for_inference`
  - **MOVE** batched `get_layer_radial_emb` from `UMASFastGPUBackend` to `UMASFastPytorchBackend`
  - **DELETE** `UMASFastGPUBackend.prepare_model_for_inference` (now just inherits)
  - **DELETE** `UMASFastGPUBackend.get_layer_radial_emb` (now just inherits)

## Benefits

1. **Cleaner inheritance**: GPU backend only overrides what's truly GPU-specific (Triton kernels)
2. **Performance for pytorch mode**: Users get unified radial speedup without needing GPU
3. **Less code duplication**: No repeated `get_layer_radial_emb` in subclass
4. **Logical separation**: Unified radial = PyTorch optimization, Triton = GPU optimization

## Testing

Run benchmarks for both backends to verify:
```bash
./run_benchmarks.sh --compile
```

Expected: Both `umas_fast_pytorch` and `umas_fast_gpu` should show similar radial computation time.
