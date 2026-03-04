# AllScAIP — All-to-all Scaled Attention Interatomic Potential

AllScAIP is an attention-based machine learning interatomic potential model. It predicts energies, forces, and stresses from atomic structures using a combination of neighborhood-level and node-level self-attention mechanisms built on top of a Differentiable kNN radius graph.

## File Structure

```
allscaip/
├── AllScAIP.py                          # Model entry point: backbone and prediction heads
├── configs.py                           # Nested dataclass configuration hierarchy
├── custom_types.py                      # GraphAttentionData dataclass (torch.compile-compatible)
│
├── modules/
│   ├── base_attention.py                # Multi-head scaled dot-product attention base class
│   ├── input_block.py                   # Atomic/edge featurization into initial embeddings
│   ├── graph_attention_block.py         # Single transformer block (neighborhood + node attention + FFNs)
│   ├── neighborhood_attention.py        # Neighbor-level self-attention (source and destination)
│   └── node_attention.py               # Node-level self-attention with sincx masking
│
└── utils/
    ├── nn_utils.py                      # Feedforward network construction helpers
    ├── radius_graph_v2.py               # BiKNN radius graph with soft/hard ranking and envelope
    ├── data_preprocess.py               # AtomicData → GraphAttentionData preprocessing pipeline
    └── fix_ckpt.py                      # Checkpoint patching for compile/max_atoms settings
```

## Checkpoint Usage (`utils/fix_ckpt.py`)

The default checkpoint ships with `use_compile=False`, which works out of the box with the ASE calculator and handles arbitrary system sizes without padding overhead.

When to enable compile:

- **Variable system sizes** (e.g., different molecules via ASE calculator) — keep `use_compile=False`. Padding every input to `max_atoms` wastes compute when system sizes vary widely.
- **Similar system sizes** (e.g., running MD on a fixed system) — enable compile with `max_atoms` set to the maximum system size. The padding overhead is minimal when actual sizes are close to `max_atoms`.
- **Batch inference / finetuning with implicit batching** — always enable compile. The data loader samples systems up to `max_atoms` per batch, so padding overhead is negligible and you get the full benefit of `torch.compile`.

Use `fix_ckpt.py` to patch a checkpoint:

```bash
# Enable compile and set max_atoms to 200 (adjust based on available VRAM)
python fix_ckpt.py path/to/checkpoint.pt \
    --enable-compile \
    --max-atoms 200
```

This writes a patched checkpoint to `*_fixed.pt` with `use_compile=True`, `use_padding=True`, and the specified `max_atoms`. Larger `max_atoms` values consume more VRAM but allow bigger systems in a single pass; reduce it if you run into OOM errors.
