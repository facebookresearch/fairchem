# AllScAIP — All-to-all Scalable Attention Interatomic Potential

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
│   └── node_attention.py                # Node-level self-attention with sincx masking
│
└── utils/
    ├── nn_utils.py                      # Feedforward network construction helpers
    ├── allscaip_radius_graph.py         # BiKNN radius graph with soft/hard ranking and envelope
    └── data_preprocess.py               # AtomicData → GraphAttentionData preprocessing pipeline
```

## Available Pretrained Models

| Model Name | Description |
|---|---|
| `allscaip-md-conserving-all-omol` | Medium conserving (gradient-based forces) |
| `allscaip-md-direct-all-omol` | Medium direct (predicted forces) |

## Quick Start

Run an MD simulation using AllScAIP with the ASE calculator:

```python
from ase import units
from ase.build import molecule
from ase.io import Trajectory
from ase.md.langevin import Langevin

from fairchem.core import FAIRChemCalculator

calc = FAIRChemCalculator.from_model_checkpoint(
    "allscaip-md-conserving-all-omol", task_name="omol"
)

atoms = molecule("H2O")
atoms.calc = calc

dyn = Langevin(
    atoms,
    timestep=0.1 * units.fs,
    temperature_K=400,
    friction=0.001 / units.fs,
)
trajectory = Trajectory("my_md.traj", "w", atoms)
dyn.attach(trajectory.write, interval=1)
dyn.run(steps=1000)
```

## Inference with `torch.compile`

AllScAIP pads inputs to a fixed `max_atoms` size to enable `torch.compile`. The compile and padding settings are controlled via `InferenceSettings` and passed to the calculator:

- **Variable system sizes** (e.g., different molecules via ASE calculator): keep `compile=False`. Padding every input to `max_atoms` wastes compute when system sizes vary widely.
- **Similar system sizes** (e.g., running MD on a fixed system): set `compile=True` with `max_atoms` equal to the maximum system size. The padding overhead is minimal when actual sizes are close to `max_atoms`.
- **Batch inference / finetuning with implicit batching**: always enable compile. The data loader samples systems up to `max_atoms` per batch, so padding overhead is negligible and you get the full benefit of `torch.compile`.

```python
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

# Without compile (default): works for any system size
calc = FAIRChemCalculator.from_model_checkpoint(
    "allscaip-md-conserving-all-omol", task_name="omol"
)

# With compile: requires max_atoms, best for fixed-size MD simulations
settings = InferenceSettings(compile=True, max_atoms=200)
calc = FAIRChemCalculator.from_model_checkpoint(
    "allscaip-md-conserving-all-omol",
    task_name="omol",
    inference_settings=settings,
)
```
