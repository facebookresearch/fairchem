---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Inference using ASE and Predictor Interface

Inference is done using [MLIPPredictUnit](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/units/mlip_unit/mlip_unit.py#L867). The [FairchemCalculator](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/calculate/ase_calculator.py#L3) (an ASE calculator) is simply a convenience wrapper around the MLIPPredictUnit.

:::{tip}
For simple cases such as demos or education, the ASE calculator is very easy to use. For more complex cases such as running MD or batched inference, we recommend using the predictor directly for better performance.
:::

```{code-cell} python3
from __future__ import annotations

from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="oc20")
```

````{admonition} Need to install fairchem-core or get UMA access or getting permissions/401 errors?
:class: dropdown


1. Install the necessary packages using pip, uv etc
```{code-cell} ipython3
:tags: [skip-execution]

! pip install fairchem-core fairchem-data-oc fairchem-applications-cattsunami
```

2. Get access to any necessary huggingface gated models
    * Get and login to your Huggingface account
    * Request access to https://huggingface.co/facebook/UMA
    * Create a Huggingface token at https://huggingface.co/settings/tokens/ with the permission "Permissions: Read access to contents of all public gated repos you can access"
    * Add the token as an environment variable using `huggingface-cli login` or by setting the HF_TOKEN environment variable.

```{code-cell} ipython3
:tags: [skip-execution]

# Login using the huggingface-cli utility
! huggingface-cli login

# alternatively,
import os
os.environ['HF_TOKEN'] = 'MY_TOKEN'
```

````

## Choosing an inference mode

Choose a mode based on how many predictions you will make, whether the chemical
composition stays constant, and whether the tensor shapes stay constant. Model
parameters are automatically frozen after inference preparation in every mode;
positions and cells remain differentiable for conservative forces, stress, and
Hessians.

| Mode | Recommended workload | Shape assumptions | Main trade-off |
| ---- | -------------------- | ----------------- | -------------- |
| `default` | One-off calculations, a small number of calls, heterogeneous systems, or batching | No assumptions | Avoids compile startup and uses activation checkpointing for broad compatibility |
| `turbo` | MD, relaxation, or another long rollout of one fixed-composition system | Atom count and composition are fixed; edge counts may change | Pays a dynamic compile cost, then accelerates repeated calls |
| `turbo-fixed` | Many calls with stable or explicitly padded atom, edge, and batch shapes | Tensor shapes must repeat | Uses shape specialization and CUDA graphs for the best steady-state speed, but a new shape can trigger an expensive compile |

`traineval` also exists for matching training and evaluation behavior. It is not
intended as a performance mode.

### One-off or heterogeneous inference

Use `default` for a single calculation, a small number of calculations, or
batched systems with different compositions. It is the API default, so the
argument can be omitted.

```{code-cell} python3
predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2", device="cuda", inference_settings="default"
)
```

The default mode avoids `torch.compile` startup, keeps TF32 disabled, does not
merge composition-specific experts, and enables activation checkpointing to
limit memory usage.

### MD and relaxation

Use `turbo` for a long trajectory of one system. This mode enables TF32, merges
MOLE experts, disables activation checkpointing, and uses dynamic compilation.
Dynamic compilation is important for ordinary MD because changes in the neighbor
list can change the edge tensor shape even when the atoms and composition are
unchanged.

```{code-cell} python3
predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2", device="cuda", inference_settings="turbo"
)
```

MOLE merging requires atomic numbers, total charge, and spin to remain constant.
Batching different systems is therefore not supported in this mode.

### Repeated fixed-shape inference

Use `turbo-fixed` only when atom, edge, and batch tensor shapes are stable or
explicitly padded into stable buckets. It selects
`torch.compile(mode="reduce-overhead", dynamic=False)` and CUDA graphs.

```{code-cell} python3
predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2", device="cuda", inference_settings="turbo-fixed"
)
```

This mode is not automatically selected for MD. A fixed atom count does not imply
a fixed edge count: atoms crossing the neighbor cutoff can produce a new shape.
Each new shape may require tens of seconds of compilation and another CUDA graph.
Use `turbo` unless graph padding or the application guarantees shape reuse.

### Other common workloads

- For high-throughput screening of different compositions, start with `default`
  and batch compatible systems. Advanced users can enable dynamic compilation
  while keeping `merge_mole=False` if enough calls will amortize compilation.
- For very large or memory-constrained systems, keep
  `activation_checkpointing=True`. This can be combined with other custom
  settings at a throughput cost.
- For Hessians or strict numerical comparisons, keep `tf32=False`. Conservative
  forces remain energy gradients in every mode.

## Custom modes for advanced users

The named modes are instances of the
[inference settings API](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/units/mlip_unit/api/inference.py).
Use a custom `InferenceSettings` object when a workload does not match a named
mode.

| Setting | Description |
| ------- | ----------- |
| `tf32` | Enables TF32 for eligible CUDA matrix multiplications. Tensor storage and accumulation remain FP32, but matrix inputs use reduced mantissa precision. This improves H100 throughput and can slightly change energies, forces, stress, and Hessians. |
| `activation_checkpointing` | Recomputes chunks during force backpropagation to reduce activation memory. Enable it for large systems or memory pressure; disable it for maximum speed when memory permits. |
| `merge_mole` | Pre-merges MOLE expert weights. It reduces parameter memory and compute but requires one fixed composition, charge, and spin. |
| `compile` | Enables `torch.compile`. The first prediction can take tens of seconds, so use it only when repeated calls amortize that cost. |
| `compile_mode` | Passed to `torch.compile`. `None` uses the PyTorch default. `"reduce-overhead"` enables CUDA-graph-oriented optimizations and is used by `turbo-fixed`. |
| `compile_dynamic` | When `True`, one graph can accept a range of tensor shapes and is suitable for ordinary MD. When `False`, PyTorch specializes to exact shapes; novel shapes can trigger recompilation. |
| `external_graph_gen` | Accepts externally generated edges. Leave disabled unless integrating another graph generator or developing graph code. |
| `internal_graph_gen_version` | Selects the internal neighbor-list implementation. Version 2 is the default and supports graph parallelism; version 3 uses NVIDIA Alchemi for supported single-GPU workloads. |
| `edge_chunk_size` | Experimental edge padding bucket size. It can reduce dynamic recompilation by limiting the number of distinct edge shapes. |
| `use_quaternion_wigner` | Uses quaternion Wigner-D computation when `True`; `False` selects the Euler-angle implementation. |
| `base_precision_dtype` | Sets model, input, and activation storage precision. FP32 is the default; FP64 is available for higher-precision workloads. |
| `execution_mode` | Selects a model backend. `None` automatically chooses a compatible backend; advanced users can request modes such as `umas_fast_gpu`. |

For example, this custom mode keeps dynamic compilation and activation
checkpointing for a repeated, memory-constrained simulation:

```{code-cell} python3
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

settings = InferenceSettings(
    tf32=True,
    activation_checkpointing=True,
    merge_mole=True,
    compile=True,
    compile_dynamic=True,
)

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2", device="cuda", inference_settings=settings
)
```

## Enabling gradient stress or Hessian prediction

Some tasks, for example omol, odac, or oc20/25, were not trained using stress labels. Similarly, no tasks were supervised to predict Hessians. However, predictions of untrained derivatives of energy, such as stress and Hessians, can be enabled by using the following inference settings flags,

| Setting Flag  | Description |
| ----- | ----- |
| predict_untrained_forces | A set of task/dataset names (e.g., `{"omol", "oc20"}`) for which forces will be computed via autograd even though the checkpoint was not trained with a forces head for those tasks. |
| predict_untrained_stress | A set of task/dataset names for which stress tensors will be computed via autograd even though the checkpoint was not trained with a stress head for those tasks. The default empty set disables this. |
| predict_untrained_hessian | A set of task/dataset names for which the Hessian matrix will be computed via autograd. |

For example, to enable stress and Hessian predictions with `omol` level of theory, the following settings can be used,

```{code-cell} python3
settings = InferenceSettings(
    predict_untrained_stress={'omol'},
    predict_untrained_hessian={'omol'}
)

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2", device="cuda", inference_settings=settings
)
```

## Multi-GPU Inference

UMA supports Graph Parallel inference natively. The graph is chunked into each rank and both the forward and backwards communication is handled by the built-in graph parallel algorithm with torch distributed. Because Multi-GPU inference requires special setup of communication protocols within a node and across nodes, we leverage [ray](https://www.ray.io/) to launch Ray Actors for each GPU-rank under the hood. This allows us to seamlessly scale to any infrastructure that can run Ray.

To make things simple for the user that wants to run multi-gpu inference locally, we provide a drop-in replacement for MLIPPredictUnit, called [ParallelMLIPPredictUnit](https://github.com/facebookresearch/fairchem/blob/85bd83535fedbc1d99eee4c12e175603ccc44ef7/src/fairchem/core/units/mlip_unit/predict.py#L415)

:::{note}
To enable multi-GPU inference, you need to install Ray manually or through the fairchem extra dependencies option.
:::

```bash
pip install fairchem-core[extras]
```

For example, we can create a predictor with 8 GPU workers in a very similar way to MLIPPredictUnit and perform an MD calculation with the ASE calculator. This mode of operation is also compatible with our LAMMPS integration.

```python
from ase import units
from ase.md.langevin import Langevin
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import time

from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2", inference_settings="turbo", device="cuda", workers=1
)
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = get_fcc_crystal_by_num_atoms(8000)
atoms.calc = calc

dyn = Langevin(
    atoms,
    timestep=0.1 * units.fs,
    temperature_K=400,
    friction=0.001 / units.fs,
)
# warmup 10 steps
dyn.run(steps=10)
start_time = time.time()
dyn.attach(
    lambda: print(
        f"Step: {dyn.get_number_of_steps()}, E: {atoms.get_potential_energy():.3f} eV, "
        f"QPS: {dyn.get_number_of_steps()/(time.time()-start_time):.2f}"
    ),
    interval=1,
)
dyn.run(steps=1000)
```

:::{tip}
This will automatically create a Ray server on your local machine and use a local client to connect to it. If you have set up a Ray cluster, you can leverage it to run parallel inference on as many nodes as you like.
:::
