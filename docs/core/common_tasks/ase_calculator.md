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

Inference using ASE and Predictor interface
------------------

Inference is done using [MLIPPredictUnit](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/units/mlip_unit/mlip_unit.py#L867). The [FairchemCalculator](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/calculate/ase_calculator.py#L3) (an ASE calculator) is simply a convenience wrapper around the MLIPPredictUnit.

For simple cases such as doing demos or education, the ASE calculator is very easy to use but for any more complex cases such as running MD, batched inference etc, we do not recommend using the calculator interface but using the predictor directly.

```{code-cell} python3
from __future__ import annotations

from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
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

## Default mode

UMA is designed for both general-purpose usage (single or batched systems) and single-system long rollout (MD simulations, relaxations, etc.). For general-purpose use, we suggest using the [default settings](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/units/mlip_unit/api/inference.py#L92). This is a good trade-off between accuracy, speed, and memory consumption and should suffice for most applications. In this setting, on a single 80GB H100 GPU, we expect a user should be able to compute on systems as large as 50k-100k neighbors (depending on their atomic density). Batching is also supported in this mode.

## Turbo mode

For long rollout trajectory use-cases, such as molecular dynamics (MD) or relaxations, we provide a special mode called **turbo**, which optimizes for speed but restricts the user to using a single system where the atomic composition is held constant. Turbo mode is approximately 1.5-2x faster than default mode, depending on the situation. However, batching is not supported in this mode. It can be easily activated as shown below.

```{code-cell} python3
predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p1", device="cuda", inference_settings="turbo"
)
```

## Custom modes for advanced users

The advanced user might quickly see that **default** mode and **turbo** mode are special cases of our [inference settings api](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/units/mlip_unit/api/inference.py#L47). You can customize it for your application if you understand what you are doing. The following table provides more information.

| Setting Flag  | Description |
| ----- | ----- |
| tf32 | enables torch [tf32](https://docs.pytorch.org/docs/stable/notes/cuda.html) format for matrix multiplication. This will speed up inference at a slight trade-off for precision. In our tests, it makes minimal difference to most applications. It is able to preserve equivariance, energy conservation for long rollouts. However, if you are computing higher order derivatives such as Hessians or other calculations that requires strict numerical precision, we recommend turning this off |
| activation_checkpointing | this uses a custom chunked activation checkpointing algorithm and allows significant savings in memory for a small inference speed penalty. If you are predicting on systems >1000 atoms, we recommend keeping this on. However, if you want the absolute fastest inference possible for small systems, you can turn this off |
| merge_mole | This is useful in long rollout applications where the system composition stays constant. By pre-merge the MoLE weights, we can save both memory and compute. |
| compile | This uses torch.compile to significantly speed up computation. Due to the way pytorch traces the internal graph, it requires a long compile time during the first iteration and can even recompile anytime it detected a significant change in input dimensions. It is not recommended if you are computing frequently on very different atomic systems. |
| external_graph_gen | Only use this if you want to use an external graph generator. This should be rarely used except for development |

For example, for an MD simulation use-case for a system of ~500 atoms, we can choose to use a custom mode like the following:

```{code-cell} python3
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

settings = InferenceSettings(
    tf32=True,
    activation_checkpointing=False,
    merge_mole=True,
    compile=True,
    external_graph_gen=False,
    internal_graph_gen_version=2,
)

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p1", device="cuda", inference_settings=settings
)
```

## Multi-GPU Inference

UMA supports Graph Parallel inference natively. The graph is chunked into each rank and both the forward and backwards communication is handled by the built-in graph parallel algorithm with torch distributed. Because Multi-GPU inference requires special setup of communication protocols within a node and across nodes, we leverage [ray](https://www.ray.io/) to launch Ray Actors for each GPU-rank under the hood. This allows us to seemlessly scale to any infrastructure that can run Ray.

To make things simple for the user that wants to run multi-gpu inference locally, we provide a drop-in replacement for MLIPPredictUnit, called [ParallelMLIPPredictUnit](https://github.com/facebookresearch/fairchem/blob/85bd83535fedbc1d99eee4c12e175603ccc44ef7/src/fairchem/core/units/mlip_unit/predict.py#L415)

To enable this you need to install Ray manually or through the fairchem extra dependencies option

```
pip install fairchem-core[extras]
```

For example, we can create a predictor with 8 GPU workers in a very similiar way to MLIPPredictUnit and perform a md calculation with the ase calculator. This mode of operation is also compatible with our LAMMPs integration.

```
from ase import units
from ase.md.langevin import Langevin
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import time

from fairchem.core.datasets.common_structures import get_fcc_carbon_xtal

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p1", inference_settings="turbo", device="cuda", workers=1
)
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = get_fcc_carbon_xtal(8000)
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

This will automatically create a Ray server on your local machine and use a local client to connect to it. If you have setup a Ray cluster, you can leverage it to run parallel inference on as many nodes as you like. We are actively working on optimziations to scale inference to large systems.
