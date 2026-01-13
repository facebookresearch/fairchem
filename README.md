[//]: # (<h1 align="center">)

[//]: # ()
[//]: # (<p align="center">)

[//]: # (  <img width="559" height="200" src="https://github.com/user-attachments/assets/25cd752c-3c56-469d-8524-4e493646f6b2"?)

[//]: # (</p>)

[//]: # ()
[//]: # (</h1>)

<h4 align="center">

![tests](https://github.com/facebookresearch/fairchem/actions/workflows/test.yml/badge.svg?branch=main&event=push)
![PyPI - Version](https://img.shields.io/pypi/v/fairchem-core)
![Static Badge](https://img.shields.io/badge/python-3.10%2B-blue)
[![codecov](https://codecov.io/gh/facebookresearch/fairchem/graph/badge.svg)](https://codecov.io/gh/facebookresearch/fairchem)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15587498.svg)](https://doi.org/10.5281/zenodo.15587498)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new/facebookresearch/fairchem?quickstart=1)

</h4>

# `fairchem` by the FAIR Chemistry team

`fairchem` is the [FAIR](https://ai.meta.com/research/) Chemistry's centralized repository of all its data, models,
demos, and application efforts for materials science and quantum chemistry.

> :warning: **FAIRChem version 2 is a breaking change from version 1 and is not compatible with our previous pretrained models and code.**
> If you want to use an older model or code from version 1 you will need to install [version 1](https://pypi.org/project/fairchem-core/1.10.0/),
> as detailed [here](#looking-for-fairchem-v1-models-and-code).

> [!CAUTION]
> UMA models and legacy inorganic bulk models trained using OMat24 are trained with DFT and DFT+U total energy labels.
> These are not compatible with Materials Project calculations. If you are using UMA or models trained on OMat24 only
> for such calculations, you can find a OMat24 specific calculations of reference unary compounds and MP2020-style
> anion and GGA/GGA+U mixing corrections in the [OMat24 Hugging Face repo](https://huggingface.co/datasets/facebook/OMAT24).
> Do not use MP2020 corrections or use the MP references compounds when using OMat24 trained models. Additional care
> must be taken when computing energy differences, such as formation and energy above hull and comparing with calculations
> in the Materials Project since DFT pseudopotentials are different and magnetic ground states may differ as well.


## Latest news
Oct 2025 - [check out our seamless Multi-node, Multi-GPU and LAMMPs interfaces to run large scale dynamics!](#multi-gpu-inference-and-lammps)

## Read our latest release post!
Read about the [UMA model and OMol25 dataset](https://ai.meta.com/blog/meta-fair-science-new-open-source-releases/) release.

[![Meta FAIR Science Release](https://github.com/user-attachments/assets/acddd09b-ed6f-4d05-9a4b-9ba5e2301150)](https://ai.meta.com/blog/meta-fair-science-new-open-source-releases/?ref=shareable)

## Try the demo!
If you want to explore model capabilities check out our
[educational demo](https://facebook-fairchem-uma-demo.hf.space/)

[![Educational Demo](https://github.com/user-attachments/assets/7005d1bb-4459-403d-b299-d41fdd8c48ec)](https://facebook-fairchem-uma-demo.hf.space/)

## Installation
Although not required, we highly recommend installing using a package manager and virtualenv such as [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer), it is much faster and better at resolving dependencies than standalone pip.

Install fairchem-core using pip
```bash
pip install fairchem-core
```

If you want to contribute or make modifications to the code, clone the repo and install in edit mode
```bash
git clone git@github.com:facebookresearch/fairchem.git

pip install -e fairchem/packages/fairchem-core[dev]
```

## Quick Start
The easiest way to use pretrained models is via the [ASE](https://wiki.fysik.dtu.dk/ase/) `FAIRChemCalculator`.
A single uma model can be used for a wide range of applications in chemistry and materials science by picking the
appropriate task name for domain specific prediction.

### Instantiate a calculator from a pretrained model
Make sure you have a Hugging Face account, have already applied for model access to the
[UMA model repository](https://huggingface.co/facebook/UMA), and have logged in to Hugging Face using an access token.
You can use the following to save an auth token,
```bash
huggingface-cli login
```

Models are referenced by their name, below are the currently supported models:

| Model Name | Description |
|---|---|
| uma-s-1p1 | Latest version of the UMA small model, fastest of the UMA models while still SOTA on most benchmarks (6.6M/150M active/total params) |
| uma-m-1p1 | Best in class UMA model across all metrics, but slower and more memory intensive than uma-s (50M/1.4B active/total params) |

### Set the task for your application and calculate

- **oc20:** use this for catalysis
- **omat:** use this for inorganic materials
- **omol:** use this for molecules
- **odac:** use this for MOFs
- **omc:** use this for molecular crystals

#### Relax an adsorbate on a catalytic surface,
```python
from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="oc20")

# Set up your system as an ASE atoms object
slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")

slab.calc = calc

# Set up LBFGS dynamics object
opt = LBFGS(slab)
opt.run(0.05, 100)
```

#### Relax an inorganic crystal,
```python
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = bulk("Fe")
atoms.calc = calc

opt = FIRE(FrechetCellFilter(atoms))
opt.run(0.05, 100)
```

#### Run molecular MD,
```python
from ase import units
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omol")

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

#### Calculate a spin gap,
```python
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")

#  singlet CH2
singlet = molecule("CH2_s1A1d")
singlet.info.update({"spin": 1, "charge": 0})
singlet.calc = FAIRChemCalculator(predictor, task_name="omol")

#  triplet CH2
triplet = molecule("CH2_s3B1d")
triplet.info.update({"spin": 3, "charge": 0})
triplet.calc = FAIRChemCalculator(predictor, task_name="omol")

triplet.get_potential_energy() - singlet.get_potential_energy()
```

#### Multi-GPU Inference and LAMMPs
If you have multiple gpus (or multiple nodes), we handle all the parallelism for you under the hood by a single flag (workers=N). For example, you can run the following 8000 atom md simulation with ~10 qps (8x H100 GPU), ~10x faster than single-gpu inference! Current benchmarks show we can run uma-s @ ~1 ns/per day with 100k+ atoms systems in real MD scenarios (more on this to come!). This is also compatible with LAMMPs to perform large scale MD. See our [docs](https://fair-chem.github.io/core/common_tasks/summary.html) for more details. This requires the Ray package to be installed and comes with the extras bundle.
```
pip install fairchem-core[extras]
```

```python
from ase import units
from ase.md.langevin import Langevin
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import time

from fairchem.core.datasets.common_structures import get_fcc_carbon_xtal

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p1", inference_settings="turbo", device="cuda", workers=8
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

### LICENSE
`fairchem` is available under a [MIT License](LICENSE.md). Models/checkpoint licenses vary by application area.
