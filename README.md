<h1 align="center"> <code>fairchem</code> by FAIR Chemistry </h1>

<p align="center">
  <img width="559" height="200" src="https://github.com/FAIR-Chem/fairchem/assets/45150244/5872c21c-8f39-41af-b703-af9817f0affe"?
</p>


<h4 align="center">

![tests](https://github.com/FAIR-Chem/fairchem/actions/workflows/test.yml/badge.svg?branch=main)
![PyPI - Version](https://img.shields.io/pypi/v/fairchem-core)
![Static Badge](https://img.shields.io/badge/python-3.10%2B-blue)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new/FAIR-Chem/fairchem?quickstart=1)

`fairchem` is the [FAIR](https://ai.meta.com/research/) Chemistry's centralized repository of all its data, models,
demos, and application efforts for materials science and quantum chemistry.

> :warning: **FAIRChem version 2 is not compatible with our previous pretrained models.**
> If you want to use an older model you will need to install [version 1](https://pypi.org/project/fairchem-core/1.10.0/),
> as detailed [here](#looking-for-fairchem-v1).

### Try the demo!
If you want to explore model capabilities check out our
[educational demo](https://huggingface.co/spaces/facebook/fairchem_uma_demo)
![output](https://github.com/user-attachments/assets/7005d1bb-4459-403d-b299-d41fdd8c48ec)

### Installation
Install fairchem-core using pip,
```bash
pip install git+https://github.com/facebookresearch/fairchem.git@fairchem_core-2.0.0#subdirectory=packages/fairchem-core
```
**PyPI package coming soon!**

### Quick Start
The easiest way to use pretrained models is via the `FAIRChemCalculator` [ASE](https://wiki.fysik.dtu.dk/ase/).
A single uma model can be used for a wide range of applications in chemistry and materials science by picking the
appropriate task name for domain specific prediction.

#### Instantiate a calculator from a pretrained model
Make sure you have a Hugging Face account, have already applied for model access to the 
[UMA model repository](https://huggingface.co/facebook/UMA), and have logged in using to Hugging Face using an access token.

#### Set the task for your application and calculate

- **oc20:** use this for catalysis
- **omat:** use this for inorganic materials
- **omol:** use this for molecules
- **odac:** use this for MOFs
- **omc:** use this for molecular crystals

Relax adsorbate on a catalytic surface,
```python
from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator

calc = FAIRChemCalculator(hf_hub_filename="uma_sm.pt", device="cuda", task_name="oc20")

# Set up your system as an ASE atoms object
slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")

slab.calc = calc

# Set up LBFGS dynamics object
opt = LBFGS(slab)
opt.run(0.05, 100)
```

Or relax an inorganic crystal,
```python
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import FAIRChemCalculator

calc = FAIRChemCalculator(hf_hub_filename="uma_sm.pt", device="cuda", task_name="omat")

atoms = bulk("Fe")
atoms.calc = calc

opt = LBFGS(FrechetCellFilter(atoms))
opt.run(0.05, 100)
```

Run molecular MD,
```python
from ase import units
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.build import molecule
from fairchem.core import FAIRChemCalculator

calc = FAIRChemCalculator(hf_hub_filename="uma_sm.pt", device="cuda", task_name="omol")

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

### Why a single repository?
A single repository simplifies testing and ensures consistency across our interconnected core, data and application
packages.
The repo is organized into several directories to help you find what you are looking for:

- [`fairchem.core`](src/fairchem/core): State-of-the-art machine learning models for materials science and chemistry
- [`fairchem.data`](src/fairchem/data): Dataset downloads and input generation codes
- [`fairchem.demo`](src/fairchem/demo): Python API for the [Open Catalyst Demo](https://open-catalyst.metademolab.com/)
- [`fairchem.applications`](src/fairchem/applications): Follow up applications and works (AdsorbML, CatTSunami, etc.)

### Looking for fairchem v1?

You can still use models from fairchem version 1, by installing version 1,

```bash
pip install fairchem-core==1.10
```

And using the `OCPCalculator`
```python
from fairchem.core import OCPCalculator

calc = OCPCalculator(
    model_name="EquiformerV2-31M-S2EF-OC20-All+MD",
    local_cache="pretrained_models",
    cpu=False,
)
```

### LICENSE
`fairchem` is available under a [MIT License](LICENSE.md).
