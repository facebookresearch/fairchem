---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# UMA Quick Start w/ ASE

The easiest way to use pretrained models is via the [ASE](https://wiki.fysik.dtu.dk/ase/) `FAIRChemCalculator`.
A single UMA model can be used for a wide range of applications in chemistry and materials science by picking the
appropriate task name for domain specific prediction.

:::{tip}
Make sure you have a Hugging Face account, have already applied for model access to the
[UMA model repository](https://huggingface.co/facebook/UMA), and have logged in to Hugging Face using an access token.
:::

## Available Tasks

Choose the task that matches your application domain:

:::::{grid} 1 2 3 5
::::{card} oc20
:link: ../catalysts/datasets/summary.md

```{image} ../assets/icons/catalysis.svg
:alt: Catalysis
:width: 60px
:align: center
```

Heterogeneous Catalysis
+++
[Learn more →](../catalysts/datasets/summary.md)
::::

::::{card} omat
:link: ../inorganic_materials/datasets/summary.md

```{image} ../assets/icons/inorganic.svg
:alt: Inorganic Materials
:width: 60px
:align: center
```

Inorganic Materials
+++
[Learn more →](../inorganic_materials/datasets/summary.md)
::::

::::{card} omol
:link: ../molecules/datasets/summary.md

```{image} ../assets/icons/molecules.svg
:alt: Molecules
:width: 60px
:align: center
```

Molecules & Polymers
+++
[Learn more →](../molecules/datasets/summary.md)
::::

::::{card} omc
:link: ../molecules/datasets/omc25.md

```{image} ../assets/icons/molecular-crystals.svg
:alt: Molecular Crystals
:width: 60px
:align: center
```

Molecular Crystals
+++
[Learn more →](../molecules/datasets/omc25.md)
::::

::::{card} odac
:link: ../dac/datasets/summary.md

```{image} ../assets/icons/mofs-dac.svg
:alt: MOFs for DAC
:width: 60px
:align: center
```

MOFs for Direct Air Capture
+++
[Learn more →](../dac/datasets/summary.md)
::::
:::::

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

## Relax an adsorbate on a catalytic surface

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

## Relax an inorganic crystal

```python
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = bulk("Fe")
atoms.calc = calc

opt = LBFGS(FrechetCellFilter(atoms))
opt.run(0.05, 100)
```

## Run molecular MD

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

## Calculate a spin gap

:::{note}
For molecular systems using the `omol` task, you can specify charge and spin multiplicity via the `atoms.info` dictionary. This is important for modeling charged or open-shell systems.
:::

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
