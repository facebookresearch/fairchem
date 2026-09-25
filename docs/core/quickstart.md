---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Hello World with UMA

This tutorial takes you from an ASE structure to a UMA prediction. You will
load one pretrained model, use it for two chemistry domains, and learn how to
substitute your own structure.

:::{note} Before you start
Complete the [installation and Hugging Face access steps](./install.md). The
first run downloads the gated `uma-s-1p2p1` checkpoint.
:::

## The four-step workflow

Every basic calculation follows the same pattern:

1. Represent the system as an ASE `Atoms` object.
2. Load a pretrained UMA model.
3. Choose the task matching the system and attach a `FAIRChemCalculator`.
4. Ask ASE for energies and forces or use an ASE simulation method.

## Load UMA once

```{code-cell} ipython3
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2p1",
    device="cuda",
)
```

The predictor contains the shared UMA model. The calculator created for each
system supplies the domain-specific task.

## Example 1: calculate a molecular spin gap

For the `omol` task, set the molecule's total charge and spin multiplicity in
`atoms.info`. Here we compare singlet and triplet states of CH₂.

```{code-cell} ipython3
from ase.build import molecule

singlet = molecule("CH2_s1A1d")
singlet.info.update({"charge": 0, "spin": 1})
singlet.calc = FAIRChemCalculator(predictor, task_name="omol")

triplet = molecule("CH2_s3B1d")
triplet.info.update({"charge": 0, "spin": 3})
triplet.calc = FAIRChemCalculator(predictor, task_name="omol")

spin_gap = triplet.get_potential_energy() - singlet.get_potential_energy()
print(f"Triplet-singlet energy difference: {spin_gap:.3f} eV")
```

## Example 2: relax an inorganic crystal

The `omat` task predicts stress as well as energy and forces, so ASE can relax
both the atoms and the unit cell.

```{code-cell} ipython3
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

iron = bulk("Fe")
iron.calc = FAIRChemCalculator(predictor, task_name="omat")

optimizer = FIRE(FrechetCellFilter(iron), logfile=None)
optimizer.run(fmax=0.05, steps=100)

print(f"Relaxed energy: {iron.get_potential_energy():.3f} eV")
print("Relaxed cell (Å):")
print(iron.cell)
```

## Try your own structure

ASE reads many common chemistry file formats, including XYZ, CIF, POSCAR, and
trajectory files. Replace the filename and task below with values appropriate
for your system.

```{code-cell} ipython3
:tags: [skip-execution]

from ase.io import read

atoms = read("my-structure.xyz")

# Required for molecules evaluated with the omol task.
atoms.info.update({"charge": 0, "spin": 1})

atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

Choose the task by scientific domain, not merely by which task accepts the
structure:

| Domain | Task | Start here |
| --- | --- | --- |
| Molecules and polymers | `omol` | [OMol25](../molecules/datasets/omol25.md) |
| Inorganic materials | `omat` | [OMat24](../inorganic_materials/datasets/omat24.md) |
| Heterogeneous catalysis | `oc20`, `oc22`, or `oc25` | [Catalysis datasets](../catalysts/datasets/summary.md) |
| Molecular crystals | `omc` | [OMC25](../molecules/datasets/omc25.md) |
| MOFs and direct air capture | `odac` | [ODAC datasets](../dac/datasets/summary.md) |

:::{warning}
Different tasks reproduce different levels of theory. Do not compare energies
across tasks as though they came from one reference calculation.
:::

## Next steps

- [Explore UMA's capabilities](./introduction.md#what-you-can-do-with-uma) by
  domain.
- Review task limitations in the [UMA model guide](./uma.md).
- Learn about inference settings in the [ASE calculator guide](./common_tasks/ase_calculator.md).
- Try the [playground](https://aidemos.atmeta.com/uma?view=playground).
