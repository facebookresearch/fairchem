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

Using pre-trained models in ASE
----------

1. First, install `fairchem` in a fresh python environment using one of the approaches in [installation documentation](install).

2. Try the following code to perform a relaxation of an adsorbate on a catalytic surface,
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

To learn more about what this simulation means and how it fits into catalysis, see the [catalysis tutorial](../catalysts/tutorials/intro)!