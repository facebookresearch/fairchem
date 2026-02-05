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

# Formation Energy

:::{tip} What You Will Learn
Calculate formation energies for inorganic materials using UMA with Materials Project-compatible corrections.
:::

We're going to start simple here - let's run a local relaxation (optimize the unit cell and positions) using a pre-trained UMA model to compute formation energies for inorganic materials.

Note predicting formation energy using models that models trained solely on OMat24 must use OMat24 compatible references and corrections for mixing PBE and PBE+U calculations. We use MP2020-style corrections fitted to OMat24 DFT calculations. For more information see the [documentation](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/thermodynamic-stability/anion-and-gga-gga+u-mixing) at the Materials Project. The necessary references can be found using the `fairchem.data.omat` package!

````{admonition} Need to install fairchem-core or get UMA access or getting permissions/401 errors?
:class: dropdown


1. Install the necessary packages using pip, uv etc
```{code-cell} ipython3
:tags: [skip-execution]

! pip install fairchem-core fairchem-data-omat
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

```{code-cell} ipython3
from __future__ import annotations

import pprint

from ase.build import bulk
from ase.optimize import FIRE
from quacc.recipes.mlp.core import relax_job
from quacc import flow
from fairchem.core.calculate import FAIRChemCalculator, FormationEnergyCalculator

# Make an Atoms object of a bulk Cu structure
atoms = bulk("Cu")

# Run a structure relaxation
@flow
def relax_flow(*args, **kwargs):
  return relax_job(*args, **kwargs)

result = relax_flow(
    atoms,
    method="fairchem",
    name_or_path="uma-s-1p1",
    task_name="omat",
    relax_cell=True,
    opt_params={"fmax": 1e-3, "optimizer": FIRE},
)

# Get the realxed atoms!
atoms = result["atoms"]

# Create an calculator using uma-s-1p1
calculator = FAIRChemCalculator.from_model_checkpoint("uma-s-1p1", task_name="omat")

# Now use the FormationEnergyCalculator to calculate the formation energy
# This will now return MP-style corrected formation energies
# For the omat task, this defaults to apply MP2020 style corrections with OMat24 compatibility
form_e_calc = FormationEnergyCalculator(calculator, apply_corrections=True)
atoms.calc = form_e_calc
form_energy = atoms.get_potential_energy()
```

```{code-cell} ipython3
pprint.pprint(f"Total energy: {result['results']['energy']} eV \n Formation energy {form_energy} eV")
```

Compare the results to the value of [-3.038 eV/atom reported](https://next-gen.materialsproject.org/materials/mp-1265?chemsys=Mg-O#thermodynamic_stability) in the the Materials Project!
*Note that we expect differences due to the different DFT settings used to calculate the OMat24 training data.*

Congratulations; you ran your first relaxation and predicted the formation energy of MgO using UMA and `quacc`!
