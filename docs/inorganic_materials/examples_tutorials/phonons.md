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

Phonons
------------------

Phonon calculations are very important for inorganic materials science to
* Calculate thermal conductivity
* Understand the vibrational modes, and thus entropy and free energy, of a material
* Predict the stability of a material at finite temperature (e.g. 300 K)
among many others!
We can run a similarly straightforward calculation that
1. Runs a relaxation on the unit cell and atoms
2. Repeats the unit cell a number of times to make it sufficiently large to capture many interesting vibrational models
3. Generatives a number of finite displacement structures by moving each atom of the unit cell a little bit in each direction
4. Running single point calculations on each of (3)
5. Gathering all of the calculations and calculating second derivatives (the hessian matrix!)
6. Calculating the eigenvalues/eigenvectors of the hessian matrix to find the vibrational modes of the material
7. Analyzing the thermodynamic properties of the vibrational modes.

Note that this analysis assumes that all vibrational modes are harmonic, which is a pretty reasonable approximately for low/moderate temperature materials, but becomes less realistic at high temperatures.

```{code-cell} ipython3
from __future__ import annotations

from ase.build import bulk
from quacc.recipes.mlp.phonons import phonon_flow

# Make an Atoms object of a bulk Cu structure
atoms = bulk("Cu")

# Run a phonon (hessian) calculation with our favorite MLP potential
result = phonon_flow(
    atoms,
    method="fairchem",
    job_params={
        "all": dict(
            name_or_path="uma-s-1p1",
            task_name="omat",
        ),
    },
    min_lengths=10.0,  # set the minimum unit cell size smaller to be compatible with limited github runner ram
)
```

```{code-cell} ipython3
print(
    f'The entropy at { result["results"]["thermal_properties"]["temperatures"][-1]:.0f} K is { result["results"]["thermal_properties"]["entropy"][-1]:.2f} kJ/mol'
)
```

Congratulations, you ran your first phonon calculation!

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
