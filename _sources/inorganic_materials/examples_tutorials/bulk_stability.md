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

Stability
------------------

We're going to start simple here - let's run a local relaxation (optimize the unit cell and positions) using a pre-trained EquiformerV2-31M-OMAT24-MP-sAlex checkpoint. This checkpoint has a few fun properties
1. It's a relatively small (31M) parameter model
2. It was pre-trained on the OMat24 dataset, and then fine-tuned on the MPtrj and Alexandria datasets, so it should emit energies and forces that are consistent with the MP GGA (PBE/PBE+U) level of theory

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

```{code-cell} ipython3
from __future__ import annotations

import pprint

from ase.build import bulk
from ase.optimize import LBFGS
from quacc.recipes.mlp.core import relax_job

# Make an Atoms object of a bulk Cu structure
atoms = bulk("Cu")

# Run a structure relaxation
result = relax_job(
    atoms,
    method="fairchem",
    name_or_path="uma-s-1p1",
    task_name="omat",
    opt_params={"fmax": 1e-3, "optimizer": LBFGS},
)
```

```{code-cell} ipython3
pprint.pprint(result)
```

Congratulations; you ran your first relaxation using an OMat24-trained checkpoint and `quacc`!
