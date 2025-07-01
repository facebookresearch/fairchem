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

# AdsorbML tutorial

The [AdsorbML](https://arxiv.org/abs/2211.16486) paper showed that pre-trained machine learning potentials were now viable to find and prioritize the best adsorption sites for a given surface. The results were quite impressive, especially if you were willing to do a DFT single-point calculation on the best calculations.

The latest UMA models are now total-energy models, and the results for the adsorption energy are even more impressive ([see the paper for details and benchmarks](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/)). The AdsorbML package helps you with automated multi-adsorbate placement, and will automatically run calculations using the ML models to find the best sites to sample.

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

## Define desired adsorbate+slab system

```{code-cell} ipython3
from __future__ import annotations

import pandas as pd
from fairchem.data.oc.core import Adsorbate, Bulk, Slab

bulk_src_id = "mp-30"
adsorbate_smiles = "*CO"

bulk = Bulk(bulk_src_id_from_db=bulk_src_id)
adsorbate = Adsorbate(adsorbate_smiles_from_db=adsorbate_smiles)
slabs = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=(1, 1, 1))

# There may be multiple slabs with this miller index.
# For demonstrative purposes we will take the first entry.
slab = slabs[0]
```

## Run heuristic/random adsorbate placement and ML relaxations

Now that we've defined the bulk, slab, and adsorbates of interest, we can quickly use the pre-trained UMA model as a calculator and the helper script `fairchem.core.components.calculate.recipes.adsorbml.run_adsorbml`. More details on the automated pipeline can be found at https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/components/calculate/recipes/adsorbml.py#L316.

```{code-cell} ipython3
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.components.calculate.recipes.adsorbml import run_adsorbml

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1")
calc = FAIRChemCalculator(predictor, task_name="oc20")

outputs = run_adsorbml(
    slab=slab,
    adsorbate=adsorbate,
    calculator=calc,
    optimizer_cls=LBFGS,
    fmax=0.02,
    steps=20,  # Increase to 200 for practical application, 20 is used for demonstrations
    num_placements=10,  # Increase to 100 for practical application, 10 is used for demonstrations
    reference_ml_energies=True,  # True if using a total energy model (i.e. UMA)
    relaxed_slab_atoms=None,
    place_on_relaxed_slab=False,
)
```

```{code-cell} ipython3
top_candidates = outputs["adslabs"]
global_min_candidate = top_candidates[0]
```

```{code-cell} ipython3
top_candidates = outputs["adslabs"]
pd.DataFrame(top_candidates)
```

## Write VASP input files

If you want to verify the results, you should run VASP. This assumes you have access to VASP pseudopotentials. The default VASP flags (which are equivalent to those used to make OC20) are located in `ocdata.utils.vasp`. Alternatively, you may pass your own vasp flags to the `write_vasp_input_files` function as `vasp_flags`. Note that to run this you need access to the VASP pseudopotentials and need to have those set up in ASE.

```{code-cell} ipython3
:tags: [skip-execution]

import os

from fairchem.data.oc.utils.vasp import write_vasp_input_files

# Grab the 5 systems with the lowest energy
top_5_candidates = top_candidates[:5]

# Write the inputs
for idx, config in enumerate(top_5_candidates):
    os.makedirs(f"data/{idx}", exist_ok=True)
    write_vasp_input_files(config["atoms"], outdir=f"data/{idx}/")
```
