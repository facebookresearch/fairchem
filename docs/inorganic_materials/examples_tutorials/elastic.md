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

Elastic Tensors
------------------

Let's do something more interesting that normally takes quite a bit of work in DFT: calculating an elastic constant! Elastic properties are important to understand how strong or easy to deform a material is, or how a material might change if compressed or expanded in specific directions (i.e. the Poisson ratio!).

We don't have to change much code from above, we just use a built-in recipe to calculate the elastic tensor from `quacc`. This recipe
1. (optionally) Relaxes the unit cell using the MLIP
2. Generates a number of deformed unit cells by applying strains
3. For each deformation, a relaxation using the MLIP and (optionally) a single point calculation is run
4. Finally, all of the above calculations are used to calculate the elastic properties of the material

For more documentation, see the quacc docs for [quacc.recipes.mlp.elastic_tensor_flow](https://quantum-accelerators.github.io/quacc/reference/quacc/recipes/mlp/elastic.html#quacc.recipes.mlp.elastic.elastic_tensor_flow)

```{code-cell} ipython3
from __future__ import annotations

from ase.build import bulk
from quacc.recipes.mlp.elastic import elastic_tensor_flow

# Make an Atoms object of a bulk Cu structure
atoms = bulk("Cu")

# Run an elastic property calculation with our favorite MLP potential
result = elastic_tensor_flow(
    atoms,
    job_params={
        "all": dict(
            method="fairchem",
            get_predict_unit_kwargs={"model_name": "uma-s-1"},
            task_name="omat",
        ),
    },
)
```

```{code-cell} ipython3
result["elasticity_doc"].bulk_modulus
```

Congratulations, you ran your first elastic tensor calculation!
