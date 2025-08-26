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

Calculation workflows with FAIRChem models
------------------------------------------

This repo is integrated with workflow tools like [QuAcc](https://github.com/Quantum-Accelerators/quacc) to make complex molecular simulation workflows easy. You can use any MLP recipe (relaxations, single-points, elastic calculations, etc) and simply specify the `fairchem` model type. Below is an example that uses the default elastic_tensor_flow flow.

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
            name_or_path="uma-s-1p1",
            task_name="omat",
        ),
    },
)
```

One of the nice things about QuAcc is that you can use plugins for whatever your favorite workflow engine is (fireworks, parssl, prefect, etc). Some of these methods can scale to hundreds of thousands of parallel calculations and are used by the FAIR chemistry team regularly!
