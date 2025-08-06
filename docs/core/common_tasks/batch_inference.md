# Batch inference with UMA models

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
If your application requires predictions over many systems you can run batch inference using
UMA models to use compute more efficiently and improve GPU utilization. Below we show some easy ways to run batch
inference over batches created at runtime or loading from a dataset. If you want to learn more about the different
inference settings supported have a look at the
[Prediction interface documentation](https://fair-chem.github.io/core/common_tasks/ase_calculator.html)

Generate batches at runtime
-----------------------------
The recommended way to create batches at runtime is to convert ASE `Atoms` objects into `AtomicData`
as follows,

```python
from ase.build import bulk, molecule
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

atoms_list = [bulk("Pt"), bulk("Cu"), bulk("NaCl", crystalstructure="rocksalt", a=2.0)]

# you need to assign the task_name desired
atomic_data_list = [
    AtomicData.from_ase(atoms, task_name="omat") for atoms in atoms_list
]
batch = atomicdata_list_to_batch(atomic_data_list)

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
preds = predictor.predict(batch)
```

The predictions are returned in a dictionary with single `torch.Tensor` value for each property predicted.
system level properties can be accessed using the same index for the system in the `atomic_data_list`, atom level
properties like forces can be obtained for a single system in the batch using the `batch.batch` attribute,
```python
# energy of the first system in the batch
preds["energy"][0]

# forces of the first system in the batch
preds["forces"][batch.batch == 0]
```

## Batch inference using a dataset and a dataloader

If you are running predictions over more structures than you can fit in memory, you can run inference using
a torch Dataloader,

```python
from torch.utils.data import DataLoader
from fairchem.core.datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

dataset = AseDBDataset(
    config=dict(src="path/to/your/dataset.aselmdb", a2g_args=dict(task_name="omol"))
)
loader = DataLoader(dataset, batch_size=200, collate_fn=atomicdata_list_to_batch)
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")

for batch in loader:
    preds = predictor.predict(batch)
```

## Inference over heterogenous batches

For the odd cases where you want to batch systems to be computed with different task predictions
(ie molecules and materials), you can take advantage of UMA models and do it in a single batch
as follows,

```python
from ase.build import bulk, molecule
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

# a molecule
h2o = molecule("H2O")
h2o.info.update({"charge": 0, "spin": 1})

# a bulk
pt = bulk("Pt")

# a catalytic surface
slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")

atomic_data_list = [
    # note that we put the molecule in a large box
    AtomicData.from_ase(
        h2o, task_name="omol", r_data_keys=["spin", "charge"], molecule_cell_size=12
    ),
    AtomicData.from_ase(pt, task_name="omat"),
    AtomicData.from_ase(slab, task_name="oc20"),
]
batch = atomicdata_list_to_batch(atomic_data_list)

predictions = predictor.predict(batch)
```
