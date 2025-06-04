# Batch inference with UMA models

If your application requires predictions over many systems you can run batch inference using
UMA models to use compute more efficiently and improve GPU utilization. Below we show some easy ways to run batch
inference  over batches created at runtime or loading from a dataset. If you want to learn more about the different
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

predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cuda")
preds = predictor.predict(batch)
