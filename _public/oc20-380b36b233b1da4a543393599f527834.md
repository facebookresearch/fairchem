# OC20 Leaderboard

This leaderboard evaluates performance on the **Open Catalyst 2020 (OC20)** dataset - a large-scale dataset for catalyst discovery containing DFT relaxations across a wide variety of adsorbate-catalyst combinations.
See the [OC20 paper](https://arxiv.org/abs/2010.09990) for more details.

The leaderboard supports two tasks:

- **S2EF (Structure to Energy and Forces)**: Predict energy and per-atom forces given an atomic structure.
- **IS2RE (Initial Structure to Relaxed Energy)**: Predict the relaxed energy given the initial structure.

Both tasks are evaluated across four test splits:
- **ID**: In-domain test set
- **OOD-Ads**: Out-of-domain adsorbates
- **OOD-Cat**: Out-of-domain catalysts
- **OOD-Both**: Out-of-domain adsorbates and catalysts

## Download

| Benchmarks | URL |
|----------|----------|
| S2EF   | [Test](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_test_lmdbs.tar.gz)     |
| IS2RE    | [Train+Val+Test](https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz)     |

## Install the necessary packages

```bash
pip install "fairchem-core>=2.5.0"
```

## S2EF

Predictions must be saved as ".npz" files containing the following keys for each split (`id`, `ood_ads`, `ood_cat`, `ood_both`):

```
{split}_ids <class 'numpy.ndarray'>
{split}_energy <class 'numpy.ndarray'>
{split}_forces <class 'numpy.ndarray'>
{split}_chunk_idx <class 'numpy.ndarray'>
```

Where,
- `{split}_ids` corresponds to the unique system identifiers
- `{split}_energy` is the predicted energy for each system
- `{split}_forces` is the predicted forces, concatenated across all systems
- `{split}_chunk_idx` is the cumulative atom count used to split the concatenated forces back into per-system arrays

As an example:

```python
import numpy as np
from fairchem.core.datasets import AseDBDataset
from fairchem.core import pretrained_mlip, FAIRChemCalculator

### Define your MLIP calculator
predictor = pretrained_mlip.get_predict_unit(args.checkpoint, device="cuda")
calc = FAIRChemCalculator(predictor, task_name="oc20")

results = {}
for split in ["id", "ood_ads", "ood_cat", "ood_both"]:
    dataset = AseDBDataset({"src": f"path/to/oc20/s2ef/test/{split}"})

    ids = []
    energy = []
    forces = []
    natoms = []
    for idx in range(len(dataset)):
        atoms = dataset.get_atoms(idx)
        atoms.calc = calc
        ids.append(atoms.info["sid"])
        natoms.append(len(atoms))
        energy.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())

    forces = np.concatenate(forces)
    chunk_idx = np.cumsum(natoms)[:-1]

    results[f"{split}_ids"] = np.array(ids)
    results[f"{split}_energy"] = np.array(energy)
    results[f"{split}_forces"] = forces
    results[f"{split}_chunk_idx"] = chunk_idx

np.savez_compressed("oc20_s2ef_predictions.npz", **results)
```

:::{warning}
The above example can be very slow on a single GPU and we encourage users to parallelize this however they like. We provide the example as a means to understand the expected format for the leaderboard.
:::

## IS2RE

Predictions must be saved as ".npz" files containing the following keys for each split (`id`, `ood_ads`, `ood_cat`, `ood_both`):

```
{split}_ids <class 'numpy.ndarray'>
{split}_energy <class 'numpy.ndarray'>
```

Where,
- `{split}_ids` corresponds to the unique system identifiers
- `{split}_energy` is the predicted relaxed energy for each system

Once a prediction file is generated, proceed to the [leaderboard](https://huggingface.co/spaces/facebook/fairchem_leaderboard), fill in the submission form, upload your file, select the corresponding evaluation type ("OC20 S2EF Test" or "OC20 IS2RE Test") and hit submit. Stay on the page until you see the success message.
