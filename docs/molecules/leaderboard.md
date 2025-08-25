# OMol25 Leaderboard

As part of the OMol25 release, we introduce a community leaderboard for researchers to submit their predictions to for evaluation - https://huggingface.co/spaces/facebook/fairchem_leaderboard.
Below we outline the steps to generate predictions and submit them to the leaderboard.

The leaderboard is broken into two different sections - "S2EF" and "Evaluations".
Structure to Energy and Forces (S2EF) is the most straightforward evaluation for MLIPs - given a structure, how well can you predict the total energy and per-atom forces.
Evaluations correspond to several chemistry relevant tasks (spin gap, ligand-strain, etc.) introduced in OMol25 to evaluate MLIPs beyond simple energy and force metrics (see the [paper](https://arxiv.org/pdf/2505.08762) for more details).

The simplest way to get started is to have an ASE-compatible MLIP calculator that can make energy and force predictions.

## S2EF
The leadebroard supports S2EF evaluations for both the OMol25 "Validation" and "Test" sets. The download links for these datasets can be found [here](https://huggingface.co/facebook/OMol25/blob/main/DATASET.md)
Validation and Test sets should contain 2,762,021 and 2,805,046 samples, respectively.

Predictions must be saved as ".npz" files and shall contain the following information:
```
ids <class 'numpy.ndarray'>
energy <class 'numpy.ndarray'>
forces <class 'numpy.ndarray'>
natoms <class 'numpy.ndarray'>
```
Where,
- `ids` corresponds to the unique identifier, `atoms.info["source"]`
- `energy` is the predicted energy
- `forces` is the predicted forces, concatenated across all systems
- `natoms` is the number of atoms corresponding to each prediction

As an example:

```python
from fairchem.core.datasets import AseDBDataset
from fairchem.core import pretrained_mlip, FAIRChemCalculator

### Define your MLIP calculator
predictor = pretrained_mlip.get_predict_unit(args.checkpoint, device="cuda")
calc = calc = FAIRChemCalculator(predictor, task_name="omol")

### Read in the dataset you wish to submit predictions to
dataset = AseDBDataset({"src": "path/to/omol/test_data"})

ids = []
energy = []
forces = []
natoms = []
for idx in range(len(dataset)):
    atoms = dataset.get_atoms(idx)
    atoms.calc = calc
    ids.append(atoms.info["source"])
    natoms.append(len(atoms))
    energy.append(atoms.get_potential_energy())
    forces.append(atoms.get_forces())

forces = np.concatenate(forces)

np.savez_compressed(
    "test_predictions.npz",
    ids=ids,
    energy=energy,
    forces=forces,
    natoms=natoms,
)
```

> :warning: DISCLAIMER: The above example can be very slow on a single GPU and we encourage users to parallelize this however they like. We provide the example as a means to understand the expected format for the leaderboard.

Once a prediction file is generated, proceed to the leaderboard, fill in the submission form, upload your file, select "Validation" or "Test" and hit submit. Stay on the page until you see the success message.

## Evaluations

The following evaluations are currently available on the OMol25 leaderboard:
* Ligand pocket: Protein-ligand interaction energy as a proxy to the binding energy, central to many biological processes.
* Ligand strain: Ligand-strain energy is an important task to understanding protein-ligand binding.
* Conformers: Identifying the lowest energy conformer is a crucial part of many biological and pharmaceutical tasks.
* Protonation: As a proxy to pKa prediction, we evaluate energy differences of structures differing by one proton.
* IE/EA: The addition, removal, and transfer of electrons is central to many redox processes.
* Distance scaling: Short range and long range intermolecular interactions are essential for observable properties like phase changes, density, etc.
* Spin gap: Differences between spin states can play a critical role of molecular optic devices and photactive catalysts.

For a detailed descripion of each task we refer people to the original [manuscript](https://arxiv.org/pdf/2505.08762).
The download links for evaluation inputs can be found [here](https://huggingface.co/facebook/OMol25/blob/main/DATASET.md).

To generate prediction files for the different tasks, we have released a set of [recipes](https://github.com/facebookresearch/fairchem/blob/omol_evals/src/fairchem/core/components/calculate/recipes/omol.py) to be used with ASE-compatible calculators.
Each evaluation task has its own unique structure, a detailed description of the expected output is provided in the recipe docstrings. The following recipes should be used to evaluate the corresponding task:

* [Ligand pocket](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/components/calculate/recipes/omol.py#L321)
* [Ligand strain](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/components/calculate/recipes/omol.py#L370)
* [Conformers](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/components/calculate/recipes/omol.py#L138)
* [Protonation](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/components/calculate/recipes/omol.py#L186)
* [IE/EA](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/components/calculate/recipes/omol.py#L235)
* [Distance scaling](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/components/calculate/recipes/omol.py#L437)
* [Spin gap](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/components/calculate/recipes/omol.py#L282)

As an example:

```python
import json
import pickle
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.components.calculate.recipes.omol import spin_gap

### Define your MLIP calculator
predictor = pretrained_mlip.get_predict_unit(args.checkpoint, device="cuda")
calc = calc = FAIRChemCalculator(predictor, task_name="omol")

### Load the desired evaluation task input data
with open("path/to/spin_gap_inputs.pkl", "rb") as f:
    spin_gap_data = pickle.load(f)

results = spin_gap(spin_gap_data, calc)
with open("spin_gap_results.json") as f:
    json.dump(results, f)
```
> :warning: DISCLAIMER: Conformers, Protonation, Ligand strain, and Distance scaling can be extremely slow on a single GPU and we encourage userse to parallelize this however they like.

>:warning: We parallelize this using the [OMolRunner](https://github.com/facebookresearch/fairchem/blob/d790a4665dc55d880b23074d75be4f5160840e57/src/fairchem/core/components/calculate/omol_runner.py#L24) to break up the inputs into chunks and parallelize them accordingly.
>The [OMolReducer](https://github.com/facebookresearch/fairchem/blob/omol_evals/src/fairchem/core/components/benchmark/omol_reducer.py) can then be used to combine results.

Once a prediction file is generated, proceed to the leaderboard, fill in the submission form, upload your file, select the corresponding evaluation task and hit submit. Stay on the page until you see the success message.
