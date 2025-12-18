# Batched atomic simulations with an `InferenceBatcher`

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

```{admonition} Learning Objectives
:class: note
The `InferenceBatcher` class and underlying concurrent batching implementations are experimental and under current development. The following tutorial is intended to provide a basic understanding of the class and its usage, but the API may change. If you have suggestions for improvements, please open an issue or submit a pull request.
```

When running many independent ASE calculations (relaxations, molecular dynamics, etc.) on small to medium-sized systems, you can significantly improve GPU utilization by batching model inference calls together. The `InferenceBatcher` class provides a high-level API to do this with minimal code changes.

The key idea is simple: instead of running each simulation sequentially, `InferenceBatcher` collects inference requests from multiple concurrent simulations and batches them together for more efficient GPU computation.

## Basic setup

To use `InferenceBatcher`, you need to:

1. Create a predict unit as usual
2. Wrap it with `InferenceBatcher`
3. Use `batcher.batch_predict_unit` instead of the original predict unit in your simulation functions

```python
from fairchem.core import pretrained_mlip
from fairchem.core.calculate import FAIRChemCalculator, InferenceBatcher

# Create a predict unit
predict_unit = pretrained_mlip.get_predict_unit("uma-s-1p1")

# Wrap it with InferenceBatcher
batcher = InferenceBatcher(
    predict_unit, concurrency_backend_options=dict(max_workers=32)
)
```

The `max_workers` parameter controls how many concurrent simulations can run concurrently.

## Writing simulation functions

The only requirement for using `InferenceBatcher` is to write your simulation logic as a function that takes an `Atoms` object and a predict unit as arguments:

```python
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS


def run_relaxation(atoms, predict_unit):
    """Run a structure relaxation and return the final energy."""
    calc = FAIRChemCalculator(predict_unit, task_name="omat")
    atoms.calc = calc
    opt = LBFGS(FrechetCellFilter(atoms), logfile=None)
    opt.run(fmax=0.02, steps=100)
    return atoms.get_potential_energy()
```

## Running batched relaxations

Once you have your simulation function, you can run it in batched mode using the executor's `map` or `submit` methods:

### Using `executor.map`

```python
from functools import partial

# Create a list of structures to relax
prim_atoms = [
    bulk("Cu"),
    bulk("MgO", "rocksalt", a=4.2),
    bulk("Si", "diamond", a=5.43),
    bulk("NaCl", "rocksalt", a=3.8),
]

atoms_list = [make_supercell(atoms, 3 * np.identity(3)) for atoms in prim_atoms]

for atoms in atoms_list:
    atoms.rattle(0.1)

# Create a partial function with the batch predict unit
run_relaxation_batched = partial(
    run_relaxation, predict_unit=batcher.batch_predict_unit
)

# Run all relaxations in parallel with batched inference
relaxed_energies = list(batcher.executor.map(run_relaxation_batched, atoms_list))
```

### Using `executor.submit` for more control

If you need more control over the execution or want to process results as they complete:

```python
# Create a new list of structures to relax
atoms_list = [make_supercell(atoms, 3 * np.identity(3)) for atoms in prim_atoms]

for atoms in atoms_list:
    atoms.rattle(0.1)

# Submit all jobs
futures = [
    batcher.executor.submit(run_relaxation, atoms, batcher.batch_predict_unit)
    for atoms in atoms_list
]

# Collect results
relaxed_energies = [future.result() for future in futures]
```

## Running batched molecular dynamics

The same pattern works for molecular dynamics simulations:

```python
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def run_nvt_md(atoms, predict_unit, temperature, traj_fname):
    """Run NVT molecular dynamics simulation."""
    calc = FAIRChemCalculator(predict_unit, task_name="omat")
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature, force_temp=True)
    dyn = Langevin(
        atoms,
        timestep=2 * units.fs,
        temperature_K=temperature,
        friction=0.1,
        trajectory=traj_fname,
        loginterval=5,
    )
    dyn.run(100)


# Run batched MD simulations
run_md_batched = partial(
    run_nvt_md, predict_unit=batcher.batch_predict_unit, temperature=300
)

futures = [
    batcher.executor.submit(run_md_batched, atoms, traj_fname=f"traj_{i}.traj")
    for i, atoms in enumerate(atoms_list)
]

# Wait for all simulations to complete
[future.result() for future in futures]
```

## When to use an `InferenceBatcher`

`InferenceBatcher` is most beneficial when:

- Running many independent simulations on small to medium-sized systems
- GPU utilization is low with serial execution
- Each individual simulation has many inference steps (relaxations, MD)

When running batch inference over static structures, consider using the [batch inference approach](batch_inference.md) with `AtomicData` directly instead. For single large systems, consider using the `MLIPParallelPredictUnit` for graph parallel inference.
