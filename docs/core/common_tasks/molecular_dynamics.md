# Molecular Dynamics with ASE

FAIRChem models implement the standard
[ASE calculator interface](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html).
You can therefore use them with ASE molecular dynamics directly, or use
FAIRChem's Hydra-configured `MDRunner` when you need reproducible configuration,
structured output, checkpointing, or cluster submission.

This guide uses a 32-atom periodic FCC Cu crystal for both NVT and NPT. The
system is small enough to run as an example, while still exercising periodic
boundaries and stress prediction. The checked-in examples use UMA's `omat`
task; their configuration and MD behavior are also tested in CI with ASE's EMT
calculator, without downloading a model.

:::{danger} Security warning
FAIRChem uses Hydra to instantiate Python objects named by `_target_` in YAML.
Only run configurations that you wrote yourself or obtained from a trusted
source.
:::

## Run NVT directly with ASE

An MD calculator must provide energy and forces. Initialize momenta before
using a deterministic thermostat; doing so is also preferable for stochastic
thermostats because it avoids an artificial heating transient.

```python
import numpy as np
from ase import units
from ase.build import bulk
from ase.io import Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from fairchem.core import FAIRChemCalculator, pretrained_mlip

# A 2 x 2 x 2 conventional FCC cell: 32 atoms with periodic boundaries.
atoms = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2p1", device="cuda", inference_settings="turbo"
)
atoms.calc = FAIRChemCalculator(predictor, task_name="omat")

rng = np.random.RandomState(42)
MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=rng)
Stationary(atoms)  # remove center-of-mass momentum

dyn = Langevin(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=300,
    friction=0.01 / units.fs,
)
trajectory = Trajectory("cu-nvt.traj", "w", atoms)
logger = MDLogger(dyn, atoms, "cu-nvt.log", header=True, mode="w")
dyn.attach(trajectory.write, interval=10)
dyn.attach(logger, interval=10)
dyn.run(1000)
trajectory.close()
```

`turbo` inference is a good fit for MD because composition, task, charge, and
spin normally remain fixed throughout a trajectory. It enables TF32 in
addition to the compiled fast path, so use the default inference mode when the
small precision trade-off is not appropriate.

## Choose an ensemble

`MDRunner` provides the following ASE dynamics adapters:

| Ensemble | Adapter | Important parameters |
| --- | --- | --- |
| NVE | `VelocityVerletThermostat` | Initial velocities and `timestep_fs` |
| NVT | `NoseHooverNVT` | `temperature_K`, `tdamp_fs` |
| NVT | `BussiThermostat` | `temperature_K`, `taut_fs` |
| NVT | `LangevinThermostat` | `temperature_K`, `friction_per_fs` |
| NPT | `BerendsenNPT` | Temperature, pressure, damping, and compressibility |

NPT additionally requires a fully periodic system with a nonzero cell and a
calculator that implements stress. UMA's `omat` task supplies stress. Do not
assume that every UMA task or third-party ASE calculator does so.

Berendsen coupling is useful for bringing a system toward a target temperature
and pressure, but it does not generate the exact NPT ensemble. Use an integrator
appropriate to the property being measured for production sampling.

## Run with Hydra

The repository contains two runnable configurations:

- [`configs/uma/md/nvt.yaml`](https://github.com/facebookresearch/fairchem/blob/main/configs/uma/md/nvt.yaml)
  uses Langevin NVT at 300 K.
- [`configs/uma/md/npt.yaml`](https://github.com/facebookresearch/fairchem/blob/main/configs/uma/md/npt.yaml)
  uses Berendsen NPT at 300 K and 1 bar.

Run either configuration from the repository root:

```bash
fairchem -c configs/uma/md/nvt.yaml
fairchem -c configs/uma/md/npt.yaml
```

Hydra overrides let you reuse the configuration without editing it:

```bash
fairchem -c configs/uma/md/nvt.yaml \
  temperature_K=500 runner.steps=10000 runner.trajectory_interval=100
```

The important pieces of each configuration are:

```yaml
runner:
  _target_: fairchem.core.components.calculate.MDRunner
  calculator:
    _target_: fairchem.core.FAIRChemCalculator.from_model_checkpoint
    name_or_path: uma-s-1p2p1
    task_name: omat
    inference_settings: turbo
    device: cuda
    workers: 1
  atoms:
    _target_: fairchem.core.datasets.common_structures.get_fcc_crystal_by_num_cells
    n_cells: 2
  velocity_seed: 42
  initialization_temperature_K: 300.0
```

`velocity_seed` and `initialization_temperature_K` must be specified together.
They initialize Maxwell-Boltzmann momenta for a new run and remove net linear
momentum by default. A resumed run restores its saved momenta and does not
initialize them again.

For your own structure, replace the `atoms` block with any Hydra-instantiable
ASE structure factory. For example:

```yaml
atoms:
  _target_: ase.io.read
  filename: /path/to/structure.extxyz
```

For molecular tasks, set `task_name: omol` and ensure total charge and spin
multiplicity are present as `atoms.info["charge"]` and `atoms.info["spin"]`.

## Outputs, checkpoints, and resume

Each CLI invocation creates a timestamped directory below `job.run_dir`. Its
`results` directory contains:

- `init_atoms.extxyz`: the structure and initialized velocities at step zero;
- `trajectory.parquet`: positions, cell, velocities, predictions, and
  thermodynamic properties at `trajectory_interval`;
- `thermo.log`: ASE thermodynamic logging at `log_interval`; and
- `metadata.json`: run and output metadata.

`checkpoint_interval` writes a rolling checkpoint containing atoms, velocities,
thermostat state, step count, and generated `resume_config.yaml` and
`portable_config.yaml` files. Resume on the same system with:

```bash
fairchem -c /path/to/preemption_state/resume_config.yaml
```

When `heartbeat_interval` is enabled, creating a file named `STOPFAIR` alongside
the run's `checkpoints` directory requests a checkpoint and graceful stop.

## Adapt the configuration to a cluster

The example YAMLs deliberately contain no FAIR-specific paths or scheduler
settings. For a single-GPU SLURM job, supply values appropriate to your site:

```bash
fairchem -c configs/uma/md/nvt.yaml \
  job.run_dir=/shared/path/md-runs \
  job.scheduler.mode=SLURM \
  job.scheduler.num_nodes=1 \
  job.scheduler.ranks_per_node=1 \
  job.scheduler.slurm.account=my-account \
  job.scheduler.slurm.partition=gpu \
  job.scheduler.slurm.qos=normal \
  job.scheduler.slurm.mem_gb=80 \
  job.scheduler.slurm.cpus_per_task=8 \
  job.scheduler.slurm.timeout_hr=24
```

Account, partition, QoS, filesystem paths, memory, CPU count, and wall time are
site-specific. Omit optional scheduler values such as QoS when your cluster
does not use them. The submission host must have access to the environment and
Hugging Face credentials; multi-node runs also need a shared `run_dir` and
model cache.

To split one large atomic graph across multiple GPUs, install
`fairchem-core[ray]`, request a Ray-backed allocation, and make the calculator
worker count equal the total allocated GPUs. For example, on one eight-GPU
node, add these overrides:

```bash
job.scheduler.use_ray=true \
job.scheduler.num_nodes=1 \
job.scheduler.ranks_per_node=8 \
runner.calculator.workers=8
```

Do not set `ranks_per_node` greater than one without the Ray-backed runner for a
single trajectory: the regular SPMD launcher would start an independent copy of
`MDRunner` on every rank. To run many independent small simulations, use
[InferenceBatcher](inference_batcher.md). For very large or established MD
workflows, consider the [LAMMPS integration](lammps.md).

## Validate a simulation

The short examples demonstrate the interface; they are not converged
production calculations. Before using a trajectory scientifically:

- choose a timestep appropriate to the fastest motion in the system;
- equilibrate before collecting observables;
- monitor energy drift, temperature, pressure, and cell volume;
- verify that the structure remains within the model's training domain; and
- test system-size, timestep, thermostat, and sampling convergence.
