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

# LAMMPS Integration

We provide an integration with the [LAMMPS](https://www.lammps.org) Molecular Simulator through the [`fix external`](https://docs.lammps.org/fix_external.html) command. This simple integration hands control of the neighborlist (graph) generation, parallelism, energy, force, and stress calculations all to UMA.

:::{danger} Security Warning
**Never run YAML configuration files from untrusted sources.** FAIRChem uses [Hydra](https://hydra.cc/) to instantiate Python objects from YAML configs via the `_target_` key. A maliciously crafted config file can execute arbitrary code on your machine. Only use configs that you have written yourself or that come from trusted sources. This is analogous to the security risks of Python's `pickle` and `torch.load()`.
:::

:::{tip}
The main advantage is that we can optimize UMA for distributed parallel inference directly without modifying LAMMPS. The user would also not need to deal with building LAMMPS from source (see conda install option below) nor [Kokkos](https://docs.lammps.org/Speed_kokkos.html), which is notoriously difficult to build correctly.
:::

There is some Python overhead, but for very fast empirical force fields where Python would be a limiting factor, this is negligible at the speeds of current MLIPs (10s - 100s of ms per step). This is the same reason nearly all modern LLM inference uses Python engines. Additionally, to easily scale to multi-node parallelism regimes, we designed the architecture using a client-server interface so LAMMPS would only see the client and the server code running inference can be optimized completely independently later.

Since the `fix external` integration simply wraps the UMA predictor interface, the way inference is run is identical to using the [MLIPPredictUnit, ASE Calculator or ParallelMLIPPredictUnit for Multi-GPU inference](https://facebookresearch.github.io/fairchem/core/common_tasks/ase_calculator.html).

## Usage Notes

:::{warning}
Please note the following differences from regular LAMMPS workflows:
:::

- We currently only support `metal` [units](https://docs.lammps.org/units.html), i.e., energy in `eV` and forces in `eV/A`
- Users can write LAMMPS scripts in the usual way (see lammps_in_example.file)
- Users should **NOT** define other types of forces such as "pair_style", "bond_style" in their scripts. These forces will get added together with UMA forces and most likely produce false results
- UMA uses atomic numbers so we try to guess the atomic number from the provided atomic masses in your LAMMPS scripts. Just make sure you provide the right masses for your atom types - this makes it easy so that you don't need to redefine atomic element mappings with LAMMPS

:::{note}
This assumption fails if you use isotopes or non-standard atomic masses, but we don't expect our models to work in those cases anyway.
:::

## Install and Run

Users can install LAMMPS however they like, but the simplest is to install via conda ([https://docs.lammps.org/Install_conda.html](https://docs.lammps.org/Install_conda.html)) if you don't need any bells and whistles.

For conda install, activate the conda env with LAMMPS and install fairchem into it. For manual LAMMPS installs, you need to provide python paths so LAMMPS can find fairchem.

:::{note}
We separate the LAMMPS integration code into a standalone package (`fairchem-lammps`). Please note fairchem-lammps uses the GnuV2 License as is required by any code that uses LAMMPS, instead of the MIT License used by the FAIRChem repository.
:::

```bash
# first install conda and lammps following the instructions above, ie: conda install lammps
# then activate the environment and install fairchem
conda activate lammps-env
pip install fairchem-core[extras,ray]
pip install fairchem-lammps
```

Assuming you have a classic LAMMPS .in script, make the following changes:

1. Remove all other forces from your LAMMPS script (e.g., pair_style, etc.)
2. Make sure the units are in "metal"
3. Make sure there is only 1 run command at the bottom of the script, if you have multiple run segments, ie: NVT followed by NPT, you can separate them into separate scripts

To run, use the Python entrypoint `lmp_fc` (shortcut name for the [python lammps_fc.py script](https://github.com/facebookresearch/fairchem/pull/1454)):

```bash
lmp_fc lmp_in="lammps_in_example.file" task_name="omol"
```

## Pre-flight validation

Run a pre-flight check before committing GPU time to a production trajectory.
Use the same structure, UMA task, charge, and spin as the intended simulation:

```bash
lmp_fc_preflight mode=check \
  structure=/absolute/path/system.extxyz \
  task=omol charge=0 spin=1 \
  model=uma-s-1p1 \
  output=/absolute/path/preflight.json
```

The structure can be any single-frame, fully periodic format readable by ASE.
The check converts it to a controlled atomic LAMMPS NVE calculation. It passes
only when:

- atom types round-trip to the original elements;
- energy and forces returned by UMA are finite;
- the callback agrees with direct UMA inference on the ASE structure within
  1 meV/atom energy error, 0.005 eV/A force MAE, and 0.02 eV/A maximum force
  error by default;
- LAMMPS applies the callback energy and forces within `1e-5` eV and
  `1e-5` eV/A; and
- five 0.5 fs NVE steps finish with finite thermodynamics.

The check is an integration test, not proof that the timestep, ensemble, or
trajectory is scientifically valid. Validate those choices separately for the
target system. For installation debugging only, a generated periodic crystal
can be used with `generated_atoms=32`; do not use it to select production
settings for unrelated chemistry.

The check performs only a handful of model evaluations and is intended to
finish in minutes. The configuration benchmark is deliberately longer because
it loads and times each candidate independently.

## Select GPU settings on the target system

Performance depends on atom count, neighbor density, GPU model, available GPU
count, and trajectory length. Do not assume that turbo mode or more GPUs is
faster. Benchmark the production structure on the GPUs that will run it:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 lmp_fc_preflight mode=benchmark \
  structure=/absolute/path/system.extxyz \
  task=omol charge=0 spin=1 \
  model=uma-s-1p1 \
  expected_steps=100000 \
  output=/absolute/path/lammps-benchmark.json
```

The benchmark measures these supported decisions:

| Profile | TF32 | Compile | Activation checkpointing | Purpose |
| --- | --- | --- | --- | --- |
| `fp32_eager` | off | off | off | numerical and decision baseline |
| `tf32_eager` | on | off | off | settings shipped in the LAMMPS YAML |
| `fp32_compiled` | off | on | off | compiled FP32 execution |
| `turbo` | on | on | off | maximum normal single-system speed |
| `memory_saving` | off | off | on | fallback for systems that otherwise OOM |

On one GPU it also compares internal graph generators v2 and v3. Version 3
uses the NVIDIA Alchemi neighbor list and can be faster on one GPU. Worker-count
tests use v2, which is designed for parallel inference. Each parallel worker
occupies one visible GPU; never request more workers than visible GPUs.
By default every visible worker count is tested. Use, for example,
`worker_counts=[1,2,4]` to restrict an exploratory run; one GPU is always added
as the reference. All options come from the packaged Hydra configuration and
can be recorded in a site-specific YAML config or supplied as overrides.

The tool includes model startup and compile costs when projecting the requested
run length. It recommends additional compilation or GPUs only when the measured
projected runtime improves by at least 10%, and rejects timing results with a
coefficient of variation above 10%. The JSON report contains every measurement,
failure, numerical comparison, software/hardware version, resolved execution
backend, and the exact Hydra overrides for the selected configuration.
If the eager FP32 baseline does not fit, the first passing FP32 case becomes
the numerical reference.

A candidate must also stay within all three numerical limits relative to the
FP32 reference: `max_energy_error_meV_per_atom=1.0`,
`max_force_mae_eV_per_A=0.005`, and
`max_force_error_eV_per_A=0.02`. Energy error is normalized per atom rather
than compared to the extensive total energy. These conservative defaults are
explicit Hydra settings, so a domain-specific workflow can tighten them. Do
not loosen them without validating the effect on the target observables. The
separate `1e-5` criteria above test lossless transfer across the LAMMPS callback;
they are not the FP32-versus-accelerated-mode acceptance limits.

### Automatic GPU execution backend

The LAMMPS YAML does not set `execution_mode`. This is intentional. With
`execution_mode=None`, compatible UMA-S CUDA runs automatically select the
optimized `umas_fast_gpu` backend, including its Triton kernels. It requires
merged MoLE weights and no activation checkpointing; otherwise inference falls
back to a compatible backend. The resolved backend is recorded in the benchmark
report. Explicit backend selection is a development diagnostic and should not
normally be added to a user's LAMMPS command.

The benchmark also leaves the graph-parallel communication mode and partition
at their supported defaults. All-to-all partitions, experimental edge padding,
quaternion selection, and CPU thread tuning are advanced development knobs and
are not automatically recommended by this workflow.

### Why the LAMMPS YAML disables compile

Compilation has a substantial first-evaluation cost and may recompile when
neighbor-graph shapes change. Eager execution is therefore a safer default for
short runs and unfamiliar systems. Long simulations with stable graph shapes
can recover that cost and run faster. `expected_steps` lets the benchmark
measure that tradeoff instead of applying a universal rule.

## Multi-GPU Parallelism

Our LAMMPS integration uses graph parallelism through the multi-GPU inference
API. Multi-GPU execution has communication and Ray startup overhead, so it is
primarily useful for systems that do not fit on one GPU or for which the
pre-flight benchmark measures a meaningful speedup.

:::{note}
Multi-GPU inference requires Ray. Install it with `pip install fairchem-core[ray]`.
On clusters whose temporary path is long, set `RAY_TMPDIR` to a short node-local
path: Ray's Unix-domain socket names must fit the platform's path limit.
:::

:::{tip}
The only change required is to pass the `ParallelMLIPPredictUnit` [here](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/lammps/lammps_fc_config.yaml#L20) instead of the regular predict unit when initializing the LAMMPS fairchem script. No need to install anything new such as Kokkos or add communication code.
:::

For example:

```bash
lmp_fc lmp_in="lammps_in_example.file" task_name="omol" \
  predict_unit='${parallel_predict_unit}' \
  parallel_predict_unit.num_workers=4
```

Set `num_workers` to no more than the GPUs made visible to the process. The
public LAMMPS CI runner has one T4 GPU. It validates the LAMMPS callback on
CUDA and Ray/graph-parallel orchestration on CPU/Gloo, but it does **not**
continuously validate multi-GPU CUDA/NCCL execution. A conditional two-GPU
integration test runs when such a test host is available. Until the project has
a maintained multi-GPU runner, treat a passing pre-flight benchmark on the
target multi-GPU host—not public CI—as the required correctness and performance
signal before applying a graph-parallel recommendation.
