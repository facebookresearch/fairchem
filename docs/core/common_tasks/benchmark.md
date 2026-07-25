# Running Model Benchmarks

Model benchmarks involve evaluating a model on downstream property predictions involving several model evaluations to calculate a single or set of related properties. For example, calculating structure relaxations, elastic tensors, phonons, or adsorption energy.

:::{danger} Security Warning
**Never run YAML configuration files from untrusted sources.** FAIRChem uses [Hydra](https://hydra.cc/) to instantiate Python objects from YAML configs via the `_target_` key. A maliciously crafted config file can execute arbitrary code on your machine. Only use configs that you have written yourself or that come from trusted sources. This is analogous to the security risks of Python's `pickle` and `torch.load()`.
:::

## Available Benchmark Configurations

To benchmark UMA models on standard datasets, you can find benchmark configuration files in `configs/uma/benchmark`. Example files include:
- `adsorbml.yaml`
- `hea-is2re.yaml`
- `kappa103.yaml`
- `matbench-discovery-discovery.yaml`
- `mdr-phonon.yaml`

:::{note}
To run these UMA benchmarks you will need to obtain the target data.
:::

## Running Benchmarks

Run the benchmark script using the fairchem CLI, specifying the benchmark config:

```bash
fairchem --config configs/uma/benchmark/benchmark.yaml
```

Replace `benchmark.yaml` with the desired benchmark config file.

:::{tip}
Benchmark results are saved to a **results** directory under the **run_dir** specified in the configuration file. Additionally, benchmark metrics are logged using the specified logger. We currently only support Weights and Biases.
:::

## Benchmark Configuration File Format

Evaluation configuration files are written in Hydra YAML format and specify how a model evaluation should be run. UMA evaluation configuration files, which can be used as templates to evaluate other models if needed, are located in `configs/uma/evaluate/`.

### Top-Level Keys

The benchmark configuration files follow the same format as model training and evaluation configuration files, with the addition of a **reducer** flag to specify how final metrics are calculated from the results of a given benchmark calculation protocol.

A benchmark configuration file should define the following top level keys:

- **job**: Contains all settings related to the evaluation job itself, including model, data, and logger configuration. For additional details see the description given in the Evaluation page.
- **runner**: Contains settings for a `CalculateRunner` which implements a downstream property calculation or simulation.
- **reducer**: Contains the settings for a `BenchmarkReducer` class which defines how to aggregate the results calculated by the `CalculateRunner` and computes metrics based on given target values.

### CalculateRunners

The benchmark details including the type of calculations and the model checkpoint are specified under the runner flag. The specific benchmark calculations are based on the chosen `CalculateRunner` (for example a `RelaxationRunner`). Several `CalculateRunner` implementations are found in the `fairchem.core.components.calculate` submodule.

:::{admonition} Implementing New Calculations
:class: dropdown

It is straightforward to write your own calculations in a `CalculateRunner`. Although implementation is very flexible and open ended, we suggest that you have a look at the interface set up by the `CalculateRunner` base class. At a minimum you will need to implement the following methods:

```python
def calculate(self, job_num: int = 0, num_jobs: int = 1) -> R:
    """Implement your calculations here by iterating over the self.input_data attribute"""


def write_results(
    self, results: R, results_dir: str, job_num: int = 0, num_jobs: int = 1
) -> None:
    """Write the results returned by your calculations in the method above"""
```

You will also see `save_state` and `load_state` abstract methods that you can use to checkpoint calculations. However, in most cases if calculations are fast enough you will not need these and you can simply implement them as empty methods.
:::


### BenchmarkReducers

A `CalculateRunner` will run calculations over a given set of structures and write out results. In order to compute benchmark metrics, a `BenchmarkReducer` is used to aggregate all these results, compute metrics and report them. Implementations of `BenchmarkReducer` classes are found in the `fairchem.core.components.benchmark` submodule.

:::{admonition} Implementing Custom Metrics
:class: dropdown

If you want to implement your own benchmark metric calculation you can write a `BenchmarkReducer` class. At a minimum, you will need to implement the following methods:

```python
def join_results(self, results_dir: str, glob_pattern: str) -> R:
    """Join your results from multiple files into a single result object."""


def save_results(self, results: R, results_dir: str) -> None:
    """Save joined results to a single file"""


def compute_metrics(self, results: R, run_name: str) -> M:
    """Compute metrics using the joined results and target data in your BenchmarkReducer."""


def save_metrics(self, metrics: M, results_dir: str) -> None:
    """Save the computed metrics to a file."""


def log_metrics(self, metrics: M, run_name: str):
    """Log metrics to the configured logger."""
```

:::{tip}
If it makes sense for your benchmark metrics and are happy working with dictionaries and pandas `DataFrames`, a lot of boilerplate code is implemented in the `JsonDFReducer`. We recommend that you start there by deriving your class from it, and focusing only on implementing the `compute_metrics` method.
:::
:::

## Inference & Training Perf Check

The perf check (`configs/uma/benchmark/perf_check/`) provides lightweight benchmarks for validating inference and training fidelity and performance. Unlike the model benchmarks above, these do **not** require real datasets — they use built-in test systems or auto-generated fake data, making them ideal for quick sanity checks after configuration changes.

### Inference Benchmark

The inference benchmark compares a candidate inference configuration against a high-precision **fp64 baseline** to measure numerical accuracy and throughput.

**What it measures:**

- **Accuracy**: energy absolute error, force MAE, force max error, stress MAE and max error (per system)
- **Performance**: queries per second (QPS), peak GPU memory, wall-clock time

**How it works:**

1. Loads a pretrained checkpoint (default: `uma-s-1p2`)
2. Runs inference in fp64 with conservative settings to establish a baseline (cached for reuse)
3. Runs inference with the candidate settings you provide
4. Compares predictions and reports error metrics in a formatted table

Three built-in test systems are used:

| System | Description | Task |
|--------|-------------|------|
| `small_molecule` | Water box with 20 molecules | `omol` |
| `medium_bulk` | FCC crystal with 200 atoms | `omat` |
| `large_bulk` | FCC crystal with 1000 atoms | `omat` |

**Usage:**

```bash
# Run with default settings
fairchem -c configs/uma/benchmark/perf_check/benchmark.yaml

# Test a specific execution mode
fairchem -c configs/uma/benchmark/perf_check/benchmark.yaml \
  runner.inference_settings.execution_mode=umas_fast_gpu

# Enable TF32 and torch.compile
fairchem -c configs/uma/benchmark/perf_check/benchmark.yaml \
  runner.inference_settings.tf32=True runner.inference_settings.compile=True

# Run on CPU
fairchem -c configs/uma/benchmark/perf_check/benchmark.yaml runner.device=cpu
```

**Available overrides:**

| Override | Default | Description |
|----------|---------|-------------|
| `runner.device` | `cuda` | Device to run on (`cuda` or `cpu`) |
| `runner.warmup_iters` | `10` | Number of warmup iterations (excluded from timing) |
| `runner.timed_iters` | `50` | Number of timed iterations for throughput measurement |
| `runner.inference_settings.tf32` | `False` | Enable TF32 tensor cores |
| `runner.inference_settings.compile` | `False` | Enable `torch.compile` |
| `runner.inference_settings.activation_checkpointing` | `True` | Enable activation checkpointing |
| `runner.inference_settings.merge_mole` | `False` | Merge MOLE experts |
| `runner.inference_settings.execution_mode` | `general` | Inference execution mode |
| `checkpoint.model_name` | `uma-s-1p2` | Pretrained model to benchmark |

### Training Benchmark

The training benchmark compares an **fp32 baseline** against a candidate training configuration (typically bf16) to measure numerical fidelity and training throughput.

**What it measures:**

- **Fidelity**: loss absolute/relative error, gradient norm relative error (measured at step 0 before optimizer updates)
- **Convergence**: baseline and candidate final-step losses
- **Performance**: training steps per second, peak GPU memory

**How it works:**

1. Auto-generates fake datasets for each task (oc20, omol, omat, odac, omc) using Lennard-Jones energies and repulsive forces, normalized to realistic per-task statistics. Generated data is cached for reuse.
2. Trains for a configurable number of steps in fp32 to establish a baseline (cached for reuse)
3. Trains with the candidate settings and compares first-step loss and gradient norms
4. Reports fidelity metrics and throughput in a formatted table

**Usage:**

```bash
# Run with default settings
fairchem -c configs/uma/benchmark/perf_check/training_benchmark.yaml

# Enable bf16 candidate
fairchem -c configs/uma/benchmark/perf_check/training_benchmark.yaml \
  runner.bf16=True

# Increase throughput measurement steps
fairchem -c configs/uma/benchmark/perf_check/training_benchmark.yaml \
  runner.throughput_steps=20

# Run on CPU
fairchem -c configs/uma/benchmark/perf_check/training_benchmark.yaml \
  runner.device=cpu
```

**Available overrides:**

| Override | Default | Description |
|----------|---------|-------------|
| `runner.device` | `cuda` | Device to run on (`cuda` or `cpu`) |
| `runner.bf16` | `False` | Enable bf16 mixed precision for the candidate run |
| `runner.throughput_steps` | `50` | Number of training steps for throughput measurement |
| `runner.seed` | `42` | Random seed for reproducibility |
| `runner.training_config` | `configs/uma/benchmark/perf_check/training_inner.yaml` | Inner training config path |
