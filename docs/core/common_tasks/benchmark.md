## Running model benchmarks

Model benchmarks involve evaluating a model on downstream property predictions involving several model evaluations to calculate a single or set of related properties. For example calculating structure relaxations, elastic tensors, phonons, or adsportion energy.

To benchmark UMA models on standard datasets, you can find benchmark configuration files in `configs/uma/benchmark`. Example files include:
- `adsorbml.yaml`
- `hea-is2re.yaml`
- `kappa103.yaml`
- `matbench-discovery-discovery.yaml`
- `mdr-phonon.yaml`

Note that to run these UMA benchmarks you will need to obtain the target data.

1. **Run the Benchmark Script**  
   Use the same runner script, specifying the benchmark config:
   ```bash
   fairchem --config configs/uma/benchmark/benchmark.yaml
   ```
   Replace `benchmark.yaml` with the desired benchmark config file.

2. **Output**  
   Benchmark results will are saved to a *results* directory under the *run_dir* specified in the configuration file. Additionally benchmark metrics are logged using the specified logger. We currently only support Weights and Biases.

## Benchmark Configuration File Format

Evaluation configuration files are written in Hydra YAML format and specify how a model evaluation should be run. UMA evaluation configuration files, which can be used as templates to evaluate other models if needed, are located in `configs/uma/evaluate/`.

### Top-Level Keys

The benchmark configuration files follow the same format as model training and evaluation configuration files, with the addition of a **reducer** flag to specify how final metrics are calculated from the results of a given benchmark calculation protocol.

A benchmark configuration files should define the following top level keys:

- **job**: Contains all settings related to the evaluation job itself, including model, data, and logger configuration. For additional details see the description given in the Evaluation page.
- **runner**: Contains settings for a `CalculateRunner` which implements a downstream property calculation or simulation.
- **reducer**: Contains the settings for a `BenchmarkReducer` class which defines how to aggregate the results of calculated by the `CalculateRunner` and computes metrics based on given target values.

#### `CalculateRunner`s:
The benchmark details including the type of calculations and the model checkpoint are specified under the runner flag. The specific benchmark calculations are based on the chosen `CalculateRunner` (for example a `RelaxationRunner`). Several `CalculateRunner` implementations are found in the `fairchem.core.components.calculate` submodule.

### Implementing new calculations in a `CalculateRunner`
It is straightforward to write your own calculations in a `CalculateRunner`. Although implementation is very flexible and open ended, we suggest that you have a look at the interface set up by the `CalculateRunner` base class. At a minimum you will need to implement the following methods:

```python
    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> R:
      """Implement your calculations here by iterating over the self.input_data attribute"""

    def write_results(
        self, results: R, results_dir: str, job_num: int = 0, num_jobs: int = 1
    ) -> None:
      """Write the results returned by your calculations in the method above"""
```

You will also see a `save_state` and `load_state` abstract methods that you can use to checkpoint calculations, however in most cases if calculations are fast enough you wont need this and you can simply implement those as empty methods.


#### `BenchmarkReducer`s:
A `CalculateRunner` will run calculations over a given set of structures and write out results. In order to compute benchmark metrics, a `BenchmarkReducer` is used to aggregate all these results, compute metrics and report them. Implementations of `BenchmarkReducer` classes are found in the `fairchem.core.components.benchmark` submodule

### Implenting metrics in a `BenchmarkReducer`

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

If it makes sense for your benchmark metrics and are happy working with dictionaries and pandas `DataFrames`, a lot of boilerplate code is implemented in the `JsonDFReducer`. We recommend that you start there by deriving your class from it, and focusing only on implementing the `compute_metrics` method.
