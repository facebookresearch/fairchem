# Training Models from Scratch

This repo is used to train large state-of-the-art graph neural networks from scratch on datasets like OC20, OMol25, or OMat24, among others.

:::{tip}
We now provide a simple CLI to handle this using your own custom datasets, but we suggest fine-tuning one of the existing checkpoints first before trying a from-scratch training.
:::

## FAIRChem Training Framework Overview

The FAIRChem training framework currently uses a simple SPMD (Single Program Multiple Data) paradigm. It is made of several components:

1. **User CLI and Launcher** - The `fairchem` CLI can run jobs locally using [torch distributed elastic](https://docs.pytorch.org/docs/stable/distributed.elastic.html) or on [SLURM](https://slurm.schedmd.com/documentation.html). More environments may be supported in the future.

2. **Configuration** - We strictly use [Hydra YAMLs](https://hydra.cc/docs/intro/) for configuration.

3. **Runner Interface** - The core program code that is replicated to run on all ranks. An optional Reducer is also available for evaluation jobs. Runners are distinct user functions that run on a single rank (i.e., GPU). They describe separate high-level tasks such as Train, Eval, Predict, Relaxations, MD, etc. Anyone can write a new runner if its functionality is sufficiently different than the ones that already exist.

4. **Trainer** - We use [TorchTNT](https://docs.pytorch.org/tnt/stable/) as a light-weight training loop. This allows us to cleanly separate the data loading from the training loop.

:::{note}
TNT is PyTorch's replacement for PyTorch Lightning - which has become severely bloated and difficult to use over the years; so we opted for the simpler option. Units are concepts in TorchTNT that provide a basic interface for training, evaluation, and prediction. These replace trainers in fairchemv1. You should write a new unit when the model paradigm is significantly different, e.g., training a Multitask-MLIP is one unit, training a diffusion model should be another unit.
:::


## FAIRChem v2 CLI

FAIRChem uses a single [CLI](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/_cli.py) for running jobs. It accepts a single argument, the location of the Hydra YAML.

:::{warning}
This is intentional to make sure all configuration is fully captured and avoid bloating of the command line interface. Because of the flexibility of Hydra YAMLs, you can still provide additional parameters and overrides using the [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/).
:::

The CLI can launch jobs locally using [torch distributed elastic](https://docs.pytorch.org/docs/stable/distributed.elastic.html) OR on [SLURM](https://slurm.schedmd.com/documentation.html).

### FAIRChem v2 Config Structure

A FAIRChem config is composed of only 2 valid top-level keys: **job** (Job Config) and **runner** (Runner Config). Additionally, you can add key/values that are used by the OmegaConf interpolation syntax to replace fields. Other than these, no other top-level keys are permitted.

- **JobConfig** represents configuration parameters that describe the overall job (mostly infra parameters) such as number of nodes, log locations, loggers, etc. This is a structured config and must strictly adhere to the JobConfig class.
- **Runner Config** describes the user code. This part of config is recursively instantiated at the start of a job using Hydra instantiation framework.

### Example Configurations

**Local run:**

```yaml
job:
  device_type: CUDA
  scheduler:
    mode: LOCAL
    ranks_per_node: 4
  run_name: local_training_run
```

**SLURM run:**

```yaml
job:
  device_type: CUDA
  scheduler:
    mode: SLURM
    ranks_per_node: 8
    num_nodes: 4
    slurm:
      account: ${cluster.account}
      qos: ${cluster.qos}
      mem_gb: ${cluster.mem_gb}
      cpus_per_task: ${cluster.cpus_per_task}
  run_dir: /path/to/output
  run_name: slurm_run_example
```

### Config Object Instantiation

To keep our configs explicit (configs should be thought of as an extension of code), we prefer to use the Hydra instantiation framework throughout; the config is always fully described by a corresponding Python class and should never be a standalone dictionary.

:::{admonition} Good vs Bad Config Patterns
:class: dropdown

**Bad pattern** - We have no idea where to find the code that uses runner or where variables x and y are actually used:

```yaml
runner:
  x: 5
  y: 6
```

**Good pattern** - Now we know which class runner corresponds to and that x, y are just initializer variables of runner. If we need to check the definition or understand the code, we can simply go to runner.py:

```yaml
runner:
  _target_: fairchem.core.components.runner.Runner
  x: 5
  y: 6
```
:::

### Runtime Instantiation with Partial Functions

While we want to use static instantiation as much as possible, there will be many cases where certain objects require runtime inputs to create. For example, if we want to create a PyTorch optimizer, we can give it all the arguments except the model parameters (because it's only known at runtime).

```yaml
optimizer:
  _target_: torch.optim.AdamW
  params: ?? # this is only known at runtime
  lr: 8e-4
  weight_decay: 1e-3
```

:::{tip}
In this case we can use a partial function. Instead of creating an optimizer object, we create a Python partial function that can then be used to instantiate the optimizer in code later:
:::

```yaml
optimizer_fn:
  _target_: torch.optim.AdamW
  _partial_: true
  lr: 8e-4
  weight_decay: 1e-3
```

```python
# later in the runner
optimizer = optimizer_fn(model.parameters())
```

## Training UMA

The UMA model is completely defined [here](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/core/models/uma). It is also called "escn_md" during internal development since it was based on the eSEN architecture.

Training, evaluation, and inference are all defined in the [mlip unit](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/units/mlip_unit/mlip_unit.py).

To train a model, we need to initialize a [TrainRunner](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/components/train/train_runner.py) with a [MLIPTrainEvalUnit](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/units/mlip_unit/mlip_unit.py).

Due to the complexity of UMA and training a multi-architecture, multi-dataset, multi-task model, we leverage [config groups](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/) syntax in Hydra to organize UMA training into the [following sections](https://github.com/facebookresearch/fairchem/tree/main/configs/uma/training_release):

- **backbone** - selects the specific backbone architecture (e.g., uma-sm, uma-md, uma-large)
- **cluster** - quickly switch settings between different SLURM clusters or local environment
- **dataset** - select the dataset to train on
- **element_refs** - select the element references
- **tasks** - select the task set (e.g., for direct or conservative training)

:::{tip}
We can switch between different combinations of configs easily this way!
:::

### Example Commands

**Get training started locally using local settings and the debug dataset:**

```bash
fairchem -c configs/uma/training_release/uma_sm_direct_pretrain.yaml cluster=h100_local dataset=uma_debug
```

**Train UMA conservative with 16 nodes on SLURM:**

```bash
fairchem -c configs/uma/training_release/uma_sm_conserve_finetune.yaml cluster=h100 job.scheduler.num_nodes=16 run_name="uma_conserve_train"
```
