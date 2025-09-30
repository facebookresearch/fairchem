# Training models from scratch

This repo is uses to train large state-of-the-art graph neural networks from scratch on datasets like OC20, OMol25, or OMat24, among others. We now provide a simple CLI to handle this using your own custom datasets, but we suggest fine-tuning one of the existing checkpoints first before trying a from-scratch training.

## Fairchem training framework overview

Fairchem training framework currently is a simple SPMD (Single Program Multiple Data) paradigm training framework. It is made of several components:

1. A user cli (`fairchem`) and launcher - can run jobs locally using [torch distributed elastic](https://docs.pytorch.org/docs/stable/distributed.elastic.html) or on [Slurm](https://slurm.schedmd.com/documentation.html). More environments may be supported in the future.

2. Configuration - we strictly use [Hydra yamls](https://hydra.cc/docs/intro/) for configuration.

3. A Runner interface - the core program code that is replicated to run on all ranks. An optional Reducer is also available for evaluation jobs.
Runners are distinct user functions are that run on a single rank (ie: GPU). They describe separate high level tasks such as Train, Eval, Predict, Relaxations, MD etc. Anyone can write a new runner if its functionality is sufficiently different than the ones that already exist.

4. Trainer - we use [TorchTNT](https://docs.pytorch.org/tnt/stable/) as a light-weight training loop. This allow us to cleanly separate the dataloading loading from the training loop. TNT is Pytorch's replacement for Pytorch Lightning - which has become severely bloated and difficult to use over the years; so we opt'd for the simpler option. Units are concepts in TorchTNT that provide a basic interface for training, evaluation and predict. These replace trainers in fairchemv1. You should write a new unit when the model paradigm is significantly different, ie: Training a Multitask-MLIP is one unit, training a diffusion model should be another Unit.


## Fairchemv2

Fairchem uses a single [cli](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/_cli.py) for running jobs. It accepts a single argument, the location of the Hydra yaml. This is intentional to make sure all configuration is fully captured and avoid bloating of the command line interface. Because of the flexibility of Hydra yamls, use can still provide additional parameters and overrides using the [hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/).

The cli can launch jobs locally using [torch distributed elastic](https://docs.pytorch.org/docs/stable/distributed.elastic.html) OR on [Slurm](https://slurm.schedmd.com/documentation.html).

### Fairchemv2 config structure

A fairchem config is composed of only 2 valid top level keys: "job" (Job Config) and "runner" (Runner Config). Additionally you can add key/values that are used by the OmegaConf interpolation syntax to replace fields. Other than these, no other top-level keys are permitted.
JobConfig represents configuration parameters that describe the overall job (mostly infra parameters) such as number of nodes, log locations, loggers etc. This is a structured config and must strictly adhere to the JobConfig class.
Runner Config describe the user code. This part of config is recursively instantiated at the start of a job using hydra instantiation framework.

### Example configurations for a local run:

```
job:
  device_type: CUDA
  scheduler:
    mode: LOCAL
    ranks_per_node: 4
  run_name: local_training_run
```

Example configurations for a slurm run:

```
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
To keep our configs explict (configs should be thought of as extension of code), we prefer to use the hydra instantiation framework throughout; the config is always fully described by a corresponding python class and should never be a standalone dictionary.

```
# this is bad
# because we have no idea what where to find the code
# that uses runner or where variables x and y are actually used
runner:
	x: 5
	y: 6

# this is good
# now we know which class runner corresponds to and that x,y are
# just initializer variables of runner. If we need to check the defintion
# or understand the code, we can simply goto runner.py
runner:
  _target_: fairchem.core.componets.runner.Runner
  x: 5
  y: 6
```

### Runtime instantiation with partial functions
While we want to use static instantiation as much as possible, there will be lots of cases where certain objects require runtime inputs to create. For example, if we want to create a pytorch optimizer, we can give it all the arguments except the model parameters (because its only known at runtime).

```
optimizer:
  _target_: torch.optim.AdamW
  params: ?? # this is only known at runtime
  lr: 8e-4
  weight_decay: 1e-3
```

In this case we can use a partial function, now instead of creating an optimizer object, we create a python partial function that can then be used to instantiate the optimizer in code later

```
optimizer_fn:
  _target_: torch.optim.AdamW
  _partial_: true
  lr: 8e-4
  weight_decay: 1e-3
 # later in the runner
 optimizer = optimizer_fn(model.parameters())
```

## Training UMA

The UMA model is completely defined [here](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/core/models/uma). It is also called "escn_md" during internal development since it was based on the eSEN architecture.

Training, eval and inference are all defined in the [mlip unit](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/units/mlip_unit/mlip_unit.py).

To train a model, we need to initialize a [TrainRunner](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/components/train/train_runner.py) with a [MLIPTrainEvalUnit](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/units/mlip_unit/mlip_unit.py).

Due to the complexity of UMA and training a multi-architecture, multi-dataset, multi-task model, we leverage [config groups](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/) syntax in Hydra to organize UMA training into the [following sections](https://github.com/facebookresearch/fairchem/tree/main/configs/uma/training_release):

* backbone - selects the specific backbone architecture, ie: uma-sm, uma-md, uma-large etc.
* cluster - quickly switch settings between different slurm clusters or local env
* dataset - select the dataset to train on
* element_refs - select the element references
* tasks - select the task set, ie: for direct or conservative training

We can switch between different combinations of configs easily this way, for example:

Getting training started locally using local settings and the debug dataset

```
fairchem -c configs/uma/training_release/uma_sm_direct_pretrain.yaml  cluster=h100_local dataset=uma_debug
```

Training UMA conservative with 16 nodes on slurm

```
fairchem -c configs/uma/training_release/uma_sm_conserve_finetune.yaml  cluster=h100 job.scheduler.num_nodes=16 run_name="uma_conserve_train"
```
