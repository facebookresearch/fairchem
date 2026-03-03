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

# Fine-tuning

This repo provides a number of scripts to quickly fine-tune a model using a custom ASE LMDB dataset. These scripts are merely for convenience and fine-tuning uses the exact same tooling and infrastructure as our standard training (see Training section). Training in the fairchem repo uses the fairchem CLI tool and configs are in [Hydra yaml](https://hydra.cc/) format.

:::{note}
Training datasets must be in the [ASE-lmdb format](https://wiki.fysik.dtu.dk/ase/ase/db/db.html#ase.db.core.connect). For UMA models, we provide a simple script to help generate ASE-lmdb datasets from a variety of input formats (CIFs, traj, extxyz, etc.) as well as a fine-tuning YAML config that can be directly used for fine-tuning.
:::

## Generating Training/Fine-tuning Datasets

First we need to generate a dataset in the aselmdb format for fine-tuning.

:::{tip}
The only requirement is that you have input files that can be read as ASE atoms objects by the `ase.io.read` routine and that they contain energy (forces, stress) in the correct format. For concrete examples, refer to the test at `tests/core/scripts/test_create_finetune_dataset.py`.
:::

First you should checkout the fairchem repo and install it to access the scripts

```{code-cell} ipython3
:tags: [skip-execution]
git clone git@github.com:facebookresearch/fairchem.git

pip install -e fairchem/src/packages/fairchem-core[dev]
```

Run this script to create the aselmdbs as well as a set of templated yamls for finetuning, we will use a few dummy structures for demonstration purposes
```{code-cell} ipython3
import os
os.chdir('../../../../fairchem')
! python src/fairchem/core/scripts/create_uma_finetune_dataset.py --train-dir docs/core/common_tasks/finetune_assets/train/ --val-dir docs/core/common_tasks/finetune_assets/val --output-dir /tmp/bulk --uma-task=omat --regression-task e
```

:::{warning}
**Task Selection:** The `uma-task` can be one of: `omol`, `odac`, `oc20`, `omat`, `omc`. While UMA was trained in a multi-task fashion, we ONLY support fine-tuning on a single UMA task at a time. Multi-task training can become very complicated! Feel free to contact us on GitHub if you have a special use-case for multi-task fine-tuning, or refer to the training configs in `/training_release` to mimic the original UMA training configs.
:::

:::{admonition} Regression Task Options
:class: dropdown

The `regression-task` can be one of:
- **e**: Energy only
- **ef**: Energy + forces
- **efs**: Energy + forces + stress

Choose based on the data you have available in the ASE db. For example, some aperiodic DFT codes only support energy/forces and not gradients, and some very fancy codes like QMC only produce energies.

**Note:** Even if you train on just energy or energy/forces, all gradients (forces/stresses) will be computable via the model gradients.
:::

This will generate a folder of LMDBs and a `uma_sm_finetune_template.yaml` that you can run directly with the fairchem CLI to start training.

:::{tip}
If you want to only create the ASE LMDBs, you can use `src/fairchem/core/scripts/create_finetune_dataset.py` which is called by `create_uma_finetune_dataset.py`.
:::

## Model Fine-tuning (Default Settings)

The previous step should have generated some YAML files to get you started on fine-tuning. You can simply run this with the `fairchem` CLI. The default is configured to run locally on 1 GPU.

```{code-cell} ipython3
:tags: [skip-execution]
! fairchem -c /tmp/bulk/uma_sm_finetune_template.yaml
```

## Advanced Configuration

The scripts provide a simple way to get started on fine-tuning, but likely for your own use cases you will need to modify the parameters. The configuration uses [Hydra-style YAMLs](https://hydra.cc/).

:::{tip}
To modify the generated YAMLs, you can either edit the files directly or use [Hydra override notation](https://hydra.cc/docs/advanced/override_grammar/basic/). Changing parameters on the command line is very simple:
:::

```{code-cell} ipython3
! fairchem -c /tmp/bulk/uma_sm_finetune_template.yaml epochs=2 lr=2e-4 job.run_dir=/tmp/finetune_dir +job.timestamp_id=some_id
```

The basic YAML configuration looks like the following:

```yaml
job:
  device_type: CUDA
  scheduler:
    mode: LOCAL
    ranks_per_node: 1
    num_nodes: 1
  debug: True
  run_dir: /tmp/uma_finetune_runs/
  run_name: uma_finetune
  logger:
    _target_: fairchem.core.common.logger.WandBSingletonLogger.init_wandb
    _partial_: true
    entity: example
    project: uma_finetune


base_model_name: uma-s-1p1
max_neighbors: 300
epochs: 1
steps: null
batch_size: 2
lr: 4e-4

train_dataloader ...
eval_dataloader ...
runner ...
```

:::{admonition} Configuration Parameters
:class: dropdown

- **base_model_name**: Refers to a model name that can be retrieved from [HuggingFace](https://huggingface.co/facebook/UMA). If you want to use your custom UMA checkpoint, provide the path directly in the runner:

  ```yaml
  model:
    _target_: fairchem.core.units.mlip_unit.mlip_unit.initialize_finetuning_model
    checkpoint_location: /path/to/your/checkpoint.pt
  ```

- **max_neighbors**: The number of neighbors used for the equivariant SO2 convolutions. 300 is the default used in UMA training, but if you don't have a lot of memory, 100 is usually fine to ensure smoothness of the potential (see the [ESEN paper](https://arxiv.org/abs/2502.12147)).

- **epochs**, **steps**: Choose to either run for an integer number of epochs or steps. Only 1 can be specified; the other must be null.

- **batch_size**: In this configuration we use the batch sampler. Start with the largest batch size that can fit on your system without running out of memory. However, don't use a batch size so large that you complete training in very few steps. The optimal batch size is usually the one that minimizes the final validation loss for a fixed compute budget.

- **lr**, **weight_decay**: These are standard learning parameters. The recommended values we use are the defaults.
:::

### Logging and Artifacts

For logging and checkpoints, all artifacts are stored in the location specified in `job.run_dir`. The visual logger we support is [Weights and Biases](https://wandb.ai/site/).

:::{warning}
Tensorboard is no longer supported. You must set up your W&B account separately and `job.debug` must be set to `False` for W&B logging to work.
:::

### Distributed Training

We support multi-GPU distributed training without additional infrastructure and multi-node distributed training on [SLURM](https://slurm.schedmd.com/documentation.html) only.

**Multi-GPU locally:** Simply set `job.scheduler.ranks_per_node=N` where N is the number of GPUs you want to train on.

**Multi-node on SLURM:** Change `job.scheduler.mode=SLURM` and set both `job.scheduler.ranks_per_node` and `job.scheduler.num_nodes` to the desired values.

:::{note}
The `run_dir` must be in a shared network accessible mount for multi-node training to work.
:::

### Resuming Runs

To resume from a checkpoint in the middle of a run, find the checkpoint folder at the step you want and use the same fairchem command:

```{code-cell} ipython3
:tags: [skip-execution]
! fairchem -c /tmp/finetune_dir/some_id/checkpoints/final/resume.yaml
```

### Running Inference on the Fine-tuned Model

Inference is run in the same way as the UMA models, except you need to load the checkpoint from a local path.

:::{warning}
You must use the same task that you used for fine-tuning!
:::

```{code-cell} ipython3
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator

predictor = load_predict_unit("/tmp/finetune_dir/some_id/checkpoints/final/inference_ckpt.pt")
calc = FAIRChemCalculator(predictor, task_name="omat")
```
