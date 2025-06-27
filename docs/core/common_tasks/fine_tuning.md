# Fine-tuning

This repo provides a number of scripts to quickly fine-tune a model using a custom ASE LMDB dataset. These scripts are merely for convenience and finetuning uses the exact same tooling and infra as our standard training (See Training section). Training in the fairchem repo uses the fairchem cli tool and configs are in [Hydra yaml](https://hydra.cc/) format. Training dataset must be in the [ASE-lmdb format](https://wiki.fysik.dtu.dk/ase/ase/db/db.html#ase.db.core.connect). For UMA models, we provide a simple script to help generate ASE-lmdb datasets from a variety of input formats as such (cifs, traj, extxyz etc) as well as a finetuning yaml config that can be directly used for finetuning.

## Generating training/fine-tuning datasets
First we need to generate a dataset in the aselmdb format for finetuning. The only requirement is you need to have input files that can be read as ASE atoms object by the ase.io.read routine and that they contain energy (forces, stress) in the correct format. For concrete examples refer to this to the test at `tests/core/scripts/test_create_finetune_dataset.py`.

Run this script to create the aselmdbs as well as a set of templated yamls for finetuning

```
python src/fairchem/core/scripts/create_uma_finetune_dataset.py --train-dir <path/to/train_ases> --val-dir <path/to/val_ases> --output-dir <path/to/output_dir> --uma-task <uma-task for finetuning> --regression-task <regression-task for finetuning>
```

* The `uma-task` can be one of the uma tasks: ie: `omol`, `odac`, `oc20`, `omat`, `omc`. While UMA was trained in the multi-task fashion, we ONLY support finetuning on a single UMA task at a time. Multi-task training can become very complicated! Feel free to contact us on github if you have a special use-case for multi-task finetuning or refer to the training configs in /training_release to mimic the original UMA training configs.

* The `regression-task` can be one of e, ef, efs (energy, energy+force, energy+force+stress), depending on the data you have available in the ASE db. For example, some aperiodic DFT codes only support energy/forces and not gradients, and some very fancy codes like QMC only produce energies. Note that even if you train on just energy or energy/forces, all gradients (forces/stresses) will be computable via the model gradients.

This will generate a folder of lmdbs and the a `uma_sm_finetune_template.yaml` that you can run directly with the fairchem cli to start training.

If you want to only create the aselmdbs, you can use `src/fairchem/core/scripts/create_finetune_dataset.py` which is called by `create_uma_finetune_dataset.py`.

## Runing training
The previous step should have generated some yaml files to get you started on finetuning. You can simply run this with the `fairchem` cli. The default is configured to run locally on a 1 GPU.

```
fairchem -c <path/to/uma_sm_finetune_template.yaml>
```

## Advanced configuration
The scripts provide a simple way to get started on finetuning, but likely for your own use cases you will need to modify the parameters. The configuration uses [hydra-style yamls](https://hydra.cc/).

To modify the generated yamls, you can either edit the files directly or use [hydra override notation](https://hydra.cc/docs/advanced/override_grammar/basic/). For example, changing a few parameters is very simple to do on the command line

```
fairchem -c <path/to/uma_sm_finetune_template.yaml> epochs=2 lr=2e-4 job.run_dir=/tmp/your_dir
```

The basic yaml configuration looks like the following:

```
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


base_model_name: uma-s-1
max_neighbors: 300
epochs: 1
steps: null
batch_size: 2
lr: 4e-4

train_dataloader ...
eval_dataloader ...
runner ...
```

* `base_model_name`: refers to a model name that can be retrieved from [huggingface](https://huggingface.co/facebook/UMA). If you want to use your custom uma checkpoint. You need to provide the path directly in the runner:

```
    model:
      _target_: fairchem.core.units.mlip_unit.mlip_unit.initialize_finetuning_model
      checkpoint_location: /path/to/your/checkpoint.pt
```

* `max_neighbors`: the number of neighbors used for the equivariant SO2 convolutions. 300 is the default used in uma training but if you don't have alot of memory, 100 is usually fine to ensure smoothness of the potential (see the [ESEN paper](https://arxiv.org/abs/2502.12147)).
* `epochs`, `steps`: choose to either run for integer number of epochs or steps, only 1 can be step, the other must be null
* `batch_size`: in this configuration we use the batch sampler, you can start with choosing the largest batch size that can fit on your system without running out of memory. However, you don't want to use a batch size so large such that you complete training in very few training steps
* `lr`, `weight_decay`: these are standard learning parameters, the recommended values we use are the defaults

### Logging and Artifacts

For logging and checkpoints, all artifacts are stored in the location specified in `job.run_dir`. The visual logger we support is [Weights and Biases](https://wandb.ai/site/). Tensorboard is no longer supported. You must set up your W&B account separately and `job.debug` must be set to `False` for W&B logging to work.

### Distributed training

We support multi-gpu distributed training without additional infra and multi-node distributed training on [SLURM](https://slurm.schedmd.com/documentation.html) only.

To train with multi-gpu locally, simply set `job.scheduler.ranks_per_node=N` where N is the number of GPUs you like to train on.

To train with multi-node on an SLURM cluster, you need to change `job.scheduler.mode=SLURM` and set both `job.scheduler.ranks_per_node` and `job.scheduler.num_nodes` to the desired values. The run_dir must be in a shared network accessible mount for this to work.

### Resuming runs

To resume from a checkpoint in the middle of a run, find the checkpoint folder at the step you want and simply run:

```
fairchem -c /path/to/checkpoints/step_xxxx/resume.yaml
```

### Running inference on the finetuned model

Inference is run in the same way as the UMA models, except you need to load the checkpoint from a local path. You must also use the same task that you used for finetuning:

```
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator

predictor = load_predict_unit("/path/to/inference_ckpt.pt")
calc = FAIRChemCalculator(predictor, task_name=task)
```
