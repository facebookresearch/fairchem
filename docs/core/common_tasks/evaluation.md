# Evaluating pretrained models

`fairchemV2` provides a number of methods used to benchmark and evaluate the UMA models that will be helpful for apples-to-apples comparisons with the paper results. More details to be provided here soon. 

## Running Model Evaluations

To evaluate a UMA model using a pre-existing configuration file, follow these steps. Example configuration files used to evaluate uma models are stored in `configs/uma/evaluate`.

1. **Run the Evaluation Script**  
   To run an evaluation simply run:
   ```bash
   fairchem --config evaluation_config.yaml
   ```
   Replace `evaluation_config.yaml` with the desired config file. For example, `configs/uma/evaluate/uma_conserving.yaml`

1. **Output**  
   Results will be logged according the specified logger. We currently only support Weights and Biases.

## Evaluation Configuration File Format

Evaluation configuration files are written in Hydra YAML format and specify how a model evaluation should be run. UMA evaluation configuration files, which can be used as templates to evaluate other models if needed, are located in `configs/uma/evaluate/`.

### Top-Level Keys

Similar to training configuration files, the only allowed top-level keys are the `job` and `runner` keys as well interpolation keys that are resolved at runtime.

- **job**: Contains all settings related to the evaluation job itself, including model, data, and logger configuration.
- **runner**: Contains settings for the evaluation runner, such as which script to use and runtime options.

Important configuration options are nested under these keys as follows:

#### Under `job`:
Specifications of how to run the actual job. The configuration options are the same here as those in a training job. Some notable flags are detailed below,
- `device_type`: The device to run model inference on (ie CUDA or CPU)
- `scheduler`: The compute scheduler specifications
- `logger`: Configuration for logging results.
  - `type`: Logger type (e.g., `wandb`).
  - `project`: Logging project name.
  - `entity`: (Optional) Logger entity/user.
- `run_dir`: Directory where results and logs will be saved.

#### Under `runner`:
The actual benchmark details such as model checkpoint and the dataset are specified under the runner flag. An evaluation run should use the `EvalRunner` class which relies on an `MLIPEvalUnit` to run inference using a pretrained model.

- `dataloader`: Dataloader specification for the evaluation dataset.
- `eval_unit`: The specification of the `MLIPEvalUnit` to be used.
  - `tasks`: The prediction task configuration. In almost all cases you can think of, these should be loaded from a model checkpoint using the `fairchem.core.units.mlip_unit.utils.load_tasks` function.
  - `model`: Defines how to load a pretrained model. We recommend using the `fairchem.core.units.mlip_unit.mlip_unit.load_inference_model` function to do so.


### Using the `defaults` key to define config groups

The `defaults` key is a Hydra feature that allows you to compose configuration files from modular config groups. Each entry under `defaults` refers to a config group (such as `model`, `data`, or other reusable components) that is merged into the final configuration at runtime. This makes it easy to swap out models, datasets, or other settings without duplicating configuration code.

For example in the UMA evaluation configs we have set up the following config groups and defaults:
```yaml
defaults:
  - _self_
  - model: omc_conserving
  - data: my_eval_data
```
This will include the configuration from `configs/uma/evaluate/model/omc_conserving.yaml` and `configs/uma/evaluate/data/my_eval_data.yaml` into the main config. The `_self_` entry ensures the current file's contents are included.

You can create new config groups or override existing ones by changing the entries under `defaults`.

```yaml
defaults:
  - cluster: Configuration settings for a particular compute cluster
  - dataset: Configuration settings for the evaluation dataset
  - checkpoint: Configuration settings of the pretrained model checkpoint 
  - _self_
```

Using config groups allows to easily override defaults in the cli. For example,

```bash
fairchem --config evaluation_config.yaml cluster=cluster_config checkpoint=checkpoint_config
```

Where `cluster_config` and `checkpoint_config` are cluster and checkpoint configuration files written to directories under cluster and checkpoint respectively. See the files in `configs/uma/evaluate` as a full example.