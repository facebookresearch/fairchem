"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import os
import pickle

import omegaconf
import torch

from fairchem.core.scripts.migrate_imports import mapping
from fairchem.core.units.mlip_unit import MLIPPredictUnit


def find_new_module_name(module):
    if module in mapping:
        return mapping[module]
    module_path = module.split(".")
    # check if parent resolves
    parent_module = ".".join(module_path[:-1])
    if parent_module == "":
        return module_path[0]
    else:
        return f"{find_new_module_name(parent_module)}.{module_path[-1]}"


def update_config(config_or_data):
    if isinstance(config_or_data, (omegaconf.dictconfig.DictConfig, dict)):
        for k, v in config_or_data.items():
            config_or_data[k] = update_config(v)
    elif isinstance(config_or_data, (omegaconf.listconfig.ListConfig, list)):
        for i, item in enumerate(config_or_data):
            config_or_data[i] = update_config(item)
    elif isinstance(config_or_data, str):
        for k, v in mapping.items():
            if k in config_or_data:
                config_or_data = config_or_data.replace(k, v)
        config_or_data = config_or_data.replace("osc", "omc")
    return config_or_data


class RenameUnpickler(pickle.Unpickler):
    def __init__(self, file, *args, **kwargs):
        super().__init__(file, *args, **kwargs)

    def find_class(self, module, name):
        return super().find_class(find_new_module_name(module), name)


def generate_stress_task_config(dataset_name, rmsd):
    return {
        "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
        "name": f"{dataset_name}_stress",
        "level": "system",
        "property": "stress",
        "loss_fn": {
            "_target_": "fairchem.core.modules.loss.DDPMTLoss",
            "loss_fn": {"_target_": "fairchem.core.modules.loss.MAELoss"},
            "reduction": "mean",
            "coefficient": 1,
        },
        "out_spec": {"dim": [1, 9], "dtype": "float32"},
        "normalizer": {
            "_target_": "fairchem.core.modules.normalization.normalizer.Normalizer",
            "mean": 0.0,
            "rmsd": rmsd,  # 1.423
        },
        "datasets": [dataset_name],
        "metrics": ["mae"],
    }


def migrate_checkpoint(
    checkpoint_path: torch.nn.Module,
    rm_static_keys: bool = True,
    map_undefined_stress_to: str | None = None,
    add_stress: bool = False,
    task_add_stress: str | None = None,
    model_version: float = 1.0,
) -> dict:
    """
    Migrates a checkpoint by updating module imports and configurations.

    This function loads a checkpoint, updates its configuration using the mapping
    defined in fairchem.core.scripts.migrate_imports,

    optionally adds stress tasks for datasets that don't have them,
    and optionally removes static keys that are no longer needed.

    Args:
        checkpoint_path: Path to the input checkpoint file
        rm_static_keys: Whether to remove static keys from the state dictionaries
        task_add_stress: If provided, adds stress tasks for datasets based on this task
        model_version: Inject this model version into model

    Returns:
        Migrated checkpoint dict
    """
    pickle.Unpickler = RenameUnpickler
    checkpoint = torch.load(checkpoint_path, pickle_module=pickle)
    checkpoint.tasks_config = update_config(checkpoint.tasks_config)
    checkpoint.model_config = update_config(checkpoint.model_config)

    if (
        checkpoint.model_config["backbone"]["model"]
        == "fairchem.experimental.foundation_models.models.message_passing.escn_omol.eSCNMDBackbone"
    ):
        checkpoint.model_config["backbone"]["use_dataset_embedding"] = False

    if (
        checkpoint.model_config["backbone"]["model"]
        == "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone"
    ):
        if "model_version" in checkpoint.model_config["backbone"]:
            assert checkpoint.model_config["backbone"]["model_version"] == model_version
        print("Setting model version to", model_version)
        checkpoint.model_config["backbone"]["model_version"] = model_version

    if task_add_stress is not None:
        target_stress_task = f"{task_add_stress}_stress"
        output_dataset_names = set()
        datasets_with_stress = set()
        target_stress_config = None
        # find output datasets
        for task in checkpoint.tasks_config:
            if "_energy" in task.name:
                output_dataset_names.add(task.name.replace("_energy", ""))
            elif "_stress" in task.name:
                datasets_with_stress.add(task.name.replace("_stress", ""))
                if task.name == target_stress_task:
                    target_stress_config = task
        assert (
            target_stress_config is not None
        ), f"Did not find existing task {target_stress_task} in {[task.name for task in checkpoint.tasks_config]}"
        # copy over the task configs to tasks that dont have stress
        for dataset_name in output_dataset_names - datasets_with_stress:
            task_config = target_stress_config.copy()
            task_config.name = f"{dataset_name}_stress"
            task_config.datasets = [dataset_name]
            checkpoint.tasks_config.append(task_config)

    if add_stress:
        checkpoint.model_config["backbone"]["regress_stress"] = True
        output_dataset_names = set()
        datasets_with_stress = set()
        datasets_to_rmsd = {}
        # find output datasets
        for task in checkpoint.tasks_config:
            if "_energy" in task.name:
                output_dataset_names.add(task.name.replace("_energy", ""))
                datasets_to_rmsd[task.name.replace("_energy", "")] = task["normalizer"][
                    "rmsd"
                ]
        for dataset_name in output_dataset_names - datasets_with_stress:
            checkpoint.tasks_config.append(
                generate_stress_task_config(
                    dataset_name, datasets_to_rmsd[dataset_name]
                )
            )

    # remove keys for registered buffers that are no longer saved
    if rm_static_keys:
        remove_keys = {"expand_index", "offset", "balance_degree_weight"}
        # list explicit keys to rename here in the weight state dictionary
        rename_keys = {
            # "module.backbone.routing_mlp": "module.backbone.mole_coefficient_mlp",
            # "module.backbone.moe_coefficient_mlp": "module.backbone.mole_coefficient_mlp",
            "backbone.dataset_embedding.dataset_emb_dict.osc.weight": "backbone.dataset_embedding.dataset_emb_dict.omc.weight",
            "module.backbone.dataset_embedding.dataset_emb_dict.osc.weight": "module.backbone.dataset_embedding.dataset_emb_dict.omc.weight",
        }
        for state_dict_name in ["model_state_dict", "ema_state_dict"]:
            state_dict = getattr(checkpoint, state_dict_name)
            for k in [key for key in state_dict if key.split(".")[-1] in remove_keys]:
                state_dict.pop(k)
                print(f"Removing {k} from {state_dict_name}")
            # rename explicitly mapped keys
            for subkey_from, subkey_to in rename_keys.items():
                for key_from in [key for key in state_dict if subkey_from in key]:
                    key_to = key_from.replace(subkey_from, subkey_to)
                    state_dict[key_to] = state_dict[key_from]
                    state_dict.pop(key_from)
                    print("rename", key_from, key_to)

    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ocp_dev/foundation_model checkpoint to fm_release"
    )

    parser.add_argument(
        "--checkpoint-in", type=str, help="checkpoint input", required=True
    )
    parser.add_argument(
        "--checkpoint-out", type=str, help="checkpoint output", required=True
    )
    parser.add_argument(
        "--remove-static-keys", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--add-stress", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--map-undefined-stress-to", type=str, required=False, default=None
    )
    parser.add_argument("--model-version", type=float, default=1.0)
    args = parser.parse_args()

    if os.path.exists(args.checkpoint_out):
        raise FileExistsError(
            f"Output checkpoint ({args.checkpoint_out}) cannot already exist"
        )

    checkpoint = migrate_checkpoint(
        args.checkpoint_in,
        args.remove_static_keys,
        args.map_undefined_stress_to,
        add_stress=args.add_stress,
        model_version=args.model_version,
    )
    torch.save(checkpoint, args.checkpoint_out)

    # test to see if checkpoint loads
    MLIPPredictUnit(args.checkpoint_out, device="cpu")

    # task_name="omol"
    # calc = FAIRChemCalculator.from_model_checkpoint(args.checkpoint_out,task_name)
    # atoms = molecule("H2O")
    # atoms.set_cell([100.0, 100.0, 100.0])  # Define a cubic cell
    # atoms.set_pbc(False)  # Enable periodic boundary conditions
    # atoms.calc = calc
    # energy = atoms.get_potential_energy()
    # stress=atoms.get_stress(task_name)
