from __future__ import annotations

import argparse
import os
import pickle

import omegaconf
import torch

from fairchem.core.scripts.migrate_imports import mapping


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
    return config_or_data


class RenameUnpickler(pickle.Unpickler):
    def __init__(self, file, *args, **kwargs):
        super().__init__(file, *args, **kwargs)

    def find_class(self, module, name):
        return super().find_class(find_new_module_name(module), name)


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
    args = parser.parse_args()

    if os.path.exists(args.checkpoint_out):
        raise FileExistsError(
            f"Output checkpoint ({args.checkpoint_out}) cannot already exist"
        )

    pickle.Unpickler = RenameUnpickler

    checkpoint = torch.load(args.checkpoint_in, pickle_module=pickle)
    checkpoint.tasks_config = update_config(checkpoint.tasks_config)
    checkpoint.model_config = update_config(checkpoint.model_config)

    torch.save(checkpoint, args.checkpoint_out)
