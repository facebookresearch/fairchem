from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import yaml

from fairchem.core.scripts.create_finetune_dataset import (
    compute_normalizer_and_linear_reference,
    launch_processing,
)
from fairchem.core.units.mlip_unit.api.inference import UMATask

logging.basicConfig(level=logging.INFO)

TEMPLATE_DIR = Path("configs/uma/finetune")
DATA_TASK_YAML = Path("data/uma_conserving_data_task_template.yaml")
UMA_SM_FINETUNE_YAML = Path("uma_sm_finetune_template.yaml")


def create_yaml(
    train_path: str,
    val_path: str,
    force_rms: float,
    linref_coeff: list,
    output_dir: str,
    dataset_name: str,
    regress_stress: bool,
):
    data_task_yaml = TEMPLATE_DIR / DATA_TASK_YAML
    with open(data_task_yaml) as file:
        template = yaml.safe_load(file)
        template["dataset_name"] = dataset_name
        template["normalizer_rmsd"] = force_rms
        template["elem_refs"] = linref_coeff
        template["train_dataset"]["splits"]["train"]["src"] = train_path
        template["val_dataset"]["splits"]["val"]["src"] = val_path
        if not regress_stress:
            # remove the stress task
            template["tasks_list"].pop()
        os.makedirs(output_dir / "data", exist_ok=True)
        with open(output_dir / DATA_TASK_YAML, "w") as yaml_file:
            yaml.dump(template, yaml_file, default_flow_style=False, sort_keys=False)

    if not regress_stress:
        uma_finetune_yaml = TEMPLATE_DIR / UMA_SM_FINETUNE_YAML
        with open(uma_finetune_yaml) as file:
            template_ft = yaml.safe_load(file)
            template_ft["regress_stress"] = False
        with open(output_dir / UMA_SM_FINETUNE_YAML, "w") as yaml_file:
            yaml.dump(template_ft, yaml_file, default_flow_style=False, sort_keys=False)
    else:
        shutil.copy2(
            TEMPLATE_DIR / UMA_SM_FINETUNE_YAML, output_dir / UMA_SM_FINETUNE_YAML
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Directory of ASE atoms objects to convert for training.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Directory of ASE atoms objects to convert for validation.",
    )
    parser.add_argument(
        "--task-to-finetune",
        type=str,
        required=True,
        choices=[t.value for t in UMATask],
        help="choose a uma task to finetune",
    )
    parser.add_argument(
        "--regress-stress",
        type=bool,
        default=False,
        help="Allow for stress regression tasks, will only work if your dataset has stress labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to save required finetuning artifacts.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for processing files.",
    )
    args = parser.parse_args()

    # Launch processing for training data
    train_path = args.output_dir / "train"
    launch_processing(args.train_dir, train_path, args.num_workers)
    force_rms, linref_coeff = compute_normalizer_and_linear_reference(
        train_path, args.num_workers
    )
    val_path = args.output_dir / "val"
    launch_processing(args.val_dir, val_path, args.num_workers)

    create_yaml(
        train_path=str(train_path),
        val_path=str(val_path),
        force_rms=float(force_rms),
        linref_coeff=linref_coeff,
        output_dir=args.output_dir,
        dataset_name=args.task_to_finetune,
        regress_stress=args.regress_stress,
    )
    logging.info(f"Generated dataset and data config yaml in {args.output_dir}")
    logging.info(
        f"To run finetuning, run fairchem -c {args.output_dir}/{UMA_SM_FINETUNE_YAML}"
    )
