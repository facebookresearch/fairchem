"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import contextmanager, nullcontext

import hydra
import numpy as np
import torch
from torchtnt.framework import PredictUnit, State

from fairchem.core.common.distutils import (
    CURRENT_DEVICE_TYPE_STR,
    get_device_for_local_rank,
)
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import InferenceSettings
from fairchem.core.units.mlip_unit.mlip_unit import load_inference_model


class MLIPPredictUnit(PredictUnit[AtomicData]):
    def __init__(
        self,
        inference_model_path: str,
        device: str = "cpu",
        overrides: dict | None = None,
        inference_settings: InferenceSettings | None = None,
        seed: int = 41,
    ):
        super().__init__()
        os.environ[CURRENT_DEVICE_TYPE_STR] = device

        self.seed(seed)

        if inference_settings is None:
            inference_settings = InferenceSettings()
        if overrides is None:
            overrides = {}
        if "backbone" not in overrides:
            overrides["backbone"] = {}
        if inference_settings.activation_checkpointing is not None:
            overrides["backbone"]["activation_checkpointing"] = (
                inference_settings.activation_checkpointing
            )
        if inference_settings.wigner_cuda is not None:
            overrides["backbone"]["use_cuda_graph_wigner"] = (
                inference_settings.wigner_cuda
            )
        if inference_settings.external_graph_gen is not None:
            overrides["backbone"][
                "otf_graph"
            ] = not inference_settings.external_graph_gen

        if inference_settings.internal_graph_gen_version is not None:
            overrides["backbone"]["radius_pbc_version"] = (
                inference_settings.internal_graph_gen_version
            )

        self.model, checkpoint = load_inference_model(
            inference_model_path, use_ema=True, overrides=overrides
        )
        tasks = [
            hydra.utils.instantiate(task_config)
            for task_config in checkpoint.tasks_config
        ]
        self.tasks = {t.name: t for t in tasks}

        assert device in ["cpu", "cuda"], "device must be either 'cpu' or 'cuda'"

        self.device = get_device_for_local_rank() if device == "cuda" else "cpu"

        self.model.eval()

        self.lazy_model_intialized = False
        self.inference_mode = inference_settings

        # store composition embedding of system the model was merged on
        self.merged_on = None

    @property
    def direct_forces(self) -> bool:
        return self.model.module.backbone.direct_forces

    @property
    def datasets(self) -> list[str]:
        return self.model.module.backbone.dataset_list

    def seed(self, seed: int):
        logging.info(f"Setting random seed to {seed}")
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def move_to_device(self):
        self.model.to(self.device)
        for task in self.tasks.values():
            task.normalizer.to(self.device)
            if task.element_references is not None:
                task.element_references.to(self.device)

    def predict_step(self, state: State, data: AtomicData) -> dict[str, torch.tensor]:
        return self.predict(data)

    def get_composition_charge_spin_dataset(self, data):
        composition_sum = data.atomic_numbers.new_zeros(
            self.model.module.backbone.max_num_elements,
            dtype=torch.int,
        ).index_add(
            0,
            data.atomic_numbers.to(torch.int),
            data.atomic_numbers.new_ones(data.atomic_numbers.shape[0], dtype=torch.int),
        )
        comp_charge_spin = (
            composition_sum,
            getattr(data, "charge", None),
            getattr(data, "spin", None),
        )
        return comp_charge_spin, getattr(data, "dataset", [None])

    def predict(
        self, data: AtomicData, undo_element_references: bool = True
    ) -> dict[str, torch.tensor]:
        if not self.lazy_model_intialized:
            # merge everything on CPU
            if self.inference_mode.merge_mole:
                # replace backbone with non MOE version
                assert (
                    data.natoms.numel() == 1
                ), f"Cannot merge model with multiple systems in batch. Must be exactly 1 system, found {data.natoms.numel()}"
                self.model.module.backbone = (
                    self.model.module.backbone.merge_MOLE_model(data.clone())
                )
                self.model.eval()
            # move to device
            self.move_to_device()
            if self.inference_mode.compile:
                logging.warning(
                    "Model is being compiled this might take a while for the first time"
                )
                self.model = torch.compile(self.model, dynamic=True)
            self.lazy_model_intialized = True

        data_device = data.to(self.device)

        if self.inference_mode.merge_mole:
            if self.merged_on is None:
                # only get embeddings after moved to final device to get right types
                self.merged_on = self.get_composition_charge_spin_dataset(data_device)
            else:
                this_sys = self.get_composition_charge_spin_dataset(data_device)
                assert (
                    data_device.natoms.numel() == 1
                ), f"Cannot run merged model on batch with multiple systems. Must be exactly 1 system, found {data_device.natoms.numel()}"
                assert (
                    self.merged_on[0][0].isclose(this_sys[0][0], rtol=1e-5).all()
                ), "Cannot run on merged model on system. Embeddings seem different..."
                assert (
                    self.merged_on[0][1] == this_sys[0][1]
                ), f"Cannot run on merged model on system. Charge is diferrent {self.merged_on[0][1]} vs {this_sys[0][1]}"
                assert (
                    self.merged_on[0][2] == this_sys[0][2]
                ), f"Cannot run on merged model on system. Spin is diferrent {self.merged_on[0][2]} vs {this_sys[0][2]}"
                assert (
                    self.merged_on[1] == this_sys[1]
                ), f"Cannot run on merged model on system. Dataset is diferrent {self.merged_on[1]} vs {this_sys[1]}"

        inference_context = torch.no_grad() if self.direct_forces else nullcontext()
        tf32_context = (
            tf32_context_manager() if self.inference_mode.tf32 else nullcontext()
        )

        pred_output = {}
        with inference_context, tf32_context:
            output = self.model(data_device)
            for task_name, task in self.tasks.items():
                pred_output[task_name] = task.normalizer.denorm(
                    output[task_name][task.property]
                )
                if undo_element_references and task.element_references is not None:
                    pred_output[task_name] = task.element_references.undo_refs(
                        data_device, pred_output[task_name]
                    )

        return pred_output


@contextmanager
def tf32_context_manager():
    # Store the original settings
    original_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    original_allow_tf32_cudnn = torch.backends.cudnn.allow_tf32
    original_float32_matmul_precision = torch.get_float32_matmul_precision()
    try:
        # Set the desired settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        yield
    finally:
        # Revert to the original settings
        torch.backends.cuda.matmul.allow_tf32 = original_allow_tf32_matmul
        torch.backends.cudnn.allow_tf32 = original_allow_tf32_cudnn
        torch.set_float32_matmul_precision(original_float32_matmul_precision)
