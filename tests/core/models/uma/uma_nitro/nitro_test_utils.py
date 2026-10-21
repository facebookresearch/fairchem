"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Shared helpers for the UMA Nitro tests.
"""

from __future__ import annotations

import copy
import os

import hydra
import pytest
import torch

from fairchem.core.calculate.pretrained_mlip import (
    pretrained_checkpoint_path_from_name,
)
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    MLIPInferenceCheckpoint,
    inference_settings_nitro,
)

_DATASETS = {"omat": "omat", "omol": "omol"}
_REFERENCES = {
    "_target_": "fairchem.core.modules.normalization.element_references."
    "ElementReferences",
    "element_references": {
        "_target_": "torch.DoubleTensor",
        "_args_": [[0.1 * z for z in range(100)]],
    },
}


def _task(dataset: str, prop: str, level: str, refs: bool) -> dict:
    return {
        "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
        "name": f"{dataset}_{prop}",
        "level": level,
        "property": prop,
        "loss_fn": None,
        "out_spec": {"dim": [1], "dtype": "float32"},
        "normalizer": {
            "_target_": "fairchem.core.modules.normalization.normalizer.Normalizer",
            "mean": 0.0,
            "rmsd": 1.342,
        },
        "element_references": copy.deepcopy(_REFERENCES) if refs else None,
        "datasets": [dataset],
    }


TASKS = [
    _task("omat", "energy", "system", True),
    _task("omol", "energy", "system", True),
    _task("omat", "forces", "atom", False),
    _task("omol", "forces", "atom", False),
    _task("omat", "stress", "system", False),
]


def model_config(model_id: str | None = "UMA-S-1.2.1", **backbone_changes) -> dict:
    """
    A UMA-S configuration (Nitro's architecture) with few experts.

    Args:
        model_id: The UMA generation tag; ``None`` omits it, as UMA 1.1 does.
        backbone_changes: Backbone fields to add or override.
    """
    backbone = {
        "model": "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone",
        "num_experts": 4,
        "max_num_elements": 100,
        "lmax": 2,
        "mmax": 2,
        "otf_graph": False,
        "cutoff": 6,
        "num_distance_basis": 32,
        "num_layers": 4,
        "ff_type": "spectral",
        "charge_balanced_channels": [0, 1, 2],
        "distance_function": "gaussian",
        "regress_forces": True,
        "direct_forces": False,
        "regress_stress": True,
        "hidden_channels": 128,
        "act_type": "gate",
        "chg_spin_emb_type": "rand_emb",
        "dataset_mapping": dict(_DATASETS),
        "moe_layer_type": "pytorch",
        "use_composition_embedding": True,
        "sphere_channels": 128,
        "edge_channels": 128,
        "norm_type": "rms_norm_sh",
        **backbone_changes,
    }
    config = {
        "_target_": "fairchem.core.models.base.HydraModel",
        "model_id": model_id,
        "supports_single_atoms": True,
        "pass_through_head_outputs": True,
        "backbone": backbone,
        "heads": {
            "energyandforcehead": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificMoEWrapper",
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "dataset_mapping": dict(_DATASETS),
                "head_kwargs": {"wrap_property": False},
            }
        },
    }
    if model_id is None:
        del config["model_id"]
    return config


def direct_head_model_config() -> dict:
    """
    A single-dataset UMA-S configuration with an unwrapped EFS head.
    """
    config = model_config()
    config["backbone"]["dataset_mapping"] = {"omat": "omat"}
    config["heads"] = {
        "efs": {"module": "fairchem.core.models.uma.escn_md.MLP_EFS_Head"}
    }
    return config


def random_checkpoint(**config_changes) -> MLIPInferenceCheckpoint:
    """
    A randomly initialized UMA-S checkpoint, as ``torch.load`` would return it.

    Args:
        config_changes: Passed to :func:`model_config`, e.g. ``model_id=None,
            model_version=1.1`` for a UMA 1.1 checkpoint.
    """
    torch.manual_seed(0)
    model = hydra.utils.instantiate(_loadable(model_config(**config_changes)))
    state = model.state_dict()
    return MLIPInferenceCheckpoint(
        model_config=model_config(**config_changes),
        model_state_dict=state,
        ema_state_dict={
            **{f"module.{k}": v for k, v in state.items()},
            "n_averaged": torch.tensor(1),
        },
        tasks_config=copy.deepcopy(TASKS),
    )


def _loadable(config: dict) -> dict:
    """
    A config HydraModel can instantiate directly (UMA 1.1 gets its model_id
    back-filled at load time, as load_inference_model does).
    """
    if "model_id" not in config:
        config = {**config, "model_id": "UMA-1.1"}
    return config


def direct_head_random_checkpoint() -> MLIPInferenceCheckpoint:
    """
    A random single-dataset checkpoint with an unwrapped EFS head.
    """
    torch.manual_seed(0)
    config = direct_head_model_config()
    model = hydra.utils.instantiate(config)
    state = model.state_dict()
    return MLIPInferenceCheckpoint(
        model_config=config,
        model_state_dict=state,
        ema_state_dict={
            **{f"module.{key}": value for key, value in state.items()},
            "n_averaged": torch.tensor(1),
        },
        tasks_config=[
            {**_task("omat", "energy", "system", True), "name": "energy"},
            {**_task("omat", "forces", "atom", False), "name": "forces"},
            {**_task("omat", "stress", "system", False), "name": "stress"},
        ],
    )


def batch(*systems):
    """
    Collate ``(atoms, task[, charge, spin])`` systems into one batch.
    """
    samples = []
    for system in systems:
        atoms, task, charge, spin = (*system, 0, 0)[:4]
        atoms = atoms.copy()
        atoms.info.update(charge=charge, spin=spin)
        samples.append(
            AtomicData.from_ase(
                atoms, task_name=task, r_edges=False, r_data_keys=["charge", "spin"]
            )
        )
    return data_list_collater(samples, otf_graph=True)


# Settings Nitro cannot honor, and the reason its error (and the fallback
# warning) gives.
UNSUPPORTED_SETTINGS = [
    ({"execution_mode": "umas_fast_gpu"}, "execution_mode"),
    ({"execution_mode": "general"}, "execution_mode"),
    ({"merge_mole": False}, "merge_mole=False"),
    ({"activation_checkpointing": True}, "activation checkpointing"),
    ({"external_graph_gen": True}, "external graph generation"),
    ({"base_precision_dtype": torch.float64}, "float32"),
    ({"tf32": False}, "tf32=False"),
    ({"predict_untrained_hessian": {"omat"}}, "Hessian"),
]


def nitro_gpu() -> bool:
    """
    Whether this machine has a GPU Nitro runs on (compute capability 8.x-12.x).
    On any other GPU, such as CI's T4s, nitro mode runs multigpu.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (
        8,
        9,
        10,
        11,
        12,
    )


# Skips tests that only make sense once Nitro itself runs.
requires_nitro_gpu = pytest.mark.skipif(
    not nitro_gpu(), reason="UMA Nitro runs only on compute capability 8.x-12.x"
)

# The load-time reason on any other GPU, checked before settings and the file.
UNVALIDATED_GPU = "supports only CUDA compute capability 8.x-12.x"


def load_time_reason(reason: str) -> str:
    """
    The fallback reason a GPU test should expect: ``reason`` on a Nitro GPU,
    the GPU check's reason elsewhere.
    """
    return reason if nitro_gpu() else UNVALIDATED_GPU


def checkpoint_path(pretrained_model_name: str) -> str:
    if os.path.exists(pretrained_model_name):
        return pretrained_model_name
    return pretrained_checkpoint_path_from_name(pretrained_model_name)


def nitro_settings(**changes) -> InferenceSettings:
    """
    The ``nitro`` preset with some fields changed.
    """
    settings = inference_settings_nitro()
    for key, value in changes.items():
        setattr(settings, key, value)
    return settings


def general_predictor(path: str, **changes) -> MLIPPredictUnit:
    """
    The general, unmerged, uncompiled HydraModel: the reference for Nitro.
    """
    settings = InferenceSettings(
        merge_mole=False, compile=False, execution_mode="general", **changes
    )
    return MLIPPredictUnit(path, "cuda", inference_settings=settings)


def task_names(predictor: MLIPPredictUnit) -> dict[str, list[str]]:
    return {
        dataset: sorted(task.name for task in tasks)
        for dataset, tasks in predictor.dataset_to_tasks.items()
    }


def _max_abs_diff(output: dict, expected: dict, key: str) -> tuple[float, float]:
    reference = expected[key].double()
    diff = (output[key].double() - reference).abs().max().item()
    return diff, reference.abs().max().item()


def assert_close(output: dict, expected: dict, rtol=1e-2) -> None:
    """
    Agreement within reduced-precision (FP16/TF32) tolerance.

    Force and stress errors scale with their magnitude (e.g. ions and spin
    states evaluated far from equilibrium), so each limit has an absolute part
    and a part relative to the reference's largest component.
    """
    assert output.keys() == expected.keys()
    for name, value in output.items():
        assert torch.isfinite(value).all(), name
    natoms = len(expected["forces"])
    energy, _ = _max_abs_diff(output, expected, "energy")
    assert energy / natoms < 5e-3, f"|dE|/atom={energy / natoms:.2e}"
    forces, scale = _max_abs_diff(output, expected, "forces")
    assert forces < 5e-2 + rtol * scale, f"max|dF|={forces:.2e} (max|F|={scale:.2e})"
    if "stress" in expected:
        stress, scale = _max_abs_diff(output, expected, "stress")
        limit = 5e-3 + rtol * scale
        assert stress < limit, f"max|dS|={stress:.2e} (max|S|={scale:.2e})"
