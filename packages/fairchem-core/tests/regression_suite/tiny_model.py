"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import hydra
import torch

from fairchem.core.regression_suite.suites.prediction_fidelity.suite import TASKS
from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint

if TYPE_CHECKING:
    from pathlib import Path

_PROPERTIES = (("energy", "system"), ("forces", "atom"), ("stress", "system"))


def _model_config() -> dict:
    """
    A UMA-S architecture shrunk to a few thousand parameters.
    """
    datasets = {name: name for name in TASKS}
    return {
        "_target_": "fairchem.core.models.base.HydraModel",
        "model_id": "UMA-S-1.2.1",
        "supports_single_atoms": True,
        "pass_through_head_outputs": True,
        "backbone": {
            "model": "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone",
            "num_experts": 1,
            "max_num_elements": 100,
            "lmax": 1,
            "mmax": 1,
            "otf_graph": False,
            "cutoff": 6,
            "num_distance_basis": 4,
            "num_layers": 1,
            "ff_type": "spectral",
            "charge_balanced_channels": [0, 1, 2],
            "distance_function": "gaussian",
            "regress_forces": True,
            "direct_forces": False,
            "regress_stress": True,
            "hidden_channels": 4,
            "act_type": "gate",
            "chg_spin_emb_type": "rand_emb",
            "dataset_mapping": dict(datasets),
            "moe_layer_type": "pytorch",
            "use_composition_embedding": True,
            "sphere_channels": 4,
            "edge_channels": 4,
            "norm_type": "rms_norm_sh",
        },
        "heads": {
            "energyandforcehead": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificMoEWrapper",
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "dataset_mapping": dict(datasets),
                "head_kwargs": {"wrap_property": False},
            }
        },
    }


def _task(dataset: str, prop: str, level: str) -> dict:
    references = {
        "_target_": "fairchem.core.modules.normalization.element_references."
        "ElementReferences",
        "element_references": {
            "_target_": "torch.DoubleTensor",
            "_args_": [[0.1 * z for z in range(100)]],
        },
    }
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
        "element_references": references if prop == "energy" else None,
        "datasets": [dataset],
    }


def write_tiny_checkpoint(path: Path, seed: int = 0) -> Path:
    """
    Save a random-weight UMA inference checkpoint for every suite task.

    The same seed gives identical weights; the global RNG is left untouched.

    Args:
        path: File to write.
        seed: Seed of the random weights.

    Returns:
        The written file.
    """
    config = _model_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = hydra.utils.instantiate(copy.deepcopy(config))
    state = model.state_dict()
    torch.save(
        MLIPInferenceCheckpoint(
            model_config=config,
            model_state_dict=state,
            ema_state_dict={
                **{f"module.{key}": value for key, value in state.items()},
                "n_averaged": torch.tensor(1),
            },
            tasks_config=[
                _task(dataset, prop, level)
                for dataset in TASKS
                for prop, level in _PROPERTIES
            ],
        ),
        path,
    )
    return path
