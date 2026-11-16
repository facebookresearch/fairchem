"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from fairchem.core.scripts.migrate_checkpoint import (
    find_new_module_name,
    migrate_checkpoint,
    update_config,
)


def test_migrate_nested_imports_and_dataset_names():
    old = "fairchem.experimental.foundation_models.modules.loss.MAELoss"
    new = "fairchem.core.modules.loss.MAELoss"
    for config in (
        {"nested": [old, {"dataset": "osc", "count": 2}]},
        OmegaConf.create({"nested": [old, {"dataset": "osc", "count": 2}]}),
    ):
        result = update_config(config)
        assert result["nested"][0] == new
        assert result["nested"][1] == {"dataset": "omc", "count": 2}
    assert find_new_module_name(old.rsplit(".", 1)[0]) == new.rsplit(".", 1)[0]
    assert find_new_module_name("unmapped.module") == "unmapped.module"


@pytest.fixture()
def checkpoint_file(tmp_path, monkeypatch):
    """
    Serialize a tiny checkpoint and restore the migration's global pickle hook.
    """
    monkeypatch.setattr(pickle, "Unpickler", pickle.Unpickler)

    def create(tasks, model="other"):
        checkpoint = SimpleNamespace(
            model_config={"backbone": {"model": model}},
            tasks_config=OmegaConf.create(tasks),
            model_state_dict={
                "backbone.expand_index": torch.ones(1),
                "backbone.weight": torch.tensor([2.0]),
                "backbone.dataset_embedding.dataset_emb_dict.osc.weight": torch.ones(1),
            },
            ema_state_dict={"module.backbone.offset": torch.ones(1)},
        )
        path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, path)
        return path

    return create


def test_migrate_removes_static_buffers_and_preserves_weights(checkpoint_file):
    path = checkpoint_file([], "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone")
    checkpoint = migrate_checkpoint(path, model_version=1.2)
    assert checkpoint.model_config["backbone"]["model_version"] == 1.2
    assert checkpoint.ema_state_dict == {}
    assert set(checkpoint.model_state_dict) == {
        "backbone.weight",
        "backbone.dataset_embedding.dataset_emb_dict.omc.weight",
    }
    assert torch.equal(
        checkpoint.model_state_dict["backbone.weight"], torch.tensor([2.0])
    )
    preserved = migrate_checkpoint(path, rm_static_keys=False)
    assert "backbone.expand_index" in preserved.model_state_dict


@pytest.mark.parametrize("uma", [False, True])
def test_add_stress_preserves_energy_tasks_and_normalization(checkpoint_file, uma):
    energy_name = "omol_energy" if uma else "energy"
    path = checkpoint_file(
        [{"name": energy_name, "datasets": ["omol"], "normalizer": {"rmsd": 2.0}}]
    )
    checkpoint = migrate_checkpoint(path, add_stress=True)
    assert checkpoint.model_config["backbone"]["regress_stress"]
    assert [task.name for task in checkpoint.tasks_config] == [
        energy_name,
        "omol_stress" if uma else "stress",
    ]
    stress = checkpoint.tasks_config[1]
    assert stress.datasets == ["omol"]
    assert stress.normalizer.rmsd == 2.0
    assert stress.out_spec.dim == [1, 9]


def test_copy_stress_task_to_dataset_missing_stress(checkpoint_file):
    path = checkpoint_file(
        [
            {"name": "omat_energy"},
            {"name": "omol_energy"},
            {"name": "omat_stress", "datasets": ["omat"], "normalizer": {"rmsd": 3}},
        ]
    )
    checkpoint = migrate_checkpoint(path, task_add_stress="omat")
    assert checkpoint.tasks_config[-1].name == "omol_stress"
    assert checkpoint.tasks_config[-1].datasets == ["omol"]
    assert checkpoint.tasks_config[-1].normalizer.rmsd == 3
    assert checkpoint.tasks_config[2].datasets == ["omat"]
    with pytest.raises(AssertionError, match="Did not find existing task"):
        migrate_checkpoint(path, task_add_stress="missing")
