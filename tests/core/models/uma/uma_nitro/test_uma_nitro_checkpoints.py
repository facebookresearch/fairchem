"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Which checkpoints UMA Nitro admits: the config comparison against uma-s-1p2p1
and the MD5 check. Fallbacks through the predict unit are in
test_uma_nitro_fallbacks.py.
"""

from __future__ import annotations

import copy

import pytest
import torch

from fairchem.core.units.mlip_unit.api.inference import UMANitroNotSupportedError
from tests.core.models.uma.uma_nitro.nitro_test_utils import (
    DIRECT_HEAD_TASKS,
    TASKS,
    _task,
    checkpoint_path,
    direct_head_model_config,
    model_config,
    random_checkpoint,
)

pytest.importorskip("triton")

from fairchem.core.models.uma.uma_nitro import UMANitroModel  # noqa: E402
from fairchem.core.models.uma.uma_nitro._checkpoints import (  # noqa: E402
    admit,
    config_mismatches,
)


def _head(config: dict) -> dict:
    return config["heads"]["energyandforcehead"]


def _deprecated_dataset_names(config: dict, tasks: list) -> None:
    config["backbone"]["dataset_mapping"]["mptrj"] = "omat"
    head = _head(config)
    head["dataset_names"] = [*sorted(head.pop("dataset_mapping")), "mptrj"]


def _second_head(config: dict, tasks: list) -> None:
    config["heads"]["copy"] = copy.deepcopy(_head(config))


_CHANGES = {
    "cutoff": (
        lambda config, tasks: config["backbone"].update(cutoff=5),
        "backbone.cutoff=5",
    ),
    "max_neighbors": (
        lambda config, tasks: config["backbone"].update(max_neighbors=100),
        "backbone.max_neighbors=100",
    ),
    "head-reduce": (
        lambda config, tasks: _head(config)["head_kwargs"].update(reduce="mean"),
        "heads.energyandforcehead.head_kwargs.reduce='mean'",
    ),
    "head-dataset-names": (
        _deprecated_dataset_names,
        "heads.energyandforcehead.dataset_names",
    ),
    "unknown-backbone-key": (
        lambda config, tasks: config["backbone"].update(radius_pbc_version=1),
        "backbone.radius_pbc_version=1 (unexpected)",
    ),
    "unknown-top-level-key": (
        lambda config, tasks: config.update(freeze_backbone=True),
        "freeze_backbone=True (unexpected)",
    ),
    "missing-key": (
        lambda config, tasks: config["backbone"].pop("use_composition_embedding"),
        "backbone.use_composition_embedding missing",
    ),
    "model-id": (
        lambda config, tasks: config.update(model_id="UMA-S-1.2"),
        "model_id='UMA-S-1.2'",
    ),
    "two-heads": (_second_head, "heads has 2 entries"),
    "dipole-task": (
        lambda config, tasks: tasks.append(_task("omat", "dipole", "system", False)),
        "'omat_dipole' predicts 'dipole'",
    ),
    "task-dataset-not-in-head": (
        lambda config, tasks: tasks.append(_task("oc20", "energy", "system", True)),
        "'oc20_energy' uses dataset 'oc20'",
    ),
}


def test_complete_configs_match():
    assert config_mismatches(model_config(), TASKS) == []
    assert config_mismatches(direct_head_model_config(), DIRECT_HEAD_TASKS) == []


def test_data_keys_may_change():
    """
    Keys fine-tunes change without changing what Nitro computes.
    """
    config = model_config(
        otf_graph=True,
        always_use_pbc=False,
        composition_dropout=0.5,
        moe_dropout=0.5,
        dataset_emb_grad=False,
        cs_emb_grad=False,
        num_experts=8,
    )
    config["supports_single_atoms"] = False
    config["backbone"]["dataset_mapping"] = {"omat": "omat", "omol": "omol", "x": "x"}
    _head(config)["dataset_mapping"] = {"omat": "omat", "omol": "omol", "y": "y"}
    assert config_mismatches(config, TASKS) == []


@pytest.mark.parametrize(("change", "named"), _CHANGES.values(), ids=_CHANGES)
def test_config_change_is_a_mismatch(change, named):
    config, tasks = model_config(), copy.deepcopy(TASKS)
    change(config, tasks)
    mismatches = config_mismatches(config, tasks)
    assert any(named in mismatch for mismatch in mismatches), mismatches


def test_admits_neither_a_changed_config_nor_an_unknown_file(tmp_path):
    """
    A changed config is rejected without hashing the file; a matching config
    is hashed, and an unknown MD5 is rejected.
    """
    changed = tmp_path / "cutoff5.pt"
    torch.save(random_checkpoint(cutoff=5), changed)
    with pytest.raises(UMANitroNotSupportedError, match="backbone.cutoff=5"):
        admit(random_checkpoint(cutoff=5), changed)
    assert not (tmp_path / "cutoff5.pt.md5").exists()

    unknown = tmp_path / "random.pt"
    torch.save(random_checkpoint(), unknown)
    with pytest.raises(UMANitroNotSupportedError, match="md5"):
        admit(random_checkpoint(), unknown)
    assert (tmp_path / "random.pt.md5").exists()


def test_overrides_are_compared(tmp_path):
    """
    Admission judges the config HydraModel builds, overrides included.
    """
    path = tmp_path / "random.pt"
    torch.save(random_checkpoint(), path)
    with pytest.raises(
        UMANitroNotSupportedError, match="overrides set backbone.max_neighbors=10"
    ):
        admit(random_checkpoint(), path, {"backbone": {"max_neighbors": 10}})


def test_model_reads_data_keys_from_overrides():
    model = UMANitroModel.from_checkpoint(
        random_checkpoint(), {"supports_single_atoms": False}
    )
    assert model.supports_single_atoms is False


@pytest.mark.pretrained("uma-s-1p2p1", "uma-s-1p2p1-ft-d4rev", "uma-s-1p2p1-ft-r2scan")
def test_admits_the_released_checkpoints(pretrained_model_name):
    """
    For uma-s-1p2p1 this also pins the reference config, and so the kernel
    constants read from it, to the real checkpoint outside the data keys.
    """
    path = checkpoint_path(pretrained_model_name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    assert config_mismatches(checkpoint.model_config, checkpoint.tasks_config) == []
    admit(checkpoint, path)
