"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Which checkpoints UMA Nitro admits: the config comparison against uma-s-1p2p1,
the admission table, and the shipped and user checkpoint lists. Fallbacks
through the predict unit are in test_uma_nitro_fallbacks.py.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from fairchem.core.units.mlip_unit.api.inference import UMANitroNotSupportedError
from tests.core.models.uma.uma_nitro.nitro_test_utils import (
    DIRECT_HEAD_TASKS,
    TASKS,
    _task,
    checkpoint_path,
    direct_head_model_config,
    list_checkpoint,
    model_config,
    random_checkpoint,
)

pytest.importorskip("triton")

from fairchem.core.models.uma.uma_nitro import UMANitroModel  # noqa: E402
from fairchem.core.models.uma.uma_nitro._checkpoints import (  # noqa: E402
    SHIPPED_LIST,
    admit,
    compatible_checkpoints,
    config_mismatches,
    user_list,
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


@pytest.fixture(scope="module")
def cutoff5_checkpoint():
    return random_checkpoint(cutoff=5)


@pytest.fixture(scope="module")
def relabeled_checkpoint():
    return random_checkpoint(model_id="UMA-S-1.2")


def _save(checkpoint, path: Path) -> Path:
    torch.save(checkpoint, path)
    return path


def _hashed(path: Path) -> bool:
    return Path(f"{path}.md5").exists()


def test_other_model_id_falls_back_unhashed_even_if_listed(
    relabeled_checkpoint, tmp_path
):
    path = _save(relabeled_checkpoint, tmp_path / "ft.pt")
    list_checkpoint(path)
    with pytest.raises(
        UMANitroNotSupportedError, match="admit only model_id UMA-S-1.2.1"
    ):
        admit(relabeled_checkpoint, path)
    assert not _hashed(path)


def test_overrides_nitro_cannot_compute_fall_back_unhashed_even_if_listed(
    nitro_random_checkpoint, tmp_path
):
    """
    Admission judges the config HydraModel builds, overrides included.
    """
    path = _save(nitro_random_checkpoint, tmp_path / "ft.pt")
    list_checkpoint(path)
    with pytest.raises(
        UMANitroNotSupportedError, match="overrides set backbone.max_neighbors=10"
    ):
        admit(nitro_random_checkpoint, path, {"backbone": {"max_neighbors": 10}})
    assert not _hashed(path)


def test_unlisted_mismatch_falls_back_without_a_listing_hint(
    cutoff5_checkpoint, tmp_path
):
    path = _save(cutoff5_checkpoint, tmp_path / "ft.pt")
    with pytest.raises(UMANitroNotSupportedError, match="backbone.cutoff=5") as error:
        admit(cutoff5_checkpoint, path)
    assert "md5sum" not in str(error.value)


def test_listed_mismatch_raises_naming_the_list(cutoff5_checkpoint, tmp_path):
    path = _save(cutoff5_checkpoint, tmp_path / "ft.pt")
    list_checkpoint(path)
    with pytest.raises(ValueError, match="backbone.cutoff=5") as error:
        admit(cutoff5_checkpoint, path)
    assert str(user_list()) in str(error.value)


@pytest.mark.skipif(shutil.which("md5sum") is None, reason="needs md5sum")
def test_unlisted_match_falls_back_with_a_hint_that_lists_it(
    nitro_random_checkpoint, tmp_path
):
    """
    Running the printed command, from any directory, lists the checkpoint.
    """
    path = _save(nitro_random_checkpoint, tmp_path / "my ft.pt")
    with pytest.raises(UMANitroNotSupportedError, match="md5sum") as error:
        admit(nitro_random_checkpoint, path)
    command = re.search(r"`([^`]+)`", str(error.value))[1]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    subprocess.run(["bash", "-c", command], cwd=elsewhere, check=True)
    admit(nitro_random_checkpoint, path)


def test_listed_match_is_admitted(nitro_random_checkpoint, tmp_path):
    path = _save(nitro_random_checkpoint, tmp_path / "ft.pt")
    list_checkpoint(path)
    admit(nitro_random_checkpoint, path)


_DIGEST = "0123456789abcdef0123456789abcdef"


@pytest.mark.parametrize(
    "content",
    [
        f"{_DIGEST}  my_ft.pt\n",
        f"{_DIGEST} *my_ft.pt\n",
        f"\\{_DIGEST}  run\\\\1.pt\n",
        f"# my fine-tunes\r\n\r\n{_DIGEST}  my_ft.pt\r\n",
        f"\ufeff{_DIGEST}  my_ft.pt\n",
        f"{_DIGEST.upper()}  my_ft.pt\n",
        f"{_DIGEST}\n",
    ],
    ids=["md5sum", "binary", "escaped-name", "crlf-comment", "bom", "upper", "bare"],
)
def test_reads_md5sum_lines(content):
    user_list().write_text(content, encoding="utf-8", newline="")
    assert compatible_checkpoints()[_DIGEST].source == user_list()


def _unreadable(path: Path) -> None:
    path.write_text(f"{_DIGEST}  my_ft.pt\n")
    path.chmod(0)


@pytest.mark.parametrize(
    ("make", "where"),
    [
        (lambda path: path.mkdir(), ""),
        (_unreadable, ""),
        (lambda path: path.write_bytes(b"\xff\xfe not utf-8\n"), ""),
        (lambda path: path.write_text(f"{_DIGEST}  ok.pt\nmy_ft.pt\n"), ":2"),
    ],
    ids=["directory", "unreadable", "undecodable", "malformed"],
)
def test_unreadable_user_list_raises_naming_it(make, where):
    if make is _unreadable and os.geteuid() == 0:
        pytest.skip("root reads any file")
    make(user_list())
    with pytest.raises(ValueError, match=re.escape(f"{user_list()}{where}")):
        compatible_checkpoints()


def test_ships_the_released_checkpoints():
    assert sorted(entry.label for entry in compatible_checkpoints().values()) == [
        "uma-s-1p2p1",
        "uma-s-1p2p1-ft-d4rev",
        "uma-s-1p2p1-ft-r2scan",
    ]


def test_shipped_entry_wins_a_shared_digest():
    digest = next(iter(compatible_checkpoints()))
    user_list().write_text(f"{digest}  my_copy.pt\n")
    assert compatible_checkpoints()[digest].source == SHIPPED_LIST


@pytest.mark.parametrize(
    ("value", "directory"),
    [
        (None, "~/.fairchem"),
        ("", "~/.fairchem"),
        ("~/my_config", "~/my_config"),
        ("/opt/fairchem", "/opt/fairchem"),
    ],
    ids=["unset", "empty", "tilde", "absolute"],
)
def test_user_list_location(monkeypatch, value, directory):
    if value is None:
        monkeypatch.delenv("FAIRCHEM_CONFIG_DIR")
    else:
        monkeypatch.setenv("FAIRCHEM_CONFIG_DIR", value)
    expected = Path(os.path.expanduser(directory), "nitro_compatible_checkpoints.txt")
    assert user_list() == expected


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
