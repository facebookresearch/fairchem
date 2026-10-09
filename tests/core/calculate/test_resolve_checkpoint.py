"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  pretrained_mlip.resolve_checkpoint name-vs-path dispatch without
        network access (Hugging Face downloads are mocked).
"""

from __future__ import annotations

import pytest

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.calculate.pretrained_mlip import (
    ResolvedCheckpoint,
    UnknownCheckpointError,
    resolve_checkpoint,
)


def test_resolve_checkpoint_path_has_no_refs(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()

    assert resolve_checkpoint(str(checkpoint)) == ResolvedCheckpoint(str(checkpoint))


@pytest.mark.parametrize("error_type", [KeyError, ValueError])
def test_resolve_checkpoint_unknown_raises(error_type):
    with pytest.raises(error_type, match="not found"):
        resolve_checkpoint("definitely-not-a-real-model")
    with pytest.raises(
        UnknownCheckpointError, match="Invalid model name or checkpoint path"
    ):
        resolve_checkpoint("definitely-not-a-real-model")


def test_resolve_checkpoint_registered_name_fetches_refs(tmp_path, monkeypatch):
    (tmp_path / "iso_atom_elem_refs.yaml").write_text("H: -0.5\n")
    (tmp_path / "form_elem_refs.yaml").write_text("refs:\n  H: -1.0\n")
    downloads = []

    def fake_hf_hub_download(filename, repo_id, subfolder, revision, cache_dir):
        downloads.append((filename, cache_dir))
        return str(tmp_path / filename)

    monkeypatch.setattr(pretrained_mlip, "hf_hub_download", fake_hf_hub_download)

    resolved = resolve_checkpoint("uma-s-1p1", cache_dir="/refs/cache")

    assert resolved.path == str(tmp_path / "uma-s-1p1.pt")
    assert resolved.atom_refs == {"H": -0.5}
    assert resolved.form_elem_refs == {"H": -1.0}
    # The checkpoint always downloads to CACHE_DIR; only refs honor cache_dir.
    assert downloads[0] == ("uma-s-1p1.pt", pretrained_mlip.CACHE_DIR)
    assert all(cache_dir == "/refs/cache" for _, cache_dir in downloads[1:])


def test_resolve_checkpoint_registry_wins_over_file(tmp_path, monkeypatch):
    (tmp_path / "uma-s-1p1").touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        pretrained_mlip,
        "hf_hub_download",
        lambda filename, **kwargs: f"/downloaded/{filename}",
    )
    monkeypatch.setattr(
        pretrained_mlip, "get_reference_energies", lambda *a: {"refs": {}}
    )

    assert resolve_checkpoint("uma-s-1p1").path == "/downloaded/uma-s-1p1.pt"
