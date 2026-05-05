"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.file_download import _CACHED_NO_EXIST

from fairchem.core._config import CACHE_DIR
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.predict import BannedCheckpointError


# Hard-coded HuggingFace coordinates rather than registry lookups, because
# retired models (e.g. uma-s-1) are intentionally removed from
# pretrained_models.json but their checkpoints still exist on the Hub.
@pytest.mark.parametrize(
    "model_label,repo_id,subfolder,filename",
    [
        ("uma-s-1", "facebook/UMA", "checkpoints", "uma-s-1.pt"),
        ("uma-s-1p2", "facebook/UMA", "checkpoints", "uma-s-1p2.pt"),
    ],
)
def test_real_uma_checkpoint_is_banned(model_label, repo_id, subfolder, filename):
    """
    The retired UMA checkpoints shipped on HuggingFace must trip the
    BannedCheckpointError. Skipped when the file is not present in the
    local HF cache so the test does not require network access in CI.
    """
    rel_path = f"{subfolder}/{filename}" if subfolder else filename
    cached = try_to_load_from_cache(
        repo_id=repo_id,
        filename=rel_path,
        cache_dir=CACHE_DIR,
    )
    if cached is None or cached is _CACHED_NO_EXIST:
        pytest.skip(f"{model_label} not in local HF cache at {CACHE_DIR}")

    with pytest.raises(BannedCheckpointError):
        MLIPPredictUnit(cached, device="cpu")
