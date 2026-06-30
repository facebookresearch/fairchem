"""Tests for fairchem.core.models.uma.compat — UMA generation classifier
and in-place ``model_id`` back-fill.
"""

from __future__ import annotations

import logging

import pytest
from omegaconf import OmegaConf

from fairchem.core.models.uma.compat import (
    UMA_1P1_MODEL_ID,
    apply_uma_compat_fixups,
    get_uma_version,
)
from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint

UMA_BACKBONE_FQN = "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone"
UMA_BACKBONE_SHORT = "escnmd_moe_backbone"


def make_fake_checkpoint(model_config) -> MLIPInferenceCheckpoint:
    """Build a real ``MLIPInferenceCheckpoint`` with empty state for tests."""
    return MLIPInferenceCheckpoint(
        model_config=model_config,
        model_state_dict={},
        ema_state_dict={},
        tasks_config={},
    )


def uma_cfg(*, model_version=None, model_id=None, backbone_model=UMA_BACKBONE_FQN):
    cfg = {"backbone": {"model": backbone_model}}
    if model_version is not None:
        cfg["backbone"]["model_version"] = model_version
    if model_id is not None:
        cfg["model_id"] = model_id
    return cfg


# ---------------------------------------------------------------------------
# UMA 1.0 / untagged (neither model_id nor model_version) — hard fail
# ---------------------------------------------------------------------------


def test_unidentified_raises():
    """No model_id and no model_version -> cannot classify -> raise.

    This is the on-disk signature of a deprecated UMA 1.0 checkpoint (which ships
    with neither field) and of an untagged freshly-trained model.
    """
    cfg = uma_cfg()  # no model_id, no model_version
    ckpt = make_fake_checkpoint(cfg)
    with pytest.raises(RuntimeError) as exc_info:
        apply_uma_compat_fixups(ckpt, checkpoint_location="/path/to/uma-s-1.pt")
    msg = str(exc_info.value)
    assert "no model_id" in msg
    assert "fairchem-core<=2.21.0" in msg  # names the deprecated-1.0 possibility
    assert "/path/to/uma-s-1.pt" in msg


# ---------------------------------------------------------------------------
# UMA 1.1 — classification + back-fill
# ---------------------------------------------------------------------------


def test_uma_1p1_string_model_version_backfills(caplog):
    cfg = uma_cfg(model_version="1.1")
    ckpt = make_fake_checkpoint(cfg)
    with caplog.at_level(logging.WARNING):
        apply_uma_compat_fixups(ckpt, checkpoint_location="/p.pt")
    assert ckpt.model_config["model_id"] == UMA_1P1_MODEL_ID
    assert any("UMA 1.1" in r.getMessage() for r in caplog.records)


def test_uma_1p1_float_model_version_backfills(caplog):
    cfg = uma_cfg(model_version=1.1)
    ckpt = make_fake_checkpoint(cfg)
    with caplog.at_level(logging.WARNING):
        apply_uma_compat_fixups(ckpt)
    assert ckpt.model_config["model_id"] == UMA_1P1_MODEL_ID


def test_uma_1p1_idempotent_bare(caplog):
    cfg = uma_cfg(model_version="1.1", model_id=UMA_1P1_MODEL_ID)
    ckpt = make_fake_checkpoint(cfg)
    with caplog.at_level(logging.WARNING):
        apply_uma_compat_fixups(ckpt)
    assert ckpt.model_config["model_id"] == UMA_1P1_MODEL_ID
    # No backfill warning on idempotent path.
    assert not any("back-filled" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("subsize_id", ["UMA-S-1.1", "UMA-M-1.1", "UMA-L-1.1"])
def test_uma_1p1_model_id_not_reclassified(subsize_id, caplog):
    """Only 1.2 is special by model_id. A 1.1-style model_id is treated as
    unknown_uma, and model_id left untouched (never re-back-filled)."""
    cfg = uma_cfg(model_version="1.1", model_id=subsize_id)
    ckpt = make_fake_checkpoint(cfg)
    assert get_uma_version(cfg) == "unknown_uma"
    with caplog.at_level(logging.WARNING):
        apply_uma_compat_fixups(ckpt)
    assert ckpt.model_config["model_id"] == subsize_id  # untouched
    assert not any("back-filled" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# UMA 1.2 — no-op
# ---------------------------------------------------------------------------


def test_uma_1p2_no_op():
    cfg = uma_cfg(model_id="UMA-S-1.2")
    ckpt = make_fake_checkpoint(cfg)
    apply_uma_compat_fixups(ckpt)
    assert ckpt.model_config["model_id"] == "UMA-S-1.2"


def test_uma_1p2_bare_no_op():
    cfg = uma_cfg(model_id="UMA-1.2")
    ckpt = make_fake_checkpoint(cfg)
    apply_uma_compat_fixups(ckpt)
    assert ckpt.model_config["model_id"] == "UMA-1.2"


# ---------------------------------------------------------------------------
# Unknown / user-customized / non-UMA
# ---------------------------------------------------------------------------


def test_unknown_uma_id_warns_no_op(caplog):
    """Future UMA 1.3 must not be silently treated as 1.2."""
    cfg = uma_cfg(model_id="UMA-1.3")
    ckpt = make_fake_checkpoint(cfg)
    with caplog.at_level(logging.WARNING):
        assert get_uma_version(cfg) == "unknown_uma"
        apply_uma_compat_fixups(ckpt)
    assert ckpt.model_config["model_id"] == "UMA-1.3"  # untouched


def test_user_customized_model_id_preserved(caplog):
    cfg = uma_cfg(model_version="1.1", model_id="my-cool-finetune")
    ckpt = make_fake_checkpoint(cfg)
    with caplog.at_level(logging.WARNING):
        apply_uma_compat_fixups(ckpt)
    assert ckpt.model_config["model_id"] == "my-cool-finetune"


def test_non_uma_no_op(caplog):
    cfg = {"backbone": {"model": "fairchem.core.models.esen.esen_backbone.ESEN"}}
    ckpt = make_fake_checkpoint(cfg)
    with caplog.at_level(logging.INFO):
        assert get_uma_version(cfg) == "not_uma"
        apply_uma_compat_fixups(ckpt)
    assert "model_id" not in ckpt.model_config
    assert not any("UMA" in r.getMessage() for r in caplog.records)


def test_short_registry_name_backbone():
    cfg = uma_cfg(model_version="1.1", backbone_model=UMA_BACKBONE_SHORT)
    ckpt = make_fake_checkpoint(cfg)
    assert get_uma_version(cfg) == "1.1"
    apply_uma_compat_fixups(ckpt)
    assert ckpt.model_config["model_id"] == UMA_1P1_MODEL_ID


def test_none_model_config():
    ckpt = make_fake_checkpoint(None)
    assert get_uma_version(None) == "not_uma"
    apply_uma_compat_fixups(ckpt)  # must not raise


def test_empty_backbone_dict_treated_as_not_uma():
    cfg = {"backbone": {}}
    ckpt = make_fake_checkpoint(cfg)
    assert get_uma_version(cfg) == "not_uma"
    apply_uma_compat_fixups(ckpt)
    assert "model_id" not in ckpt.model_config


def test_empty_string_model_id_treated_as_absent():
    cfg = uma_cfg(model_version="1.1", model_id="")
    ckpt = make_fake_checkpoint(cfg)
    apply_uma_compat_fixups(ckpt)
    assert ckpt.model_config["model_id"] == UMA_1P1_MODEL_ID


def test_whitespace_model_id_treated_as_absent():
    cfg = uma_cfg(model_version="1.1", model_id="   ")
    ckpt = make_fake_checkpoint(cfg)
    apply_uma_compat_fixups(ckpt)
    assert ckpt.model_config["model_id"] == UMA_1P1_MODEL_ID


# ---------------------------------------------------------------------------
# DictConfig + override-bypass policy
# ---------------------------------------------------------------------------


def test_dictconfig_backfill_under_struct_mode():
    """OmegaConf struct mode must not block the back-fill (uses open_dict)."""
    cfg = OmegaConf.create(uma_cfg(model_version="1.1"))
    OmegaConf.set_struct(cfg, True)
    ckpt = make_fake_checkpoint(cfg)
    apply_uma_compat_fixups(ckpt)
    assert ckpt.model_config["model_id"] == UMA_1P1_MODEL_ID


def test_dictconfig_uma_1p2_classified():
    cfg = OmegaConf.create(uma_cfg(model_id="UMA-S-1.2"))
    assert get_uma_version(cfg) == "1.2"


def test_dictconfig_unidentified_raises():
    cfg = OmegaConf.create(uma_cfg())
    ckpt = make_fake_checkpoint(cfg)
    with pytest.raises(RuntimeError, match="no model_id"):
        apply_uma_compat_fixups(ckpt)


# ---------------------------------------------------------------------------
# 1.2 size-variant classification
# ---------------------------------------------------------------------------


def test_uma_1p2_m_variant_classified():
    assert get_uma_version(uma_cfg(model_id="UMA-M-1.2")) == "1.2"
