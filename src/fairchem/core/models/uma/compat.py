"""UMA checkpoint compatibility shim.

Classifies a loaded checkpoint by UMA generation (from ``model_id``, falling back
to ``backbone.model_version`` for legacy 1.1) and fixes up its ``model_config``
in place at load time (in ``load_inference_model``, before instantiation):

* **UMA 1.1** ships without a ``model_id`` — back-fill it to ``"UMA-1.1"``.
* **UMA 1.2+** are already tagged, non-UMA is ignored — no-op.

The MoE ``include_self`` quirk is not handled here: the backbone derives it from
``model_id`` (``eSCNMDMoeBackbone.set_MOLE_coefficients``; only 1.2 uses True).

Transitional: remove the 1.1 back-fill once UMA 1.1 is deprecated, and delete
this module entirely once both UMA 1.1 and 1.2 are deprecated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from omegaconf import DictConfig, OmegaConf, open_dict

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint


UmaVersion = Literal["1.1", "1.2", "unidentified", "unknown_uma", "not_uma"]

UMA_1P1_MODEL_ID = "UMA-1.1"


# Last fairchem-core release that supports UMA 1.0 (git describe --tags on
# this repo: fairchem_core-2.21.0-2-g28c7f4ba6).
_LAST_UMA_1P0_FAIRCHEM_VERSION = "2.21.0"

# Hydra registry short-name for the UMA MoE backbone.
_UMA_BACKBONE_SHORT_NAME = "escnmd_moe_backbone"
# Suffix of the fully-qualified backbone class path used in shipped checkpoints.
_UMA_BACKBONE_FQN_SUFFIX = "uma.escn_moe.eSCNMDMoeBackbone"


def get_uma_version(model_config: dict | DictConfig | None) -> UmaVersion:
    """Classify a checkpoint by UMA generation from the three real signatures.

    Shipped: 1.0 = neither field set; 1.1 = ``model_version`` only; 1.2 =
    ``model_id`` only (matched on its trailing ``-1.2`` so a future
    model_id-tagged 1.2.1 / 1.3 is not mistaken for 1.2).

    Returns ``"1.1"`` / ``"1.2"`` for a recognized generation; ``"unidentified"``
    when neither field is set (the UMA 1.0 / untagged signature, rejected on
    load); ``"unknown_uma"`` for a UMA backbone matching nothing known (future
    release or user-customized ``model_id``); ``"not_uma"`` otherwise.
    """
    if not isinstance(model_config, (dict, DictConfig)):
        return "not_uma"
    backbone = model_config.get("backbone", {})
    model = backbone.get("model") if isinstance(backbone, (dict, DictConfig)) else None
    if not isinstance(model, str) or not (
        model == _UMA_BACKBONE_SHORT_NAME or model.endswith(_UMA_BACKBONE_FQN_SUFFIX)
    ):
        return "not_uma"

    model_id = model_config.get("model_id")
    model_id = model_id.strip() if isinstance(model_id, str) else ""
    model_version = backbone.get("model_version")

    if not model_id and model_version is None:
        return "unidentified"  # UMA 1.0 (ships with neither) or untagged-new
    if not model_id:  # legacy: identified by model_version (only 1.1 ships this way)
        return "1.1" if model_version in (1.1, "1.1") else "unknown_uma"
    return "1.2" if model_id.endswith("-1.2") else "unknown_uma"


def _raise_unidentified_uma(
    model_config: dict | DictConfig,
    checkpoint_location: str | None,
) -> None:
    path_str = (
        checkpoint_location if checkpoint_location is not None else "<unknown path>"
    )
    raise RuntimeError(
        f"UMA checkpoint at {path_str!r} has no model_id. If this is a deprecated "
        f"UMA 1.0 checkpoint, install fairchem-core<={_LAST_UMA_1P0_FAIRCHEM_VERSION}; "
        f"otherwise set a model_id on the model config (e.g. model_id: UMA-1.2.1)."
    )


def _backfill_uma_1p1_model_id(model_config: dict | DictConfig) -> None:
    """Set ``model_id = "UMA-1.1"`` in-memory (transient, not persisted).

    Only reached for a UMA 1.1 checkpoint, which by construction has no model_id.
    """
    if isinstance(model_config, DictConfig):
        with open_dict(model_config):
            OmegaConf.update(model_config, "model_id", UMA_1P1_MODEL_ID, force_add=True)
    else:
        model_config["model_id"] = UMA_1P1_MODEL_ID


def apply_uma_compat_fixups(
    checkpoint: MLIPInferenceCheckpoint,
    checkpoint_location: str | None = None,
) -> None:
    """Reject untagged checkpoints and back-fill the UMA 1.1 ``model_id``.

    The only legacy fixup needed: UMA 1.1 ships without a ``model_id``, so it is
    back-filled to ``"UMA-1.1"`` in-memory. UMA 1.2+ are already tagged → no-op;
    non-UMA → no-op. A checkpoint with neither ``model_id`` nor ``model_version``
    (UMA 1.0 / untagged) is rejected. Idempotent. ``include_self`` is derived
    from ``model_id`` by the backbone, not here.
    """
    model_config = getattr(checkpoint, "model_config", None)
    version = get_uma_version(model_config)

    if version == "unidentified":
        _raise_unidentified_uma(model_config, checkpoint_location)
    if version == "1.1":
        _backfill_uma_1p1_model_id(model_config)
    # 1.2 / 1.2.1+ / unknown_uma / not_uma: no-op.
