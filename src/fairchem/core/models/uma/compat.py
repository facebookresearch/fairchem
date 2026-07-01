"""UMA checkpoint compatibility shim.

Classifies a loaded checkpoint by UMA generation (from ``model_id``, falling back
to ``backbone.model_version`` for legacy 1.1) and fixes up its ``model_config``
in place at load time (in ``load_inference_model``, before instantiation):

* **UMA 1.1** ships without a ``model_id`` — back-fill it to ``"UMA-1.1"``.
* **UMA 1.2+** are already tagged, non-UMA is ignored — no-op.

The MoE ``include_self`` bug is not handled here: the backbone derives it from
``model_id`` (``eSCNMDMoeBackbone.set_MOLE_coefficients``; only 1.2 uses True).

Transitional: remove the 1.1 back-fill once UMA 1.1 is deprecated, and delete
this module entirely once both UMA 1.1 and 1.2 are deprecated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint


UmaVersion = Literal["not_uma", "unidentified", "1.1", "tagged"]

UMA_1P1_MODEL_ID = "UMA-1.1"


# Last fairchem-core release that supports UMA 1.0 (git describe --tags on
# this repo: fairchem_core-2.21.0-2-g28c7f4ba6).
_LAST_UMA_1P0_FAIRCHEM_VERSION = "2.21.0"

# Hydra registry short-name for the UMA MoE backbone.
_UMA_BACKBONE_SHORT_NAME = "escnmd_moe_backbone"
# Suffix of the fully-qualified backbone class path used in shipped checkpoints.
_UMA_BACKBONE_FQN_SUFFIX = "uma.escn_moe.eSCNMDMoeBackbone"


def get_uma_version(model_config: dict | None) -> UmaVersion:
    """Classify what fix-up a checkpoint needs (see :func:`apply_uma_compat_fixups`).

    * ``"not_uma"`` — not a UMA backbone.
    * ``"1.1"`` — legacy UMA 1.1: no ``model_id`` but ``backbone.model_version ==
      1.1`` → back-fill ``model_id``.
    * ``"unidentified"`` — UMA backbone with neither ``model_id`` nor
      ``model_version`` (UMA 1.0 / untagged-new) → rejected on load.
    * ``"tagged"`` — already has a ``model_id`` (UMA 1.2+ or custom) → no-op. The
      1.2 ``include_self`` rule lives in the backbone, keyed on ``model_id``.
    """
    if not isinstance(model_config, dict):
        return "not_uma"
    backbone = model_config.get("backbone", {})
    model = backbone.get("model") if isinstance(backbone, dict) else None
    if not isinstance(model, str) or not (
        model == _UMA_BACKBONE_SHORT_NAME or model.endswith(_UMA_BACKBONE_FQN_SUFFIX)
    ):
        return "not_uma"

    model_id = model_config.get("model_id")
    if isinstance(model_id, str) and model_id.strip():
        return "tagged"  # 1.2+ / custom — backbone reads model_id for include_self
    if backbone.get("model_version") in (1.1, "1.1"):
        return "1.1"  # legacy: no model_id, model_version says 1.1
    return "unidentified"  # UMA 1.0 / untagged (no model_id, no model_version)


def _raise_unidentified_uma(checkpoint_location: str | None) -> None:
    path_str = (
        checkpoint_location if checkpoint_location is not None else "<unknown path>"
    )
    raise RuntimeError(
        f"UMA checkpoint at {path_str!r} has no model_id. If this is a deprecated "
        f"UMA 1.0 checkpoint, install fairchem-core<={_LAST_UMA_1P0_FAIRCHEM_VERSION}; "
        f"otherwise set a model_id on the model config (e.g. model_id: UMA-1.2.1)."
    )


def apply_uma_compat_fixups(
    checkpoint: MLIPInferenceCheckpoint,
    checkpoint_location: str | None = None,
) -> None:
    """Reject untagged checkpoints and back-fill the UMA 1.1 ``model_id``.

    The only legacy fixup needed: UMA 1.1 ships without a ``model_id``, so it is
    back-filled to ``"UMA-1.1"`` in-memory (transient, not persisted). UMA 1.2+
    are already tagged → no-op; non-UMA → no-op. A checkpoint with neither
    ``model_id`` nor ``model_version`` (UMA 1.0 / untagged) is rejected.
    Idempotent. ``include_self`` is derived from ``model_id`` by the backbone.
    """
    model_config = getattr(checkpoint, "model_config", None)
    version = get_uma_version(model_config)

    if version == "unidentified":
        _raise_unidentified_uma(checkpoint_location)
    if version == "1.1":
        # Back-fill model_id (UMA 1.1 by construction has none).
        model_config["model_id"] = UMA_1P1_MODEL_ID
    # "tagged" (1.2+) / "not_uma": no-op.
