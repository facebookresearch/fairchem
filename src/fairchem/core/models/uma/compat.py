"""UMA-specific checkpoint compatibility shim.

This module classifies a loaded :class:`MLIPInferenceCheckpoint` by UMA
generation and applies in-place fixups to the checkpoint's ``model_config``
before the model is instantiated. ``model_id`` is the source of truth for the
generation. Concrete behaviors:

* **UMA 1.0 / untagged checkpoints are rejected.** UMA 1.0 ships with neither a
  ``model_id`` nor a ``backbone.model_version``; a checkpoint with neither cannot
  be classified (it is either a deprecated 1.0 or an untagged freshly-trained
  model), so it raises with an actionable message (set ``model_id``, or
  ``pip install 'fairchem-core<=2.21.0'`` to use a genuine 1.0).
* **UMA 1.1 gets ``model_id`` back-filled** to ``"UMA-1.1"`` if missing.
  Shipped UMA 1.1 checkpoints carry ``backbone.model_version=1.1`` but no
  top-level ``model_id`` (1.2+ ships ``model_id``). Back-filling makes
  ``HydraModel.model_id`` reflect the generation for downstream consumers. This
  is a transitional shim: deprecate/remove once UMA 1.1 is unsupported.

This shim does nothing else: UMA 1.2+ are already tagged (no-op) and non-UMA
checkpoints are ignored.

``include_self`` (the MoE composition-reduction quirk) is NOT handled here. It is
a property of the generation, derived by the backbone from its ``model_id``:
``eSCNMDMoeBackbone.set_MOLE_coefficients`` uses ``include_self = (model_id ==
"UMA-S-1.2")`` — only UMA 1.2 uses ``True``; 1.1 and 1.2.1+ use ``False``.
``HydraModel`` propagates ``model_id`` to the backbone after construction.

``model_version`` is no longer read for behavior — only as the legacy classifier
fallback that identifies shipped 1.1.

Call site
---------
The fixup runs at a single chokepoint:
:func:`fairchem.core.units.mlip_unit.utils.load_inference_model`, applied right
after ``torch.load`` and before ``hydra.utils.instantiate``. Both inference
(``MLIPPredictUnit.__init__``, via ``preloaded_checkpoint``) and finetuning
(``initialize_finetuning_model``) reach it. It runs *before* any caller
``overrides`` are merged, so a caller may still override ``model_id`` post-fixup
(used by tests).

Future generations
------------------
When a new generation ships, update the generation branches in
:func:`get_uma_version` (and the ``include_self`` rule in
``eSCNMDMoeBackbone.set_MOLE_coefficients`` if its composition behavior differs),
refresh ``uma_changelog.md``, and add a test in
``tests/core/models/uma/test_compat.py``.

Known gaps
----------
* ``load_tasks`` (utils.py) does its own ``torch.load`` but only instantiates
  tasks, not the model — out of scope.
* ``convert_train_checkpoint_to_inference_checkpoint`` writes a fresh inference
  checkpoint from train state *without* tagging it; identity/flag are re-derived
  in-memory when that file is later loaded through ``load_inference_model``, not
  persisted to disk.
"""

from __future__ import annotations

import logging
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
        f"UMA checkpoint identity required but not found at {path_str!r}: the "
        f"checkpoint has neither a top-level `model_id` nor a "
        f"`backbone.model_version`, so its generation cannot be determined.\n"
        f"\n"
        f"This is either (a) a freshly-trained UMA model that was not tagged, or "
        f"(b) a deprecated UMA 1.0 checkpoint (which shipped with neither field).\n"
        f"\n"
        f"To fix (a), set a `model_id` on the model config when training (e.g. "
        f"`model_id: UMA-1.2.1`), or tag an existing checkpoint in place:\n"
        f"    ckpt = torch.load(path, weights_only=False)\n"
        f"    ckpt.model_config['model_id'] = 'UMA-1.2.1'\n"
        f"    torch.save(ckpt, path)\n"
        f"\n"
        f"For (b), install the last fairchem-core release that supported UMA 1.0:\n"
        f"    pip install 'fairchem-core<={_LAST_UMA_1P0_FAIRCHEM_VERSION}'"
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
        logging.warning(
            f"UMA 1.1 checkpoint at {checkpoint_location!r} had no model_id; "
            f"back-filled model_id={UMA_1P1_MODEL_ID!r} (in-memory, not persisted)."
        )
    # 1.2 / 1.2.1+ / unknown_uma / not_uma: no-op.
