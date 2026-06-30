"""UMA-specific checkpoint compatibility shim.

This module classifies a loaded :class:`MLIPInferenceCheckpoint` by UMA
generation and applies in-place fixups to the checkpoint's ``model_config``
before the model is instantiated. ``model_id`` is the source of truth for the
generation. Concrete behaviors:

* **Unidentified / UMA 1.0 are rejected.** A UMA backbone with neither a
  ``model_id`` nor a ``backbone.model_version`` cannot be classified — this is
  also the on-disk signature of a deprecated UMA 1.0 checkpoint — so it raises
  with an actionable message (set ``model_id``, or ``pip install
  'fairchem-core<=2.21.0'`` to use a genuine 1.0). An *explicit*
  ``model_version == 1.0`` raises with the 1.0-specific deprecation message.
* **UMA 1.1 gets ``model_id`` back-filled** to ``"UMA-1.1"`` if missing.
  Shipped UMA 1.1 checkpoints carry ``backbone.model_version=1.1`` but no
  top-level ``model_id`` (1.2+ ships ``model_id``). Back-filling makes
  ``HydraModel.model_id`` reflect the generation for downstream consumers. This
  is a transitional shim: deprecate/remove once UMA 1.1 is unsupported.
* **``include_self_bug`` is set per generation** on the backbone config. The
  ``include_self`` flag in the MoE composition reduction
  (``escn_moe.eSCNMDMoeBackbone.set_MOLE_coefficients``) differs per generation.
  The shim writes the authoritative value into ``backbone.include_self_bug``
  before instantiation, so the backbone consumes a plain bool and never inspects
  the version itself:

  | Generation | Identified by | ``include_self_bug`` |
  |---|---|---|
  | 1.1 | ``model_version == 1.1`` (model_id back-filled) | ``False`` |
  | 1.2 | ``model_id`` (e.g. ``UMA-S-1.2``) | ``True`` |
  | 1.2.1+ | ``model_id`` (e.g. ``UMA-1.2.1``) | ``False`` |

  Only UMA 1.2 uses ``True`` — the shim sets ``include_self_bug = (version ==
  "1.2")``, where ``version`` comes from :func:`get_uma_version` (so any 1.2
  size variant — ``UMA-S-1.2`` / ``UMA-M-1.2`` / bare ``UMA-1.2`` — is covered).
  1.0 never reaches this (rejected first).

``model_version`` is no longer read for behavior — only as the legacy classifier
fallback that identifies shipped 1.1. Its backbone default is unchanged.

Call site
---------
The fixup runs at a single chokepoint:
:func:`fairchem.core.units.mlip_unit.utils.load_inference_model`, applied right
after ``torch.load`` and before ``hydra.utils.instantiate``. Both inference
(``MLIPPredictUnit.__init__``, via ``preloaded_checkpoint``) and finetuning
(``initialize_finetuning_model``) reach it. It runs *before* any caller
``overrides`` are merged, so a caller may still override ``model_id`` /
``backbone.include_self_bug`` post-fixup (used by tests).

Future generations
------------------
When a new generation ships, update the ``include_self_bug`` rule in
:func:`apply_uma_compat_fixups` (only 1.2 is currently special) and the
generation branches in :func:`get_uma_version`, refresh the table above +
``uma_changelog.md``, and add a test in
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


UmaVersion = Literal["1.0", "1.1", "1.2", "unidentified", "unknown_uma", "not_uma"]

UMA_1P1_MODEL_ID = "UMA-1.1"


# Last fairchem-core release that supports UMA 1.0 (git describe --tags on
# this repo: fairchem_core-2.21.0-2-g28c7f4ba6).
_LAST_UMA_1P0_FAIRCHEM_VERSION = "2.21.0"

# Hydra registry short-name for the UMA MoE backbone.
_UMA_BACKBONE_SHORT_NAME = "escnmd_moe_backbone"
# Suffix of the fully-qualified backbone class path used in shipped checkpoints.
_UMA_BACKBONE_FQN_SUFFIX = "uma.escn_moe.eSCNMDMoeBackbone"


def _is_uma_backbone(backbone_model: object) -> bool:
    return isinstance(backbone_model, str) and (
        backbone_model == _UMA_BACKBONE_SHORT_NAME
        or backbone_model.endswith(_UMA_BACKBONE_FQN_SUFFIX)
    )


def _match_version(value: object, target: float) -> bool:
    """Return True iff ``value`` equals ``target`` (handles float or str)."""
    try:
        return float(value) == target
    except (TypeError, ValueError):
        return False


def _normalize_model_id(value: object) -> str | None:
    """Return a stripped model_id, or None if absent/blank/non-string."""
    if not isinstance(value, str):
        return None
    return value.strip() or None


def get_uma_version(model_config: dict | DictConfig | None) -> UmaVersion:
    """Classify a checkpoint by UMA generation.

    Shipped signatures: 1.0 = neither field set; 1.1 = ``model_version`` only;
    1.2 = ``model_id`` only. A ``model_id`` is matched on its trailing version,
    so a future model_id-tagged 1.2.1 / 1.3 is not mistaken for 1.2.

    Returns ``"1.0"`` / ``"1.1"`` / ``"1.2"`` for a recognized generation;
    ``"unidentified"`` when neither field is set (rejected on load);
    ``"unknown_uma"`` for a UMA backbone that matches nothing known (future
    release or user-customized ``model_id``); ``"not_uma"`` otherwise.
    """
    if not isinstance(model_config, (dict, DictConfig)):
        return "not_uma"
    backbone = model_config.get("backbone", {})
    if not isinstance(backbone, (dict, DictConfig)) or not _is_uma_backbone(
        backbone.get("model")
    ):
        return "not_uma"

    model_id = _normalize_model_id(model_config.get("model_id"))
    model_version = backbone.get("model_version")

    if model_id is None and model_version is None:
        return "unidentified"  # 1.0 / untagged -> rejected on load
    if model_id is None:  # legacy: identified by model_version
        if _match_version(model_version, 1.0):
            return "1.0"
        if _match_version(model_version, 1.1):
            return "1.1"
        return "unknown_uma"
    if model_id.endswith("-1.2"):
        return "1.2"
    if model_id.endswith("-1.1"):
        return "1.1"
    return "unknown_uma"  # 1.2.1+, 1.3, or a user-customized model_id


def _raise_uma_1p0(
    model_config: dict | DictConfig,
    checkpoint_location: str | None,
) -> None:
    mid = (
        model_config.get("model_id")
        if isinstance(model_config, (dict, DictConfig))
        else None
    )
    backbone = (
        model_config.get("backbone", {})
        if isinstance(model_config, (dict, DictConfig))
        else {}
    )
    mv = (
        backbone.get("model_version")
        if isinstance(backbone, (dict, DictConfig))
        else None
    )
    path_str = (
        checkpoint_location if checkpoint_location is not None else "<unknown path>"
    )
    raise RuntimeError(
        f"UMA 1.0 checkpoints are no longer supported in this version of "
        f"fairchem-core. Detected UMA 1.0 from checkpoint at {path_str!r} "
        f"(model_id={mid!r}, backbone.model_version={mv!r}).\n"
        f"\n"
        f"UMA 1.0 has a known semantic divergence in "
        f"`fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone` (its "
        f"composition-reduction `include_self` behavior differs from 1.1/1.2). "
        f"Loading 1.0 weights with "
        f"the current code would produce numerically different results "
        f"than the original release, so this is a hard failure.\n"
        f"\n"
        f"To use this checkpoint, install the last fairchem-core release "
        f"that supported UMA 1.0:\n"
        f"    pip install 'fairchem-core<={_LAST_UMA_1P0_FAIRCHEM_VERSION}'\n"
        f"\n"
        f"To use a current release, switch to UMA 1.1 or UMA 1.2 (see "
        f"`facebook/UMA` on Hugging Face, or "
        f"`fairchem.core.calculate.pretrained_mlip.available_models`)."
    )


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


def _set_backbone_include_self_bug(
    model_config: dict | DictConfig, value: bool
) -> None:
    """Authoritatively set ``backbone.include_self_bug`` (always overwrite)."""
    backbone = model_config.get("backbone")
    if not isinstance(backbone, (dict, DictConfig)):
        return
    if isinstance(backbone, DictConfig):
        with open_dict(backbone):
            backbone["include_self_bug"] = value
    else:
        backbone["include_self_bug"] = value


def _backfill_uma_1p1_model_id(model_config: dict | DictConfig) -> bool:
    """Set ``model_config['model_id'] = "UMA-1.1"`` if absent. Returns True if mutated."""
    existing = _normalize_model_id(model_config.get("model_id"))
    if existing is not None:
        return False
    if isinstance(model_config, DictConfig):
        with open_dict(model_config):
            OmegaConf.update(
                model_config,
                "model_id",
                UMA_1P1_MODEL_ID,
                merge=False,
                force_add=True,
            )
    else:
        model_config["model_id"] = UMA_1P1_MODEL_ID
    return True


def apply_uma_compat_fixups(
    checkpoint: MLIPInferenceCheckpoint,
    checkpoint_location: str | None = None,
) -> None:
    """Classify ``checkpoint`` and apply in-place UMA-generation fixups.

    See module docstring for the full contract. Idempotent. Safe to call on
    non-UMA checkpoints (no-op).
    """
    model_config = getattr(checkpoint, "model_config", None)
    version = get_uma_version(model_config)

    if version == "not_uma":
        return

    if version == "1.0":
        _raise_uma_1p0(model_config, checkpoint_location)

    if version == "unidentified":
        _raise_unidentified_uma(model_config, checkpoint_location)

    if version == "1.1":
        mutated = _backfill_uma_1p1_model_id(model_config)
        if mutated:
            logging.warning(
                f"UMA 1.1 checkpoint at {checkpoint_location!r} had no model_id; "
                f"back-filled model_id={UMA_1P1_MODEL_ID!r}. This back-fill is "
                "in-memory only (re-derived on every load), not persisted."
            )
    elif version == "1.2":
        mid = model_config.get("model_id")
        logging.info(
            f"Loaded UMA 1.2 checkpoint: model_id={mid!r}, "
            f"path={checkpoint_location!r}"
        )
    elif version == "unknown_uma":
        logging.warning(
            f"UMA checkpoint at {checkpoint_location!r} could not be classified "
            "into a known generation; defaulting include_self_bug=False."
        )

    # Authoritatively set the per-generation MoE composition flag for every
    # surviving UMA backbone (1.1 / 1.2 / unknown_uma). Only 1.2 uses True.
    # Always overwrite so a re-saved finetune config cannot carry a stale value.
    _set_backbone_include_self_bug(model_config, version == "1.2")
