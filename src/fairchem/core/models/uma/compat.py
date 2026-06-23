"""UMA-specific checkpoint compatibility shim.

This module classifies a loaded :class:`MLIPInferenceCheckpoint` by UMA
generation and applies in-place fixups to the checkpoint's ``model_config``
before the model is instantiated. Two concrete behaviors:

* **UMA 1.0 is deprecated and raises.** UMA 1.0 has a known semantic
  divergence with later releases (the ``include_self`` flag in the
  composition reduction at ``escn_moe.py``'s ``set_MOLE_coefficients``
  branches on ``np.isclose(self.model_version, 1.0)``). Silent loads
  would produce numerically different results than UMA 1.1/1.2, so we
  hard-fail rather than ship a deprecation cycle.
* **UMA 1.1 gets ``model_id`` back-filled** to the constant ``"UMA-1.1"``
  if missing. Shipped UMA 1.1 checkpoints carry ``backbone.model_version=1.1``
  but no top-level ``model_id`` (only 1.2 does). Back-filling here makes
  ``HydraModel.model_id`` reflect the generation for downstream consumers
  (logging, serving, surgery scripts).

Sub-size (S / M / L) is intentionally not encoded — inside a single major.minor
release UMA variants are similar enough that downstream code does not need
to dispatch on size. The back-fill is the bare ``"UMA-1.1"``.

Call sites
----------
The fixup must run at every raw I/O boundary that loads a checkpoint:

1. :func:`fairchem.core.units.mlip_unit.utils.load_inference_model` — the
   generic Hydra loader (safety net for any direct caller).
2. :class:`fairchem.core.units.mlip_unit.predict.MLIPPredictUnit` —
   ``MLIPPredictUnit.__init__`` does its own ``torch.load`` and reads the
   config before delegating to ``load_inference_model``. Hooking here means
   UMA 1.0 raises before the ~1 GB tensor allocation, and downstream calls
   see the patched config.
3. :func:`fairchem.core.units.mlip_unit.mlip_unit.convert_train_checkpoint_to_inference_checkpoint`
   — converts a DCP train checkpoint into a fresh inference checkpoint. The
   fixup runs on the new ``MLIPInferenceCheckpoint`` before it is written to
   disk, so finetune-derived inference checkpoints come out correctly tagged.

:func:`apply_uma_compat_fixups` is idempotent — calling it from all three
sites (and accidentally multiple times) is safe.

Override-bypass policy
----------------------
The fixup runs *before* any caller-supplied ``overrides`` are merged into
``model_config``. A user cannot bypass the UMA 1.0 gate by passing
``overrides={"backbone": {"model_version": 1.1}}``. A user CAN, post-fixup,
force a different ``model_id`` via ``overrides={"model_id": "MY-ID"}``.

Known gaps
----------
* ``load_tasks`` (utils.py) does its own ``torch.load`` but only
  instantiates tasks, not the model — out of scope for this gate.
* ``MLIPTrainEvalUnit._execute_load_state`` resumes a DCP train run via
  ``dcp.load`` directly into an already-instantiated model — also out of
  scope. The first subsequent ``save_state`` routes through
  ``convert_train_checkpoint_to_inference_checkpoint`` where the fixup
  *does* fire, so the persisted inference checkpoint is still tagged.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Literal

import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint


UmaVersion = Literal["1.0", "1.1", "1.2", "unknown_uma", "not_uma"]

UMA_1P1_MODEL_ID = "UMA-1.1"

# Last fairchem-core release that supports UMA 1.0 (git describe --tags on
# this repo: fairchem_core-2.21.0-2-g28c7f4ba6).
_LAST_UMA_1P0_FAIRCHEM_VERSION = "2.21.0"

# Accepts "UMA-1.X" or "UMA-S-1.X" / "UMA-M-1.X" / "UMA-L-1.X" historical forms.
_UMA_MODEL_ID_RE = re.compile(r"^UMA(?:-[SML])?-1\.(\d+)$")

# Hydra registry short-name for the UMA MoE backbone.
_UMA_BACKBONE_SHORT_NAME = "escnmd_moe_backbone"
# Suffix of the fully-qualified backbone class path used in shipped checkpoints.
_UMA_BACKBONE_FQN_SUFFIX = "uma.escn_moe.eSCNMDMoeBackbone"


def _is_uma_backbone(backbone_model: object) -> bool:
    if not isinstance(backbone_model, str):
        return False
    if backbone_model == _UMA_BACKBONE_SHORT_NAME:
        return True
    if backbone_model.endswith(_UMA_BACKBONE_FQN_SUFFIX):
        return True
    # Defensive: resolve via the registry in case of subclasses.
    try:
        from fairchem.core.common.registry import registry
        from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone

        cls = registry.get_model_class(backbone_model)
    except Exception:
        return False
    return isinstance(cls, type) and issubclass(cls, eSCNMDMoeBackbone)


def _match_version(value: object, target: float) -> bool:
    """Return True iff ``value`` parses to a float ≈ ``target``."""
    try:
        return bool(np.isclose(float(value), target))
    except (TypeError, ValueError):
        return False


def _normalize_model_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def get_uma_version(model_config: dict | DictConfig | None) -> UmaVersion:
    """Classify a checkpoint's ``model_config`` by UMA generation.

    Returns one of:

    * ``"1.0"`` / ``"1.1"`` / ``"1.2"`` — a recognized UMA generation.
    * ``"unknown_uma"`` — a UMA backbone with a ``model_id`` or
      ``model_version`` that does not match any known generation
      (e.g. a future UMA 1.3 release).
    * ``"not_uma"`` — not a UMA checkpoint (esen, AllScAIP, ...), a
      missing/corrupt ``model_config``, or no backbone declared.
    """
    if not isinstance(model_config, (dict, DictConfig)):
        return "not_uma"

    backbone = model_config.get("backbone", {})
    if not isinstance(backbone, (dict, DictConfig)):
        return "not_uma"

    if not _is_uma_backbone(backbone.get("model")):
        return "not_uma"

    mid = _normalize_model_id(model_config.get("model_id"))
    if mid is not None:
        match = _UMA_MODEL_ID_RE.match(mid)
        if match:
            minor = int(match.group(1))
            if minor == 1:
                return "1.1"
            if minor == 2:
                return "1.2"
            logging.warning(
                f"Unknown UMA minor version in model_id={mid!r}; "
                "treating as unknown_uma."
            )
            return "unknown_uma"
        # A user-customized model_id is respected — we don't reclassify
        # the checkpoint based on backbone.model_version in that case
        # (the user has explicitly named their derivative).
        return "unknown_uma"

    mv = backbone.get("model_version")
    if _match_version(mv, 1.1):
        return "1.1"
    if _match_version(mv, 1.2):
        return "1.2"
    if mv is None or _match_version(mv, 1.0):
        return "1.0"
    logging.warning(
        f"Unknown UMA backbone.model_version={mv!r}; treating as unknown_uma."
    )
    return "unknown_uma"


def _raise_uma_1p0(
    model_config: dict | DictConfig,
    checkpoint_location: str | None,
) -> None:
    mid = model_config.get("model_id") if isinstance(model_config, (dict, DictConfig)) else None
    backbone = model_config.get("backbone", {}) if isinstance(model_config, (dict, DictConfig)) else {}
    mv = backbone.get("model_version") if isinstance(backbone, (dict, DictConfig)) else None
    path_str = checkpoint_location if checkpoint_location is not None else "<unknown path>"
    raise RuntimeError(
        f"UMA 1.0 checkpoints are no longer supported in this version of "
        f"fairchem-core. Detected UMA 1.0 from checkpoint at {path_str!r} "
        f"(model_id={mid!r}, backbone.model_version={mv!r}).\n"
        f"\n"
        f"UMA 1.0 has a known semantic divergence in "
        f"`fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone` (the "
        f"composition-reduction `include_self` flag branches on "
        f"`np.isclose(self.model_version, 1.0)`). Loading 1.0 weights with "
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

    if version == "1.0":
        _raise_uma_1p0(model_config, checkpoint_location)

    if version == "1.1":
        mutated = _backfill_uma_1p1_model_id(model_config)
        if mutated:
            logging.warning(
                f"UMA 1.1 checkpoint at {checkpoint_location!r} had no "
                f"model_id; back-filled model_id={UMA_1P1_MODEL_ID!r}. "
                "This will be persisted to disk if the checkpoint is "
                "subsequently saved (e.g. at the end of a finetune)."
            )
        return

    if version == "1.2":
        mid = model_config.get("model_id") if isinstance(model_config, (dict, DictConfig)) else None
        logging.info(
            f"Loaded UMA 1.2 checkpoint: model_id={mid!r}, "
            f"path={checkpoint_location!r}"
        )
        return

    if version == "unknown_uma":
        logging.warning(
            f"UMA checkpoint at {checkpoint_location!r} could not be classified "
            "into a known generation (1.1 or 1.2); no fixups applied."
        )
        return

    # not_uma: silent no-op.
