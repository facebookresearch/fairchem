"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Which checkpoints UMA Nitro admits.

Nitro's kernels hard-code uma-s-1p2p1's architecture, so a checkpoint is
admitted only when the config HydraModel would build from it matches that
architecture (``config_mismatches``) and its MD5 is a known one (``admit``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fairchem.core.common.checksums import cached_file_md5
from fairchem.core.units.mlip_unit.api.inference import (
    NITRO_CHECKPOINT,
    NITRO_MODEL_ID,
    UMANitroNotSupportedError,
)
from fairchem.core.units.mlip_unit.utils import update_configs

_SUPPORTED_CHECKPOINTS = {
    "3497615fd30a24c5b35cd3b41a682e6e": NITRO_CHECKPOINT,
    "eb6fed512b3f0801b27dfdd6e8c1f9ee": "uma-s-1p2p1-ft-d4rev",
    "fc041fe67c6e1501bf8b71cc490826f7": "uma-s-1p2p1-ft-r2scan",
}

SUPPORTED_PROPERTIES = frozenset({"energy", "forces", "stress"})

_ANY = object()

_REFERENCE_CONFIG = {
    "_target_": "fairchem.core.models.base.HydraModel",
    "model_id": NITRO_MODEL_ID,
    "pass_through_head_outputs": True,
    "supports_single_atoms": _ANY,
    "backbone": {
        "model": "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone",
        "max_num_elements": 100,
        "sphere_channels": 128,
        "hidden_channels": 128,
        "edge_channels": 128,
        "lmax": 2,
        "mmax": 2,
        "num_layers": 4,
        "cutoff": 6,
        "max_neighbors": 300,
        "num_distance_basis": 32,
        "distance_function": "gaussian",
        "use_pbc": True,
        "use_pbc_single": True,
        "direct_forces": False,
        "regress_forces": True,
        "regress_stress": True,
        "norm_type": "rms_norm_sh",
        "act_type": "gate",
        "ff_type": "spectral",
        "chg_spin_emb_type": "rand_emb",
        "charge_balanced_channels": [0, 1, 2],
        "moe_layer_type": "pytorch",
        "use_composition_embedding": True,
        "use_global_embedding": False,
        "dataset_mapping": _ANY,
        "otf_graph": _ANY,
        "always_use_pbc": _ANY,
        "composition_dropout": _ANY,
        "moe_dropout": _ANY,
        "dataset_emb_grad": _ANY,
        "cs_emb_grad": _ANY,
        "num_experts": _ANY,
    },
}

_WRAPPED_HEAD = {
    "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificMoEWrapper",
    "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
    "head_kwargs": {"wrap_property": False},
    "dataset_mapping": _ANY,
}
_BARE_HEAD = {"module": "fairchem.core.models.uma.escn_md.MLP_EFS_Head"}
_HEAD_SHAPES = {head["module"]: head for head in (_WRAPPED_HEAD, _BARE_HEAD)}


def _differences(config: Mapping, reference: Mapping, prefix: str = "") -> list[str]:
    """
    Describe every key of ``config`` that differs from ``reference``.

    Args:
        config: A checkpoint config, or a section of one.
        reference: The matching section of a reference config.
        prefix: The dotted path of ``config`` in the checkpoint config.

    Returns:
        One readable entry per differing, missing or unexpected key.
    """
    differences = []
    for key in sorted(set(config) | set(reference), key=str):
        path = f"{prefix}{key}"
        if key not in reference:
            differences.append(f"{path}={config[key]!r} (unexpected)")
        elif reference[key] is _ANY:
            continue
        elif key not in config:
            differences.append(f"{path} missing (expected {reference[key]!r})")
        elif isinstance(reference[key], Mapping) and isinstance(config[key], Mapping):
            differences += _differences(config[key], reference[key], f"{path}.")
        elif config[key] != reference[key]:
            differences.append(f"{path}={config[key]!r} (expected {reference[key]!r})")
    return differences


def config_mismatches(model_config: Mapping, tasks_config: Sequence) -> list[str]:
    """
    Describe how a checkpoint's config differs from what Nitro computes.

    Args:
        model_config: The checkpoint's model config.
        tasks_config: The checkpoint's task configs.

    Returns:
        Readable mismatches such as ``backbone.cutoff=5 (expected 6)``; empty
        when Nitro computes what HydraModel would.
    """
    model = {key: value for key, value in model_config.items() if key != "heads"}
    mismatches = _differences(model, _REFERENCE_CONFIG)
    heads = model_config.get("heads") or {}
    if len(heads) != 1:
        return [*mismatches, f"heads has {len(heads)} entries (expected 1)"]
    ((name, head),) = heads.items()
    shape = _HEAD_SHAPES.get(head.get("module"), _WRAPPED_HEAD)
    mismatches += _differences(head, shape, f"heads.{name}.")
    mappings = [model_config.get("backbone", {}).get("dataset_mapping") or {}]
    if shape is _WRAPPED_HEAD:
        mappings.append(head.get("dataset_mapping") or {})
    for task in tasks_config:
        if task["property"] not in SUPPORTED_PROPERTIES:
            mismatches.append(
                f"task {task['name']!r} predicts {task['property']!r} (expected "
                "energy, forces or stress)"
            )
        mismatches += [
            f"task {task['name']!r} uses dataset {dataset!r}, which a "
            "dataset_mapping lacks"
            for dataset in task["datasets"]
            if not all(dataset in mapping for mapping in mappings)
        ]
    return mismatches


def admit(checkpoint, path, overrides: dict | None = None) -> None:
    """
    Check that Nitro computes what HydraModel would for this checkpoint.

    The config is compared before the file is hashed, since the comparison is
    free and the hash reads the whole file. The MD5 sum is cached in a
    ``<path>.md5`` sidecar (``cached_file_md5``).

    Args:
        checkpoint: The loaded ``MLIPInferenceCheckpoint``.
        path: The checkpoint file.
        overrides: Config overrides the predictor applies to HydraModel.

    Raises:
        UMANitroNotSupportedError: If the checkpoint, with its overrides, is
            not one Nitro computes.
    """
    model_config, tasks = checkpoint.model_config, checkpoint.tasks_config
    effective = update_configs(model_config, overrides) if overrides else model_config
    mismatches = config_mismatches(effective, tasks)
    if mismatches and overrides and not config_mismatches(model_config, tasks):
        raise UMANitroNotSupportedError(
            f"overrides set {'; '.join(mismatches)}, which UMA Nitro's kernels "
            f"cannot compute (they hard-code {NITRO_CHECKPOINT}'s architecture)"
        )
    if mismatches:
        raise UMANitroNotSupportedError(
            f"{path} differs from {NITRO_CHECKPOINT}'s architecture, which UMA "
            f"Nitro's kernels hard-code: {'; '.join(mismatches)}"
        )
    digest = cached_file_md5(path)
    if digest not in _SUPPORTED_CHECKPOINTS:
        supported = ", ".join(
            f"{name} (md5 {checksum})"
            for checksum, name in _SUPPORTED_CHECKPOINTS.items()
        )
        raise UMANitroNotSupportedError(
            f"{path} is not supported in nitro mode (md5 {digest}); supported "
            f"checkpoints are {supported}"
        )
