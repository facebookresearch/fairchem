"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import hydra
from omegaconf import OmegaConf

# Allowlisted module prefixes whose callables may be used as an instantiate
# ``_target_``. This is a positive allowlist: any target that does not fall
# under one of these prefixes is refused. It complements hydra's built-in
# blocklist (a denylist of known-dangerous callables, always on in
# hydra-core>=1.3.4) with a stricter allowlist for configs that originate from
# untrusted sources, such as downloaded checkpoints.
#
# ``fairchem`` (rather than ``fairchem.core``) is used so that other fairchem
# namespace packages (e.g. fairchem.data, fairchem.applications) keep working.
#
# NOTE: hydra plans to add a native ``target_whitelist``/``_target_whitelist_``
# API; once released, this wrapper can delegate to it instead.
_SAFE_TARGET_PREFIXES = (
    "fairchem",
    "torch",
    "torchtnt",
    "ase",
)


class UnsafeTargetError(Exception):
    """
    Raised when a config contains an instantiate ``_target_`` that is not in
    the allowlist.
    """


def _is_allowed_target(target: str) -> bool:
    return any(
        target == prefix or target.startswith(prefix + ".")
        for prefix in _SAFE_TARGET_PREFIXES
    )


def _iter_targets(node: Any):
    """
    Recursively yield every ``_target_`` string found in a config tree.

    Args:
        node: A mapping, sequence, or scalar from a (possibly nested) config.

    Yields:
        Each ``_target_`` value encountered anywhere in the tree.
    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key == "_target_" and isinstance(value, str):
                yield value
            else:
                yield from _iter_targets(value)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for item in node:
            yield from _iter_targets(item)


def validate_config_targets(config: Any) -> None:
    """
    Validate every ``_target_`` in a config tree against the allowlist.

    Recursion is independent of hydra's ``_recursive_`` flag, so nested targets
    are always checked even when recursive instantiation is disabled.

    Args:
        config: A hydra/OmegaConf config (or plain dict/list) to validate.

    Raises:
        UnsafeTargetError: If any ``_target_`` is not under an allowed prefix.
    """
    if OmegaConf.is_config(config):
        container = OmegaConf.to_container(config, resolve=False)
    else:
        container = config

    for target in _iter_targets(container):
        if not _is_allowed_target(target):
            raise UnsafeTargetError(
                f"Refusing to instantiate target '{target}': it is not in the "
                f"allowlist. Only targets under {_SAFE_TARGET_PREFIXES} may be "
                f"instantiated. If this is a legitimate target, add its module "
                f"prefix to _SAFE_TARGET_PREFIXES in fairchem.core.common.safe_hydra."
            )


def safe_instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Validate a config's ``_target_`` values against the allowlist, then
    instantiate it with hydra.

    This is a drop-in replacement for ``hydra.utils.instantiate`` intended for
    configs that may originate from untrusted sources (e.g. downloaded
    checkpoints), guarding against arbitrary code execution via crafted
    ``_target_`` entries.

    Args:
        config: The config to instantiate.
        *args: Positional arguments forwarded to ``hydra.utils.instantiate``.
        **kwargs: Keyword arguments forwarded to ``hydra.utils.instantiate``.

    Returns:
        The instantiated object(s).

    Raises:
        UnsafeTargetError: If the config contains a disallowed ``_target_``.
    """
    validate_config_targets(config)
    return hydra.utils.instantiate(config, *args, **kwargs)
