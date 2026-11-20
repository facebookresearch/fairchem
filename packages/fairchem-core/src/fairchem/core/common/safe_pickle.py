"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import io
import logging
import pickle
import warnings

logger = logging.getLogger(__name__)

# Allowlisted modules whose classes may be unpickled.
# Only standard scientific-Python and ASE types are permitted.
_SAFE_MODULE_PREFIXES = (
    "builtins",
    "collections",
    "numpy",
    "pandas",
    "ase",
    "pymatgen",
    "torch",
    "fairchem",
    "scipy",
    "datetime",
    "pathlib",
    "re",
    "enum",
    "copy",
    "decimal",
    "fractions",
    "numbers",
    "operator",
    "functools",
    "itertools",
    "spglib",
)


class _RestrictedUnpickler(pickle.Unpickler):
    """
    A restricted unpickler that only allows classes from known-safe modules.

    This prevents arbitrary code execution via crafted pickle files by
    refusing to instantiate classes from modules outside the allowlist.
    """

    def find_class(self, module: str, name: str) -> type:
        if not any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in _SAFE_MODULE_PREFIXES
        ):
            raise pickle.UnpicklingError(
                f"Refusing to unpickle class {module}.{name}: "
                f"module '{module}' is not in the allowlist. "
                f"If this is a legitimate class, add its module prefix to "
                f"_SAFE_MODULE_PREFIXES in fairchem.core.common.safe_pickle."
            )
        return super().find_class(module, name)


def safe_pickle_load(file_or_path, *, warn: bool = True):
    """
    Load a pickle file using a restricted unpickler that only allows
    classes from known-safe modules.

    Args:
        file_or_path: A file-like object (open in 'rb' mode) or a file path string.
        warn: If True, emit a deprecation warning recommending safer formats.

    Returns:
        The deserialized object.

    Raises:
        pickle.UnpicklingError: If the pickle contains classes from disallowed modules.
    """
    if warn:
        warnings.warn(
            "Loading data from a pickle file. Pickle files can execute arbitrary code "
            "and should only be loaded from trusted sources. Consider migrating to a "
            "safer format such as Parquet, CSV, or JSON.",
            stacklevel=2,
        )

    if isinstance(file_or_path, (str, bytes)):
        from pathlib import Path

        with open(Path(file_or_path), "rb") as f:
            return _RestrictedUnpickler(f).load()
    elif isinstance(file_or_path, (io.IOBase, io.RawIOBase, io.BufferedIOBase)):
        return _RestrictedUnpickler(file_or_path).load()
    else:
        # Assume it's a file-like object
        return _RestrictedUnpickler(file_or_path).load()
