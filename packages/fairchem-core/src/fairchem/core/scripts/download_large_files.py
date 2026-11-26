"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Backwards-compatible alias: this module moved to
# fairchem.core.common.download_large_files. Released fairchem-data-oc and
# fairchem-data-odac versions import it from here.

from __future__ import annotations

from fairchem.core.common.download_large_files import *  # noqa: F403
from fairchem.core.common.download_large_files import (  # noqa: F401
    download_file_group,
    fairchem_root,
    parse_args,
)

if __name__ == "__main__":
    download_file_group(parse_args().file_group)
