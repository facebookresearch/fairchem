"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Backwards-compatibility shim: this utility moved to
``fairchem.core.common.download_large_files``. Import from there;
this module will be removed in a future release.
"""

from __future__ import annotations

from fairchem.core.common.download_large_files import *  # noqa: F403
from fairchem.core.common.download_large_files import (
    download_file_group,
    parse_args,
)

if __name__ == "__main__":
    args = parse_args()
    download_file_group(args.file_group)
