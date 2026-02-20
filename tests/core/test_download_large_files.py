"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from fairchem.core.scripts import download_large_files as dl_large


@patch.object(dl_large, "urlretrieve")
def test_download_large_files(url_mock):
    def urlretrieve_mock(x, y):
        if not os.path.exists(os.path.dirname(y)):
            raise ValueError(
                f"The path to {y} does not exist. fairchem directory structure has changed,"
            )

    url_mock.side_effect = urlretrieve_mock
    dl_large.download_file_group("ALL")


@pytest.mark.serial()
def test_download_large_files_round_trip():
    """
    Verify download_file_group actually downloads missing files.
    """
    fc_root = dl_large.fairchem_root()
    install_dir = fc_root.parent.parent
    targets = [install_dir / f for f in dl_large.FILE_GROUPS["adsorbml"]]

    if any(f.exists() for f in targets):
        pytest.fail("adsorbml files already present — expected them to be absent")

    try:
        dl_large.download_file_group("adsorbml")
        for f in targets:
            assert f.exists()
            assert f.stat().st_size > 0
    finally:
        for f in targets:
            f.unlink(missing_ok=True)
