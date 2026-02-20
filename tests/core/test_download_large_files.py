"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from unittest.mock import patch

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
