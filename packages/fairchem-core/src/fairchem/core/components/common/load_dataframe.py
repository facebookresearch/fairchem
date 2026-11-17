"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pandas as pd
from monty.serialization import loadfn


def load_json_to_df(
    path: str,
    index_name: str | None,
    index_rename: str | None = None,
    sort_index: bool = True,
) -> pd.DataFrame:
    """Read a json file into a pandas DataFrame, optionally reset the index and sort

    Args:
        path: path to json or compressed json file (ie json.gz)
        index_name: name of column to set as index.
        index_rename: if given will rename the index to this
        sort_index: sort the dataframe by its index

    Returns:
        pd.Dataframe
    """

    obj = loadfn(path)
    dataframe = pd.DataFrame(obj)

    if index_name is not None:
        dataframe = dataframe.set_index(index_name)

    if index_rename is not None:
        dataframe.index.name = index_name

    if sort_index:
        dataframe = dataframe.sort_index()

    return dataframe
