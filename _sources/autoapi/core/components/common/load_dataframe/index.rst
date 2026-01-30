core.components.common.load_dataframe
=====================================

.. py:module:: core.components.common.load_dataframe

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.components.common.load_dataframe.load_json_to_df


Module Contents
---------------

.. py:function:: load_json_to_df(path: str, index_name: str | None, index_rename: str | None = None, sort_index: bool = True) -> pandas.DataFrame

   Read a json file into a pandas DataFrame, optionally reset the index and sort

   :param path: path to json or compressed json file (ie json.gz)
   :param index_name: name of column to set as index.
   :param index_rename: if given will rename the index to this
   :param sort_index: sort the dataframe by its index

   :returns: pd.Dataframe


