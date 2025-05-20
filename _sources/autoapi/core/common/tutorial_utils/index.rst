core.common.tutorial_utils
==========================

.. py:module:: core.common.tutorial_utils

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.common.tutorial_utils.describe_fairchem
   core.common.tutorial_utils.train_test_val_split


Module Contents
---------------

.. py:function:: describe_fairchem()

   Print some system information that could be useful in debugging.


.. py:function:: train_test_val_split(ase_db, ttv=(0.8, 0.1, 0.1), files=('train.db', 'test.db', 'val.db'), seed=42)

   Split an ase db into train, test and validation dbs.

   ase_db: path to an ase db containing all the data.
   ttv: a tuple containing the fraction of train, test and val data. This will be normalized.
   files: a tuple of filenames to write the splits into. An exception is raised if these exist.
          You should delete them first.
   seed: an integer for the random number generator seed

   Returns the absolute path to files.


