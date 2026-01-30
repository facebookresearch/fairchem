core.scripts.migrate_imports
============================

.. py:module:: core.scripts.migrate_imports

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.scripts.migrate_imports.mapping
   core.scripts.migrate_imports.extensions


Functions
---------

.. autoapisummary::

   core.scripts.migrate_imports.replace_strings_in_file
   core.scripts.migrate_imports.main


Module Contents
---------------

.. py:data:: mapping

.. py:data:: extensions
   :value: ['.yaml', '.py']


.. py:function:: replace_strings_in_file(file_path, replacements, dry_run)

   Replaces input strings with output strings in a given file.

   :param file_path: Path to the file to process.
   :type file_path: str
   :param replacements: Dictionary of input strings to output strings.
   :type replacements: dict
   :param dry_run: Whether to perform a dry run (print changes without making them).
   :type dry_run: bool


.. py:function:: main()

