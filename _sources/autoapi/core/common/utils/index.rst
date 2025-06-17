core.common.utils
=================

.. py:module:: core.common.utils

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.common.utils.DEFAULT_ENV_VARS


Classes
-------

.. autoapisummary::

   core.common.utils.UniqueKeyLoader
   core.common.utils.SeverityLevelBetween


Functions
---------

.. autoapisummary::

   core.common.utils.conditional_grad
   core.common.utils._import_local_file
   core.common.utils.setup_experimental_imports
   core.common.utils._get_project_root
   core.common.utils.setup_imports
   core.common.utils.debug_log_entry_exit
   core.common.utils.setup_logging
   core.common.utils.setup_env_vars
   core.common.utils._resolve_scale_factor_submodule
   core.common.utils._report_incompat_keys
   core.common.utils.match_state_dict
   core.common.utils.load_state_dict
   core.common.utils.get_commit_hash
   core.common.utils.get_commit_hash_for_repo
   core.common.utils.load_model_and_weights_from_checkpoint
   core.common.utils.get_timestamp_uid
   core.common.utils.tensor_stats
   core.common.utils.get_weight_table
   core.common.utils.get_checkpoint_format
   core.common.utils.get_deep
   core.common.utils.get_subdirectories_sorted_by_time


Module Contents
---------------

.. py:data:: DEFAULT_ENV_VARS

.. py:class:: UniqueKeyLoader(stream)

   Bases: :py:obj:`yaml.SafeLoader`


   .. py:method:: construct_mapping(node, deep=False)


.. py:function:: conditional_grad(dec)

   Decorator to enable/disable grad depending on whether force/energy predictions are being made


.. py:function:: _import_local_file(path: pathlib.Path, *, project_root: pathlib.Path) -> None

   Imports a Python file as a module

   :param path: The path to the file to import
   :type path: Path
   :param project_root: The root directory of the project (i.e., the "ocp" folder)
   :type project_root: Path


.. py:function:: setup_experimental_imports(project_root: pathlib.Path) -> None

   Import selected directories of modules from the "experimental" subdirectory.

   If a file named ".include" is present in the "experimental" subdirectory,
   this will be read as a list of experimental subdirectories whose module
   (including in any subsubdirectories) should be imported.

   :param project_root: The root directory of the project (i.e., the "ocp" folder)


.. py:function:: _get_project_root() -> pathlib.Path

   Gets the root folder of the project (the "ocp" folder)
   :return: The absolute path to the project root.


.. py:function:: setup_imports(config: dict | None = None) -> None

.. py:function:: debug_log_entry_exit(func)

.. py:class:: SeverityLevelBetween(min_level: int, max_level: int)

   Bases: :py:obj:`logging.Filter`


   Filter instances are used to perform arbitrary filtering of LogRecords.

   Loggers and Handlers can optionally use Filter instances to filter
   records as desired. The base filter class only allows events which are
   below a certain point in the logger hierarchy. For example, a filter
   initialized with "A.B" will allow events logged by loggers "A.B",
   "A.B.C", "A.B.C.D", "A.B.D" etc. but not "A.BB", "B.A.B" etc. If
   initialized with the empty string, all events are passed.


   .. py:attribute:: min_level


   .. py:attribute:: max_level


   .. py:method:: filter(record) -> bool

      Determine if the specified record is to be logged.

      Returns True if the record should be logged, or False otherwise.
      If deemed appropriate, the record may be modified in-place.



.. py:function:: setup_logging() -> None

.. py:function:: setup_env_vars() -> None

.. py:function:: _resolve_scale_factor_submodule(model: torch.nn.Module, name: str)

.. py:function:: _report_incompat_keys(model: torch.nn.Module, keys: torch.nn.modules.module._IncompatibleKeys, strict: bool = False) -> tuple[list[str], list[str]]

.. py:function:: match_state_dict(model_state_dict: collections.abc.Mapping[str, torch.Tensor], checkpoint_state_dict: collections.abc.Mapping[str, torch.Tensor]) -> dict

.. py:function:: load_state_dict(module: torch.nn.Module, state_dict: collections.abc.Mapping[str, torch.Tensor], strict: bool = True) -> tuple[list[str], list[str]]

.. py:function:: get_commit_hash() -> str

.. py:function:: get_commit_hash_for_repo(git_repo_path: str) -> str | None

.. py:function:: load_model_and_weights_from_checkpoint(checkpoint_path: str) -> torch.nn.Module

.. py:function:: get_timestamp_uid() -> str

.. py:function:: tensor_stats(name: str, x: torch.Tensor) -> dict

.. py:function:: get_weight_table(model: torch.nn.Module) -> tuple[list, list]

.. py:function:: get_checkpoint_format(config: dict) -> str

.. py:function:: get_deep(dictionary: dict, keys: str, default: str | None = None)

.. py:function:: get_subdirectories_sorted_by_time(directory: str) -> list

   Get all subdirectories in a directory sorted by their last modification time.
   :param directory: The path to the directory to search.
   :type directory: str

   :returns: A list of tuples containing the subdirectory path and its last modification time.
   :rtype: list


