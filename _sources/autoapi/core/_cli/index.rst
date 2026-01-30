core._cli
=========

.. py:module:: core._cli

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core._cli.get_canonical_config
   core._cli.get_hydra_config_from_yaml
   core._cli.main


Module Contents
---------------

.. py:function:: get_canonical_config(config: omegaconf.DictConfig) -> omegaconf.DictConfig

.. py:function:: get_hydra_config_from_yaml(config_yml: str, overrides_args: list[str]) -> omegaconf.DictConfig

.. py:function:: main(args: argparse.Namespace | None = None, override_args: list[str] | None = None)

