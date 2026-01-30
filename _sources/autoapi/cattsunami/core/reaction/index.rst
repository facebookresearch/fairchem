cattsunami.core.reaction
========================

.. py:module:: cattsunami.core.reaction

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   cattsunami.core.reaction.Reaction


Module Contents
---------------

.. py:class:: Reaction(reaction_db_path: str, adsorbate_db_path: str, reaction_id_from_db: int | None = None, reaction_str_from_db: str | None = None, reaction_type: str | None = None)

   Initialize Reaction object


   .. py:attribute:: reaction_db_path


   .. py:method:: get_desorption_mapping(reactant)

      Get mapping for desorption reaction



