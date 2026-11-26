"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# Module-path renames applied when migrating checkpoints (and source
# trees) written against pre-2.0 fairchem layouts to the current
# fairchem.core layout. Consumed by
# fairchem.core.scripts.migrate_checkpoint and by the standalone
# scripts/migrate_imports.py at the fairchem-core package root.

mapping = {
    "fairchem.experimental.foundation_models.units": "fairchem.core.units.mlip_unit",
    "fairchem.experimental.foundation_models.components.train": "fairchem.core.components.train",
    "fairchem.experimental.foundation_models.components.common": "fairchem.core.components.common",
    "fairchem.experimental.foundation_models.models.nn": "fairchem.core.models.puma.nn",
    "fairchem.experimental.foundation_models.models.common": "fairchem.core.models.puma.common",
    "fairchem.experimental.foundation_models.models.message_passing.escn_md": "fairchem.core.models.puma.escn_md",
    "fairchem.experimental.foundation_models.models.message_passing.escn_omol": "fairchem.core.models.puma.escn_md",
    "fairchem.experimental.foundation_models.models.message_passing.escn_moe": "fairchem.core.models.puma.escn_moe",
    # "fairchem.core.models.puma.escn_moe": "fairchem.core.models.puma.escn_mole",
    # "fairchem.experimental.foundation_models.models.message_passing.escn_moe": "fairchem.core.models.puma.escn_mole",
    # "fairchem.core.models.puma.escn_mole.eSCNMDMoeBackbone": "fairchem.core.models.puma.escn_mole.eSCNMDMOLEBackbone",
    "fairchem.experimental.foundation_models.modules.element_references": "fairchem.core.modules.normalization.element_references",
    "fairchem.experimental.foundation_models.modules.loss": "fairchem.core.modules.loss",
    "fairchem.experimental.foundation_models.multi_task_dataloader.transforms.data_object": "fairchem.core.modules.transforms",
    "fairchem.experimental.foundation_models.multi_task_dataloader.max_atom_distributed_sampler": "fairchem.core.datasets.samplers.max_atom_distributed_sampler",
    "fairchem.experimental.foundation_models.multi_task_dataloader.mt_collater": "fairchem.core.datasets.collaters.mt_collater",
    "fairchem.experimental.foundation_models.multi_task_dataloader.mt_concat_dataset": "fairchem.core.datasets.mt_concat_dataset",
    "fairchem.experimental.foundation_models.tests.units": "tests.core.units.mlip_unit",
    "fairchem.experimental.foundation_models.components.evaluate": "fairchem.core.components.evaluate",
    "tests/units/": "tests/core/units/mlip_unit/",
    "fairchem.core.models.puma.nn": "fairchem.core.models.uma.nn",
    "fairchem.core.models.puma.common": "fairchem.core.models.uma.common",
    "fairchem.core.models.puma.escn_md": "fairchem.core.models.uma.escn_md",
    "fairchem.core.models.puma.escn_moe": "fairchem.core.models.uma.escn_moe",
}
