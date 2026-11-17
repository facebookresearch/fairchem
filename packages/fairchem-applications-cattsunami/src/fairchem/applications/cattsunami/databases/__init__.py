"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import os

DISSOCIATION_REACTION_DB_PATH = os.path.join(
    __path__[0], "dissociation_reactions_22May24.pkl"
)
DESORPTION_REACTION_DB_PATH = os.path.join(__path__[0], "desorptions_9Aug23.pkl")
TRANSFER_REACTION_DB_PATH = os.path.join(__path__[0], "transfers_5Sept23.pkl")
