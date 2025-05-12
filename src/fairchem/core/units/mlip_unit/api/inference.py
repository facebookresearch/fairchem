"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MLIPInferenceCheckpoint:
    model_config: dict
    model_state_dict: dict
    ema_state_dict: dict
    tasks_config: dict


@dataclass
class InferenceSettings:
    tf32: bool | None = None
    activation_checkpointing: bool | None = None
    merge_mole: bool | None = None
    compile: bool | None = None
    wigner_cuda: bool | None = None
    external_graph_gen: bool | None = None


def inference_settings_default():
    return InferenceSettings(
        tf32=True,
        activation_checkpointing=True,
        merge_mole=False,
        compile=False,
        wigner_cuda=False,
    )


def inference_settings_turbo():
    return InferenceSettings(
        tf32=True,
        activation_checkpointing=True,
        merge_mole=True,
        compile=True,
        wigner_cuda=True,
        external_graph_gen=True,
    )
