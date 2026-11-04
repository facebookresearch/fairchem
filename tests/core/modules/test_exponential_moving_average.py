"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gc
import io

import pytest
import torch

from fairchem.core.modules.exponential_moving_average import ExponentialMovingAverage


@pytest.mark.parametrize("explicit_parameters", [False, True])
@pytest.mark.parametrize("use_num_updates", [False, True])
def test_ema_update_store_restore_and_checkpoint(explicit_parameters, use_num_updates):
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    frozen = torch.nn.Parameter(torch.tensor([9.0]), requires_grad=False)
    parameters = [parameter, frozen]
    ema = ExponentialMovingAverage(
        parameters, decay=0.5, use_num_updates=use_num_updates
    )
    args = (parameters,) if explicit_parameters else ()
    parameter.data.fill_(4.0)
    ema.update(*args)
    expected = 2 * (2 / 11) + 4 * (9 / 11) if use_num_updates else 3.0
    ema.store(*args)
    ema.copy_to(*args)
    assert parameter.item() == pytest.approx(expected)
    assert frozen.item() == 9
    buffer = io.BytesIO()
    torch.save(ema.state_dict(), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored = ExponentialMovingAverage(parameters, decay=0.1)
    restored.load_state_dict(checkpoint)
    assert restored.decay == 0.5
    assert restored.num_updates == (1 if use_num_updates else None)
    restored.restore(*args)
    assert parameter.item() == 4
    checkpoint["shadow_params"][0].zero_()
    restored.copy_to(*args)
    assert parameter.item() == pytest.approx(expected)


@pytest.mark.parametrize("decay", [-0.1, 1.1])
def test_ema_rejects_invalid_decay(decay):
    parameter = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="Decay"):
        ExponentialMovingAverage([parameter], decay)
    ema = ExponentialMovingAverage([parameter], 0.5)
    state = ema.state_dict()
    state["decay"] = decay
    with pytest.raises(ValueError, match="Decay"):
        ema.load_state_dict(state)


def test_ema_reports_collected_parameters_and_accepts_replacements():
    parameter = torch.nn.Parameter(torch.ones(1))
    ema = ExponentialMovingAverage([parameter], 0.5)
    del parameter
    gc.collect()
    with pytest.raises(ValueError, match="garbage collected"):
        ema.update()
    replacement = torch.nn.Parameter(torch.tensor([3.0]))
    ema.update([replacement])
    ema.copy_to([replacement])
    assert replacement.item() == 2
