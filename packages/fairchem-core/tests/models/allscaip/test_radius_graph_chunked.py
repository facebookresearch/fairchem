"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.allscaip.utils.allscaip_radius_graph import (
    build_radius_graph,
    build_radius_graph_chunked,
    segment_argsort,
    segment_argsort_vectorized,
)


@pytest.mark.parametrize(
    "soft,low_memory,pairwise",
    [
        (False, False, True),
        (True, False, True),
        (True, True, True),
        (False, False, False),
    ],
)
def test_chunked_graph_matches_dense_edges_distances_and_derivatives(
    soft, low_memory, pairwise
):
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.1, 1.2, 0.2], [0.3, 0.1, 1.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    kwargs = dict(
        pos=pos,
        cell=torch.eye(3, dtype=pos.dtype) * 10,
        image_id=torch.zeros(1, 3, dtype=pos.dtype),
        cutoff=4.0,
        start_index=7,
        device=pos.device,
        k=8,
        soft=soft,
        use_low_mem=low_memory,
        compute_dist_pairwise=pairwise,
    )
    dense = build_radius_graph(**kwargs)
    chunked = build_radius_graph_chunked(**kwargs, chunk_size=2)
    for expected, actual in zip(dense, chunked):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)
    assert chunked[0].min() == 7
    assert chunked[0].max() == 10
    assert chunked[4].shape[-1] == 3
    if soft:
        gradients = [
            torch.autograd.grad(result[5].sum(), pos, retain_graph=True)[0]
            for result in (dense, chunked)
        ]
        assert torch.isfinite(gradients[1]).all()
        torch.testing.assert_close(gradients[1], gradients[0])


def test_segment_ranks_restart_at_each_segment_including_missing_ids():
    values = torch.tensor([3.0, 1.0, 2.0, 5.0, 4.0])
    segments = torch.tensor([0, 0, 0, 2, 2])
    expected = torch.tensor([2, 0, 1, 1, 0])
    for function in (segment_argsort, segment_argsort_vectorized):
        assert torch.equal(function(values, segments, num_segments=4), expected)
