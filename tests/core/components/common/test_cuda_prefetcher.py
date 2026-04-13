"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.components.common.cuda_prefetcher import CUDAPrefetcher


class _FakeBatch:
    """Minimal batch object with .to() support."""

    def __init__(self, data: torch.Tensor):
        self.data = data
        self.pos = data

    def to(self, device, non_blocking=False):
        return _FakeBatch(self.data.to(device, non_blocking=non_blocking))


class _FakeDataLoader:
    """Fake dataloader that yields _FakeBatch objects."""

    def __init__(self, batches: list[_FakeBatch]):
        self._batches = batches
        self.batch_sampler = [list(range(len(batches)))]
        self.dataset = "fake_dataset"

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def test_len_delegates_to_underlying_dataloader():
    dl = _FakeDataLoader([_FakeBatch(torch.zeros(1))] * 5)
    prefetcher = CUDAPrefetcher(dl, torch.device("cpu"))
    assert len(prefetcher) == 5


def test_getattr_delegates_to_underlying_dataloader():
    dl = _FakeDataLoader([_FakeBatch(torch.zeros(1))])
    prefetcher = CUDAPrefetcher(dl, torch.device("cpu"))
    assert prefetcher.batch_sampler == dl.batch_sampler
    assert prefetcher.dataset == "fake_dataset"


def test_empty_dataloader_len():
    dl = _FakeDataLoader([])
    prefetcher = CUDAPrefetcher(dl, torch.device("cpu"))
    assert len(prefetcher) == 0


@pytest.mark.gpu()
def test_single_batch():
    device = torch.device("cuda")
    batch = _FakeBatch(torch.tensor([1.0, 2.0, 3.0]))
    dl = _FakeDataLoader([batch])
    prefetcher = CUDAPrefetcher(dl, device)
    batches = list(prefetcher)
    assert len(batches) == 1
    assert torch.equal(batches[0].data, torch.tensor([1.0, 2.0, 3.0], device=device))


@pytest.mark.gpu()
def test_multiple_batches_preserves_order():
    device = torch.device("cuda")
    tensors = [torch.tensor([float(i)]) for i in range(5)]
    dl = _FakeDataLoader([_FakeBatch(t) for t in tensors])
    prefetcher = CUDAPrefetcher(dl, device)
    results = [b.data.item() for b in prefetcher]
    assert results == [0.0, 1.0, 2.0, 3.0, 4.0]


@pytest.mark.gpu()
def test_reiter():
    """Test that the prefetcher can be iterated multiple times."""
    device = torch.device("cuda")
    dl = _FakeDataLoader([_FakeBatch(torch.tensor([1.0]))])
    prefetcher = CUDAPrefetcher(dl, device)
    assert len(list(prefetcher)) == 1
    assert len(list(prefetcher)) == 1


@pytest.mark.gpu()
def test_cuda_prefetcher_moves_to_gpu():
    device = torch.device("cuda")
    tensors = [torch.tensor([float(i)]) for i in range(3)]
    dl = _FakeDataLoader([_FakeBatch(t) for t in tensors])
    prefetcher = CUDAPrefetcher(dl, device)
    for i, batch in enumerate(prefetcher):
        assert batch.data.device.type == "cuda"
        assert batch.data.item() == float(i)
