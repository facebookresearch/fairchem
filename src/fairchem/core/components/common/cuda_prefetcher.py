"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator


class _PrefetchIterator:
    """
    Iterator that prefetches the next batch onto GPU via a dedicated CUDA stream.
    """

    def __init__(self, dataloader_iter: Iterator, device: torch.device):
        self._dataloader_iter = dataloader_iter
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._next_batch = None
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self._dataloader_iter)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self._stream):
            self._next_batch = batch.to(self._device, non_blocking=True)

    def __next__(self):
        if self._next_batch is None:
            raise StopIteration
        torch.cuda.current_stream(self._device).wait_stream(self._stream)
        batch = self._next_batch
        self._preload()
        return batch

    def __iter__(self):
        return self


class CUDAPrefetcher:
    """
    Wraps a DataLoader to prefetch one batch ahead onto GPU via a separate CUDA stream.

    This overlaps HtoD memcpy with GPU compute, reducing serialization overhead.
    Requires pin_memory=True on the underlying DataLoader for non_blocking transfers.
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader, device: torch.device):
        self._dataloader = dataloader
        self._device = device

    def __iter__(self) -> _PrefetchIterator:
        return _PrefetchIterator(iter(self._dataloader), self._device)

    def __len__(self) -> int:
        return len(self._dataloader)

    def __getattr__(self, name: str):
        return getattr(self._dataloader, name)
