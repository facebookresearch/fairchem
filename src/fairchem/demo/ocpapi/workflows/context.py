"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from contextvars import ContextVar


@contextmanager
def set_context_var(
    context_var: ContextVar,
    value: Any,
) -> Generator[None, None, None]:
    """
    Sets the input convext variable to the input value and yields control
    back to the caller. When control returns to this function, the context
    variable is reset to its original value.

    Args:
        context_var: The context variable to set.
        value: The value to assign to the variable.
    """
    token = context_var.set(value)
    try:
        yield
    finally:
        context_var.reset(token)
