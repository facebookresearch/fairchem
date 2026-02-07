"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch.autograd import Function
from torch.profiler import ProfilerActivity, profile, record_function
from torchtnt.framework.callback import Callback

from fairchem.core.common import distutils
from fairchem.core.common.logger import WandBSingletonLogger

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torchtnt.framework import State, TTrainUnit

    from fairchem.core.common.logger import Logger


# Thread-local storage for backward label contexts (dict mapping block_id -> context)
_backward_label_contexts = threading.local()

# Global counter for unique block IDs
_block_id_counter = 0
_block_id_lock = threading.Lock()


def _get_next_block_id() -> int:
    """Get a unique block ID."""
    global _block_id_counter
    with _block_id_lock:
        _block_id_counter += 1
        return _block_id_counter


def _get_label_dict() -> dict:
    """Get thread-local dict of active backward record_function contexts."""
    if not hasattr(_backward_label_contexts, "contexts"):
        _backward_label_contexts.contexts = {}
    return _backward_label_contexts.contexts


def _is_profiler_active() -> bool:
    """Check if the PyTorch profiler is currently active."""
    try:
        # Try the internal API first (available in some PyTorch versions)
        return torch.profiler._utils._is_profiler_enabled()
    except AttributeError:
        # Fallback: check if there's an active profiler by checking kineto availability
        # This is a reasonable heuristic - if kineto is not available, profiler won't work anyway
        try:
            return torch.autograd.profiler._is_profiler_enabled
        except AttributeError:
            # If we can't determine, assume profiler is active to be safe
            return True


class _NoOpContextManager:
    """A no-op context manager that returns tensors unchanged."""

    def mark_input(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def mark_output(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def mark(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


_NO_OP_CTX = _NoOpContextManager()


class _BackwardLabelStart(Function):
    """Inserted at block OUTPUT. In backward, this runs FIRST and OPENS the label context."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, name: str, block_id: int) -> torch.Tensor:
        ctx.name = name
        ctx.block_id = block_id
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        contexts = _get_label_dict()
        # Only open if not already open for this block_id
        if ctx.block_id not in contexts:
            rf_ctx = record_function(f"{ctx.name}_backward")
            rf_ctx.__enter__()
            contexts[ctx.block_id] = rf_ctx
        return grad_output, None, None


class _BackwardLabelEnd(Function):
    """Inserted at block INPUT. In backward, this runs LAST and CLOSES the label context."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, name: str, block_id: int) -> torch.Tensor:
        ctx.name = name
        ctx.block_id = block_id
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        contexts = _get_label_dict()
        # Close the context for this specific block_id
        if ctx.block_id in contexts:
            rf_ctx = contexts.pop(ctx.block_id)
            rf_ctx.__exit__(None, None, None)
        return grad_output, None, None


class _BackwardRecordFunction(Function):
    """Autograd function that wraps backward pass in a record_function context."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, name: str) -> torch.Tensor:
        ctx.name = name
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        with record_function(f"{ctx.name}_backward"):
            return grad_output, None


def mark_backward(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Mark a tensor to have a labeled backward pass in profiler traces.

    The backward pass through this tensor will be wrapped in a record_function
    with "{name}_backward", making it easy to identify in profiler traces.

    Example:
    ```python
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        x = self.layer(input)
        x = mark_backward(x, "my_layer")
        loss = x.sum()
        loss.backward()
    # Trace will show "my_layer_backward" for the backward pass
    ```

    Args:
        tensor: Tensor to attach backward marker to (must have requires_grad=True)
        name: Label for the backward pass (will have "_backward" appended)

    Returns:
        The tensor with backward hook attached
    """
    if not tensor.requires_grad:
        return tensor
    return _BackwardRecordFunction.apply(tensor, name)


def record_backward(name: str):
    """Decorator that labels both forward and backward passes in profiler traces.

    This is the simplest way to add profiler labels - just decorate your function
    and both forward and backward passes will be automatically labeled.

    The backward region will encompass ALL backward operations between the function's
    outputs and inputs.

    Example:
    ```python
    @record_backward("attention")
    def compute_attention(q, k, v):
        attn = torch.softmax(q @ k.T / math.sqrt(k.size(-1)), dim=-1)
        return attn @ v

    # Usage:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = compute_attention(q, k, v)
        out.sum().backward()

    # Trace will show:
    # - "attention" for forward pass
    # - "attention_backward" for backward pass (covering all backward ops in the function)
    ```

    Args:
        name: Label for the function. Forward pass uses this name,
              backward pass uses "{name}_backward".

    Returns:
        Decorated function with profiler labels.
    """

    def decorator(fn):
        def wrapper(*args, **kwargs):
            # Skip profiling overhead if profiler is not active
            if not _is_profiler_active():
                return fn(*args, **kwargs)

            block_id = _get_next_block_id()
            # Mark input tensors with END markers (they close the backward label)
            marked_args = _apply_end_marker(args, name, block_id)
            marked_kwargs = _apply_end_marker(kwargs, name, block_id)

            with record_function(name):
                outputs = fn(*marked_args, **marked_kwargs)

            # Mark output tensors with START markers (they open the backward label)
            return _apply_start_marker(outputs, name, block_id)

        return wrapper

    return decorator


def _apply_start_marker(obj, name: str, block_id: int):
    """Recursively apply _BackwardLabelStart to tensors."""
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad:
            return _BackwardLabelStart.apply(obj, name, block_id)
        return obj
    elif isinstance(obj, tuple):
        return tuple(_apply_start_marker(item, name, block_id) for item in obj)
    elif isinstance(obj, list):
        return [_apply_start_marker(item, name, block_id) for item in obj]
    elif isinstance(obj, dict):
        return {k: _apply_start_marker(v, name, block_id) for k, v in obj.items()}
    else:
        return obj


def _apply_end_marker(obj, name: str, block_id: int):
    """Recursively apply _BackwardLabelEnd to tensors."""
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad:
            return _BackwardLabelEnd.apply(obj, name, block_id)
        return obj
    elif isinstance(obj, tuple):
        return tuple(_apply_end_marker(item, name, block_id) for item in obj)
    elif isinstance(obj, list):
        return [_apply_end_marker(item, name, block_id) for item in obj]
    elif isinstance(obj, dict):
        return {k: _apply_end_marker(v, name, block_id) for k, v in obj.items()}
    else:
        return obj


class RecordFunctionWithBackward:
    """Context manager that labels both forward and backward passes in profiler traces.

    Use this to wrap a block of code and label its backward pass. You must call:
    - `ctx.mark_input(tensor)` on input tensor(s) at the START of the block
    - `ctx.mark_output(tensor)` on output tensor(s) at the END of the block

    This creates a proper backward region that encompasses ALL backward operations
    between the output and input markers.

    For a simpler API, use `ctx.mark(tensor)` on just the output - this will only
    label the gradient flow through that specific tensor, not the entire block.

    Example:
    ```python
    with RecordFunctionWithBackward("attention") as ctx:
        q = ctx.mark_input(q)
        k = ctx.mark_input(k)
        v = ctx.mark_input(v)
        attn = torch.softmax(q @ k.T, dim=-1) @ v
        attn = ctx.mark_output(attn)

    # Trace shows "attention" and "attention_backward" (covering all backward ops)
    ```

    Args:
        name: Label for the code block.
    """

    def __init__(self, name: str):
        self.name = name
        self._record_fn_ctx = None
        self._block_id = None

    def mark_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mark an input tensor - inserts END marker that closes the backward label."""
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            return tensor
        if self._block_id is None:
            self._block_id = _get_next_block_id()
        # Insert _BackwardLabelEnd node - this runs LAST in backward (closes the context)
        return _BackwardLabelEnd.apply(tensor, self.name, self._block_id)

    def mark_output(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mark an output tensor - inserts START marker that opens the backward label."""
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            return tensor
        if self._block_id is None:
            self._block_id = _get_next_block_id()
        # Insert _BackwardLabelStart node - this runs FIRST in backward (opens the context)
        return _BackwardLabelStart.apply(tensor, self.name, self._block_id)

    def mark(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mark a tensor for backward labeling (simple mode - only labels this gradient path).

        For full block coverage, use mark_input() and mark_output() instead.
        """
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            return tensor
        return _BackwardRecordFunction.apply(tensor, self.name)

    def __enter__(self):
        self._record_fn_ctx = record_function(self.name)
        self._record_fn_ctx.__enter__()
        self._block_id = _get_next_block_id()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._record_fn_ctx.__exit__(exc_type, exc_val, exc_tb)
        return False


@contextmanager
def record_function_with_backward(name: str):
    """Context manager that labels both forward and backward passes in profiler traces.

    Note: You must call `ctx.mark(tensor)` on output tensors for backward labeling.
    For automatic marking, use the `@record_backward` decorator instead.

    Example:
    ```python
    with record_function_with_backward("my_layer") as ctx:
        y = self.layer(x)
        y = ctx.mark(y)

    # Trace shows "my_layer" and "my_layer_backward"
    ```

    Args:
        name: Label for the code block.

    Yields:
        Context with `mark(tensor)` method.
    """
    # Skip profiling overhead if profiler is not active
    if not _is_profiler_active():
        yield _NO_OP_CTX
        return

    ctx = RecordFunctionWithBackward(name)
    with ctx:
        yield ctx


def get_default_profiler_handler(
    run_id: str, output_dir: str, logger: Logger, all_ranks: bool = False
):
    """Get a standard callback handle for the pytorch profiler"""

    def trace_handler(p):
        if all_ranks or distutils.is_master():
            trace_name = f"{run_id}_rank_{distutils.get_rank()}.pt.trace.json"
            output_path = os.path.join(output_dir, trace_name)
            logging.info(f"Saving trace in {output_path}")
            p.export_chrome_trace(output_path)
            if logger:
                logger.log_artifact(
                    name=trace_name, type="profile", file_location=output_path
                )

    return trace_handler


def get_profile_schedule(wait: int = 5, warmup: int = 5, active: int = 2):
    """Get a profile schedule and total number of steps to run
    check pytorch docs on the meaning of these paramters:
    https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule
    Example usage:
    ```
    trace_handler = get_default_profiler_handler(run_id = self.config["cmd"]["timestamp_id"],
                                                    output_dir = self.config["cmd"]["results_dir"],
                                                    logger = self.logger)
    profile_schedule, total_profile_steps = get_profile_schedule()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=profile_schedule,
        on_trace_ready=trace_handler
    ) as p:
        for i in steps:
            <code block to profile>
            if i < total_profile_steps:
                p.step()
    """
    total_profile_steps = wait + warmup + active
    profile_schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active)

    return profile_schedule, total_profile_steps


class ProfilerCallback(Callback):
    def __init__(
        self,
        job_config: DictConfig,
        wait_steps: int = 5,
        warmup_steps: int = 5,
        active_steps: int = 2,
        all_ranks: bool = False,
        activities: tuple = (ProfilerActivity.CPU, ProfilerActivity.CUDA),
    ) -> None:
        profile_dir = os.path.join(job_config.metadata.log_dir, "profiles")
        os.makedirs(profile_dir, exist_ok=True)
        logger = (
            WandBSingletonLogger.get_instance()
            if WandBSingletonLogger.initialized()
            else None
        )
        handler = get_default_profiler_handler(
            run_id=job_config.run_name,
            output_dir=profile_dir,
            logger=logger,
            all_ranks=all_ranks,
        )
        schedule, self.total_steps = get_profile_schedule(
            wait_steps, warmup_steps, active_steps
        )
        self.profiler = profile(
            activities=activities, schedule=schedule, on_trace_ready=handler
        )

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        self.profiler.start()

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        step = unit.train_progress.num_steps_completed
        if step <= self.total_steps:
            self.profiler.step()
        else:
            self.profiler.stop()
