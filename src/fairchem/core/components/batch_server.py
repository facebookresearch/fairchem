"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import io
import json
import logging
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import asdict, dataclass, field
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

import ray
import torch
from ray import serve

from fairchem.core.components.serve_utils import (
    get_app_handle_with_retry,
    get_ray_connection_info,
    wait_for_serve_ready,
)
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from fairchem.core.units.mlip_unit.api.model_spec import (
    ModelSpec,
    ModelSpecNotRegisteredError,
)

# ``ModelSpec`` and the Ray Serve lifecycle helpers live in leaf modules so that
# ``units.mlip_unit.predict`` can import them without importing this module (a
# cycle: batch_server -> units.mlip_unit -> predict -> batch_server). They are
# re-exported here because this module was their original home.
__all__ = [
    "AutobatchConfig",
    "AutobatchResult",
    "BatchConfig",
    "BatchPredictServer",
    "DeploymentConfig",
    "ModelSpec",
    "ModelSpecNotRegisteredError",
    "MultiplexedBatchPredictServer",
    "get_app_handle_with_retry",
    "get_ray_connection_info",
    "probe_optimal_batch_size",
    "setup_batch_predict_server",
    "setup_multiplexed_batch_predict_server",
    "update_batch_config",
    "update_served_predict_unit",
    "wait_for_serve_ready",
]

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit import MLIPPredictUnit


def _to_cpu(obj: Any) -> Any:
    """Return a CPU-resident copy of ``obj`` so it can be deserialized on CPU-only Ray workers.

    Uses ``torch.save`` + ``torch.load(map_location="cpu")`` which transparently
    handles arbitrary object graphs containing tensors, ``nn.Module`` instances,
    OmegaConf containers, etc., without needing to walk and mutate the structure.
    """

    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


# Centralized batching defaults. Kept here (not on the server class
# __init__s) so the two setup helpers can't drift apart silently.
DEFAULT_MAX_BATCH_SIZE = 512
DEFAULT_BATCH_WAIT_TIMEOUT_S = 0.1
MAX_NUM_MODELS_PER_REPLICA = 3
MODEL_SPEC_CACHE_CAPACITY = MAX_NUM_MODELS_PER_REPLICA * 4


@dataclass
class DeploymentConfig:
    """Typed mirror of the most common ``@serve.deployment`` / ``.options()`` kwargs.

    Fields default to ``None`` and are dropped before being forwarded to
    Ray Serve, so unspecified options inherit Ray Serve's own defaults
    rather than this class re-asserting them.
    """

    num_replicas: int | None = None
    ray_actor_options: dict | None = None
    # ``autoscaling_config`` accepts a dict or a ``ray.serve.schema.AutoscalingConfig``;
    # ``logging_config`` accepts a dict or ``ray.serve.schema.LoggingConfig``.
    autoscaling_config: Any = None
    max_ongoing_requests: int | None = None
    max_queued_requests: int | None = None
    max_replicas_per_node: int | None = None
    graceful_shutdown_timeout_s: float | None = None
    graceful_shutdown_wait_loop_s: float | None = None
    health_check_period_s: float | None = None
    health_check_timeout_s: float | None = None
    logging_config: Any = None
    user_config: dict | None = None
    placement_group_bundles: list | None = None
    placement_group_strategy: str | None = None
    request_router_config: Any = None

    def to_options_kwargs(self) -> dict[str, Any]:
        """Return ``{name: value}`` for fields that were explicitly set (non-None)."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class BatchConfig:
    """Typed mirror of kwargs accepted by ``BatchPredictServer.__init__`` and
    ``MultiplexedBatchPredictServer.__init__``.
    """

    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    batch_wait_timeout_s: float = DEFAULT_BATCH_WAIT_TIMEOUT_S
    split_oom_batch: bool = False

    def to_init_kwargs(self) -> dict[str, Any]:
        """Return ``{name: value}`` for fields that were explicitly set (non-None)."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_user_config(self) -> dict[str, Any]:
        """
        Return this config as a Ray Serve ``user_config`` payload.

        Ray Serve pushes ``user_config`` to *every* replica and invokes
        ``reconfigure`` there, which is the only supported way to change
        batching across a multi-replica deployment: a ``DeploymentHandle``
        call reaches exactly one replica.

        The payload must be JSON-serializable, which all fields already are.
        """
        return asdict(self)


class BatchPredictServerMixin:
    """
    Shared batched-inference logic mixed into Ray Serve deployment classes.

    This mixin is **not** decorated with ``@serve.deployment`` so that it
    can be used as a regular base class.  The concrete subclasses
    ``BatchPredictServer`` and ``MultiplexedBatchPredictServer`` apply the
    decorator themselves.
    """

    def configure_batching(
        self,
        max_batch_size: int,
        batch_wait_timeout_s: float,
    ) -> None:
        """
        Configure batching parameters at runtime.

        Args:
            max_batch_size: Maximum number of atoms in a batch.
            batch_wait_timeout_s: Timeout in seconds to wait for a full batch.
        """
        self.predict.set_max_batch_size(max_batch_size)
        self.predict.set_batch_wait_timeout_s(batch_wait_timeout_s)

    def reconfigure(self, user_config: dict) -> None:
        """
        Apply batching parameters pushed by Ray Serve to every replica.

        Ray Serve calls this on each replica after ``__init__`` and again
        whenever the deployment's ``user_config`` changes, so it is the only
        mechanism that updates batching fleet-wide. Prefer it over calling
        ``configure_batching`` through a ``DeploymentHandle``, which reaches
        exactly one replica and silently leaves the rest on their old config.

        Args:
            user_config: Payload produced by :meth:`BatchConfig.to_user_config`.
                Unknown keys are ignored; absent keys leave the current value
                in place.
        """
        if not user_config:
            return

        max_batch_size = user_config.get("max_batch_size")
        batch_wait_timeout_s = user_config.get("batch_wait_timeout_s")
        split_oom_batch = user_config.get("split_oom_batch")

        if max_batch_size is not None:
            self.predict.set_max_batch_size(max_batch_size)
        if batch_wait_timeout_s is not None:
            self.predict.set_batch_wait_timeout_s(batch_wait_timeout_s)
        if split_oom_batch is not None:
            self.split_oom_batch = bool(split_oom_batch)

        logging.info(
            "Reconfigured batching: max_batch_size=%s batch_wait_timeout_s=%s "
            "split_oom_batch=%s",
            max_batch_size,
            batch_wait_timeout_s,
            split_oom_batch,
        )

    def get_predict_unit_attribute(self, attribute_name: str, **kwargs) -> Any:
        # Move the returned value to CPU so that callers running on
        # CPU-only Ray workers can deserialize it without requiring CUDA
        # (e.g. ``atom_refs`` typically contains tensors stored on the
        # server's device).
        return _to_cpu(getattr(self.predict_unit, attribute_name))

    def validate_atoms_data(self, atoms_info: dict, task_name: str, **kwargs) -> dict:
        """
        Run the predict unit's validation and return the (possibly mutated) atoms.info.

        Validation may set defaults (e.g. charge, spin) on ``atoms.info``.
        Because the caller needs those mutations applied locally, this method
        accepts and returns the ``atoms.info`` dict rather than a full
        ``Atoms`` object.

        Args:
            atoms_info: Copy of ``atoms.info`` from the caller's Atoms object.
            task_name: Task name passed through to the predict unit.

        Returns:
            The mutated ``atoms.info`` dict with any defaults applied.
        """
        from ase import Atoms

        # Build a minimal Atoms stub just for validation — only atoms.info
        # is read/mutated by validate_atoms_data implementations.
        stub = Atoms()
        stub.info = atoms_info
        self.predict_unit.validate_atoms_data(stub, task_name)
        return stub.info

    def update_predict_unit(self, predict_unit) -> None:
        """
        Update the predict unit with a new checkpoint.

        Args:
            predict_unit: New MLIPPredictUnit instance (Ray resolves any ObjectRef
                before invoking this method, so the argument is always the actual object)
        """
        self.predict_unit = predict_unit
        logging.info("predict_unit updated")

    def _run_batched_inference(
        self,
        items: list[AtomicData],
        predict_unit: MLIPPredictUnit,
        undo_element_references: bool,
    ) -> list[dict]:
        """
        Run batched inference with OOM splitting.
        """
        data_deque: deque[list[AtomicData]] = deque([items])
        results: list[dict] = []
        while data_deque:
            oom = False
            current = data_deque.popleft()
            batch = atomicdata_list_to_batch(current)
            try:
                preds = predict_unit.predict(
                    batch, undo_element_references=undo_element_references
                )
                results.extend(self._split_predictions(preds, batch))
            except torch.OutOfMemoryError as err:
                if not self.split_oom_batch:
                    raise torch.OutOfMemoryError(
                        "Out of memory during batched inference. "
                        "This can happen when the batch contains systems of very different sizes. "
                        "Consider reducing max_batch_size or setting split_oom_batch=True "
                        "to automatically split OOM batches. "
                        "Note: split_oom_batch is useful for heterogeneous batches but may "
                        "impact performance for homogeneous workloads."
                    ) from err
                if len(current) == 1:
                    raise torch.OutOfMemoryError(
                        "Out of memory for a single system left in batch. "
                        "Try reducing max_batch_size or using a model with lower memory requirements."
                    ) from err
                logging.warning(
                    "Caught out of memory error. Splitting batch and retrying."
                )
                oom = True
                torch.cuda.empty_cache()
            if oom:
                mid = len(current) // 2
                data_deque.appendleft(current[mid:])
                data_deque.appendleft(current[:mid])
        return results

    @staticmethod
    def _normalize_undo_flags(
        undo_element_references: bool | list[bool],
        num_requests: int,
    ) -> list[bool]:
        """
        Normalize a batched ``undo_element_references`` argument to one flag
        per request.

        ``@serve.batch`` collects *every* argument of the decorated function
        into a list, so a scalar-looking parameter arrives as a list with one
        entry per co-batched request. A non-empty list is always truthy, so
        forwarding it straight to ``predict_unit.predict`` would silently undo
        element references even when every caller asked not to.

        Args:
            undo_element_references: The value as received by the batch
                function: a list (one entry per request) when at least one
                caller supplied the argument, or the scalar default otherwise.
            num_requests: Number of requests in the batch window.

        Returns:
            A list of ``num_requests`` booleans.

        Raises:
            ValueError: If a list was received whose length does not match
                ``num_requests``.
        """
        if isinstance(undo_element_references, (list, tuple)):
            if len(undo_element_references) != num_requests:
                raise ValueError(
                    "Batched undo_element_references has length "
                    f"{len(undo_element_references)} but the batch contains "
                    f"{num_requests} request(s)."
                )
            return [bool(flag) for flag in undo_element_references]
        return [bool(undo_element_references)] * num_requests

    def _run_grouped_inference(
        self,
        data_list: list[AtomicData],
        undo_flags: list[bool],
        predict_unit: MLIPPredictUnit,
    ) -> list[dict]:
        """
        Run inference for one model, grouped by ``undo_element_references``.

        Requests that disagree on ``undo_element_references`` cannot share a
        forward pass, so they are dispatched as separate sub-batches and the
        results are reassembled in the original request order.

        Args:
            data_list: AtomicData objects collected by ``@serve.batch``.
            undo_flags: One ``undo_element_references`` flag per request.
            predict_unit: The predict unit to run all of these requests with.

        Returns:
            List of prediction dictionaries, one per input, in original order.
        """
        results: list[dict | None] = [None] * len(data_list)

        groups: dict[bool, list[int]] = defaultdict(list)
        for index, flag in enumerate(undo_flags):
            groups[flag].append(index)

        for flag, indices in groups.items():
            group_results = self._run_batched_inference(
                [data_list[index] for index in indices], predict_unit, flag
            )
            for index, prediction in zip(indices, group_results):
                results[index] = prediction

        return results

    async def __call__(
        self,
        data: AtomicData,
        undo_element_references: bool = True,
        **kwargs,
    ) -> dict:
        """
        Main entry point for inference requests.

        Args:
            data: Single AtomicData object
            undo_element_references: Whether to undo element references in predictions

        Returns:
            Prediction dictionary for this system
        """
        predictions = await self.predict(data, undo_element_references)
        return predictions

    def _split_predictions(
        self,
        predictions: dict,
        batch: AtomicData,
    ) -> list[dict]:
        """
        Split batched predictions back into individual system predictions.

        Args:
            predictions: Dictionary of batched prediction tensors
            batch: The batched AtomicData used for inference

        Returns:
            List of prediction dictionaries, one per system
        """
        split_preds = []
        for i in range(len(batch)):
            system_predictions = {}

            for key, pred in predictions.items():
                if pred.shape[0] == len(batch):
                    # Per-system prediction
                    system_predictions[key] = pred[i : i + 1]
                elif pred.shape[0] == len(batch.batch):
                    # Per-atom prediction
                    mask = batch.batch == i
                    system_predictions[key] = pred[mask]
                else:
                    raise ValueError(
                        f"Cannot split prediction for key '{key}': "
                        f"unexpected shape {pred.shape} for batch size {len(batch)} "
                        f"and num_atoms {batch.num_atoms}"
                    )

                # Move to CPU before returning so the caller (which may be a
                # CPU-only Ray worker) can deserialize the result without
                # requiring CUDA.
                if hasattr(system_predictions[key], "detach"):
                    system_predictions[key] = system_predictions[key].detach().cpu()

            split_preds.append(system_predictions)

        return split_preds


@serve.deployment(
    logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
    max_ongoing_requests=300,
)
class BatchPredictServer(BatchPredictServerMixin):
    """
    Ray Serve deployment that batches incoming inference requests
    for a single pre-loaded model.
    """

    def __init__(
        self,
        predict_unit_ref,
        max_batch_size: int,
        batch_wait_timeout_s: float,
        split_oom_batch: bool = False,
    ):
        """
        Initialize with a Ray object reference to a PredictUnit.

        Args:
            predict_unit_ref: Ray object reference to an MLIPPredictUnit instance
            max_batch_size: Maximum number of atoms in a batch.
                The actual number of atoms will likely be larger than this as batches
                are split when num atoms exceeds this value.
            batch_wait_timeout_s: Timeout in seconds to wait for a prediction
            split_oom_batch: If True, automatically split batches that cause OOM errors
                and retry with smaller sub-batches. This is useful when running batches
                with very different sized systems (e.g., mixed molecules and bulk
                materials), but may impact performance for homogeneous workloads.
                Defaults to False.
        """
        self.predict_unit = ray.get(predict_unit_ref)
        self.split_oom_batch = split_oom_batch
        self.configure_batching(max_batch_size, batch_wait_timeout_s)

        logging.info(
            "BatchPredictServer initialized with predict_unit from object store"
        )

    @serve.batch(
        batch_size_fn=lambda batch: sum(sample.natoms.sum() for sample in batch).item()
    )
    async def predict(
        self,
        data_list: list[AtomicData],
        undo_element_references: bool | list[bool] = True,
    ) -> list[dict]:
        """
        Process a batch of AtomicData objects with the pre-loaded model.

        Args:
            data_list: List of AtomicData objects (automatically batched by
                Ray Serve).
            undo_element_references: Whether to undo element references. Ray
                Serve batches this into one value per request; requests that
                disagree are dispatched as separate sub-batches.

        Returns:
            List of prediction dictionaries, one per input, in original order.
        """
        undo_flags = self._normalize_undo_flags(undo_element_references, len(data_list))
        return self._run_grouped_inference(data_list, undo_flags, self.predict_unit)

    async def is_multiplexed(self) -> bool:
        return False


@serve.deployment(
    logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
    max_ongoing_requests=300,
)
class MultiplexedBatchPredictServer(BatchPredictServerMixin):
    """
    Ray Serve deployment that supports multiplexed model loading with batching.

    Unlike ``BatchPredictServer`` which serves a single pre-loaded model,
    this deployment loads models on demand using ``@serve.multiplexed``.
    Different clients request models with a :class:`ModelSpec`; its deterministic
    ``model_id`` is used only for Serve routing and LRU cache identity. The spec
    travels with each request and supplies the loader configuration.

    **Batching with per-model routing.**  ``@serve.batch`` collects requests
    from concurrent ``__call__`` invocations.  Because
    ``serve.get_multiplexed_model_id()`` is only reliable in per-request
    context (``__call__``), each request captures its ``model_id`` there and
    passes it explicitly as a second positional argument to ``predict()``.
    Inside the batch function, requests are grouped by ``model_id``, each
    group is processed with the correct cached ``predict_unit`` via
    ``await self.get_model(model_id)`` (LRU cache hit), and results are
    reassembled in original request order.
    """

    def __init__(
        self,
        max_batch_size: int,
        batch_wait_timeout_s: float,
        split_oom_batch: bool = False,
    ):
        """
        Initialize the multiplexed predict server.

        Args:
            max_batch_size: Maximum number of atoms in a batch.
            batch_wait_timeout_s: Timeout in seconds to wait for a prediction.
            split_oom_batch: If True, automatically split batches that cause OOM errors
                and retry with smaller sub-batches. This is useful when running batches
                with very different sized systems (e.g., mixed molecules and bulk
                materials), but may impact performance for homogeneous workloads.
                Defaults to False.
        """
        self.split_oom_batch = split_oom_batch
        self._specs: OrderedDict[str, ModelSpec] = OrderedDict()
        self._active_spec_counts: dict[str, int] = defaultdict(int)
        self._spec_capacity = MODEL_SPEC_CACHE_CAPACITY
        self.configure_batching(max_batch_size, batch_wait_timeout_s)
        logging.info("MultiplexedBatchPredictServer initialized")

    def _register_spec(self, spec: ModelSpec, *, pin: bool = False) -> str:
        """Record a model configuration long enough for the multiplexed loader."""
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"spec must be a ModelSpec, got {type(spec).__name__}")

        model_id = spec.model_id
        existing = self._specs.get(model_id)
        if existing is not None and existing.canonical_dict() != spec.canonical_dict():
            raise ValueError(f"ModelSpec hash collision for model_id={model_id!r}")
        self._specs[model_id] = spec
        self._specs.move_to_end(model_id)
        if pin:
            self._active_spec_counts[model_id] += 1
        self._evict_specs()
        return model_id

    def _release_spec(self, model_id: str) -> None:
        """Unpin an in-flight spec and restore the steady-state cache bound."""
        remaining = self._active_spec_counts[model_id] - 1
        if remaining > 0:
            self._active_spec_counts[model_id] = remaining
        else:
            self._active_spec_counts.pop(model_id, None)
        self._evict_specs()

    def _evict_specs(self) -> None:
        """Evict least-recent specs that are not needed by active requests."""
        while len(self._specs) > self._spec_capacity:
            evictable_id = next(
                (
                    candidate_id
                    for candidate_id in self._specs
                    if candidate_id not in self._active_spec_counts
                ),
                None,
            )
            if evictable_id is None:
                return
            del self._specs[evictable_id]

    async def is_multiplexed(self) -> bool:
        return True

    @serve.batch(
        batch_size_fn=lambda batch: sum(sample.natoms.sum() for sample in batch).item(),
        max_concurrent_batches=1,
    )
    async def predict(
        self,
        data_list: list[AtomicData],
        model_id_list: list[str],
        spec_list: list[ModelSpec],
        undo_element_references: bool | list[bool] = True,
    ) -> list[dict]:
        """
        Process a batch of AtomicData objects, grouped by model_id.

        Requests for different models accumulate in the same ``@serve.batch``
        window and are then dispatched in sequential per-model groups.  The
        ``model_id`` for each request is passed explicitly from ``__call__``
        (where ``serve.get_multiplexed_model_id()`` is still valid) rather than
        being read inside this function (where only one request context is
        active).

        Args:
            data_list: List of AtomicData objects (automatically batched by
                Ray Serve).
            model_id_list: Corresponding model IDs, one per request.
            spec_list: Corresponding model specs, used to keep in-flight requests
                registered even when the bounded spec table evicts older entries.
            undo_element_references: Whether to undo element references. Ray
                Serve batches this into one value per request; requests that
                disagree are dispatched as separate sub-batches.

        Returns:
            List of prediction dictionaries, one per input, in original order.
        """
        num_requests = len(data_list)
        if len(model_id_list) != num_requests or len(spec_list) != num_requests:
            raise ValueError(
                "Batched request arguments have mismatched lengths: "
                f"{num_requests} data, {len(model_id_list)} model_id(s), "
                f"{len(spec_list)} spec(s)."
            )
        undo_flags = self._normalize_undo_flags(undo_element_references, num_requests)

        # Group (original_index, data, spec, undo_flag) tuples by model_id.
        groups: dict[str, list[tuple[int, AtomicData, ModelSpec, bool]]] = defaultdict(
            list
        )
        for i, (data, model_id, spec, undo_flag) in enumerate(
            zip(data_list, model_id_list, spec_list, undo_flags)
        ):
            groups[model_id].append((i, data, spec, undo_flag))

        results: list[dict | None] = [None] * num_requests

        for model_id, indexed_items in groups.items():
            group_spec = indexed_items[0][2]
            registered_model_id = self._register_spec(group_spec)
            if registered_model_id != model_id:
                raise ValueError(
                    f"Batched ModelSpec identity mismatch: received {model_id!r}, "
                    f"derived {registered_model_id!r}."
                )
            predict_unit = await self.get_model(model_id)  # LRU cache hit

            indices, group_data, _, group_flags = zip(*indexed_items)
            group_results = self._run_grouped_inference(
                list(group_data), list(group_flags), predict_unit
            )
            for orig_idx, pred in zip(indices, group_results):
                results[orig_idx] = pred

        return results

    @serve.multiplexed(max_num_models_per_replica=MAX_NUM_MODELS_PER_REPLICA)
    async def get_model(self, model_id: str):
        """Load or retrieve the predict unit identified by a registered spec."""
        try:
            spec = self._specs[model_id]
        except KeyError as err:
            raise ModelSpecNotRegisteredError(
                f"No ModelSpec is registered for model_id={model_id!r} on this "
                "replica. Send the ModelSpec with the request before loading it."
            ) from err
        self._specs.move_to_end(model_id)

        loader_kwargs = {
            "inference_settings": spec.loader_settings(),
            "device": spec.resolve_device(),
        }
        overrides = spec.loader_overrides()
        if overrides:
            loader_kwargs["overrides"] = overrides

        # ``ModelSpec`` resolves ``source="auto"`` at construction time, so the
        # loader choice here must match the one already baked into ``model_id``
        # rather than being re-derived from this replica's filesystem.
        if spec.source == "path":
            from fairchem.core.units.mlip_unit import load_predict_unit

            predict_unit = load_predict_unit(spec.checkpoint, **loader_kwargs)
        else:
            from fairchem.core.calculate import pretrained_mlip

            predict_unit = pretrained_mlip.get_predict_unit(
                spec.checkpoint, **loader_kwargs
            )

        logging.info(
            "MultiplexedBatchPredictServer loaded model_id=%r spec=%s",
            model_id,
            json.dumps(spec.canonical_dict(), sort_keys=True),
        )
        return predict_unit

    async def get_predict_unit_attribute(
        self, attribute_name: str, spec: ModelSpec
    ) -> Any:
        """Get an attribute after registering and loading the requested model."""
        model_id = self._register_spec(spec, pin=True)
        try:
            predict_unit = await self.get_model(model_id)
            return _to_cpu(getattr(predict_unit, attribute_name))
        finally:
            self._release_spec(model_id)

    async def validate_atoms_data(
        self, atoms_info: dict, task_name: str, spec: ModelSpec
    ) -> dict:
        """Run model-specific validation after registering the requested model."""
        from ase import Atoms

        model_id = self._register_spec(spec, pin=True)
        try:
            predict_unit = await self.get_model(model_id)
            stub = Atoms()
            stub.info = atoms_info
            predict_unit.validate_atoms_data(stub, task_name)
            return stub.info
        finally:
            self._release_spec(model_id)

    async def __call__(
        self,
        data: AtomicData,
        spec: ModelSpec,
        undo_element_references: bool = True,
    ) -> dict:
        """Register the request's spec and forward its identity into the batch."""
        model_id = self._register_spec(spec, pin=True)
        try:
            routed_model_id = serve.get_multiplexed_model_id()
            if routed_model_id != model_id:
                raise ValueError(
                    "Ray Serve multiplexed_model_id does not match the request's "
                    f"ModelSpec: routed={routed_model_id!r}, expected={model_id!r}."
                )
            return await self.predict(data, model_id, spec, undo_element_references)
        finally:
            self._release_spec(model_id)


def _init_ray_and_serve(
    ray_actor_options: dict,
    num_replicas: int,
) -> None:
    """
    Ensure Ray and Ray Serve are initialised.
    """
    cpus_per_actor = ray_actor_options.get("num_cpus", min(cpu_count(), 8))
    ray_actor_options["num_cpus"] = cpus_per_actor

    requested_cpus = cpus_per_actor * num_replicas

    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,
            logging_config=ray.LoggingConfig(log_level="WARNING"),
            num_cpus=requested_cpus,
        )
        logging.info("Ray initialized")

    # If the deployment's CPU request exceeds what's currently available,
    # replicas will queue until capacity frees up. Warn (don't raise)
    # because multi-node Ray clusters can auto-grow as workers join, and
    # autoscaling deployments only need capacity for ``min_replicas`` at
    # startup. ``available_resources()`` already accounts for CPUs used by
    # the Serve controller/proxy and other live actors, so no manual
    # overhead adjustment is needed.
    available_cpus = ray.available_resources().get("CPU", 0)
    if requested_cpus > available_cpus:
        logging.warning(
            f"Ray Serve deployment requests {cpus_per_actor} CPU(s) x "
            f"{num_replicas} replica(s) = {requested_cpus} CPU(s), but only "
            f"{available_cpus:g} CPU(s) are currently available on the Ray "
            "cluster. Replicas will queue until workers join or autoscaling "
            "adds capacity. If the cluster is fixed-size and small, reduce "
            "ray_actor_options['num_cpus'] / num_replicas."
        )

    # ``serve.start`` is idempotent for matching options. We always call it so
    # that ``proxy_location`` is enforced (``serve.status()`` returns OK even
    # when serve isn't running yet, so it can't be used as a "started" probe).
    # Clients use Python deployment handles (``serve.get_app_handle``), not HTTP,
    # so disable the HTTP proxy entirely. This avoids ProxyActor startup
    # failures on worker nodes where port 8000 is already bound.
    serve.start(
        proxy_location="Disabled",
        logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
    )
    logging.info("Ray Serve started (proxy_location=Disabled)")


def _effective_replicas(deployment_config: dict) -> int:
    """
    Best-effort estimate of the number of replicas that need to be
    schedulable at startup, for CPU-sizing warnings.

    For autoscaling configs, returns ``min_replicas`` (autoscaling will
    grow to ``max_replicas`` later if capacity becomes available).
    Otherwise returns ``num_replicas`` if it is an int, else ``1``.
    """
    ac = deployment_config.get("autoscaling_config") or {}
    if ac:
        for key in ("min_replicas", "max_replicas"):
            if key in ac:
                try:
                    return max(1, int(ac[key]))
                except (TypeError, ValueError):
                    pass
    nr = deployment_config.get("num_replicas")
    if isinstance(nr, int):
        return max(1, nr)
    return 1


def _prepare_deployment_config(
    deployment_config: DeploymentConfig | dict | None,
    default_num_gpus: float,
) -> DeploymentConfig:
    """
    Normalize ``deployment_config`` and settle the replica's GPU allocation.

    Args:
        deployment_config: Caller-supplied config, or ``None``.
        default_num_gpus: GPUs per replica to use when the caller did not pin
            ``ray_actor_options["num_gpus"]``.

    Returns:
        A :class:`DeploymentConfig` with ``ray_actor_options`` populated.
    """
    if not isinstance(deployment_config, DeploymentConfig):
        deployment_config = DeploymentConfig(**(deployment_config or {}))
    actor_opts = dict(deployment_config.ray_actor_options or {})
    if "num_gpus" not in actor_opts:
        actor_opts["num_gpus"] = default_num_gpus
        logging.info(
            "Replicas will request num_gpus=%s. Pass num_gpus=... or set "
            "ray_actor_options['num_gpus'] to override.",
            default_num_gpus,
        )
    deployment_config.ray_actor_options = actor_opts
    return deployment_config


def _check_predict_unit_device(predict_unit: MLIPPredictUnit, num_gpus: float) -> None:
    """
    Reject a predict unit pinned to a CUDA ordinal the replica will not have.

    Ray sets ``CUDA_VISIBLE_DEVICES`` per replica, so a replica granted GPUs
    only ever sees them as ``cuda:0`` upward. A unit whose tensors were
    serialized from, say, ``cuda:1`` fails to deserialize there with an opaque
    "invalid device ordinal" error, so fail here with an actionable one.

    Args:
        predict_unit: The unit about to be placed in the object store.
        num_gpus: GPUs granted to each replica.

    Raises:
        ValueError: If the unit is pinned to a CUDA ordinal at or beyond the
            number of GPUs the replica will be able to see.
    """
    device = torch.device(predict_unit.device)
    if device.type != "cuda" or num_gpus <= 0:
        return

    ordinal = device.index or 0
    if ordinal >= num_gpus:
        raise ValueError(
            f"predict_unit is on {predict_unit.device!r}, but each replica is "
            f"granted num_gpus={num_gpus} and Ray remaps CUDA_VISIBLE_DEVICES so "
            f"the replica only sees ordinals 0..{int(num_gpus) - 1}. Deserializing "
            "the unit there would fail with 'invalid device ordinal'. Load the "
            "predict unit on 'cuda:0' (or 'cpu') before serving it, or raise "
            "num_gpus."
        )


@dataclass
class _DeployedApp:
    """
    State needed to redeploy an app in place.

    Retained so :func:`update_batch_config` can push a new ``user_config``
    while holding every other deployment option byte-identical, which is what
    makes Ray Serve treat the redeploy as a lightweight update (``reconfigure``
    on the existing replicas) rather than a rolling restart.
    """

    deployment: Any
    options_kwargs: dict[str, Any]
    bind_args: tuple
    bind_kwargs: dict[str, Any]
    route_prefix: str
    batch_config: BatchConfig


# Process-local: only the process that deployed an app can redeploy it, since
# redeploying requires the bound arguments (e.g. the predict unit ObjectRef).
_DEPLOYED_APPS: dict[str, _DeployedApp] = {}


def _deploy_app(
    deployment: Any,
    options_kwargs: dict[str, Any],
    bind_args: tuple,
    bind_kwargs: dict[str, Any],
    batch_config: BatchConfig,
    deployment_name: str,
    route_prefix: str,
) -> serve.handle.DeploymentHandle:
    """
    Deploy (or redeploy) an app with batching pushed through ``user_config``.

    Batch settings are sent as ``user_config`` in addition to the constructor
    kwargs so that Ray Serve applies them to every replica via ``reconfigure``,
    including replicas that autoscaling adds later.
    """
    merged_options = {
        **options_kwargs,
        "user_config": {
            **(options_kwargs.get("user_config") or {}),
            **batch_config.to_user_config(),
        },
    }
    app = deployment.options(**merged_options).bind(*bind_args, **bind_kwargs)
    # ``serve.run`` blocks until the app reaches RUNNING, so a failed update
    # raises here instead of being silently dropped.
    handle = serve.run(app, name=deployment_name, route_prefix=route_prefix)
    _DEPLOYED_APPS[deployment_name] = _DeployedApp(
        deployment=deployment,
        options_kwargs=options_kwargs,
        bind_args=bind_args,
        bind_kwargs=bind_kwargs,
        route_prefix=route_prefix,
        batch_config=batch_config,
    )
    return handle


def update_batch_config(
    deployment_name: str,
    batch_config: BatchConfig | dict,
) -> BatchConfig:
    """
    Broadcast new batching parameters to every replica of a deployment.

    A ``DeploymentHandle`` method call reaches exactly one replica, so calling
    ``configure_batching.remote(...)`` leaves every other replica on its old
    settings. This pushes the new values through ``user_config`` instead, which
    Ray Serve applies fleet-wide via ``reconfigure``. Because only
    ``user_config`` changes, replicas are reconfigured in place rather than
    restarted, so no model is reloaded.

    Args:
        deployment_name: Name passed to the original ``setup_*`` call.
        batch_config: New :class:`BatchConfig` (or equivalent dict).

    Returns:
        The applied :class:`BatchConfig`.

    Raises:
        KeyError: If ``deployment_name`` was not deployed from this process.
    """
    if not isinstance(batch_config, BatchConfig):
        batch_config = BatchConfig(**(batch_config or {}))

    record = _DEPLOYED_APPS.get(deployment_name)
    if record is None:
        raise KeyError(
            f"No deployment named {deployment_name!r} was created by this "
            "process, so its batching config cannot be updated. Batch config "
            "updates must be issued from the process that called "
            "setup_batch_predict_server / setup_multiplexed_batch_predict_server."
        )

    _deploy_app(
        deployment=record.deployment,
        options_kwargs=record.options_kwargs,
        bind_args=record.bind_args,
        bind_kwargs=record.bind_kwargs,
        batch_config=batch_config,
        deployment_name=deployment_name,
        route_prefix=record.route_prefix,
    )
    logging.info(
        "Broadcast batch config to all replicas of %r: %s",
        deployment_name,
        batch_config,
    )
    return batch_config


def update_served_predict_unit(
    deployment_name: str,
    predict_unit: MLIPPredictUnit,
) -> None:
    """
    Replace the model served by every replica of a single-model deployment.

    Unlike ``update_predict_unit`` invoked through a handle -- which mutates
    one replica and leaves the others, plus any replica autoscaling adds later,
    serving the old checkpoint -- this rebinds the deployment so Ray Serve rolls
    the new checkpoint out to the whole fleet.

    Args:
        deployment_name: Name passed to the original ``setup_*`` call.
        predict_unit: The replacement predict unit.

    Raises:
        KeyError: If ``deployment_name`` was not deployed from this process.
    """
    record = _DEPLOYED_APPS.get(deployment_name)
    if record is None:
        raise KeyError(
            f"No deployment named {deployment_name!r} was created by this "
            "process, so its checkpoint cannot be updated."
        )

    predict_unit_ref = ray.put(predict_unit)
    _deploy_app(
        deployment=record.deployment,
        options_kwargs=record.options_kwargs,
        # The predict unit ref is the first bound positional argument.
        bind_args=(predict_unit_ref, *record.bind_args[1:]),
        bind_kwargs=record.bind_kwargs,
        batch_config=record.batch_config,
        deployment_name=deployment_name,
        route_prefix=record.route_prefix,
    )
    logging.info("Rolled new predict unit out to all replicas of %r", deployment_name)


def setup_batch_predict_server(
    predict_unit: MLIPPredictUnit,
    deployment_config: DeploymentConfig | dict | None = None,
    batch_config: BatchConfig | dict | None = None,
    deployment_name: str = "predict-server",
    route_prefix: str = "/predict",
    num_gpus: float | None = None,
) -> serve.handle.DeploymentHandle:
    """
    Deploy a ``BatchPredictServer`` that serves a single pre-loaded model.

    Args:
        predict_unit: An MLIPPredictUnit instance to use for inference.
        deployment_config: :class:`DeploymentConfig` (or equivalent dict) of
            kwargs forwarded to ``BatchPredictServer.options(...)``. Any field
            on :class:`DeploymentConfig` is valid (e.g. ``num_replicas``,
            ``autoscaling_config``, ``ray_actor_options``,
            ``max_ongoing_requests``, ``graceful_shutdown_timeout_s``,
            ``logging_config``).
        batch_config: :class:`BatchConfig` (or equivalent dict) of kwargs
            forwarded into ``BatchPredictServer.__init__`` via
            ``.bind(**batch_config)``. Accepts ``max_batch_size``,
            ``batch_wait_timeout_s``, and ``split_oom_batch``.
        deployment_name: Name for the Ray Serve deployment.
        route_prefix: HTTP route prefix for the deployment.
        num_gpus: GPUs to request per replica. Defaults to ``1`` when
            ``predict_unit`` is on CUDA and ``0`` otherwise. An explicit value
            in ``deployment_config["ray_actor_options"]["num_gpus"]`` wins over
            this argument.

    Returns:
        Ray Serve deployment handle.
    """
    if num_gpus is None:
        # Safe to infer here: the predict unit is a local object whose device is
        # ground truth for this deployment, unlike a driver-side CUDA probe.
        num_gpus = 1 if torch.device(predict_unit.device).type == "cuda" else 0

    dc = _prepare_deployment_config(deployment_config, num_gpus)
    if not isinstance(batch_config, BatchConfig):
        batch_config = BatchConfig(**(batch_config or {}))

    dc_kwargs = dc.to_options_kwargs()
    _check_predict_unit_device(predict_unit, dc_kwargs["ray_actor_options"]["num_gpus"])
    _init_ray_and_serve(dc_kwargs["ray_actor_options"], _effective_replicas(dc_kwargs))

    predict_unit_ref = ray.put(predict_unit)
    logging.info("Predict unit stored in Ray object store")

    handle = _deploy_app(
        deployment=BatchPredictServer,
        options_kwargs=dc_kwargs,
        bind_args=(predict_unit_ref,),
        bind_kwargs=batch_config.to_init_kwargs(),
        batch_config=batch_config,
        deployment_name=deployment_name,
        route_prefix=route_prefix,
    )
    logging.info(f"BatchPredictServer deployed: name={deployment_name}")
    return handle


def setup_multiplexed_batch_predict_server(
    deployment_config: DeploymentConfig | dict | None = None,
    batch_config: BatchConfig | dict | None = None,
    deployment_name: str = "multiplexed-predict-server",
    route_prefix: str = "/multiplex-predict",
    num_gpus: float | None = None,
) -> serve.handle.DeploymentHandle:
    """
    Deploy a ``MultiplexedBatchPredictServer`` that loads models on demand.

    Models are loaded lazily when a request arrives with a
    ``multiplexed_model_id`` set on the handle.

    Args:
        deployment_config: :class:`DeploymentConfig` (or equivalent dict)
            forwarded to ``MultiplexedBatchPredictServer.options(...)``.
        batch_config: :class:`BatchConfig` (or equivalent dict) forwarded
            into ``MultiplexedBatchPredictServer.__init__`` via
            ``.bind(**batch_config)``.
        deployment_name: Name for the Ray Serve deployment.
        route_prefix: HTTP route prefix for the deployment.
        num_gpus: GPUs to request per replica. **Set this explicitly when the
            driver and the replicas may run on different hardware.** There is
            no local model to infer from, so leaving it ``None`` falls back to
            the driver's own CUDA visibility, which is wrong when deploying
            from a CPU login node to a GPU cluster.

    Returns:
        Ray Serve deployment handle.
    """
    if num_gpus is None:
        # There is no local model to consult, so this falls back to probing the
        # *driver's* CUDA visibility -- which is wrong whenever the driver and
        # the replicas are on different hardware (e.g. submitting from a CPU
        # login node to a GPU cluster). Warn loudly rather than silently
        # deploying a GPU model onto CPU replicas.
        num_gpus = 1 if torch.cuda.is_available() else 0
        if num_gpus == 0:
            logging.warning(
                "No num_gpus was given and this driver sees no CUDA device, so "
                "replicas will be scheduled with num_gpus=0 and every model will "
                "load on CPU -- even if the cluster has idle GPUs. Pass "
                "num_gpus=1 explicitly when deploying from a CPU-only host to a "
                "GPU cluster."
            )

    dc = _prepare_deployment_config(deployment_config, num_gpus)
    if not isinstance(batch_config, BatchConfig):
        batch_config = BatchConfig(**(batch_config or {}))

    dc_kwargs = dc.to_options_kwargs()
    _init_ray_and_serve(dc_kwargs["ray_actor_options"], _effective_replicas(dc_kwargs))

    handle = _deploy_app(
        deployment=MultiplexedBatchPredictServer,
        options_kwargs=dc_kwargs,
        bind_args=(),
        bind_kwargs=batch_config.to_init_kwargs(),
        batch_config=batch_config,
        deployment_name=deployment_name,
        route_prefix=route_prefix,
    )
    logging.info(f"MultiplexedBatchPredictServer deployed: name={deployment_name}")
    return handle


@dataclass
class AutobatchConfig:
    """
    Configuration for probing-based autobatching.

    Attributes:
        min_batch_size: Minimum batch size (in atoms) to start probing from.
        max_batch_size_cap: Maximum batch size cap to avoid excessive probing.
        probe_steps: Number of probe steps to run at each batch size.
        backoff_factor: Factor to reduce batch size by after OOM (e.g., 0.8 = 80%).
        timeout_floor_s: Minimum batch wait timeout in seconds.
        timeout_ceil_s: Maximum batch wait timeout in seconds.
        timeout_latency_multiplier: Multiplier applied to the median single-request
            latency to compute the batch wait timeout.
        warmup_steps: Number of warmup inference steps before probing.
    """

    min_batch_size: int = 128
    max_batch_size_cap: int = 16384
    probe_steps: int = 3
    backoff_factor: float = 0.8
    timeout_floor_s: float = 0.01
    timeout_ceil_s: float = 1.0
    timeout_latency_multiplier: float = 2.0
    warmup_steps: int = 2


@dataclass
class AutobatchResult:
    """
    Result from autobatch probing.

    Attributes:
        max_batch_size: Optimal maximum batch size in atoms.
        batch_wait_timeout_s: Optimal batch wait timeout in seconds.
        median_latency_s: Median single-request inference latency observed during
            probing (the basis for ``batch_wait_timeout_s``).
        probe_timestamp: Unix timestamp when probing was performed.
    """

    max_batch_size: int
    batch_wait_timeout_s: float
    median_latency_s: float
    probe_timestamp: float = field(default_factory=time.time)


def _expand_probe_data(
    data_list: list[AtomicData], target_num_atoms: int
) -> list[AtomicData]:
    """
    Expand probe data by repeating items to reach target atom count.

    Args:
        data_list: List of AtomicData objects to use as base data.
        target_num_atoms: Target total number of atoms for the batch.

    Returns:
        List of AtomicData objects with total atoms >= target_num_atoms.
    """
    if not data_list:
        raise ValueError("data_list cannot be empty")

    base_num_atoms = sum(data.natoms.sum().item() for data in data_list)

    if base_num_atoms >= target_num_atoms:
        return data_list

    num_repeats = (target_num_atoms + base_num_atoms - 1) // base_num_atoms

    expanded_list = []
    for _ in range(num_repeats):
        expanded_list.extend(data_list)

    return expanded_list


def probe_optimal_batch_size(
    predict_unit: MLIPPredictUnit,
    probe_data: list[AtomicData],
    config: AutobatchConfig | None = None,
) -> AutobatchResult:
    """
    Probe for optimal batch size and timeout using runtime GPU memory behavior.

    This function performs a binary search-like probing to find the maximum
    batch size that doesn't cause OOM errors, then derives an appropriate
    batch wait timeout from the measured single-request latency.

    Args:
        predict_unit: The MLIPPredictUnit to probe with.
        probe_data: List of AtomicData objects to use for probing. If the total
            number of atoms is less than the target batch size being probed,
            the data will be repeated to reach the target size.
        config: Autobatch configuration. Uses defaults if None.

    Returns:
        AutobatchResult with optimal parameters.
    """
    if config is None:
        config = AutobatchConfig()

    if not probe_data:
        raise ValueError("probe_data cannot be empty")

    device = predict_unit.device

    if "cuda" not in str(device):
        logging.info("Autobatch probing skipped for CPU device, using defaults")
        return AutobatchResult(
            max_batch_size=config.min_batch_size,
            batch_wait_timeout_s=config.timeout_ceil_s,
            median_latency_s=0.1,
        )

    logging.info("Starting autobatch probing...")

    free_hardware, total_mem = (
        (0, 0) if not torch.cuda.is_available() else torch.cuda.mem_get_info()
    )
    torch_unused_cache = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    free_mem = free_hardware + torch_unused_cache

    logging.info(
        f"GPU memory: {free_mem / 1e9:.2f}GB free / {total_mem / 1e9:.2f}GB total"
    )

    logging.info(f"Running {config.warmup_steps} warmup steps...")
    warmup_batch = atomicdata_list_to_batch(probe_data)
    for _ in range(config.warmup_steps):
        try:
            predict_unit.predict(warmup_batch, undo_element_references=False)
        except Exception as exc:
            logging.warning(f"Warmup step failed: {exc}")
    torch.cuda.empty_cache()

    # Measure single-request latency (the probe data as-is, unexpanded). The
    # batch_wait_timeout must track how long a *single* request takes so that
    # requests arriving close together get batched without a lone request
    # waiting an eternity. Deriving it from the large probed batches instead
    # (which can be seconds) would saturate ``timeout_ceil_s`` and make every
    # sub-batch-filling request wait that long -- crippling low-concurrency
    # throughput.
    single_request_latencies: list[float] = []
    for step in range(config.probe_steps):
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            predict_unit.predict(warmup_batch, undo_element_references=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            single_request_latencies.append(elapsed)
            logging.debug(f"  Single-request step {step + 1}: {elapsed:.4f}s")
        except Exception as exc:
            logging.warning(f"  Single-request latency probe failed: {exc}")
    torch.cuda.empty_cache()

    # Binary search for optimal batch size. Only success/OOM matters here;
    # the timeout is derived separately from the single-request latency above.
    low = config.min_batch_size
    high = config.max_batch_size_cap
    best_batch_size = low

    logging.info(f"Probing batch sizes in range [{low}, {high}]...")

    while low <= high:
        mid = (low + high) // 2
        success = True

        logging.debug(f"Testing batch size: {mid} atoms")

        for step in range(config.probe_steps):
            try:
                expanded_data = _expand_probe_data(probe_data, mid)
                batch = atomicdata_list_to_batch(expanded_data)

                torch.cuda.synchronize()
                start = time.perf_counter()
                predict_unit.predict(batch, undo_element_references=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                logging.debug(f"  Step {step + 1}: {elapsed:.4f}s")
            except torch.OutOfMemoryError:
                logging.debug(f"  OOM at batch size {mid}")
                success = False
                torch.cuda.empty_cache()
                break
            except Exception as exc:
                logging.warning(f"  Probe failed at batch size {mid}: {exc}")
                success = False
                break

        if success:
            best_batch_size = mid
            low = mid + 1
            logging.debug(f"  Success at {mid}, trying larger...")
        else:
            high = mid - 1
            logging.debug(f"  Failed at {mid}, trying smaller...")

    final_batch_size = int(best_batch_size * config.backoff_factor)
    final_batch_size = max(final_batch_size, config.min_batch_size)

    # Compute timeout from the single-request latency, not the large probed
    # batches. The timeout is how long the server lingers to accumulate a
    # batch; keying it to one request's latency keeps a lone request from
    # stalling while still letting near-simultaneous requests coalesce.
    if single_request_latencies:
        sorted_latencies = sorted(single_request_latencies)
        median_latency = sorted_latencies[len(sorted_latencies) // 2]
        timeout = median_latency * config.timeout_latency_multiplier
        timeout = max(config.timeout_floor_s, min(timeout, config.timeout_ceil_s))
    else:
        median_latency = 0.1
        timeout = config.timeout_ceil_s

    result = AutobatchResult(
        max_batch_size=final_batch_size,
        batch_wait_timeout_s=timeout,
        median_latency_s=median_latency,
    )

    logging.info(
        f"Autobatch probing complete: max_batch_size={result.max_batch_size}, "
        f"timeout={result.batch_wait_timeout_s:.4f}s, "
        f"median_latency={result.median_latency_s:.4f}s"
    )

    return result
