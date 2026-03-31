"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

import ray
import torch
from ray import serve
from ray.serve.schema import ApplicationStatus

from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit import MLIPPredictUnit


@serve.deployment(
    logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
    max_ongoing_requests=300,
)
class BatchPredictServer:
    """
    Ray Serve deployment that batches incoming inference requests.
    """

    def __init__(
        self,
        predict_unit_ref,
        max_batch_size: int,
        batch_wait_timeout_s: float,
        split_oom_batch: bool = True,
    ):
        """
        Initialize with a Ray object reference to a PredictUnit.

        Args:
            predict_unit_ref: Ray object reference to an MLIPPredictUnit instance
            max_batch_size: Maximum number of atoms in a batch.
                The actual number of atoms will likely be larger than this as batches
                are split when num atoms exceeds this value.
            batch_wait_timeout_s: Timeout in seconds to wait for a prediction
            split_oom_batch: If true will split batch if an OOM error is raised
        """
        self.predict_unit = ray.get(predict_unit_ref)
        self.split_oom_batch = split_oom_batch
        self.configure_batching(max_batch_size, batch_wait_timeout_s)

        logging.info("BatchedPredictor initialized with predict_unit from object store")

    def configure_batching(
        self,
        max_batch_size: int,
        batch_wait_timeout_s: float,
    ):
        self.predict.set_max_batch_size(max_batch_size)
        self.predict.set_batch_wait_timeout_s(batch_wait_timeout_s)

    def get_predict_unit_attribute(self, attribute_name: str, **kwargs) -> Any:
        return getattr(self.predict_unit, attribute_name)

    def validate_atoms_data(self, atoms_info: dict, task_name: str) -> dict:
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

    def update_predict_unit(self, predict_unit_ref) -> None:
        """Update the predict unit with a new checkpoint.

        Args:
            predict_unit_ref: Ray object reference to a new MLIPPredictUnit instance
        """
        self.predict_unit = ray.get(predict_unit_ref)
        logging.info("BatchPredictServer predict_unit updated from object store")

    @serve.batch(
        batch_size_fn=lambda batch: sum(sample.natoms.sum() for sample in batch).item()
    )
    async def predict(
        self, data_list: list[AtomicData], undo_element_references: bool = True
    ) -> list[dict]:
        """
        Process a batch of AtomicData objects.

        Args:
            data_list: List of AtomicData objects (automatically batched by Ray Serve)
            undo_element_references: Whether to undo element references in predictions

        Returns:
            List of prediction dictionaries, one per input
        """
        data_deque = deque([data_list])
        prediction_list = []
        while len(data_deque) > 0:
            oom = False
            data_list = data_deque.popleft()
            batch = atomicdata_list_to_batch(data_list)

            try:
                predictions = self.predict_unit.predict(
                    batch, undo_element_references=undo_element_references
                )
                prediction_list.extend(self._split_predictions(predictions, batch))
            except torch.OutOfMemoryError as err:
                if not self.split_oom_batch:
                    raise torch.OutOfMemoryError(
                        "Reduce max_batch_size or set split_oom_batch=True to automatically split OOM batches."
                    ) from err

                if len(data_list) == 1:
                    raise torch.OutOfMemoryError(
                        "Out of memory for a single system left in batch."
                    ) from err

                logging.warning(
                    "Caught out of memory error. Splitting batch and retrying."
                )
                oom = True
                torch.cuda.empty_cache()

            if oom:
                mid = len(data_list) // 2
                data_deque.appendleft(data_list[mid:])
                data_deque.appendleft(data_list[:mid])

        return prediction_list

    async def __call__(
        self, data: AtomicData, undo_element_references: bool = True
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

            split_preds.append(system_predictions)

        return split_preds


@serve.deployment(
    logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
    max_ongoing_requests=300,
)
class MultiplexedBatchPredictServer(BatchPredictServer):
    """
    Ray Serve deployment that supports multiplexed model loading.

    Unlike ``BatchPredictServer`` which serves a single pre-loaded model,
    this deployment loads models on demand using ``@serve.multiplexed``.
    Different clients can request different models by passing a ``model_id``
    and an LRU cache keeps up to ``max_num_models_per_replica`` models
    resident on each replica.
    """

    def __init__(
        self,
        max_batch_size: int,
        batch_wait_timeout_s: float,
        split_oom_batch: bool = True,
    ):
        """
        Initialize the multiplexed predict server.

        Args:
            max_batch_size: Maximum number of atoms in a batch.
            batch_wait_timeout_s: Timeout in seconds to wait for a prediction.
            split_oom_batch: If true will split batch if an OOM error is raised.
        """
        # Skip BatchPredictServer.__init__ — no predict_unit_ref needed.
        self.split_oom_batch = split_oom_batch
        self.configure_batching(max_batch_size, batch_wait_timeout_s)
        logging.info("MultiplexedBatchPredictServer initialized")

    @serve.multiplexed(max_num_models_per_replica=3)
    async def get_model(self, model_id: str):
        """
        Load (or retrieve from cache) a predict unit by model key.

        Args:
            model_id: Key in the format ``"checkpoint_name_or_path:settings"``
                where ``settings`` is one of the recognized inference setting
                names (e.g. ``"default"``, ``"turbo"``) or an empty string for
                the default settings.
        """
        parts = model_id.split(":", 1)
        checkpoint = parts[0]
        settings_name = parts[1] if len(parts) > 1 and parts[1] else "default"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if os.path.isfile(checkpoint):
            from fairchem.core.units.mlip_unit import load_predict_unit

            predict_unit = load_predict_unit(
                checkpoint,
                inference_settings=settings_name,
                device=device,
            )
        else:
            from fairchem.core.calculate import pretrained_mlip

            predict_unit = pretrained_mlip.get_predict_unit(
                checkpoint,
                inference_settings=settings_name,
                device=device,
            )

        self.predict_unit = predict_unit
        logging.info(f"MultiplexedBatchPredictServer loaded model_id={model_id!r}")

    async def get_predict_unit_attribute(
        self, attribute_name: str, model_id: str | None = None
    ) -> Any:
        """
        Get an attribute from a loaded predict unit.

        Uses the ``multiplexed_model_id`` set on the request by the caller
        to resolve the correct model first.
        """
        model_id = model_id or serve.get_multiplexed_model_id()
        await self.get_model(model_id)
        return getattr(self.predict_unit, attribute_name)

    async def validate_atoms_data(self, atoms_info: dict, task_name: str) -> dict:
        """
        Run model-specific validation after loading the correct model.
        """
        model_id = serve.get_multiplexed_model_id()
        await self.get_model(model_id)
        return super().validate_atoms_data(self, atoms_info, task_name)

    async def __call__(
        self, data: AtomicData, undo_element_references: bool = True
    ) -> dict:
        """
        Main entry point for multiplexed inference requests.

        The ``multiplexed_model_id`` is read from the Ray Serve request
        context (set by the caller via ``handle.options(multiplexed_model_id=...)``).

        Args:
            data: Single AtomicData object.
            undo_element_references: Whether to undo element references.

        Returns:
            Prediction dictionary for this system.
        """
        model_id = serve.get_multiplexed_model_id()
        await self.get_model(model_id)
        predictions = await self.predict(data, undo_element_references)
        return predictions


def _init_ray_and_serve(
    ray_actor_options: dict,
    num_replicas: int,
) -> None:
    """
    Ensure Ray and Ray Serve are initialised.
    """
    cpus_per_actor = ray_actor_options.get("num_cpus", min(cpu_count(), 8))
    ray_actor_options["num_cpus"] = cpus_per_actor

    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,
            logging_config=ray.LoggingConfig(log_level="WARNING"),
            num_cpus=cpus_per_actor * num_replicas,
        )
        logging.info("Ray initialized")

    try:
        serve.status()
        logging.info("Ray Serve already running")
    except Exception:
        serve.start(
            logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
        )
        logging.info("Ray Serve started")


def _build_deployment_options(
    ray_actor_options: dict,
    num_replicas: int,
    autoscaling_config: dict | None,
) -> dict:
    """
    Build the ``deployment_options`` dict for ``Server.options()``.
    """
    deployment_options: dict[str, Any] = {
        "ray_actor_options": ray_actor_options,
    }
    if autoscaling_config is not None:
        deployment_options["autoscaling_config"] = autoscaling_config
    else:
        deployment_options["num_replicas"] = num_replicas
    return deployment_options


def setup_batch_predict_server(
    predict_unit: MLIPPredictUnit,
    max_batch_size: int = 512,
    batch_wait_timeout_s: float = 0.1,
    split_oom_batch: bool = True,
    num_replicas: int = 1,
    ray_actor_options: dict | None = None,
    deployment_name: str = "fairchem-inference",
    route_prefix: str = "/predict",
    autoscaling_config: dict | None = None,
) -> serve.handle.DeploymentHandle:
    """
    Deploy a ``BatchPredictServer`` that serves a single pre-loaded model.

    Args:
        predict_unit: An MLIPPredictUnit instance to use for inference.
        max_batch_size: Maximum number of atoms in a batch.
        batch_wait_timeout_s: Maximum wait time before processing partial batch.
        split_oom_batch: Whether to split batches that cause OOM errors.
        num_replicas: Number of deployment replicas. Ignored if
            *autoscaling_config* is provided.
        ray_actor_options: Additional Ray actor options
            (e.g., ``{"num_gpus": 1, "num_cpus": 4}``).
        deployment_name: Name for the Ray Serve deployment.
        route_prefix: HTTP route prefix for the deployment.
        autoscaling_config: Optional autoscaling configuration dict.

    Returns:
        Ray Serve deployment handle.
    """
    if ray_actor_options is None:
        ray_actor_options = {}

    if "cuda" in predict_unit.device and "num_gpus" not in ray_actor_options:
        ray_actor_options["num_gpus"] = 1

    _init_ray_and_serve(ray_actor_options, num_replicas)

    predict_unit_ref = ray.put(predict_unit)
    logging.info("Predict unit stored in Ray object store")

    deployment_options = _build_deployment_options(
        ray_actor_options, num_replicas, autoscaling_config
    )
    deployment = BatchPredictServer.options(**deployment_options).bind(
        predict_unit_ref,
        max_batch_size=max_batch_size,
        batch_wait_timeout_s=batch_wait_timeout_s,
        split_oom_batch=split_oom_batch,
    )

    handle = serve.run(deployment, name=deployment_name, route_prefix=route_prefix)
    logging.info(f"BatchPredictServer deployed: name={deployment_name}")
    return handle


def setup_multiplexed_batch_predict_server(
    max_batch_size: int = 512,
    batch_wait_timeout_s: float = 0.1,
    split_oom_batch: bool = True,
    num_replicas: int = 1,
    ray_actor_options: dict | None = None,
    deployment_name: str = "fairchem-inference",
    route_prefix: str = "/predict",
    autoscaling_config: dict | None = None,
) -> serve.handle.DeploymentHandle:
    """
    Deploy a ``MultiplexedBatchPredictServer`` that loads models on demand.

    Models are loaded lazily when a request arrives with a
    ``multiplexed_model_id`` set on the handle.

    Args:
        max_batch_size: Maximum number of atoms in a batch.
        batch_wait_timeout_s: Maximum wait time before processing partial batch.
        split_oom_batch: Whether to split batches that cause OOM errors.
        num_replicas: Number of deployment replicas. Ignored if
            *autoscaling_config* is provided.
        ray_actor_options: Additional Ray actor options
            (e.g., ``{"num_gpus": 1, "num_cpus": 4}``).
        deployment_name: Name for the Ray Serve deployment.
        route_prefix: HTTP route prefix for the deployment.
        autoscaling_config: Optional autoscaling configuration dict.

    Returns:
        Ray Serve deployment handle.
    """
    if ray_actor_options is None:
        ray_actor_options = {}

    if torch.cuda.is_available() and "num_gpus" not in ray_actor_options:
        ray_actor_options["num_gpus"] = 1

    _init_ray_and_serve(ray_actor_options, num_replicas)

    deployment_options = _build_deployment_options(
        ray_actor_options, num_replicas, autoscaling_config
    )
    deployment = MultiplexedBatchPredictServer.options(**deployment_options).bind(
        max_batch_size=max_batch_size,
        batch_wait_timeout_s=batch_wait_timeout_s,
        split_oom_batch=split_oom_batch,
    )

    handle = serve.run(deployment, name=deployment_name, route_prefix=route_prefix)
    logging.info(f"MultiplexedBatchPredictServer deployed: name={deployment_name}")
    return handle


def wait_for_serve_ready(
    app_name: str = "fairchem-inference",
    poll_interval_seconds: float = 2,
) -> bool:
    """
    Wait for Ray Serve to be fully ready to accept requests.

    Blocks until the Ray Serve controller is running and the specified
    application reaches RUNNING status.

    Args:
        app_name: Name of the Ray Serve application to wait for.
        poll_interval_seconds: How often to check status.

    Returns:
        True if server is ready.

    Raises:
        RuntimeError: If server fails to deploy.
    """

    # Phase 1: Wait for Ray Serve controller
    logging.info("Waiting for Ray Serve controller to start...")
    while True:
        try:
            status = serve.status()
            logging.info("Ray Serve controller is running")
            break
        except Exception as e:
            error_msg = str(e)
            if (
                "SERVE_CONTROLLER_ACTOR" in error_msg
                or "Failed to look up actor" in error_msg
            ):
                logging.debug(f"Ray Serve controller not ready yet: {error_msg}")
                time.sleep(poll_interval_seconds)
            else:
                raise

    # Phase 2: Wait for the application to be deployed and running
    logging.info(f"Waiting for application '{app_name}' to be ready...")
    while True:
        try:
            status = serve.status()

            if app_name not in status.applications:
                logging.debug(f"Application '{app_name}' not found yet, waiting...")
                time.sleep(poll_interval_seconds)
                continue

            app_status = status.applications[app_name]

            if app_status.status == ApplicationStatus.RUNNING:
                logging.info(f"Application '{app_name}' is RUNNING and ready")
                return True
            elif app_status.status == ApplicationStatus.DEPLOYING:
                logging.debug(f"Application '{app_name}' is still deploying...")
                time.sleep(poll_interval_seconds)
            elif app_status.status in (
                ApplicationStatus.DEPLOY_FAILED,
                ApplicationStatus.UNHEALTHY,
            ):
                raise RuntimeError(
                    f"Application '{app_name}' failed to deploy. "
                    f"Status: {app_status.status}, Message: {app_status.message}"
                )
            else:
                logging.debug(f"Application '{app_name}' status: {app_status.status}")
                time.sleep(poll_interval_seconds)

        except RuntimeError:
            raise
        except Exception as e:
            logging.warning(f"Error checking serve status: {e}")
            time.sleep(poll_interval_seconds)


def get_ray_connection_info(head_file: str) -> dict[str, str | None]:
    """
    Read Ray connection info from a head.json file.

    Args:
        head_file: Path to head.json file from a Ray cluster.

    Returns:
        Dictionary with ``ray_address``, ``namespace_serve_fairchem``, and
        ``local`` keys. For local clusters ``ray_address`` is *None*.
    """
    with open(head_file) as f:
        head_info = json.load(f)

    namespace_serve_fairchem = head_info.get("namespace_serve_fairchem")
    is_local = head_info.get("local", False)

    if is_local:
        return {
            "ray_address": None,
            "namespace_serve_fairchem": namespace_serve_fairchem,
            "local": True,
        }

    hostname = head_info.get("hostname")
    client_port = head_info.get("client_port")

    if not hostname or not client_port:
        raise ValueError(
            f"Invalid head.json: missing hostname or client_port in {head_file}"
        )

    return {
        "ray_address": f"ray://{hostname}:{client_port}",
        "namespace_serve_fairchem": namespace_serve_fairchem,
        "local": False,
    }
