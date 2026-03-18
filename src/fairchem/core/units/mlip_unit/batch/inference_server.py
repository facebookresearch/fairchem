"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

FAIRChem Inference Server - Multiplexed Ray Serve deployment.

This server runs on the Ray cluster and handles all FAIRChem inference requests:
- Automatic batching of inference requests
- Model multiplexing (multiple checkpoints, LRU eviction)
- Scale-to-zero when idle
- GPU-efficient inference

Model identification:
- model_key format: "{checkpoint_name}:{inference_settings}"
- task_name comes from AtomicData.dataset at inference time (not in model_key)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from ray import serve

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit import MLIPPredictUnit

logger = logging.getLogger(__name__)

def batch_size_natoms(items):
    return sum(int(sample["atomic_data"].natoms) for sample in items)

class FAIRChemInferenceServer:
    """
    Multiplexed FAIRChem inference server.

    Handles multiple model checkpoints with LRU eviction.
    Automatically batches incoming inference requests.

    Model key format: "{checkpoint_name}:{inference_settings}"
    Task name comes from AtomicData.dataset at inference time.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_metadata_cache = {}
        logger.info(f"FAIRChemInferenceServer initialized on device: {self.device}")

    @serve.multiplexed(max_num_models_per_replica=3)
    async def get_model(self, model_key: str) -> MLIPPredictUnit:
        """
        Load model on demand. Models are cached with LRU eviction.

        model_key format: "{checkpoint_name_or_path}:{inference_settings}"
        e.g., "uma-s-1p1:default" or "/path/to/model.pt:default"

        If checkpoint_name looks like a file path (ends with .pt or contains /),
        it will be loaded directly from that path using load_predict_unit.
        Otherwise, it will be loaded from the pretrained model registry.

        Note: task_name is NOT part of model_key - it comes from
        AtomicData.dataset at inference time.
        """
        from fairchem.core.calculate import pretrained_mlip
        from fairchem.core.units.mlip_unit import load_predict_unit

        parts = model_key.split(":")
        checkpoint_name = parts[0]
        inference_settings = parts[1] if len(parts) > 1 else "default"

        start_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None

        if start_time:
            start_time.record()

        logger.info(f"Loading model '{model_key}'")

        # Check if this is a local checkpoint path or a pretrained model name
        is_local_path = checkpoint_name.endswith(".pt") or "/" in checkpoint_name or "\\" in checkpoint_name

        if is_local_path:
            logger.info(f"Loading local checkpoint from path: {checkpoint_name}")
            predict_unit = load_predict_unit(
                checkpoint_name,
                inference_settings=inference_settings,
                device=self.device,
            )
        else:
            predict_unit = pretrained_mlip.get_predict_unit(
                checkpoint_name,
                inference_settings=inference_settings,
                device=self.device,
            )

        # Cache metadata for this model
        self._cache_model_metadata(model_key, predict_unit)

        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            load_time = start_time.elapsed_time(end_time)
            logger.info(f"Successfully loaded model '{model_key}' in {load_time:.1f}ms.")
        else:
            logger.info(f"Successfully loaded model '{model_key}'")

        return predict_unit

    def _cache_model_metadata(self, model_key: str, predict_unit: MLIPPredictUnit):
        """Cache model metadata for client queries."""
        self._model_metadata_cache[model_key] = {
            "form_elem_refs": getattr(predict_unit, "form_elem_refs", {}),
            "atom_refs": getattr(predict_unit, "atom_refs", {}),
            "dataset_to_tasks": {
                name: [
                    {"name": t.name, "property": t.property, "level": t.level}
                    for t in tasks
                ]
                for name, tasks in predict_unit.dataset_to_tasks.items()
            },
        }

    def get_model_metadata(self, model_key: str) -> dict:
        """Get cached metadata for a model."""
        return self._model_metadata_cache.get(model_key, {})

    async def fetch_model_metadata(self, model_key: str) -> dict[str, Any]:
        """
        Fetch metadata for a model, loading it if necessary.

        This ensures the model is loaded and returns its metadata
        (dataset_to_tasks, form_elem_refs, atom_refs).
        """
        # Ensure model is loaded (will use cached if already loaded)
        await self.get_model(model_key)

        # Return the cached metadata
        return self._model_metadata_cache.get(model_key, {})

    async def _predict_batch_impl(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Batched inference implementation - processes a batch of requests.

        This is the core implementation; the @serve.batch decorator is applied
        dynamically in start_serve() to allow runtime configuration.

        Each request dict contains:
        - model_key: str - the model identifier (checkpoint:settings)
        - atomic_data: AtomicData - the input structure (with .dataset set to task_name)
        - undo_element_references: bool - whether to undo element references

        NOTE: The @serve.batch decorator batches requests by timing, NOT by model_key.
        We must group requests by model_key to ensure each model only processes its
        own requests. Otherwise, requests for different models (e.g., force-predicting
        UMA vs. optical property model) could be incorrectly batched together.
        """
        from collections import defaultdict

        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

        if not requests:
            return []

        # Group requests by model_key to ensure each model processes only its requests
        # This is critical: @serve.batch batches by timing, not by model_key!
        requests_by_model: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for idx, req in enumerate(requests):
            model_key = req["model_key"]
            requests_by_model[model_key].append((idx, req))

        # Pre-allocate results array to preserve original request ordering
        results: list[dict[str, Any] | None] = [None] * len(requests)

        # Process each model group separately
        for model_key, indexed_requests in requests_by_model.items():
            predict_unit = await self.get_model(model_key)

            indices = [idx for idx, _ in indexed_requests]
            model_requests = [req for _, req in indexed_requests]

            # Extract atomic data from this model's requests
            data_list = [req["atomic_data"] for req in model_requests]
            undo_refs = model_requests[0].get("undo_element_references", True)

            # Batch the data for this model
            batch = atomicdata_list_to_batch(data_list)

            # Run inference
            with torch.no_grad():
                predictions = predict_unit.predict(batch, undo_element_references=undo_refs)

            # Split predictions back to individual results
            split_preds = self._split_predictions(predictions, batch, len(model_requests))

            # Place results back in original positions
            for idx, pred in zip(indices, split_preds, strict=False):
                results[idx] = pred

        return results

    def _split_predictions(
        self,
        predictions: dict[str, torch.Tensor],
        batch: Any,
        num_systems: int,
    ) -> list[dict[str, Any]]:
        """Split batched predictions back to per-system results."""
        results = []

        # Get batch indices
        batch_indices = batch.batch if hasattr(batch, "batch") else None
        _natoms = batch.natoms if hasattr(batch, "natoms") else None

        for i in range(num_systems):
            system_pred = {}

            for key, value in predictions.items():
                if value is None:
                    system_pred[key] = None
                elif key in ("energy", "free_energy"):
                    # Per-system scalar
                    system_pred[key] = float(value[i].cpu())
                elif key in ("forces", "stress"):
                    # Per-atom or per-system tensor
                    if batch_indices is not None and key == "forces":
                        mask = batch_indices == i
                        system_pred[key] = value[mask].cpu().numpy()
                    elif key == "stress" and value.dim() > 1:
                        system_pred[key] = value[i].cpu().numpy()
                    else:
                        system_pred[key] = value.cpu().numpy()
                else:
                    # Try to handle generically
                    if value.dim() == 0:
                        system_pred[key] = float(value.cpu())
                    elif value.shape[0] == num_systems:
                        system_pred[key] = value[i].cpu().numpy()
                    else:
                        # Assume per-atom, use batch indices
                        if batch_indices is not None:
                            mask = batch_indices == i
                            system_pred[key] = value[mask].cpu().numpy()
                        else:
                            system_pred[key] = value.cpu().numpy()

            results.append(system_pred)

        return results

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Single request entry point - handles both predictions and metadata requests.

        Prediction requests get auto-batched by @serve.batch.
        Metadata requests are handled directly.
        """
        # Handle metadata request
        if request.get("request_type") == "metadata":
            return await self.fetch_model_metadata(request["model_key"])

        # Prediction request - gets auto-batched by @serve.batch
        # Ray Serve automatically handles batching/unbatching, so we get a single result back
        # Note: predict_batch is defined dynamically in start_serve with @serve.batch
        return await self.predict_batch(request)


# Create the deployment
# app = FAIRChemInferenceServer.bind()


def _find_free_port() -> int:
    """Find an available port on localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def start_serve(
    http_port: int | None = None,
    route_prefix: str = "/inference",
    num_workers: int | None = None,
    log_level: str = "WARNING",
    batch_config: dict[str, Any] | None = None,
    deployment_config: dict[str, Any] | None = None,
) -> int:
    """
    Start Ray Serve with the FAIRChem inference server.

    Should be called once when the Ray cluster starts.

    Args:
        http_port: Port for HTTP server. If None, finds an available port automatically.
        route_prefix: URL prefix for the inference endpoint
        num_workers: Number of workers in the cluster. Used to set max_replicas.
            If None, defaults to 1 replica.
        log_level: Log level for Ray Serve deployment (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Default: "WARNING"
        batch_config: Dict with batch settings. Expected keys:
            - max_batch_size: Max batch size in atoms
            - batch_wait_timeout_s: Max time to wait for batch to fill
            - max_concurrent_batches: Max concurrent batches per replica
        deployment_config: Dict matching @serve.deployment() kwargs structure.
            Passed directly to the deployment decorator. Runtime values
            (max_replicas, max_ongoing_requests, logging_config) are merged in.

    Returns:
        The port number the HTTP server is running on.
    """
    import copy
    import math

    from ray import serve


    # Use provided configs or empty dicts
    batch_config = batch_config or {}
    deployment_config = copy.deepcopy(deployment_config) if deployment_config else {}

    # Merge in runtime-computed values
    if "autoscaling_config" not in deployment_config:
        deployment_config["autoscaling_config"] = {}

    # Set max replicas from num_workers (not ray.cluster_resources() which may not
    # reflect all nodes if called before they've joined)
    if "max_replicas" not in deployment_config["autoscaling_config"]:
        max_replicas = num_workers or 1
        logger.info(f"Setting max_replicas={max_replicas} based on num_workers={num_workers}")
    elif isinstance(deployment_config["autoscaling_config"].get("max_replicas"), float):
        max_replicas = int(deployment_config["autoscaling_config"]["max_replicas"] * num_workers)
        logger.info(f"Setting max_replicas={max_replicas} from fraction * num_workers")
    else:
        max_replicas = deployment_config["autoscaling_config"].get("max_replicas")

    deployment_config["autoscaling_config"]["max_replicas"] = max_replicas

    # If min_replicas is a fraction, interpret as fraction of max_replicas
    min_replicas = deployment_config["autoscaling_config"].get("min_replicas", 1)
    if isinstance(min_replicas, float):
        deployment_config["autoscaling_config"]["min_replicas"] = math.ceil(min_replicas * max_replicas)
        logger.info(f"Setting min_replicas={deployment_config['autoscaling_config']['min_replicas']} from fraction {min_replicas} * {max_replicas}")

    deployment_config["logging_config"] = {
        "log_level": log_level,
        "enable_access_log": log_level == "DEBUG",
    }

    # Start Serve
    serve.start(
        detached=True,
        http_options={
            "location": "NoServer",  # We don't want Serve to start its own HTTP server since we'll use Ray Serve's internal API
        },
    )

    # Create a configured subclass with the batch decorator applied
    # This allows batch settings to be configured at deployment time
    @serve.deployment(**deployment_config)
    class ConfiguredFAIRChemInferenceServer(FAIRChemInferenceServer):
        @serve.batch(
            batch_size_fn=batch_size_natoms,
            **batch_config
        )
        async def predict_batch(
            self,
            requests: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            return await self._predict_batch_impl(requests)

    # Create deployment with GPU-aware autoscaling
    deployment = ConfiguredFAIRChemInferenceServer.bind()

    # Deploy the application
    serve.run(
        deployment,
        name="fairchem_inference",
        route_prefix=route_prefix,
    )

    logger.info(f"FAIRChem inference server started on port {http_port}")
    logger.info(f"Endpoint: http://localhost:{http_port}{route_prefix}")

    return http_port


def wait_for_serve_ready(
    app_name: str = "fairchem_inference",
    poll_interval_seconds: float = 2,
) -> bool:
    """
    Wait for Ray Serve to be fully ready to accept requests.

    This function blocks until:
    1. Ray Serve controller is running (SERVE_CONTROLLER_ACTOR exists)
    2. The specified application is deployed and in RUNNING status

    Should be called after start_serve() to ensure the server is ready
    before allowing flows to start using it.

    Args:
        app_name: Name of the Ray Serve application to wait for.
            Default: "fairchem_inference"
        poll_interval_seconds: How often to check status. Default: 2

    Returns:
        True if server is ready.

    Raises:
        RuntimeError: If server fails to deploy (DEPLOY_FAILED or UNHEALTHY status).

    Example:
        start_serve()
        wait_for_serve_ready()  # Blocks until server is ready
        # Now safe to run flows that use the inference server
    """
    import time

    from ray import serve
    from ray.serve.schema import ApplicationStatus

    # Phase 1: Wait for Ray Serve controller to exist
    logger.info("Waiting for Ray Serve controller to start...")
    while True:
        try:
            # Try to get serve status - this will fail if controller doesn't exist
            status = serve.status()
            logger.info("Ray Serve controller is running")
            break
        except Exception as e:
            error_msg = str(e)
            if "SERVE_CONTROLLER_ACTOR" in error_msg or "Failed to look up actor" in error_msg:
                logger.debug(f"Ray Serve controller not ready yet: {error_msg}")
                time.sleep(poll_interval_seconds)
            else:
                # Some other error - re-raise
                raise

    # Phase 2: Wait for the application to be deployed and running
    logger.info(f"Waiting for application '{app_name}' to be ready...")
    while True:
        try:
            status = serve.status()

            # Check if our application exists
            if app_name not in status.applications:
                logger.debug(f"Application '{app_name}' not found yet, waiting...")
                time.sleep(poll_interval_seconds)
                continue

            app_status = status.applications[app_name]

            # Check application status
            if app_status.status == ApplicationStatus.RUNNING:
                logger.info(f"Application '{app_name}' is RUNNING and ready")
                return True
            elif app_status.status == ApplicationStatus.DEPLOYING:
                logger.debug(f"Application '{app_name}' is still deploying...")
                time.sleep(poll_interval_seconds)
            elif app_status.status in (ApplicationStatus.DEPLOY_FAILED, ApplicationStatus.UNHEALTHY):
                raise RuntimeError(
                    f"Application '{app_name}' failed to deploy. "
                    f"Status: {app_status.status}, Message: {app_status.message}"
                )
            else:
                logger.debug(f"Application '{app_name}' status: {app_status.status}")
                time.sleep(poll_interval_seconds)

        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Error checking serve status: {e}")
            time.sleep(poll_interval_seconds)
