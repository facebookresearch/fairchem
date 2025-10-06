"""
Sequential request server with parallel model execution
Usage: python server.py --workers 4 --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import time
from typing import TYPE_CHECKING

import hydra
import torch
import torch.distributed as dist
import websockets
from monty.dev import requires
from torch.distributed.elastic.utils.distributed import get_free_port
from websockets.asyncio.server import serve

if TYPE_CHECKING:
    from omegaconf import DictConfig

from fairchem.core.common import gp_utils
from fairchem.core.common.distutils import (
    assign_device_for_local_rank,
    get_device_for_local_rank,
    setup_env_local_multi_gpu,
)

try:
    import ray
    from ray import remote

    ray_installed = True
except ImportError:
    ray = None

    def remote(cls):
        # dummy
        return cls

    ray_installed = False

logging.basicConfig(level=logging.INFO)


def move_tensors_to_cpu(data):
    """
    Recursively move all PyTorch tensors in a nested data structure to CPU.

    Args:
        data: Input data structure (dict, list, tuple, tensor, or other)

    Returns:
        Data structure with all tensors moved to CPU
    """
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {key: move_tensors_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_cpu(item) for item in data)
    else:
        # Return as-is for non-tensor types (str, int, float, etc.)
        return data


@remote
class MLIPWorker:
    def __init__(
        self,
        worker_id: int,
        world_size: int,
        predictor_config: dict,
        master_port: int | None = None,
        master_address: str | None = None,
    ):
        if ray_installed is False:
            raise RuntimeError("Requires `ray` to be installed")

        self.worker_id = worker_id
        self.world_size = world_size
        self.predictor_config = predictor_config
        self.master_address = (
            ray.util.get_node_ip_address() if master_address is None else master_address
        )
        self.master_port = get_free_port() if master_port is None else master_port
        self.is_setup = False

    def get_master_address_and_port(self):
        return (self.master_address, self.master_port)

    def _distributed_setup(
        self,
        worker_id: int,
        master_port: int,
        world_size: int,
        predictor_config: dict,
        master_address: str,
    ):
        # initialize distributed environment
        # TODO, this wont work for multi-node, need to fix master addr
        setup_env_local_multi_gpu(worker_id, master_port, master_address)
        # local_rank = int(os.environ["LOCAL_RANK"])
        device = predictor_config.get("device", "cpu")
        assign_device_for_local_rank(device == "cpu", 0)
        backend = "gloo" if device == "cpu" else "nccl"
        dist.init_process_group(
            backend=backend,
            rank=worker_id,
            world_size=world_size,
        )
        gp_utils.setup_graph_parallel_groups(world_size, backend)
        self.predict_unit = hydra.utils.instantiate(predictor_config)
        logging.info(
            f"Worker {worker_id}, gpu_id: {ray.get_gpu_ids()}, loaded predict unit: {self.predict_unit}, "
            f"on port {self.master_port}, with device: {get_device_for_local_rank()}, config: {self.predictor_config}"
        )

    def predict(self, data):
        if not self.is_setup:
            self._distributed_setup(
                self.worker_id,
                self.master_port,
                self.world_size,
                self.predictor_config,
                self.master_address,
            )
            self.is_setup = True
            self.data = data.clone()
        t0 = time.time()
        if self.data is not None:
            out = self.predict_unit.predict(self.data)
        else:
            out = self.predict_unit.predict(data)
        logging.info(f"Worker {self.worker_id} prediction time: {time.time() - t0} s")
        out = move_tensors_to_cpu(out)
        logging.info(f"Worker {self.worker_id} after cpu time: {time.time() - t0} s")
        if self.worker_id == 0:
            return out
        # atomic_data = pickle.loads(data)
        # result = self.predict_unit.predict(atomic_data)
        # return pickle.dumps(result)


@requires(ray_installed, message="Requires `ray` to be installed")
class MLIPInferenceServerWebSocket:
    def __init__(self, predictor_config: dict, port=8001, num_workers=1):
        logging.basicConfig(level=logging.INFO)
        self.host = "localhost"
        self.port = port
        self.num_workers = num_workers
        self.predictor_config = predictor_config
        # Initialize a pool of MLIPWorkers
        self.master_pg_port = get_free_port()
        ray.init(logging_level=logging.INFO)
        options = {"num_gpus": 1} if predictor_config.get("device") == "cuda" else {}
        self.workers = [
            MLIPWorker.options(**options).remote(
                i,
                self.num_workers,
                self.predictor_config,
                self.master_pg_port,
                "localhost",
            )
            for i in range(self.num_workers)
        ]
        logging.info(
            "Initialized Local MLIPInferenceServerWebSocket with config: "
            f"{self.predictor_config}, port: {self.port}, workers: {self.num_workers}"
        )

        # Set up signal handlers for clean shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def handler(self, websocket):
        try:
            async for message in websocket:
                # don't unpickle here, just pass bytes to workers
                futures = [w.predict.remote(message) for w in self.workers]
                # just get the first result that is ready since they are identical
                # the rest of the futures should go out of scope and memory garbage collected
                ready_ids, _ = ray.wait(futures, num_returns=1)
                await websocket.send(ray.get(ready_ids[0]))
        except websockets.exceptions.ConnectionClosed:
            logging.info("Client disconnected")
        except Exception as e:
            logging.info(f"MLIPInferenceServer handler Error: {e}")
        finally:
            self.shutdown()

    async def start(self):
        self.stop_event = asyncio.Event()

        async with serve(self.handler, self.host, self.port, max_size=10 * 1024 * 1024):
            print(f"WebSocket server started on port {self.port}")

            with contextlib.suppress(asyncio.CancelledError):
                await self.stop_event.wait()

    def run(self):
        """Run the server (blocking)"""
        asyncio.run(self.start())

    def shutdown(self):
        """Shutdown the server and clean up Ray resources"""
        if hasattr(self, "stop_event"):
            self.stop_event.set()
        ray.shutdown()
        logging.info("MLIPInferenceServerWebSocket shutdown complete")


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="server_config",
)
def main(cfg: DictConfig):
    # Ensure logs from all Ray workers are printed to the driver
    server = MLIPInferenceServerWebSocket(
        cfg.predict_unit, cfg.server.port, cfg.server.workers
    )
    server.run()


if __name__ == "__main__":
    main()
