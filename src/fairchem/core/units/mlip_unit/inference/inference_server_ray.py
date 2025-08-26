"""
Sequential request server with parallel model execution
Usage: python server.py --workers 4 --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import pickle
import signal
from typing import TYPE_CHECKING

import hydra
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


@remote
class MLIPWorker:
    def __init__(
        self, worker_id: int, world_size: int, master_port: int, predictor_config: dict
    ):
        if ray_installed is False:
            raise RuntimeError("Requires `ray` to be installed")

        self.worker_id = worker_id
        self._distributed_setup(
            worker_id, master_port, world_size, predictor_config.get("device", "cpu")
        )
        self.predict_unit = hydra.utils.instantiate(predictor_config)
        logging.info(
            f"Worker {worker_id}, gpu_id: {ray.get_gpu_ids()}, loaded predict unit: {self.predict_unit}, "
            f"on port {master_port}, with device: {get_device_for_local_rank()}, config: {predictor_config}"
        )

    def _distributed_setup(
        self, worker_id: int, master_port: int, world_size: int, device: str
    ):
        # initialize distributed environment
        # TODO, this wont work for multi-node, need to fix master addr
        setup_env_local_multi_gpu(worker_id, master_port)
        # local_rank = int(os.environ["LOCAL_RANK"])
        assign_device_for_local_rank(device == "cpu", 0)
        backend = "gloo" if device == "cpu" else "nccl"
        dist.init_process_group(
            backend=backend,
            rank=worker_id,
            world_size=world_size,
        )
        gp_utils.setup_graph_parallel_groups(world_size, backend)

    def predict(self, data: bytes):
        atomic_data = pickle.loads(data)
        result = self.predict_unit.predict(atomic_data)
        return pickle.dumps(result)


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
                i, self.num_workers, self.master_pg_port, self.predictor_config
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

        async with serve(self.handler, self.host, self.port):
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
