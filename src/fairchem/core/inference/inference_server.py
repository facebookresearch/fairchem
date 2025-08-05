"""
Sequential request server with parallel model execution
Usage: python server.py --workers 4 --port 8000
"""

from __future__ import annotations

import logging
import pickle
import socket
import time
import traceback
from typing import TYPE_CHECKING, Protocol

import hydra
import torch.multiprocessing as mp

if TYPE_CHECKING:
    from omegaconf import DictConfig
from torch.distributed.elastic.utils.distributed import get_free_port

from fairchem.core.common import distutils, gp_utils
from fairchem.core.common.distutils import (
    get_device_for_local_rank,
    setup_env_local_multi_gpu,
)
from fairchem.core.common.utils import detach_dict_tensors

logging.basicConfig(level=logging.DEBUG)


def worker_process(
    worker_id, input_queue, output_queue, predict_config, master_port, world_size
):
    """Worker process - waits for input data and processes it"""
    setup_env_local_multi_gpu(worker_id, master_port)
    device = predict_config.get("device")
    backend = "gloo" if device == "cpu" else "nccl"
    dist_config = {
        "distributed_backend": backend,
        "world_size": world_size,
        "cpu": device == "cpu",
        "submit": False,
    }
    distutils.setup(dist_config)
    gp_utils.setup_graph_parallel_groups(world_size, backend)
    predict_unit = hydra.utils.instantiate(predict_config)
    logging.debug(
        f"Worker {worker_id} loaded predict unit: {predict_unit}, "
        f"on port {master_port}, with device: {get_device_for_local_rank()}"
    )

    while True:
        try:
            # Wait for input data from main process
            logging.debug(f"Worker {worker_id} waiting for input")
            request_data = input_queue.get()

            if request_data is None:  # Shutdown signal
                break

            # Deserialize input data - asssume pickle for now
            atomic_data_input = pickle.loads(request_data)
            logging.debug(f"Worker {worker_id} received input: {atomic_data_input}")
            # inference_input = deserialize_message(request_data, INFERENCE_INPUT_SCHEMA)

            # Run prediction, just pass in None for state for now
            result = predict_unit.predict_step(None, atomic_data_input)
            # detach and move tensors to cpu before sending back
            result = detach_dict_tensors(result)

            logging.debug(f"Worker {worker_id} predicted result: {result}")

            # Send result back
            output_queue.put(
                {
                    "worker_id": worker_id,
                    "result": result,
                    "error": None,
                }
            )

        except Exception as e:
            logging.error(f"Worker {worker_id} encountered error: {e}")
            traceback.print_exc()
            output_queue.put(
                {
                    "worker_id": worker_id,
                    "result": None,
                    "error": str(e),
                }
            )


class InferenceServerProtocol(Protocol):
    def run(self) -> None: ...


class MLIPInferenceServerMP(InferenceServerProtocol):
    def __init__(
        self,
        num_workers: int,
        port: int,
        predict_config: DictConfig,
        start_method: str = "spawn",
    ):
        mp.set_start_method(start_method)

        self.num_workers = num_workers
        self.port = port
        self.predict_config = predict_config

        # Create queues for each worker
        # use queues to pass input, need to change to using SharedMemory for large data

        self.input_queues = [mp.Queue() for _ in range(num_workers)]
        self.output_queues = [mp.Queue() for _ in range(num_workers)]
        self.workers = []

    def start_workers(self):
        """Start all worker processes"""
        port = get_free_port()
        for i in range(self.num_workers):
            worker = mp.Process(
                target=worker_process,
                args=(
                    i,
                    self.input_queues[i],
                    self.output_queues[i],
                    self.predict_config,
                    port,
                    self.num_workers,
                ),
            )
            worker.start()
            self.workers.append(worker)
        logging.info(f"MLIP Inference Server: Started {self.num_workers} workers")

    def process_request(self, request_data):
        """Send request to all workers and collect results"""
        start_time = time.time()

        # Send same data to all workers
        for input_queue in self.input_queues:
            input_queue.put(request_data)

        # Just collect result from the first worker to speed this up?
        results = []
        for output_queue in self.output_queues:
            result = output_queue.get()
            results.append(result)

        inference_time = (time.time() - start_time) * 1000

        # All responses should not error
        if all(r["error"] is None for r in results):
            response_data = {
                "predictions": results[0]["result"],
                "error": None,
                "inference_time": inference_time,
            }
        else:
            # Handle errors
            response_data = {
                "predictions": [],
                "error": [result.get("error", "") for result in results],
                "inference_time": inference_time,
            }

        return response_data

    def handle_client(self, client_socket, addr):
        """Handle single client connection - process requests sequentially"""
        logging.info(f"Connection from {addr}")
        try:
            while True:
                # Read request length
                length_data = client_socket.recv(4)
                if not length_data:
                    break

                msg_length = int.from_bytes(length_data, "big")

                # Read request data
                data = client_socket.recv(msg_length, socket.MSG_WAITALL)
                if not data or len(data) != msg_length:
                    break

                # Process request (blocks until all workers complete)
                # The requested data is in a pickled AtomicData for now
                response_data = self.process_request(data)
                pickled_response = pickle.dumps(response_data)

                # Send response back in pickled format as well
                length = len(pickled_response).to_bytes(4, "big")
                client_socket.sendall(length + pickled_response)

        except Exception as e:
            logging.error(f"Client {addr} encountered error: {e}")
            traceback.print_exc()
        finally:
            client_socket.close()
            logging.info(f"Client {addr} disconnected")

    def run(self):
        """Start server - handles one request at a time"""
        # Start worker processes
        self.start_workers()

        # Start server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("localhost", self.port))
        server_socket.listen(1)  # Only accept 1 connection at a time

        logging.info(f"MLIP Inference Server: Listening on port {self.port}")

        try:
            while True:
                # Accept one client at a time
                client_socket, addr = server_socket.accept()

                # Handle this client completely before accepting next one
                self.handle_client(client_socket, addr)

        except KeyboardInterrupt:
            logging.info("Shutting down server...")
        finally:
            # Shutdown workers
            for input_queue in self.input_queues:
                input_queue.put(None)  # Shutdown signal

            for worker in self.workers:
                worker.join()

            server_socket.close()


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="server_config",
)
def main(cfg: DictConfig):
    # if backend method is mp, use python multiprocessing
    server: InferenceServerProtocol = MLIPInferenceServerMP(
        num_workers=cfg.server.workers,
        port=cfg.server.port,
        predict_config=cfg.predict_unit,
    )
    # otherwise use ray
    server.run()


if __name__ == "__main__":
    main()
