from __future__ import annotations

import logging
import pickle
import socket
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from fairchem.core.datasets.atomic_data import AtomicData

from fairchem.core.units.mlip_unit.inference.socket_utils import (
    SOCKET_TIMEOUT,
    recv_message,
    send_message,
)


class MLIPInferenceClient:
    def __init__(self, server_address: str, port: int):
        self.server_address = server_address
        self.port = port
        self.socket = (
            None  # socket objects cannot be pickled so we initialize it lazily
        )

    def _connect(self):
        # lazy connect to server
        if self.socket is None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(SOCKET_TIMEOUT)
            self.socket.connect((self.server_address, self.port))
            logging.info(
                f"Connected to inference server at {self.server_address}:{self.port}"
            )

    def call(self, data: AtomicData) -> dict[str, torch.tensor]:
        self._connect()

        # Serialize the data
        serialized_data = pickle.dumps(data)

        # Send the message using helper function
        send_message(self.socket, serialized_data)

        # Receive the response using helper function
        response_data = recv_message(self.socket)
        logging.debug(f"Received response length: {len(response_data)} bytes")

        # Deserialize the response data
        response = pickle.loads(response_data)
        if response.get("error") is not None:
            raise RuntimeError(f"Inference error: {response['error']}")
        return response["predictions"]

    def __del__(self):
        if self.socket is not None:
            self.socket.close()
