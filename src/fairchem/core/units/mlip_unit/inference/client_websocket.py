# client.py
from __future__ import annotations

import pickle

import websockets
from websockets.sync.client import connect as ws_connect


class AsyncMLIPInferenceWebSocketClient:
    def __init__(self, host, port, max_size=100 * 1024 * 1024):  # 100MB default
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        self.max_size = max_size

    async def connect(self):
        self.websocket = await websockets.connect(self.uri, max_size=self.max_size)

    async def close(self):
        await self.websocket.close()

    async def call(self, atomic_data):
        if not self.websocket:
            await self.connect()

        await self.websocket.send(pickle.dumps(atomic_data))

        # Receive response
        response = await self.websocket.recv()
        result = pickle.loads(response)
        return result


class SyncMLIPInferenceWebSocketClient:
    def __init__(self, host, port, max_size=100 * 1024 * 1024):  # 100MB default
        self.uri = f"ws://{host}:{port}"
        self.ws = None
        self.max_size = max_size

    def connect(self):
        self.ws = ws_connect(
            self.uri, ping_timeout=300, ping_interval=60, max_size=self.max_size
        )

    def call(self, atomic_data):
        if not self.ws:
            self.connect()

        self.ws.send(pickle.dumps(atomic_data))

        # Receive response
        response = self.ws.recv()
        result = pickle.loads(response)
        return result

    def __del__(self):
        self.close()

    def close(self):
        self.ws.close()
