"""Asynchronous Multi-Structure WebSocket Pub/Sub Connection Manager.

Manages active WebGL digital twin client connections partitioned by structure_id,
heartbeat ping/pongs, and thread-safe async JSON telemetry broadcasting.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("SHM.ConnectionManager")


class ConnectionManager:
    """Manages active WebSocket client connections partitioned by structure_id."""

    def __init__(self) -> None:
        # structure_id -> {client_id: WebSocket}
        self._connections: Dict[str, Dict[str, WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, structure_id: str = "bridge_demo_01") -> str:
        """Accept a new WebSocket connection and register under a structure_id."""
        await websocket.accept()
        client_id = str(uuid.uuid4())
        async with self._lock:
            if structure_id not in self._connections:
                self._connections[structure_id] = {}
            self._connections[structure_id][client_id] = websocket
        logger.info(f"WebSocket client {client_id} connected to structure '{structure_id}' (Total on structure: {len(self._connections[structure_id])})")
        return client_id

    async def disconnect(self, client_id: str, structure_id: str = "bridge_demo_01") -> None:
        """Unregister a disconnected client safely."""
        async with self._lock:
            if structure_id in self._connections and client_id in self._connections[structure_id]:
                del self._connections[structure_id][client_id]
                logger.info(f"WebSocket client {client_id} disconnected from structure '{structure_id}'.")
                if not self._connections[structure_id]:
                    del self._connections[structure_id]

    async def broadcast(self, message: Dict[str, Any], structure_id: Optional[str] = None) -> None:
        """Broadcast JSON message payload to clients listening to structure_id."""
        target_sid = structure_id or message.get("structure_id", "bridge_demo_01")
        async with self._lock:
            client_dict = self._connections.get(target_sid, {})
            if not client_dict:
                return
            clients_snapshot = list(client_dict.items())

        payload_str = json.dumps(message)
        stale_clients: List[str] = []

        for cid, ws in clients_snapshot:
            try:
                await ws.send_text(payload_str)
            except (WebSocketDisconnect, RuntimeError, Exception) as err:
                logger.warning(f"Failed to send telemetry to client {cid} on structure {target_sid}: {err}")
                stale_clients.append(cid)

        if stale_clients:
            async with self._lock:
                if target_sid in self._connections:
                    for cid in stale_clients:
                        self._connections[target_sid].pop(cid, None)

    def active_count(self, structure_id: Optional[str] = None) -> int:
        """Return count of active connections for structure_id or across all structures."""
        if structure_id is not None:
            return len(self._connections.get(structure_id, {}))
        return sum(len(clients) for clients in self._connections.values())
