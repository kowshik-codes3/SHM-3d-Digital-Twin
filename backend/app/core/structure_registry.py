"""Structure runtime orchestrator and multi-structure registry."""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from app.core.artifact_manager import ArtifactBundle, ArtifactManager
from app.core.structures import DEFAULT_BRIDGE_CONFIG, StructureConfig
from app.db.alert_repository import AlertRepository
from app.ingestion.base import IngestionSource
from app.ingestion.simulator_source import SimulatorSource
from app.ml.anomaly_engine import AnomalyEvent, AnomalyStateMachine, STATE_NORMAL
from app.ml.model import LSTMAutoencoder
from app.ml.scaler_utils import StructuralDataScaler
from app.ml.stream_buffer import SensorStreamBuffer
from app.ml.telemetry_formatter import TelemetryFormatter

logger = logging.getLogger("SHM.StructureRegistry")


class StructureRuntime:
    """Encapsulates the full isolated runtime pipeline for a single structure."""

    def __init__(
        self,
        config: StructureConfig,
        artifacts: ArtifactBundle,
        ingestion_source: Optional[IngestionSource] = None,
        broadcast_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
        device: Optional[torch.device] = None,
        alert_repository: Optional[AlertRepository] = None,
    ) -> None:
        self.config = config
        self.artifacts = artifacts
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.broadcast_fn = broadcast_fn
        self.alert_repository = alert_repository

        # Isolated streaming buffer & state machine
        self.buffer = SensorStreamBuffer(window_size=config.window_size, num_channels=config.num_channels)
        adj_data = config.compute_adjacency_matrix()
        self.state_machine = AnomalyStateMachine(
            config=artifacts.threshold_config,
            persistence=3,
            t_persist=0.04,
            adjacency_matrix=adj_data["matrix"],
        )
        self.scaler = artifacts.scaler
        self.model = artifacts.model
        self.model.to(self.device)
        self.model.eval()

        self.ingestion_source = ingestion_source or SimulatorSource(fs=config.fs)
        self._task: Optional[asyncio.Task] = None
        self.inference_count = 0
        self.start_time = time.time()
        self._last_persisted_status: str = STATE_NORMAL  # dedup guard
        self._last_warning_time: float = 0.0             # debounce guard for transient warning jitter

    def push_frame(self, frame: np.ndarray) -> Optional[AnomalyEvent]:
        """Ingest a 1D frame vector and run inference if buffer is full."""
        self.buffer.push_frame(frame)
        if not self.buffer.is_full():
            return None

        # Apply thermal normalization if available
        if self.artifacts.thermal_normalizer is not None and self.artifacts.thermal_normalizer.is_fitted:
            raw_window = self.buffer.get_window()  # Shape (window_size, channels)
            norm_window = self.artifacts.thermal_normalizer.transform(raw_window)
            scaled_window = self.scaler.transform(norm_window)
            input_tensor = torch.tensor(scaled_window, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            # Standardize & run forward pass
            input_tensor = self.buffer.get_standardized_tensor(scaler=self.scaler, device=self.device)

        with torch.no_grad():
            reconstructed, _ = self.model(input_tensor)

        mse_tensor = torch.mean((input_tensor - reconstructed) ** 2, dim=1).squeeze(0)
        mse_vec = mse_tensor.cpu().numpy()

        self.inference_count += 1
        event = self.state_machine.step(
            window_index=self.inference_count,
            timestamp=time.time(),
            mse_vector=mse_vec,
        )
        return event

    def _on_frame_emitted(self, frame: np.ndarray) -> None:
        """Callback invoked synchronously by ingestion source on each frame."""
        event = self.push_frame(frame)
        if event is None:
            return

        # Broadcast to live WebSocket subscribers
        if self.broadcast_fn is not None:
            payload = TelemetryFormatter.format(event, structure_id=self.config.structure_id)
            asyncio.create_task(self.broadcast_fn(payload))

        # Persist alerts to DB:
        # 1. CRITICAL is a confirmed structural alarm (passed persistence filter P>=3, t>=t_persist).
        #    Persist immediately on state transition into CRITICAL.
        # 2. WARNING is an early indicator. Persist on transition, debounced to max 1 per 30s
        #    to prevent 50 Hz ambient noise jitter from flooding the DB.
        should_persist = False
        now = event.timestamp
        if self.alert_repository is not None:
            if event.system_status == "CRITICAL" and self._last_persisted_status != "CRITICAL":
                should_persist = True
            elif event.system_status == "WARNING" and self._last_persisted_status != "WARNING":
                if (now - self._last_warning_time) >= 30.0:
                    should_persist = True
                    self._last_warning_time = now

        if should_persist:
            nodes_dict = {
                name: {
                    "mse": ns.mse,
                    "threshold": ns.threshold,
                    "ratio": ns.ratio,
                    "status": ns.status,
                }
                for name, ns in event.nodes.items()
            }
            try:
                self.alert_repository.insert_alert(
                    structure_id=self.config.structure_id,
                    structure_name=self.config.name,
                    timestamp=event.timestamp,
                    system_status=event.system_status,
                    localized_defect_node=event.localized_defect_node,
                    nodes=nodes_dict,
                )
            except Exception as exc:
                logger.warning(f"Failed to persist alert for {self.config.structure_id}: {exc}")

        self._last_persisted_status = event.system_status

    async def start(self) -> None:
        """Start background telemetry ingestion."""
        if self._task is not None and not self._task.done():
            return
        logger.info(f"Starting runtime for structure '{self.config.structure_id}' using {self.ingestion_source.name}...")
        self._task = asyncio.create_task(self.ingestion_source.start(self._on_frame_emitted))
        await asyncio.sleep(0.01)

    def stop(self) -> None:
        """Stop runtime ingestion."""
        if self.ingestion_source:
            self.ingestion_source.stop()
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info(f"Stopped runtime for structure '{self.config.structure_id}'.")

    async def stop_async(self) -> None:
        """Stop runtime ingestion and await task cancellation."""
        self.stop()
        if self._task is not None:
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass


class StructureRegistry:
    """Global registry managing multiple isolated StructureRuntime instances."""

    def __init__(self, artifact_manager: Optional[ArtifactManager] = None) -> None:
        self.artifact_manager = artifact_manager or ArtifactManager()
        self._runtimes: Dict[str, StructureRuntime] = {}
        self._configs: Dict[str, StructureConfig] = {}

    def register_structure(
        self,
        config: StructureConfig,
        version: str = "v1",
        ingestion_source: Optional[IngestionSource] = None,
        broadcast_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
        device: Optional[torch.device] = None,
        alert_repository: Optional[AlertRepository] = None,
    ) -> StructureRuntime:
        """Register and initialize runtime for a structure."""
        artifacts = self.artifact_manager.load_artifacts(config.structure_id, version, device, config=config)
        runtime = StructureRuntime(
            config=config,
            artifacts=artifacts,
            ingestion_source=ingestion_source,
            broadcast_fn=broadcast_fn,
            device=device,
            alert_repository=alert_repository,
        )
        self._runtimes[config.structure_id] = runtime
        self._configs[config.structure_id] = config
        logger.info(f"Registered structure runtime: {config.structure_id} ({config.name})")
        return runtime

    def unregister_structure(self, structure_id: str) -> bool:
        """Stop and unregister a structure runtime."""
        runtime = self._runtimes.pop(structure_id, None)
        self._configs.pop(structure_id, None)
        if runtime is not None:
            runtime.stop()
            logger.info(f"Unregistered structure runtime: {structure_id}")
            return True
        return False

    def get(self, structure_id: str) -> Optional[StructureRuntime]:
        return self._runtimes.get(structure_id)

    def get_config(self, structure_id: str) -> Optional[StructureConfig]:
        return self._configs.get(structure_id)

    def list_structures(self) -> List[Dict[str, Any]]:
        result = []
        for sid, cfg in self._configs.items():
            rt = self._runtimes.get(sid)
            result.append({
                "structure_id": cfg.structure_id,
                "name": cfg.name,
                "structure_type": cfg.structure_type.value,
                "num_channels": cfg.num_channels,
                "channels": cfg.channel_names,
                "sensors": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "channel_name": s.name,
                        "type": s.type.value,
                        "sensor_type": s.type.value,
                        "coordinates_3d": list(s.position_xyz),
                        "position_xyz": list(s.position_xyz),
                        "threshold_tau": 0.04,
                    }
                    for s in cfg.sensors
                ],
                "is_running": rt.ingestion_source.is_running if rt else False,
                "inferences_executed": rt.inference_count if rt else 0,
            })
        return result

    async def start_all(self) -> None:
        for rt in self._runtimes.values():
            await rt.start()

    def stop_all(self) -> None:
        for rt in self._runtimes.values():
            rt.stop()

    async def stop_all_async(self) -> None:
        for rt in self._runtimes.values():
            await rt.stop_async()
