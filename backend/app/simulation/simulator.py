"""Asynchronous background telemetry stream simulator with dynamic fault injection.

Generates realistic 5-channel structural vibration frames at 50 Hz using SHMSignalGenerator,
pushes frames to the ingestion stream buffer, and supports runtime fault injection APIs.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.simulation.signal_generator import SHMSignalGenerator

logger = logging.getLogger("SHM.Simulator")

CHANNEL_INDEX_MAP = {
    "Sensor_Abutment_Left": 0,
    "Sensor_Pier_Left": 1,
    "Sensor_MidSpan_Girder": 2,
    "Sensor_Pier_Right": 3,
    "Sensor_Ambient_Temp": 4,
}


class TelemetrySimulator:
    """Simulates real-time multi-channel structural vibration telemetry.

    Parameters
    ----------
    fs : Sampling frequency in Hz (default 50.0).
    base_temp : Ambient baseline temperature in °C (default 26.0).
    """

    def __init__(self, fs: float = 50.0, base_temp: float = 26.0) -> None:
        self.fs = fs
        self.base_temp = base_temp
        self.dt = 1.0 / fs
        self.sig_gen = SHMSignalGenerator()
        
        # Pre-generate healthy baseline dataset (30,000 samples = 600s)
        logger.info("Pre-generating healthy baseline signal sequence for simulator...")
        self._df = self.sig_gen.generate_healthy_baseline()
        self._data_matrix = self._df.values.astype(np.float32)
        self._num_samples = len(self._data_matrix)

        self._active_faults: Dict[str, Dict[str, Any]] = {}
        self._step_counter = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def inject_fault(
        self,
        node_id: str,
        fault_type: str = "stiffness_loss",
        severity: float = 0.5,
    ) -> None:
        """Inject a dynamic structural fault into a specific sensor channel."""
        assert node_id in CHANNEL_INDEX_MAP, f"Unknown channel ID: {node_id}"
        self._active_faults[node_id] = {
            "type": fault_type,
            "severity": float(severity),
            "start_step": self._step_counter,
        }
        logger.warning(f"Fault injected on {node_id}: {fault_type} (Severity={severity})")

    def clear_faults(self) -> None:
        """Reset all active injected faults (return to baseline healthy state)."""
        self._active_faults.clear()
        logger.info("All injected faults cleared. Stream reset to healthy baseline.")

    def generate_frame(self) -> np.ndarray:
        """Generate a single 1D 5-channel telemetry frame vector."""
        idx = self._step_counter % self._num_samples
        self._step_counter += 1
        t_val = self._step_counter * self.dt

        frame = self._data_matrix[idx].copy()

        # Apply active injected faults
        for node_id, fault in self._active_faults.items():
            c_idx = CHANNEL_INDEX_MAP[node_id]
            severity = fault["severity"]
            fault_type = fault["type"]

            if fault_type == "stiffness_loss":
                damage_osc = (2.5 * severity) * np.sin(2.0 * np.pi * 0.9 * t_val)
                frame[c_idx] += damage_osc
            elif fault_type == "settlement":
                frame[c_idx] += (1.5 * severity)
            elif fault_type == "spike":
                frame[c_idx] += np.random.choice([0.0, 2.5 * severity])

        return frame

    async def start(
        self,
        buffer_push_fn: Callable[[np.ndarray], None],
        inference_trigger_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        """Start background loop generating telemetry at fs Hz."""
        self._running = True
        logger.info(f"Starting TelemetrySimulator background task ({self.fs} Hz)...")

        try:
            while self._running:
                start_t = time.perf_counter()
                frame = self.generate_frame()
                buffer_push_fn(frame)

                if inference_trigger_fn is not None:
                    inference_trigger_fn()

                elapsed = time.perf_counter() - start_t
                sleep_time = max(0.0, self.dt - elapsed)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            self._running = False
            logger.info("TelemetrySimulator cancelled.")

    def stop(self) -> None:
        """Stop background simulation loop."""
        self._running = False
        logger.info("TelemetrySimulator stopped.")
