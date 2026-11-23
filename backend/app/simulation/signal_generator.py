"""Module to generate synthetic multi-channel telemetry for structural health monitoring."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from ..core.config import SHMConfig, SensorConfig


class SHMSignalGenerator:
    """Generates multi-harmonic, thermally-coupled telemetry with optional defects."""

    def __init__(self, config: Optional[SHMConfig] = None):
        """
        Initializes the generator with configuration parameters.
        
        Args:
            config: SHMConfig object containing structural parameters.
        """
        self.config = config or SHMConfig()

    def generate_temperature(self, t: np.ndarray) -> np.ndarray:
        """
        Generates diurnal temperature progression over time array.
        
        Formula:
            T(t) = 26.0 + 8.0 * sin(2 * pi * t / 86400.0)
            
        Args:
            t: Time vector in seconds.
            
        Returns:
            Temperature values at each timestep in degrees Celsius.
        """
        # Diurnal fluctuation parameters: temperature peaks in afternoon, cold or baseline at start
        mean_temp = (self.config.TEMP_MAX_C + self.config.TEMP_MIN_C) / 2.0
        amplitude = (self.config.TEMP_MAX_C - self.config.TEMP_MIN_C) / 2.0
        
        omega = 2.0 * np.pi / self.config.TEMP_DIURNAL_PERIOD_SEC
        # Normal sine wave centered around mean (26 C), bouncing between 18 and 34 C
        return mean_temp + amplitude * np.sin(omega * t)

    def _generate_vibration_for_sensor(
        self,
        t: np.ndarray,
        sensor_id: int,
        sensor: SensorConfig,
        ambient_temp: np.ndarray,
        scenario: Optional[str] = None
    ) -> np.ndarray:
        """
        Generates vibration acceleration time series for a single sensor.
        
        Args:
            t: Time vector in seconds.
            sensor_id: Numeric ID of the sensor channel.
            sensor: SensorConfig representing parameters.
            ambient_temp: Precomputed temperature curve.
            scenario: Optional defect scenario 'A' or 'B'.
            
        Returns:
            Acceleration trace as 1D numpy array.
        """
        n_samples = len(t)
        dt = 1.0 / self.config.FS

        # 1. Compute dynamic frequencies under thermal influence
        f1_base = np.full(n_samples, self.config.MODE_1_FREQ_HZ)
        f2_base = np.full(n_samples, self.config.MODE_2_FREQ_HZ)

        if scenario == "A":
            # Scenario A: Mode 1 frequency drop (1.5 Hz -> 1.1 Hz) post damage_start
            damage_mask = t >= self.config.DAMAGE_START_TIME_SEC
            f1_base[damage_mask] = self.config.SCENARIO_A_MOD_1_TARGET_FREQ_HZ

        # Apply temperature coupling: f(T) = f_base * (1 + b * (T - 20))
        # Frequency decreases with expansion (b < 0)
        thermal_modifier = 1.0 + self.config.TEMP_SENSITIVITY_COEFF * (ambient_temp - 20.0)
        f1 = f1_base * thermal_modifier
        f2 = f2_base * thermal_modifier

        # 2. Compute cumulative phases to avoid phase jump shocks
        phase1 = 2.0 * np.pi * np.cumsum(f1) * dt
        phase2 = 2.0 * np.pi * np.cumsum(f2) * dt

        # 3. Simulate structural response
        part_m1 = sensor.mode_participation.get("mode1", 0.0)
        part_m2 = sensor.mode_participation.get("mode2", 0.0)

        harmonic_response = (part_m1 * np.sin(phase1)) + (part_m2 * np.sin(phase2))

        # Scale harmonic amplitude
        amp_scale = np.full(n_samples, sensor.amplitude_scale)
        if scenario == "A" and sensor_id == 2:  # Sensor_MidSpan_Girder
            damage_mask = t >= self.config.DAMAGE_START_TIME_SEC
            amp_scale[damage_mask] *= self.config.SCENARIO_A_MID_SPAN_MULT_FACTOR

        signal = amp_scale * harmonic_response

        # 4. Generate random ambient noise
        base_noise_std = sensor.noise_std
        noise_scale = np.ones(n_samples)
        
        if scenario == "B" and sensor_id == 1:  # Sensor_Pier_Left
            damage_mask = t >= self.config.DAMAGE_START_TIME_SEC
            noise_scale[damage_mask] = self.config.SCENARIO_B_PIER_NOISE_MULT

        noise = np.random.normal(0.0, base_noise_std, n_samples) * noise_scale
        signal += noise

        # 5. Apply settlement offset ramp for Scenario B
        if scenario == "B" and sensor_id == 1:  # Sensor_Pier_Left
            damage_start_idx = int(self.config.DAMAGE_START_TIME_SEC * self.config.FS)
            ramp_duration_samples = int(10.0 * self.config.FS)  # 10 second transition
            damage_end_ramp_idx = damage_start_idx + ramp_duration_samples
            
            offset = np.zeros(n_samples)
            if damage_end_ramp_idx <= n_samples:
                offset[damage_start_idx:damage_end_ramp_idx] = np.linspace(
                    0.0, self.config.SCENARIO_B_PIER_OFFSET, ramp_duration_samples
                )
                offset[damage_end_ramp_idx:] = self.config.SCENARIO_B_PIER_OFFSET
            else:
                offset[damage_start_idx:] = self.config.SCENARIO_B_PIER_OFFSET
                
            signal += offset

        return signal

    def generate_healthy_baseline(self) -> pd.DataFrame:
        """
        Generates the 600-second healthy baseline dataset.
        
        Returns:
            DataFrame with datetime index and structural sensor columns.
        """
        return self._generate_dataset(scenario=None)

    def generate_damaged_scenario(self, scenario: str) -> pd.DataFrame:
        """
        Generates the 600-second damaged scenario dataset.
        
        Args:
            scenario: 'A' (Stiffness Loss) or 'B' (Pier Settlement).
            
        Returns:
            DataFrame with datetime index and structural sensor columns.
        """
        if scenario not in ("A", "B"):
            raise ValueError("Scenario must be 'A' or 'B'.")
        return self._generate_dataset(scenario=scenario)

    def _generate_dataset(self, scenario: Optional[str] = None) -> pd.DataFrame:
        """Internal helper to construct and populate DataFrame."""
        n_samples = self.config.num_samples
        t = np.arange(n_samples) / self.config.FS
        
        # Create timestamp index
        start_time = pd.Timestamp("2026-08-01 08:00:00")
        timestamps = start_time + pd.to_timedelta(t, unit="s")

        temp_data = self.generate_temperature(t)

        data = {}
        for sensor_id, sensor in self.config.SENSORS.items():
            if sensor.name == "Sensor_Ambient_Temp":
                data[sensor.name] = temp_data
            else:
                data[sensor.name] = self._generate_vibration_for_sensor(
                    t=t,
                    sensor_id=sensor_id,
                    sensor=sensor,
                    ambient_temp=temp_data,
                    scenario=scenario
                )

        df = pd.DataFrame(data, index=timestamps)
        df.index.name = "timestamp"
        return df
