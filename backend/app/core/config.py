"""Configuration module for the Structural Health Monitoring (SHM) systems."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Centralized application settings driven by environment variables and .env."""
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server Network
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 5173

    # Telemetry and Ingestion
    FS: float = 50.0
    WINDOW_SIZE: int = 50
    PERSISTENCE: int = 3
    T_PERSIST: float = 0.04
    DEVICE: str = "cpu"
    INGESTION_TYPE: str = "simulator"
    DEMO_MODE: bool = True

    # MQTT Broker & Security (credentials read from env only)
    MQTT_HOST: str = "localhost"
    MQTT_PORT: int = 1883
    MQTT_USERNAME: str = ""
    MQTT_PASSWORD: str = ""
    MQTT_TLS_ENABLED: bool = False
    MQTT_CA_CERTS: str = ""
    MQTT_QOS: int = 1
    MQTT_KEEPALIVE: int = 30

    # HTTP Frame Ingestion Auth & Limits
    SHM_API_KEY_DEFAULT: str = ""
    MAX_FRAME_PAYLOAD_BYTES: int = 2 * 1024 * 1024
    HTTP_RATE_LIMIT_RPS: float = 200.0
    IDEMPOTENCY_TTL_SECONDS: float = 300.0

    # Spool & Recovery
    SPOOL_DIR: str = "data/spool"
    SPOOL_MAX_BYTES: int = 10 * 1024 * 1024
    SPOOL_REPLAY_MINUTES: float = 10.0

    # Backpressure & Queue Sizing
    INGESTION_QUEUE_MAXSIZE: int = 512
    DETECTOR_QUEUE_MAXSIZE: int = 512

    def get_structure_api_key(self, structure_id: str) -> str:
        """Resolve per-structure HTTP ingestion API key from env."""
        env_key = f"SHM_API_KEY_{structure_id.upper().replace('-', '_')}"
        return os.environ.get(env_key, "") or os.environ.get("SHM_API_KEY_DEFAULT", self.SHM_API_KEY_DEFAULT)

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DB_PATH: str = "data/shm.db"
    ARTIFACTS_DIR: str = "artifacts"
    REPORTS_DIR: str = "reports"

    # API Prefix & CORS
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ]


settings = AppSettings()


@dataclass(frozen=True)
class SensorConfig:
    """Configuration class for individual sensors."""
    name: str
    description: str
    amplitude_scale: float
    noise_std: float
    mode_participation: Dict[str, float]  # participation factor for modes


@dataclass(frozen=True)
class SHMConfig:
    """Physical, signal processing, and telemetry configurations for the SHM system."""

    # Processing and sampling constants
    FS: float = settings.FS  # Sampling frequency in Hz
    DURATION_SEC: float = 600.0  # Simulation duration in seconds

    
    @property
    def num_samples(self) -> int:
        """Total number of discrete samples."""
        return int(self.FS * self.DURATION_SEC)

    # Butterworth bandpass filter parameters
    FILTER_ORDER: int = 4
    PASSBAND_HZ: Tuple[float, float] = (0.5, 18.0)

    # Base modal frequencies
    MODE_1_FREQ_HZ: float = 1.5  # Fundamental bending mode
    MODE_2_FREQ_HZ: float = 3.8  # Torsional/shear mode

    # Ambient environmental temperature limits
    TEMP_MIN_C: float = 18.0
    TEMP_MAX_C: float = 34.0
    TEMP_DIURNAL_PERIOD_SEC: float = 86400.0  # 24 hours in seconds
    TEMP_SENSITIVITY_COEFF: float = -0.001  # Frequency shift per deg C above baseline (20 deg C)

    # Sensor channel topology mapping channel ID to SensorConfig
    SENSORS: Dict[int, SensorConfig] = field(default_factory=lambda: {
        0: SensorConfig(
            name="Sensor_Abutment_Left",
            description="Vertical acceleration (az), high stiffness, low amplitude",
            amplitude_scale=0.05,
            noise_std=0.01,
            mode_participation={"mode1": 0.2, "mode2": 0.1}
        ),
        1: SensorConfig(
            name="Sensor_Pier_Left",
            description="Coupled horizontal & vertical acceleration (ax, az), foundation response",
            amplitude_scale=0.15,
            noise_std=0.02,
            mode_participation={"mode1": 0.5, "mode2": 0.4}
        ),
        2: SensorConfig(
            name="Sensor_MidSpan_Girder",
            description="Dynamic vertical deflection & bending acceleration (az), critical load zone",
            amplitude_scale=0.5,
            noise_std=0.03,
            mode_participation={"mode1": 1.0, "mode2": 0.6}
        ),
        3: SensorConfig(
            name="Sensor_Pier_Right",
            description="Symmetric support foundation response (coupled ax, az)",
            amplitude_scale=0.15,
            noise_std=0.02,
            mode_participation={"mode1": 0.5, "mode2": 0.4}
        ),
        4: SensorConfig(
            name="Sensor_Ambient_Temp",
            description="Diurnal ambient thermal variation field in degrees Celsius",
            amplitude_scale=0.0,
            noise_std=0.0,
            mode_participation={}
        )
    })

    # Damage simulation triggers
    DAMAGE_START_TIME_SEC: float = 480.0  # Inject damage in the final 20%
    
    # Scenario A: Stiffness loss at MidSpan
    SCENARIO_A_MOD_1_TARGET_FREQ_HZ: float = 1.1  # Mode 1 frequency drop
    SCENARIO_A_MID_SPAN_MULT_FACTOR: float = 2.5  # Amplitude magnification

    # Scenario B: Pier settlement / tilting
    SCENARIO_B_PIER_OFFSET: float = 0.15          # Step offset change in signal
    SCENARIO_B_PIER_NOISE_MULT: float = 3.0       # Noise standard deviation multiplier

    @property
    def channel_names(self) -> List[str]:
        """Ordered list of sensor channel names matching IDs 0 to 4."""
        return [self.SENSORS[i].name for i in sorted(self.SENSORS.keys())]
