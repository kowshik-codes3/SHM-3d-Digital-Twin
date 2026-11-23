"""Ingestion source factory supporting declarative runtime instantiation."""

import logging
from typing import Any, Dict, List, Optional

from app.ingestion.base import IngestionSource
from app.ingestion.http_source import HTTPSource
from app.ingestion.kafka_source import KafkaSource
from app.ingestion.mqtt_source import MQTTSource
from app.ingestion.replay_source import ReplaySource
from app.ingestion.simulator_source import SimulatorSource

logger = logging.getLogger("SHM.IngestionFactory")


class IngestionFactory:
    """Factory creating appropriate IngestionSource instances from configuration."""

    @staticmethod
    def create_source(
        source_type: str,
        params: Optional[Dict[str, Any]] = None,
        channel_names: Optional[List[str]] = None,
        fs: float = 50.0,
        structure_id: str = "default_structure",
    ) -> IngestionSource:
        """Instantiate an IngestionSource based on source_type.

        Parameters
        ----------
        source_type : str
            One of: 'simulator', 'replay', 'mqtt', 'kafka', 'http'.
        params : Optional[Dict[str, Any]]
            Source-specific configuration parameters.
        channel_names : Optional[List[str]]
            List of sensor channel names.
        fs : float
            Sampling frequency in Hz.
        structure_id : str
            Identifier for the structure.
        """
        p = params or {}
        st = source_type.lower().strip()
        ch_names = list(channel_names or [])

        if st == "simulator":
            base_temp = float(p.get("base_temp", 26.0))
            chunk_samples = int(p.get("chunk_samples", 50))
            return SimulatorSource(
                fs=fs,
                base_temp=base_temp,
                structure_id=structure_id,
                sensor_ids=ch_names or None,
                chunk_samples=chunk_samples,
            )

        elif st == "replay":
            file_path = p.get("file_path") or p.get("cache_dir")
            if not file_path:
                raise ValueError("Replay source requires 'file_path' or 'cache_dir' parameter.")
            speed = p.get("speed_multiplier", 1.0)
            loop = bool(p.get("loop", False))
            chunk_samples = int(p.get("chunk_samples", 50))
            recording_ids = p.get("recording_ids")
            return ReplaySource(
                file_path=file_path,
                fs=fs,
                speed_multiplier=speed,
                loop=loop,
                structure_id=structure_id,
                sensor_ids=ch_names or None,
                chunk_samples=chunk_samples,
                recording_ids=recording_ids,
            )

        elif st == "mqtt":
            host = p.get("host")
            port = p.get("port")
            topic_prefix = p.get("topic_prefix")
            topic = p.get("topic")
            qos = p.get("qos")
            return MQTTSource(
                channel_names=ch_names,
                structure_id=structure_id,
                host=host,
                port=port,
                topic_prefix=topic_prefix,
                topic=topic,
                qos=qos,
                fs=fs,
            )

        elif st == "kafka":
            bootstrap = str(p.get("bootstrap_servers", "localhost:9092"))
            topic = str(p.get("topic", f"shm-{structure_id}"))
            group = str(p.get("group_id", f"shm-{structure_id}-group"))
            auto_offset = str(p.get("auto_offset_reset", "latest"))
            return KafkaSource(
                channel_names=ch_names,
                bootstrap_servers=bootstrap,
                topic=topic,
                group_id=group,
                auto_offset_reset=auto_offset,
            )

        elif st == "http":
            return HTTPSource(channel_names=ch_names, structure_id=structure_id, fs=fs)

        else:
            raise ValueError(f"Unknown ingestion source type: '{source_type}'. Expected one of: simulator, replay, mqtt, kafka, http.")
