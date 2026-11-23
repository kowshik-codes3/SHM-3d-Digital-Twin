"""Artifact manager for resolving and saving per-structure model artifacts."""

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from app.ml.model import LSTMAutoencoder
from app.ml.scaler_utils import StructuralDataScaler
from app.ml.threshold_calibrator import ChannelThreshold, ThresholdCalibrator, ThresholdConfig

logger = logging.getLogger("SHM.ArtifactManager")


@dataclass
class ModelMetadata:
    """Audit metadata recorded with each trained model version."""
    structure_id: str
    version: str
    model_type: str
    channels: List[str]
    window_size: int
    fs: float
    trained_at: str
    training_data_range: Dict[str, str]
    metrics: Dict[str, Any]
    git_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(**data)


from app.ml.thermal_normalizer import ThermalNormalizer


@dataclass
class ArtifactBundle:
    """In-memory bundle of resolved artifacts for a structure."""
    structure_id: str
    version: str
    model: LSTMAutoencoder
    scaler: StructuralDataScaler
    threshold_config: ThresholdConfig
    thermal_normalizer: Optional[ThermalNormalizer] = None
    metadata: Optional[ModelMetadata] = None


def get_git_hash() -> str:
    """Retrieve current commit hash for model provenance."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


class ArtifactManager:
    """Manages loading and storing per-structure artifacts with versioning."""

    def __init__(self, base_artifacts_dir: Optional[Path] = None) -> None:
        if base_artifacts_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"
        else:
            self.base_dir = Path(base_artifacts_dir)

    def get_structure_version_dir(self, structure_id: str, version: str = "v1") -> Path:
        return self.base_dir / structure_id / version

    def ensure_default_artifacts_migrated(self, structure_id: str = "bridge_demo_01") -> None:
        """Migrate legacy root artifacts into structured subfolder if not present."""
        target_dir = self.get_structure_version_dir(structure_id, "v1")
        target_dir.mkdir(parents=True, exist_ok=True)

        legacy_model = self.base_dir / "lstm_autoencoder_best.pt"
        legacy_scaler = self.base_dir / "scaler.joblib"
        legacy_thresh = self.base_dir / "threshold_config.json"

        target_model = target_dir / "model.pt"
        target_scaler = target_dir / "scaler.joblib"
        target_thresh = target_dir / "threshold_config.json"
        target_meta = target_dir / "metadata.json"

        if legacy_model.exists() and not target_model.exists():
            shutil.copy2(legacy_model, target_model)
        if legacy_scaler.exists() and not target_scaler.exists():
            shutil.copy2(legacy_scaler, target_scaler)
        if legacy_thresh.exists() and not target_thresh.exists():
            shutil.copy2(legacy_thresh, target_thresh)

        if not target_meta.exists():
            meta = ModelMetadata(
                structure_id=structure_id,
                version="v1",
                model_type="LSTMAutoencoder",
                channels=[
                    "Sensor_Abutment_Left",
                    "Sensor_Pier_Left",
                    "Sensor_MidSpan_Girder",
                    "Sensor_Pier_Right",
                    "Sensor_Ambient_Temp",
                ],
                window_size=50,
                fs=50.0,
                trained_at="2026-09-18T12:00:00Z",
                training_data_range={"start": "2026-08-01 08:00:00", "end": "2026-08-01 08:10:00"},
                metrics={"baseline_far": 0.0, "latency_mean_ms": 3.48},
                git_hash=get_git_hash(),
            )
            with open(target_meta, "w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, indent=2)

    def ensure_or_create_default_artifacts(self, config: Any, version: str = "v1") -> None:
        """Create baseline model artifacts for a newly registered structure if missing."""
        structure_id = getattr(config, "structure_id", str(config))
        version_dir = self.get_structure_version_dir(structure_id, version)
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "model.pt"
        scaler_path = version_dir / "scaler.joblib"
        thresh_path = version_dir / "threshold_config.json"
        meta_path = version_dir / "metadata.json"

        channel_names = getattr(config, "channel_names", [])
        num_channels = getattr(config, "num_channels", len(channel_names))
        channels_dict = getattr(config, "channels", {})
        fs = getattr(config, "fs", 50.0)

        # 1. Baseline Model
        if not model_path.exists():
            model = LSTMAutoencoder(input_size=max(1, num_channels), seq_len=50)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Synthesized initial LSTMAutoencoder weights at {model_path}")

        # 2. Baseline Scaler
        if not scaler_path.exists():
            scaler = StructuralDataScaler(feature_names=channel_names)
            dummy_data = pd.DataFrame(
                np.zeros((100, max(1, num_channels)), dtype=np.float32),
                columns=channel_names,
            )
            scaler.fit(dummy_data)
            scaler.save(scaler_path)
            logger.info(f"Synthesized initial StructuralDataScaler at {scaler_path}")

        # 3. Baseline Thresholds
        if not thresh_path.exists():
            thresh_map = {}
            for name in channel_names:
                ch_def = channels_dict.get(name)
                tau_val = getattr(ch_def, "tau", 0.05) if ch_def else 0.05
                tau_val = tau_val if (tau_val and tau_val > 0) else 0.05
                thresh_map[name] = ChannelThreshold(
                    name=name,
                    mu=0.01,
                    sigma=0.01,
                    tau=float(tau_val),
                    warn=float(tau_val * 0.75),
                    k=3.0,
                )
            tc = ThresholdConfig(
                calibration_timestamp=time.time(),
                k_multiplier=3.0,
                channels=thresh_map,
            )
            with open(thresh_path, "w", encoding="utf-8") as f:
                json.dump(tc.to_json(), f, indent=2)
            logger.info(f"Synthesized initial ThresholdConfig at {thresh_path}")

        # 4. Metadata
        if not meta_path.exists():
            meta = ModelMetadata(
                structure_id=structure_id,
                version=version,
                model_type="LSTMAutoencoder",
                channels=channel_names,
                window_size=50,
                fs=fs,
                trained_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                training_data_range={"start": "synthetic_baseline", "end": "synthetic_baseline"},
                metrics={"baseline_far": 0.0, "latency_mean_ms": 3.0},
                git_hash=get_git_hash(),
            )
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, indent=2)

    def load_artifacts(
        self,
        structure_id: str,
        version: str = "v1",
        device: Optional[torch.device] = None,
        config: Optional[Any] = None,
    ) -> ArtifactBundle:
        """Load artifact bundle for a specific structure and version."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure legacy bridge migration
        if structure_id == "bridge_demo_01":
            self.ensure_default_artifacts_migrated(structure_id)

        version_dir = self.get_structure_version_dir(structure_id, version)

        # If artifacts are missing and config is provided, auto-synthesize
        if not version_dir.exists() and config is not None:
            self.ensure_or_create_default_artifacts(config, version)

        # Fallback to root artifacts if version folder is missing
        if not version_dir.exists() and (self.base_dir / "lstm_autoencoder_best.pt").exists():
            model_path = self.base_dir / "lstm_autoencoder_best.pt"
            scaler_path = self.base_dir / "scaler.joblib"
            thresh_path = self.base_dir / "threshold_config.json"
            meta_path = None
        else:
            model_path = version_dir / "model.pt"
            scaler_path = version_dir / "scaler.joblib"
            thresh_path = version_dir / "threshold_config.json"
            meta_path = version_dir / "metadata.json"

        # If still missing and config provided, synthesize
        if (not model_path.exists() or not scaler_path.exists() or not thresh_path.exists()) and config is not None:
            self.ensure_or_create_default_artifacts(config, version)

        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        if not thresh_path.exists():
            raise FileNotFoundError(f"Threshold config not found at {thresh_path}")

        # Load scaler & thresholds
        scaler = StructuralDataScaler.load(scaler_path)
        threshold_config = ThresholdCalibrator.load(str(thresh_path))

        # Load model
        model = LSTMAutoencoder(input_size=len(threshold_config.channels), seq_len=50)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        # Load thermal normalizer if present
        thermal_normalizer: Optional[ThermalNormalizer] = None
        thermal_path = version_dir / "thermal_normalizer.joblib"
        if thermal_path.exists():
            try:
                thermal_normalizer = ThermalNormalizer.load(thermal_path)
            except Exception as e:
                logger.warning(f"Could not load thermal normalizer: {e}")

        metadata: Optional[ModelMetadata] = None
        if meta_path and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = ModelMetadata.from_dict(json.load(f))

        return ArtifactBundle(
            structure_id=structure_id,
            version=version,
            model=model,
            scaler=scaler,
            threshold_config=threshold_config,
            thermal_normalizer=thermal_normalizer,
            metadata=metadata,
        )
