"""JSON telemetry payload formatter for WebSocket broadcasting (Phase 7 compatible).

Converts AnomalyEvent dataclass outputs into the structured JSON schema
expected by the real-time Digital Twin WebSocket API.
"""

import json
from typing import Any, Dict, List, Optional

from .anomaly_engine import AnomalyEvent


class TelemetryFormatter:
    """Transforms AnomalyEvent objects into compliant WebSocket JSON payloads.

    The output schema matches the Phase 7 WebSocket broadcast contract:

    {
        "timestamp": <float>,
        "system_status": "NORMAL" | "WARNING" | "CRITICAL",
        "localized_defect_node": <str | null>,
        "nodes": {
            "<channel_name>": {
                "mse": <float>,
                "threshold": <float>,
                "ratio": <float>,
                "status": "NORMAL" | "WARNING" | "CRITICAL"
            },
            ...
        }
    }
    """

    @staticmethod
    def format(event: AnomalyEvent, structure_id: Optional[str] = None) -> Dict[str, Any]:
        """Serialize an AnomalyEvent to a JSON-serializable dict.

        Args:
            event: AnomalyEvent produced by AnomalyStateMachine.step().
            structure_id: Optional structure ID identifier.

        Returns:
            Dict ready for json.dumps() or WebSocket broadcast.
        """
        nodes_payload: Dict[str, Dict[str, Any]] = {}
        for name, ns in event.nodes.items():
            nodes_payload[name] = {
                "mse":       round(ns.mse,       6),
                "threshold": round(ns.threshold, 6),
                "ratio":     round(ns.ratio,     4),
                "status":    ns.status,
            }

        payload: Dict[str, Any] = {
            "timestamp":            event.timestamp,
            "window_index":         event.window_index,
            "system_status":        event.system_status,
            "localized_defect_node": event.localized_defect_node,
            "nodes":                nodes_payload,
        }
        if structure_id:
            payload["structure_id"] = structure_id
        return payload

    @staticmethod
    def to_json(event: AnomalyEvent, indent: Optional[int] = None) -> str:
        """Serialize an AnomalyEvent directly to a JSON string.

        Args:
            event  : AnomalyEvent to serialize.
            indent : JSON indentation (None = compact, 2 = pretty-print).

        Returns:
            JSON string.
        """
        return json.dumps(
            TelemetryFormatter.format(event), indent=indent
        )

    @staticmethod
    def batch_format(events: List[AnomalyEvent]) -> list:
        """Serialize a list of AnomalyEvents to a list of dicts.

        Useful for bulk logging or REST API bulk-export endpoints.
        """
        return [TelemetryFormatter.format(e) for e in events]
