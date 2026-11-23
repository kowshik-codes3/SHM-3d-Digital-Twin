"""Alert persistence repository for the SHM Digital Twin.

Stores WARNING/CRITICAL anomaly events to SQLite and exposes
typed query methods for the REST Alert API.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .database import DatabaseManager

logger = logging.getLogger("SHM.AlertRepository")


# ── Typed result models ─────────────────────────────────────────────────────────

@dataclass
class AlertRecord:
    """Lightweight alert row (used in list responses)."""
    id: int
    structure_id: str
    structure_name: str
    timestamp: float
    system_status: str
    localized_defect_node: Optional[str]
    acknowledged: bool
    created_at: str


@dataclass
class AlertDetail(AlertRecord):
    """Full alert with per-node sensor snapshot included."""
    nodes: Dict[str, Any]  # Parsed from nodes_json


@dataclass
class AlertStats:
    """Summary counts per structure."""
    structure_id: str
    total: int
    critical: int
    warning: int
    acknowledged: int
    unacknowledged: int


# ── Repository ──────────────────────────────────────────────────────────────────

class AlertRepository:
    """CRUD gateway for the `alerts` table.

    Parameters
    ----------
    db : DatabaseManager
        Shared database manager (must be initialised before use).
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # ── Writes ──────────────────────────────────────────────────────────────────

    def insert_alert(
        self,
        structure_id: str,
        structure_name: str,
        timestamp: float,
        system_status: str,
        localized_defect_node: Optional[str],
        nodes: Dict[str, Any],
    ) -> int:
        """Persist a new alert event and return the inserted row id.

        Args:
            structure_id         : Registry key of the structure.
            structure_name       : Human-readable structure name.
            timestamp            : Unix timestamp of the anomaly event.
            system_status        : 'WARNING' or 'CRITICAL'.
            localized_defect_node: Sensor node id, or None.
            nodes                : Per-node dict (mse, threshold, ratio, status).

        Returns:
            Integer primary key of the inserted row.
        """
        nodes_json = json.dumps(nodes)
        cursor = self._db.execute(
            """
            INSERT INTO alerts
                (structure_id, structure_name, timestamp, system_status,
                 localized_defect_node, nodes_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (structure_id, structure_name, timestamp, system_status,
             localized_defect_node, nodes_json),
        )
        self._db.commit()
        alert_id = cursor.lastrowid
        logger.info(
            f"Alert #{alert_id} persisted: [{system_status}] "
            f"{structure_id} → {localized_defect_node}"
        )
        return alert_id

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged.

        Returns:
            True if a row was updated, False if alert_id was not found.
        """
        cursor = self._db.execute(
            "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
            (alert_id,),
        )
        self._db.commit()
        return cursor.rowcount > 0

    # ── Reads ────────────────────────────────────────────────────────────────────

    def get_alerts(
        self,
        structure_id: Optional[str] = None,
        status: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[AlertRecord], int]:
        """Fetch paginated alert list with optional filters.

        Returns:
            Tuple of (list of AlertRecord, total matching count).
        """
        conditions = []
        params: list = []

        if structure_id:
            conditions.append("structure_id = ?")
            params.append(structure_id)
        if status:
            conditions.append("system_status = ?")
            params.append(status.upper())
        if acknowledged is not None:
            conditions.append("acknowledged = ?")
            params.append(1 if acknowledged else 0)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        total_row = self._db.execute(
            f"SELECT COUNT(*) FROM alerts {where}", tuple(params)
        ).fetchone()
        total = total_row[0] if total_row else 0

        rows = self._db.execute(
            f"""
            SELECT id, structure_id, structure_name, timestamp,
                   system_status, localized_defect_node, acknowledged, created_at
            FROM alerts {where}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            tuple(params) + (limit, offset),
        ).fetchall()

        records = [
            AlertRecord(
                id=r["id"],
                structure_id=r["structure_id"],
                structure_name=r["structure_name"],
                timestamp=r["timestamp"],
                system_status=r["system_status"],
                localized_defect_node=r["localized_defect_node"],
                acknowledged=bool(r["acknowledged"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]
        return records, total

    def get_alert(self, alert_id: int) -> Optional[AlertDetail]:
        """Fetch full alert detail including per-node snapshot.

        Returns:
            AlertDetail or None if not found.
        """
        row = self._db.execute(
            """
            SELECT id, structure_id, structure_name, timestamp,
                   system_status, localized_defect_node, nodes_json,
                   acknowledged, created_at
            FROM alerts WHERE id = ?
            """,
            (alert_id,),
        ).fetchone()

        if row is None:
            return None

        return AlertDetail(
            id=row["id"],
            structure_id=row["structure_id"],
            structure_name=row["structure_name"],
            timestamp=row["timestamp"],
            system_status=row["system_status"],
            localized_defect_node=row["localized_defect_node"],
            acknowledged=bool(row["acknowledged"]),
            created_at=row["created_at"],
            nodes=json.loads(row["nodes_json"]),
        )

    def get_stats(self, structure_id: Optional[str] = None) -> List[AlertStats]:
        """Return alert counts grouped by structure.

        Args:
            structure_id: If provided, restrict to a single structure.

        Returns:
            List of AlertStats (one per structure).
        """
        where = "WHERE structure_id = ?" if structure_id else ""
        params = (structure_id,) if structure_id else ()

        rows = self._db.execute(
            f"""
            SELECT
                structure_id,
                COUNT(*)                                          AS total,
                SUM(CASE WHEN system_status='CRITICAL' THEN 1 ELSE 0 END) AS critical,
                SUM(CASE WHEN system_status='WARNING'  THEN 1 ELSE 0 END) AS warning,
                SUM(acknowledged)                                 AS acknowledged,
                SUM(CASE WHEN acknowledged=0 THEN 1 ELSE 0 END)  AS unacknowledged
            FROM alerts {where}
            GROUP BY structure_id
            """,
            params,
        ).fetchall()

        return [
            AlertStats(
                structure_id=r["structure_id"],
                total=r["total"],
                critical=r["critical"],
                warning=r["warning"],
                acknowledged=r["acknowledged"],
                unacknowledged=r["unacknowledged"],
            )
            for r in rows
        ]
