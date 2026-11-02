from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .models import TopologyResult
from .validation import validate_topology


CSV_COLUMNS = (
    "ID",
    "NEXTDOWNID",
    "dist_hdn",
    "status",
    "attachment_rule",
    "attachment_confidence",
    "attachment_risk",
    "edge_rule",
    "edge_confidence",
    "edge_risk",
)


def _json_payload(result: TopologyResult) -> dict:
    return {
        "schemaVersion": result.schema_version,
        "nodeIds": list(result.node_ids),
        "edges": [
            [edge.upstream_id, edge.downstream_id]
            for edge in result.edges
            if edge.downstream_id is not None
        ],
        "assignments": result.assignments,
        "diagnostics": result.diagnostics,
        "metadata": result.metadata,
    }


def _csv_rows(result: TopologyResult) -> list[dict]:
    rows = []
    for edge in result.edges:
        assignment = result.assignments.get(edge.upstream_id, {})
        rows.append(
            {
                "ID": edge.upstream_id,
                "NEXTDOWNID": edge.downstream_id or "",
                "dist_hdn": edge.distance_km if edge.distance_km is not None else "",
                "status": edge.status,
                "attachment_rule": assignment.get("selection_reason", "unattached"),
                "attachment_confidence": assignment.get(
                    "attachment_confidence", "low"
                ),
                "attachment_risk": assignment.get(
                    "attachment_risk", "unattached"
                ),
                "edge_rule": assignment.get("edge_rule", "unattached"),
                "edge_confidence": assignment.get("edge_confidence", "low"),
                "edge_risk": assignment.get("edge_risk", "unattached"),
            }
        )
    return rows


def write_artifacts(
    result: TopologyResult, output_dir: str | Path
) -> tuple[Path, Path]:
    """Write JSON and CSV derived from the same validated in-memory graph."""
    validate_topology(result)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"artifact output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "topology.json"
    csv_path = output_dir / "adjacency.csv"
    json_path.write_text(
        json.dumps(
            _json_payload(result),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(_csv_rows(result), columns=CSV_COLUMNS).to_csv(
        csv_path, index=False
    )
    return json_path, csv_path
