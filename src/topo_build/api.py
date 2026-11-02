from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any

from river_topo_build import (
    _get_node_id_series,
    build_station_topology,
    prepare_data_frames,
    read_and_prepare_data,
)

from .errors import TopologyBuildError
from .models import TopologyEdge, TopologyOptions, TopologyResult
from .validation import validate_topology


def _error_from_legacy(exc: Exception) -> TopologyBuildError:
    message = str(exc)
    if "必须声明 CRS" in message:
        code = "TOPO_CRS_MISSING"
    elif "缺少必要字段" in message or "站点 ID 字段" in message:
        code = "TOPO_REQUIRED_FIELD_MISSING"
    elif isinstance(exc, (FileNotFoundError, OSError)):
        code = "TOPO_SOURCE_NOT_FOUND"
    else:
        code = "TOPO_BUILD_FAILED"
    return TopologyBuildError(code, message)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if hasattr(value, "__geo_interface__"):
        return _json_value(value.__geo_interface__)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Real):
        value = float(value)
        if not math.isfinite(value):
            raise TopologyBuildError(
                "TOPO_VALUE_INVALID", "构图诊断包含非有限数值"
            )
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    raise TopologyBuildError(
        "TOPO_VALUE_INVALID", f"构图诊断包含不可序列化类型: {type(value).__name__}"
    )


def _normalise_assignments(assignments: dict, node_ids: tuple[str, ...]) -> dict[str, dict]:
    normalised = {}
    for index, values in assignments.items():
        node_id = node_ids[int(index)]
        normalised[node_id] = _json_value(values)
    return normalised


def _build_prepared(rivers, stations, options: TopologyOptions) -> TopologyResult:
    try:
        raw_edges, assignments, diagnostics, _ = build_station_topology(
            rivers,
            stations,
            max_snap_distance=options.max_snap_distance_m,
            attachment_profile=options.attachment_profile,
        )
    except TopologyBuildError:
        raise
    except (OSError, ValueError) as exc:
        raise _error_from_legacy(exc) from exc

    id_series = _get_node_id_series(stations)
    if id_series is None:
        raise TopologyBuildError(
            "TOPO_REQUIRED_FIELD_MISSING", "未找到可用于构图的站点 ID 字段"
        )
    node_ids = tuple(str(value).removeprefix("chn_dllg_") for value in id_series)
    edges = tuple(
        TopologyEdge(
            upstream_id=node_ids[int(source)],
            downstream_id=node_ids[int(target)] if target is not None else None,
            distance_km=float(distance) if distance is not None else None,
            status=str(status),
        )
        for source, target, distance, status in raw_edges
    )
    result = TopologyResult(
        schema_version="topology-v1",
        node_ids=node_ids,
        edges=edges,
        assignments=_normalise_assignments(assignments, node_ids),
        diagnostics={str(key): int(value) for key, value in diagnostics.items()},
        metadata={
            "algorithmVersion": "1",
            "workCrs": str(stations.crs),
        },
    )
    validate_topology(result)
    return result


def build_topology_from_frames(
    rivers, stations, options: TopologyOptions | None = None
) -> TopologyResult:
    """Build a deterministic topology from in-memory GeoDataFrames."""
    options = options or TopologyOptions()
    try:
        prepared_rivers, prepared_stations, _, _ = prepare_data_frames(
            rivers,
            stations,
            options.buffer_distance_m,
            mode="all",
            work_crs=options.work_crs,
        )
    except (OSError, ValueError) as exc:
        raise _error_from_legacy(exc) from exc
    return _build_prepared(prepared_rivers, prepared_stations, options)


def build_topology(
    river_source: str | Path,
    station_source: str | Path,
    options: TopologyOptions | None = None,
) -> TopologyResult:
    """Build a deterministic topology without plotting or writing files."""
    options = options or TopologyOptions()
    try:
        rivers, stations, _, _ = read_and_prepare_data(
            river_source,
            station_source,
            options.buffer_distance_m,
            mode="all",
            work_crs=options.work_crs,
        )
    except (OSError, ValueError) as exc:
        raise _error_from_legacy(exc) from exc
    return _build_prepared(rivers, stations, options)
