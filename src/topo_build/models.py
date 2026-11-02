from __future__ import annotations

from dataclasses import dataclass, field

from .errors import TopologyBuildError


@dataclass(frozen=True)
class TopologyEdge:
    upstream_id: str
    downstream_id: str | None
    distance_km: float | None
    status: str


@dataclass(frozen=True)
class TopologyOptions:
    buffer_distance_m: float = 50.0
    max_snap_distance_m: float = 3000.0
    attachment_profile: str = "auto"
    work_crs: str | None = None

    def __post_init__(self):
        if self.buffer_distance_m <= 0:
            raise TopologyBuildError(
                "TOPO_OPTION_INVALID", "buffer_distance_m 必须大于 0"
            )
        if self.max_snap_distance_m <= 0:
            raise TopologyBuildError(
                "TOPO_OPTION_INVALID", "max_snap_distance_m 必须大于 0"
            )
        if self.attachment_profile not in {"auto", "euclidean_nearest"}:
            raise TopologyBuildError(
                "TOPO_OPTION_INVALID", "attachment_profile 非法"
            )


@dataclass(frozen=True)
class TopologyResult:
    schema_version: str
    node_ids: tuple[str, ...]
    edges: tuple[TopologyEdge, ...]
    assignments: dict[str, dict] = field(default_factory=dict)
    diagnostics: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if self.schema_version != "topology-v1":
            raise TopologyBuildError(
                "TOPO_SCHEMA_INVALID", "schema_version 必须为 topology-v1"
            )
