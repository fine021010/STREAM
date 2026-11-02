from __future__ import annotations

from collections import deque

from .errors import TopologyBuildError
from .models import TopologyResult


def validate_topology(result: TopologyResult) -> None:
    """Validate the public topology independently of its GIS implementation."""
    node_ids = result.node_ids
    if not node_ids or any(not node_id.strip() for node_id in node_ids):
        raise TopologyBuildError("TOPO_NODE_INVALID", "nodeIds 必须非空")
    if len(set(node_ids)) != len(node_ids):
        raise TopologyBuildError("TOPO_NODE_DUPLICATE", "nodeIds 必须唯一")

    known = set(node_ids)
    source_counts = {node_id: 0 for node_id in node_ids}
    directed_edges: set[tuple[str, str]] = set()
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in result.edges:
        if edge.upstream_id not in known:
            raise TopologyBuildError("TOPO_EDGE_INVALID", "边引用未知上游节点")
        source_counts[edge.upstream_id] += 1
        if edge.downstream_id is None:
            continue
        if edge.downstream_id not in known:
            raise TopologyBuildError("TOPO_EDGE_INVALID", "边引用未知下游节点")
        if edge.upstream_id == edge.downstream_id:
            raise TopologyBuildError("TOPO_EDGE_INVALID", "拓扑不允许自环")
        pair = (edge.upstream_id, edge.downstream_id)
        if pair in directed_edges:
            raise TopologyBuildError("TOPO_EDGE_DUPLICATE", "拓扑边重复")
        directed_edges.add(pair)
        adjacency[edge.upstream_id].add(edge.downstream_id)
        adjacency[edge.downstream_id].add(edge.upstream_id)

    if any(count != 1 for count in source_counts.values()):
        raise TopologyBuildError(
            "TOPO_EDGE_INVALID", "每个节点必须恰好有一条下游或出口记录"
        )
    if len(node_ids) <= 1:
        return
    reached = {node_ids[0]}
    pending = deque(reached)
    while pending:
        for neighbour in adjacency[pending.popleft()] - reached:
            reached.add(neighbour)
            pending.append(neighbour)
    if len(reached) != len(node_ids):
        raise TopologyBuildError("TOPO_GRAPH_DISCONNECTED", "拓扑必须连通")
