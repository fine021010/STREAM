from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point


def test_options_are_immutable_and_validate_public_values():
    from topo_build import TopologyBuildError, TopologyOptions

    options = TopologyOptions()

    assert options.buffer_distance_m == 50.0
    assert options.max_snap_distance_m == 3000.0
    assert options.attachment_profile == "auto"
    with pytest.raises(FrozenInstanceError):
        options.buffer_distance_m = 10
    with pytest.raises(TopologyBuildError, match="max_snap_distance_m"):
        TopologyOptions(max_snap_distance_m=0)


def test_package_initially_exports_contract_types():
    import topo_build

    assert set(topo_build.__all__) == {
        "build_topology",
        "build_topology_from_frames",
        "TopologyBuildError",
        "TopologyEdge",
        "TopologyOptions",
        "TopologyResult",
        "write_artifacts",
    }


def test_build_topology_from_frames_uses_in_memory_gis_without_file_io(
    monkeypatch,
):
    from topo_build import build_topology_from_frames

    rivers = gpd.GeoDataFrame(
        {"HYRIV_ID": [1, 2], "NEXT_DOWN": [2, 0]},
        geometry=[
            LineString([(0, 1000), (0, 0)]),
            LineString([(0, 0), (0, -1000)]),
        ],
        crs="EPSG:3857",
    )
    stations = gpd.GeoDataFrame(
        {"STCD": ["upstream", "outlet"]},
        geometry=[Point(0, 900), Point(0, -900)],
        crs="EPSG:3857",
    )

    def fail_read(*args, **kwargs):
        raise AssertionError("in-memory construction must not read GIS files")

    monkeypatch.setattr(gpd, "read_file", fail_read)
    result = build_topology_from_frames(rivers, stations)

    assert result.node_ids == ("upstream", "outlet")
    assert [(edge.upstream_id, edge.downstream_id) for edge in result.edges] == [
        ("upstream", "outlet"),
        ("outlet", None),
    ]


def _write_simple_shapes(tmp_path: Path) -> tuple[Path, Path]:
    river_path = tmp_path / "river.shp"
    station_path = tmp_path / "stations.shp"
    gpd.GeoDataFrame(
        {"HYRIV_ID": [1, 2], "NEXT_DOWN": [2, 0]},
        geometry=[
            LineString([(0, 1000), (0, 0)]),
            LineString([(0, 0), (0, -1000)]),
        ],
        crs="EPSG:3857",
    ).to_file(river_path)
    gpd.GeoDataFrame(
        {"STCD": ["upstream", "outlet"]},
        geometry=[Point(0, 900), Point(0, -900)],
        crs="EPSG:3857",
    ).to_file(station_path)
    return river_path, station_path


def test_build_topology_has_no_plot_or_write_side_effects(tmp_path, monkeypatch):
    from topo_build import build_topology

    river_path, station_path = _write_simple_shapes(tmp_path)

    def fail_savefig(*args, **kwargs):
        raise AssertionError("online topology construction must not plot")

    monkeypatch.setattr("matplotlib.pyplot.savefig", fail_savefig)
    before = {path.name for path in tmp_path.iterdir()}

    result = build_topology(river_path, station_path)

    assert result.schema_version == "topology-v1"
    assert result.node_ids == ("upstream", "outlet")
    assert [
        (edge.upstream_id, edge.downstream_id) for edge in result.edges
    ] == [("upstream", "outlet"), ("outlet", None)]
    assert result.assignments["upstream"]["snap_point"]["type"] == "Point"
    assert len(result.assignments["upstream"]["snap_point"]["coordinates"]) == 2
    assert {path.name for path in tmp_path.iterdir()} == before


def test_validate_topology_rejects_disconnected_graph():
    from topo_build import TopologyBuildError, TopologyEdge, TopologyResult
    from topo_build.validation import validate_topology

    result = TopologyResult(
        schema_version="topology-v1",
        node_ids=("a", "b"),
        edges=(TopologyEdge("a", None, None, "outlet"),
               TopologyEdge("b", None, None, "outlet")),
    )

    with pytest.raises(TopologyBuildError) as exc_info:
        validate_topology(result)

    assert exc_info.value.code == "TOPO_GRAPH_DISCONNECTED"


def test_write_artifacts_keeps_json_and_csv_consistent(tmp_path):
    from topo_build import build_topology, write_artifacts

    river_path, station_path = _write_simple_shapes(tmp_path)
    result = build_topology(river_path, station_path)

    json_path, csv_path = write_artifacts(result, tmp_path / "artifacts")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = pd.read_csv(csv_path, dtype=str).fillna("")
    assert payload["nodeIds"] == list(result.node_ids)
    assert payload["edges"] == [["upstream", "outlet"]]
    assert rows[["ID", "NEXTDOWNID"]].values.tolist() == [
        ["upstream", "outlet"],
        ["outlet", ""],
    ]


def test_build_topology_normalizes_legacy_station_prefix(tmp_path):
    from topo_build import build_topology

    river_path, station_path = _write_simple_shapes(tmp_path)
    stations = gpd.read_file(station_path)
    stations["STCD"] = ["chn_dllg_upstream", "chn_dllg_outlet"]
    stations.to_file(station_path)

    result = build_topology(river_path, station_path)

    assert result.node_ids == ("upstream", "outlet")
    assert set(result.assignments) == {"upstream", "outlet"}
