# STREAM

Station Topology Reconstruction through Evidence-guided Attachment and Mapping.

STREAM reconstructs directed station connections from river-network geometry,
downstream reach identifiers, and station locations. Optional control-area and
station metadata support attachment correction. The method also retains
attachment evidence, confidence information, and diagnostic risk labels.

## Installation

Requires Python 3.10 or later.

```sh
python -m pip install .
```

Dependencies are declared in `pyproject.toml`: GeoPandas, Shapely, pandas,
and Matplotlib. The import name is `topo_build`.

## Input data

River and station inputs must declare their coordinate reference systems.
River records require `HYRIV_ID` (reach identifier) and `NEXT_DOWN`
(downstream reach identifier). Station identifiers are read from `STCD`,
`ID`, `id`, or `station_id`.

Optional station control areas are read from `area_gov`, `AREA_GOV`,
`DRAIN_AREA`, or `DRNA` in square kilometres; optional river contributing
areas use `UPLAND_SKM`. Input datasets with other field names must first
be adapted to this schema. Catchment polygons and reference station
connections are not required for construction.

## Usage

```python
from topo_build import TopologyOptions, build_topology, write_artifacts

result = build_topology(
    "river.shp",
    "stations.shp",
    TopologyOptions(max_snap_distance_m=3000),
)
write_artifacts(result, "artifacts")
```

The public API returns station connections, river distances, assignments,
diagnostics, and metadata without automatically plotting or writing files.
`river_topo_build.py` contains the method implementation and optional plotting
utilities. Its executable demonstration refers to local datasets that are
not included; use the API with your own data.

## Verification

The API tests create synthetic river and station data; no study dataset is
required. The wheel test verifies installation and import outside the source
directory.

```sh
python -m pip install pytest
PYTHONPATH=src python -m pytest -q test_topo_build_package.py test_topo_build_wheel.py
```

## Scope

This repository distributes the topology-construction implementation and
portable package tests. It does not contain manuscripts, patent documents,
raw station observations, third-party river datasets, or runoff-model training
outputs. Obtain study data separately from their respective providers and
comply with their access and redistribution conditions.

An open-source license has not yet been selected. Repository visibility does
not itself grant an open-source license.
