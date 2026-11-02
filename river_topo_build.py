"""以有向河段为核心构建水文站点上下游拓扑。"""

import math
import os
import re
from collections import Counter

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import Point
from shapely.ops import unary_union


def _get_node_id_series(nodes_gdf):
    for field in ("STCD", "ID", "id", "station_id"):
        if field in nodes_gdf.columns:
            return nodes_gdf[field].astype(str)
    return None


def _normalise_key(value):
    if pd.isna(value):
        return None
    try:
        number = float(value)
        if number <= 0:
            return None
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return str(value).strip()


def _station_area_km2(row):
    """读取跨数据集可选的测站控制面积；缺失时返回 ``None``。"""
    for field in ("area_gov", "AREA_GOV", "DRAIN_AREA", "DRNA"):
        if field not in row.index:
            continue
        value = pd.to_numeric(row.get(field), errors="coerce")
        if pd.notna(value) and float(value) > 0:
            return float(value)
    return None


def prepare_data_frames(river_gdf, nodes_gdf, buffer_dist, mode="all", choices=None,
                        work_crs=None, basin_id="unknown"):
    """Validate and project already-loaded GIS inputs for topology building."""
    river_gdf = river_gdf.copy()
    nodes_gdf = nodes_gdf.copy()
    if river_gdf.crs is None or nodes_gdf.crs is None:
        raise ValueError("河网和站点都必须声明 CRS")
    if nodes_gdf.empty:
        raise ValueError("站点数据为空")

    if mode == "choices":
        if not choices:
            raise ValueError("mode='choices' 需要提供非空 choices")
        ids = _get_node_id_series(nodes_gdf)
        if ids is None:
            raise ValueError("未找到可用于筛选的站点 ID 字段")
        nodes_gdf = nodes_gdf[ids.isin({str(x) for x in choices})].copy()
        if nodes_gdf.empty:
            raise ValueError("筛选后站点为空，请检查 choices")

    if basin_id == "unknown" and "BASIN_ID" in nodes_gdf.columns:
        basin_id = str(nodes_gdf.iloc[0]["BASIN_ID"])

    if work_crs is None:
        nodes_wgs84 = nodes_gdf.to_crs(4326)
        lon, lat = nodes_wgs84.geometry.x.mean(), nodes_wgs84.geometry.y.mean()
        zone = max(1, min(60, int((lon + 180) / 6) + 1))
        metric_crs = f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"
    else:
        metric_crs = work_crs
    river_gdf = river_gdf.to_crs(metric_crs).reset_index(drop=True)
    nodes_gdf = nodes_gdf.to_crs(metric_crs).reset_index(drop=True)
    invalid = nodes_gdf.geometry.isna() | nodes_gdf.geometry.is_empty
    if invalid.any():
        raise ValueError(f"存在 {int(invalid.sum())} 个空站点几何")
    total_buffer = unary_union(nodes_gdf.buffer(buffer_dist))
    print(f"原始河网: {len(river_gdf)} 条; 站点: {len(nodes_gdf)} 个; 工作 CRS: {metric_crs}")
    return river_gdf, nodes_gdf, total_buffer, basin_id


def read_and_prepare_data(river_path, nodes_path, buffer_dist, mode="all", choices=None,
                          work_crs=None):
    river_gdf = gpd.read_file(river_path)
    nodes_gdf = gpd.read_file(nodes_path)
    match = re.search(r"songliao_(\d+)", str(nodes_path))
    basin_id = f"songliao_{match.group(1)}" if match else "unknown"
    return prepare_data_frames(
        river_gdf, nodes_gdf, buffer_dist, mode=mode, choices=choices,
        work_crs=work_crs, basin_id=basin_id,
    )


def clip_river(river_gdf, total_buffer):
    """仅筛选完整河段用于绘图，不截断几何和沿河长度。"""
    selected = river_gdf[river_gdf.intersects(total_buffer)].copy().reset_index(drop=True)
    print(f"缓冲区命中的完整河段: {len(selected)} 条")
    return selected


def _validate_river_network(river_gdf):
    missing = {"HYRIV_ID", "NEXT_DOWN"}.difference(river_gdf.columns)
    if missing:
        raise ValueError(f"河网缺少必要字段: {', '.join(sorted(missing))}")
    valid = ~(river_gdf.geometry.isna() | river_gdf.geometry.is_empty)
    valid &= river_gdf.geom_type.isin(["LineString", "MultiLineString"])
    if (~valid).any():
        print(f"警告: 忽略 {int((~valid).sum())} 条无效河段")
    river_gdf = river_gdf.loc[valid].copy()
    try:
        return river_gdf.explode(index_parts=False).reset_index(drop=True)
    except TypeError:
        return river_gdf.explode().reset_index(drop=True)


def _choose_euclidean_nearest_station_segments(nodes_gdf, river_gdf):
    """基础方法：每个站点只挂接到欧氏距离最近的完整河段。"""
    assignments = {}
    spatial_index = river_gdf.sindex
    for node_idx, point in enumerate(nodes_gdf.geometry):
        try:
            nearest_result = spatial_index.nearest(point, return_all=True)
            candidate_indexes = list(dict.fromkeys(
                int(value) for value in nearest_result[1].tolist()
            ))
        except (AttributeError, IndexError, TypeError):
            candidate_indexes = list(range(len(river_gdf)))
        if not candidate_indexes:
            continue
        candidates = []
        for line_idx in candidate_indexes:
            line = river_gdf.geometry.iloc[line_idx]
            distance = float(line.distance(point))
            projection = float(line.project(point))
            endpoint_distance = min(
                projection, max(0.0, float(line.length) - projection)
            )
            candidates.append({
                "line_idx": int(line_idx),
                "distance": distance,
                "projection": projection,
                "endpoint_distance": endpoint_distance,
                "endpoint_penalty": 0.0,
            })
        selected = min(candidates, key=lambda value: (
            value["distance"], value["line_idx"]
        ))
        line_idx = selected["line_idx"]
        line = river_gdf.geometry.iloc[line_idx]
        upland = pd.to_numeric(
            river_gdf.iloc[line_idx].get("UPLAND_SKM"), errors="coerce"
        )
        upland = float(upland) if pd.notna(upland) else None
        assignments[node_idx] = {
            "line_idx": line_idx,
            "river_id": _normalise_key(river_gdf.iloc[line_idx]["HYRIV_ID"]),
            "snap_dist_m": selected["distance"],
            "nearest_dist_m": selected["distance"],
            "candidate_count": len(candidates),
            "top_k_candidates": [dict(candidate) for candidate in candidates],
            "top_k_line_indices": [
                candidate["line_idx"] for candidate in candidates
            ],
            "ambiguous_candidate_count": 1,
            "ambiguous_line_indices": [line_idx],
            "attachment_profile": "euclidean_nearest",
            # 基线明确不使用站点面积，避免构边阶段隐式跳过面积较小的站点。
            "station_area_km2": None,
            "selected_upstream_area_km2": upland,
            "area_relative_error": None,
            "area_log_error": None,
            "area_rule_applied": False,
            "selection_reason": "euclidean_nearest",
            "attachment_confidence": "high",
            "attachment_risk": "",
            "target_branch_missing": False,
            "endpoint_distance_m": selected["endpoint_distance"],
            "endpoint_penalty_m": 0.0,
            "projection_m": selected["projection"],
            "snap_point": line.interpolate(selected["projection"]),
        }
    return assignments


def _choose_unique_station_segments(nodes_gdf, river_gdf, max_distance,
                                    ambiguity_distance=500, top_k=5,
                                    adaptive_max_distance=None,
                                    endpoint_tolerance=250,
                                    attachment_profile="auto"):
    """Top-k 自适应唯一挂接，并显式量化挂接不确定性。

    ``max_distance`` 是常规搜索距离，不再是硬截断。实际搜索上限默认为
    ``max(3000, 1.5 * max_distance)``。正常候选位于最近距离加
    ``ambiguity_distance`` 内；若最近投影靠近端点，候选带自动扩大，以避免
    因河段分割位置造成误选。
    """
    assignments = {}
    spatial_index = river_gdf.sindex
    direct_upstream_indices = {}
    for line_idx, row in river_gdf.iterrows():
        next_id = _normalise_key(row.get("NEXT_DOWN"))
        if next_id is not None:
            direct_upstream_indices.setdefault(next_id, []).append(int(line_idx))
    if adaptive_max_distance is None:
        adaptive_max_distance = max(3000.0, 1.5 * max_distance)

    for node_idx, point in enumerate(nodes_gdf.geometry):
        if attachment_profile == "auto":
            station_type = str(nodes_gdf.iloc[node_idx].get("STTP", "")).upper()
            if station_type in {"ZQ", "ZZ", "RR"}:
                node_profile = "channel"
            else:
                # PP 雨量站和无站类的历史数据保持原有层级优先规则。
                node_profile = "hierarchy"
        else:
            node_profile = attachment_profile
        if node_profile not in {"hierarchy", "channel"}:
            raise ValueError("attachment_profile 必须是 auto、hierarchy 或 channel")

        candidate_indexes = list(spatial_index.query(
            point.buffer(adaptive_max_distance), predicate="intersects"
        ))
        if not candidate_indexes:
            continue
        candidates = []
        for line_idx in candidate_indexes:
            row = river_gdf.iloc[line_idx]
            distance = float(row.geometry.distance(point))
            if distance <= adaptive_max_distance:
                projection = float(row.geometry.project(point))
                endpoint_distance = min(projection, max(0.0, row.geometry.length - projection))
                endpoint_penalty = max(0.0, endpoint_tolerance - endpoint_distance)
                candidates.append({
                    "line_idx": int(line_idx),
                    "distance": distance,
                    "projection": projection,
                    "endpoint_distance": endpoint_distance,
                    "endpoint_penalty": endpoint_penalty,
                })
        if not candidates:
            continue
        candidates.sort(key=lambda c: (c["distance"], c["line_idx"]))
        candidates = candidates[:top_k]
        nearest_distance = candidates[0]["distance"]
        nearest_at_endpoint = candidates[0]["endpoint_penalty"] > 0
        adaptive_band = ambiguity_distance * (3 if nearest_at_endpoint else 1)
        ambiguous = [
            candidate for candidate in candidates
            if candidate["distance"] <= nearest_distance + adaptive_band
        ]
        # 汇流口的干流延续段可能比某条汇入支流略远，但它是最近河段的
        # 直接 NEXT_DOWN。只要仍在 Top-k 自适应范围内，就应进入最终候选。
        nearest_row = river_gdf.iloc[candidates[0]["line_idx"]]
        nearest_next = _normalise_key(nearest_row.get("NEXT_DOWN"))
        downstream_candidates = [
            candidate for candidate in candidates
            if _normalise_key(river_gdf.iloc[candidate["line_idx"]]["HYRIV_ID"]) == nearest_next
            # 直接下游规则只解决汇流口附近的分段偏移，不能跨越过大的
            # 空间差距强行把站点推到更远的公共下游河段。
            and candidate["distance"] <= nearest_distance + 2 * ambiguity_distance
        ]
        ambiguous_ids = {candidate["line_idx"] for candidate in ambiguous}
        for candidate in downstream_candidates:
            if candidate["line_idx"] not in ambiguous_ids:
                ambiguous.append(candidate)
                ambiguous_ids.add(candidate["line_idx"])

        # 河道径流/水位站原则上就在目标河道附近。先用一个小距离带约束
        # 合理候选；若测站有控制集水面积，则允许面积高度一致的河段在
        # 最近距离 + 250 m 内“救回”。这处理了测站坐标落在干流/汇流点、
        # 但测站实际代表汇入支流控制断面的情形。
        # 雨量站没有这一物理约束，继续使用完整自适应候选带。
        if node_profile == "channel":
            channel_band = max(25.0, min(100.0, nearest_distance * 0.25))
            if nearest_at_endpoint:
                channel_band = max(channel_band, 75.0)
            plausible = [
                candidate for candidate in ambiguous
                if candidate["distance"] <= nearest_distance + channel_band
            ]
        else:
            channel_band = None
            plausible = ambiguous

        station_area = _station_area_km2(nodes_gdf.iloc[node_idx])

        area_rule_applied = False
        confluence_area_increment_applied = False
        confluence_area_sum_applied = False
        confluence_incoming_area_sum_km2 = None
        confluence_incoming_area_sum_error = None
        direct_upstream_branch_priority_applied = False
        direct_upstream_branch_distance_margin_m = None
        direct_upstream_branch_boundary = False
        connected_area_rescue_applied = False
        connected_area_rescue_direction = None
        connected_area_rescue_tier = None
        sibling_area_rescue_applied = False
        mainstem_metadata_rescue_applied = False
        mainstem_metadata_field = None
        mainstem_order_gain = None
        mainstem_area_ratio = None
        mainstem_rescue_distance_m = None
        mainstem_rescue_relation = None
        mainstem_attachment_mismatch = False
        mainstem_mismatch_candidates = []
        river_name = str(nodes_gdf.iloc[node_idx].get("RVNM", "")).strip()
        hierarchy_name = str(nodes_gdf.iloc[node_idx].get("HNNM", "")).strip()
        river_name_base = river_name.replace("干流", "").strip()
        hierarchy_name_base = hierarchy_name.replace("干流", "").strip()
        # HNNM 常记录所属大水系，支流站也可能写成“松花江干流”。只有
        # RVNM 自身明确含“干流”，或 RVNM 与 HNNM 的干流名一致，才
        # 把元数据视为站点位于干流的证据。
        mainstem_fields = []
        if "干流" in river_name:
            mainstem_fields.append("RVNM")
        elif (
            "干流" in hierarchy_name
            and river_name_base
            and hierarchy_name_base
            and (
                river_name_base in hierarchy_name_base
                or hierarchy_name_base in river_name_base
            )
        ):
            mainstem_fields.append("HNNM")
        # 兼容已有审计输出；新逻辑统一记录在 connected_* 字段中。
        upstream_area_rescue_applied = False
        upstream_area_rescue_tier = None
        area_candidates = []
        best_area_error = None
        second_area_error = None
        if node_profile == "channel" and station_area is not None:
            area_rescue_limit = nearest_distance + 250.0
            short_connected_rescue_limit = nearest_distance + 500.0
            long_connected_rescue_limit = nearest_distance + 1000.0
            nearest_id = _normalise_key(nearest_row.get("HYRIV_ID"))
            nearest_next = _normalise_key(nearest_row.get("NEXT_DOWN"))
            nearest_upland = pd.to_numeric(nearest_row.get("UPLAND_SKM"), errors="coerce")
            nearest_area_error = (
                abs(float(nearest_upland) - station_area) / station_area
                if pd.notna(nearest_upland) and float(nearest_upland) > 0 else None
            )
            direct_downstream_area_errors = []
            for downstream_candidate in candidates:
                downstream_row = river_gdf.iloc[downstream_candidate["line_idx"]]
                if _normalise_key(downstream_row.get("HYRIV_ID")) != nearest_next:
                    continue
                downstream_upland = pd.to_numeric(
                    downstream_row.get("UPLAND_SKM"), errors="coerce"
                )
                if pd.notna(downstream_upland) and float(downstream_upland) > 0:
                    direct_downstream_area_errors.append(
                        abs(float(downstream_upland) - station_area) / station_area
                    )
            # 普通面积候选仍受歧义带约束；直接上游救回可检查完整 Top-k，
            # 从而覆盖候选距离超过普通 500 m 歧义带的汇流口偏移。
            for candidate in candidates:
                upland = pd.to_numeric(
                    river_gdf.iloc[candidate["line_idx"]].get("UPLAND_SKM"),
                    errors="coerce",
                )
                if pd.isna(upland) or float(upland) <= 0:
                    continue
                area_error = abs(float(upland) - station_area) / station_area
                candidate_row = river_gdf.iloc[candidate["line_idx"]]
                candidate_id = _normalise_key(candidate_row.get("HYRIV_ID"))
                is_direct_upstream = (
                    _normalise_key(candidate_row.get("NEXT_DOWN")) == nearest_id
                )
                is_direct_downstream = candidate_id == nearest_next
                # 汇流前的两条入口河段互为兄弟支流：它们不是直接上下游，
                # 但共享同一 NEXT_DOWN。径流站坐标可能更靠近较小支流，
                # 此时控制面积可作为强证据将其救回到实际控制的入口河段。
                is_sibling_incoming = (
                    nearest_next is not None
                    and candidate_id != nearest_id
                    and _normalise_key(candidate_row.get("NEXT_DOWN")) == nearest_next
                    and pd.notna(nearest_upland)
                    and float(nearest_upland) <= 0.05 * station_area
                    and pd.notna(candidate_row.get("ORD_STRA"))
                    and pd.notna(nearest_row.get("ORD_STRA"))
                    and float(candidate_row.get("ORD_STRA"))
                    >= float(nearest_row.get("ORD_STRA")) + 2
                    # 若汇流后的直接下游河段也与站点面积吻合，应优先解释为
                    # 汇流后控制断面；仅当兄弟入口至少再改善 5 个百分点时救回。
                    and (
                        not direct_downstream_area_errors
                        or min(direct_downstream_area_errors) - area_error >= 0.05
                    )
                )
                rescue_direction = (
                    "upstream" if is_direct_upstream
                    and pd.notna(nearest_upland) and float(nearest_upland) > station_area
                    else "downstream" if is_direct_downstream
                    and pd.notna(nearest_upland) and float(nearest_upland) < station_area
                    else "sibling" if is_sibling_incoming
                    else None
                )
                # 若最近河段已经流过汇流口，其汇水面积会明显大于测站控制
                # 面积，则向直接上游救回；若最近河段尚未汇入足够集水面积，
                # 则对称地检查直接下游。直接连通、面积误差和改善幅度共同
                # 约束，避免简单扩大所有河段的面积搜索范围。
                is_short_connected_rescue = (
                    candidate["distance"] > area_rescue_limit
                    and candidate["distance"] <= short_connected_rescue_limit
                    and nearest_area_error is not None
                    and nearest_area_error > 0.15
                    and rescue_direction is not None
                    and area_error <= 0.10
                    and nearest_area_error - area_error >= 0.10
                )
                # 500--1000 m 仅用于强证据救回：最近河段面积严重不符、
                # 直接上游候选误差不超过 5%，且至少改善 20 个百分点。
                is_long_connected_rescue = (
                    candidate["distance"] > short_connected_rescue_limit
                    and candidate["distance"] <= long_connected_rescue_limit
                    and nearest_area_error is not None
                    and nearest_area_error > 0.30
                    and rescue_direction is not None
                    and area_error <= 0.05
                    and nearest_area_error - area_error >= 0.20
                )
                is_standard_area_candidate = (
                    candidate["line_idx"] in ambiguous_ids
                    and candidate["distance"] <= area_rescue_limit
                )
                if (
                    is_standard_area_candidate
                    or is_short_connected_rescue
                    or is_long_connected_rescue
                ):
                    candidate["upstream_area_km2"] = float(upland)
                    candidate["area_relative_error"] = area_error
                    candidate["area_log_error"] = abs(
                        math.log(float(upland) / station_area)
                    )
                    candidate["connected_area_rescue_direction"] = (
                        rescue_direction
                        if is_short_connected_rescue or is_long_connected_rescue
                        else None
                    )
                    candidate["connected_area_rescue_tier"] = (
                        "long" if is_long_connected_rescue
                        else "short" if is_short_connected_rescue else None
                    )
                    area_candidates.append(candidate)

            area_candidates.sort(key=lambda candidate: (
                candidate["area_relative_error"], candidate["distance"],
                candidate["endpoint_penalty"], candidate["line_idx"],
            ))
            if area_candidates:
                best_area_error = area_candidates[0]["area_relative_error"]
                second_area_error = (
                    area_candidates[1]["area_relative_error"]
                    if len(area_candidates) > 1 else math.inf
                )
                # 面积误差不超过 15%，且比次优候选至少好 5 个百分点，
                # 才视为决定性物理证据。5 个百分点可区分 204 站的
                # 0.7%/9.2% 候选，同时仍把更接近的面积差异留给其他规则。
                winning_rescue_tier = area_candidates[0].get(
                    "connected_area_rescue_tier"
                )
                required_advantage = 0.10 if winning_rescue_tier == "long" else 0.05
                area_rule_applied = (
                    best_area_error <= 0.15
                    and second_area_error - best_area_error >= required_advantage
                )
                if area_rule_applied:
                    connected_area_rescue_tier = winning_rescue_tier
                    connected_area_rescue_direction = area_candidates[0].get(
                        "connected_area_rescue_direction"
                    )
                    connected_area_rescue_applied = (
                        connected_area_rescue_tier is not None
                    )
                    sibling_area_rescue_applied = (
                        connected_area_rescue_direction == "sibling"
                    )
                    if connected_area_rescue_direction == "upstream":
                        upstream_area_rescue_tier = connected_area_rescue_tier
                        upstream_area_rescue_applied = True
                    # 5 个百分点以内视为面积等效，保留给距离和端点消歧。
                    plausible = [
                        candidate for candidate in area_candidates
                        if candidate["area_relative_error"] <= best_area_error + 0.05
                    ]
                else:
                    # 大流域中，小支流汇入只占干流面积的几个百分点，固定的
                    # 5 个百分点优势会失效。若最佳候选是汇流前的同级干流，
                    # 且汇流前后面积增量可由另一条直接汇入支流解释，则用
                    # 面积守恒确定站点位于汇流前一侧。
                    best_candidate = area_candidates[0]
                    best_row = river_gdf.iloc[best_candidate["line_idx"]]
                    best_id = _normalise_key(best_row.get("HYRIV_ID"))
                    best_next = _normalise_key(best_row.get("NEXT_DOWN"))
                    best_upland = best_candidate.get("upstream_area_km2")
                    best_order = pd.to_numeric(best_row.get("ORD_STRA"), errors="coerce")
                    nearest_order = pd.to_numeric(
                        nearest_row.get("ORD_STRA"), errors="coerce"
                    )
                    same_mainstem_order = (
                        pd.notna(best_order) and pd.notna(nearest_order)
                        and float(best_order) == float(nearest_order)
                    )
                    area_increment = (
                        float(nearest_upland) - float(best_upland)
                        if pd.notna(nearest_upland) and best_upland is not None else None
                    )
                    nearest_abs_error = (
                        abs(float(nearest_upland) - station_area)
                        if pd.notna(nearest_upland) else None
                    )
                    best_abs_error = (
                        abs(float(best_upland) - station_area)
                        if best_upland is not None else None
                    )
                    sibling_increment_matches = []
                    if area_increment is not None and area_increment > 0:
                        for sibling_idx in direct_upstream_indices.get(nearest_id, []):
                            sibling_row = river_gdf.iloc[sibling_idx]
                            sibling_id = _normalise_key(sibling_row.get("HYRIV_ID"))
                            if sibling_id == best_id:
                                continue
                            sibling_area = pd.to_numeric(
                                sibling_row.get("UPLAND_SKM"), errors="coerce"
                            )
                            if pd.notna(sibling_area) and float(sibling_area) > 0:
                                sibling_increment_matches.append(
                                    abs(float(sibling_area) - area_increment) / area_increment
                                )
                    similar_upstream_candidates = [
                        candidate for candidate in area_candidates
                        if _normalise_key(
                            river_gdf.iloc[candidate["line_idx"]].get("NEXT_DOWN")
                        ) == nearest_id
                        and candidate["area_relative_error"] <= best_area_error + 0.01
                    ]
                    confluence_area_increment_applied = (
                        best_candidate["distance"] <= area_rescue_limit
                        and best_next == nearest_id
                        and same_mainstem_order
                        and best_area_error <= 0.05
                        and nearest_area_error is not None
                        and nearest_area_error >= 3 * max(best_area_error, 1e-9)
                        and area_increment is not None and area_increment > 0
                        and nearest_abs_error is not None and best_abs_error is not None
                        and (nearest_abs_error - best_abs_error) / area_increment >= 0.50
                        and bool(sibling_increment_matches)
                        and min(sibling_increment_matches) <= 0.25
                        and len(similar_upstream_candidates) == 1
                    )
                    if confluence_area_increment_applied:
                        area_rule_applied = True
                        plausible = [best_candidate]

            # 若最近河段的面积明显大于站点，说明测站很可能位于该河段的
            # 某条直接上游分支。此时先在拓扑连通的分支中用距离定位，不能
            # 让面积略优、但属于另一平行水系的候选跨分支“抢挂”。
            direct_upstream_branch_candidates = []
            if (
                nearest_area_error is not None and nearest_area_error > 0.30
                and pd.notna(nearest_upland) and float(nearest_upland) > station_area
            ):
                for candidate in candidates:
                    candidate_row = river_gdf.iloc[candidate["line_idx"]]
                    if _normalise_key(candidate_row.get("NEXT_DOWN")) != nearest_id:
                        continue
                    upland = pd.to_numeric(
                        candidate_row.get("UPLAND_SKM"), errors="coerce"
                    )
                    if pd.isna(upland) or float(upland) <= 0:
                        continue
                    area_error = abs(float(upland) - station_area) / station_area
                    if (
                        candidate["distance"] <= short_connected_rescue_limit
                        and area_error <= 0.15
                    ):
                        candidate["upstream_area_km2"] = float(upland)
                        candidate["area_relative_error"] = area_error
                        candidate["area_log_error"] = abs(
                            math.log(float(upland) / station_area)
                        )
                        direct_upstream_branch_candidates.append(candidate)

            direct_upstream_branch_candidates.sort(key=lambda candidate: (
                candidate["distance"] + candidate["endpoint_penalty"],
                candidate["area_relative_error"], candidate["line_idx"],
            ))
            if direct_upstream_branch_candidates:
                branch_candidate = direct_upstream_branch_candidates[0]
                branch_effective_distance = (
                    branch_candidate["distance"] + branch_candidate["endpoint_penalty"]
                )
                second_branch_effective_distance = (
                    direct_upstream_branch_candidates[1]["distance"]
                    + direct_upstream_branch_candidates[1]["endpoint_penalty"]
                    if len(direct_upstream_branch_candidates) > 1 else math.inf
                )
                direct_upstream_branch_distance_margin_m = (
                    second_branch_effective_distance - branch_effective_distance
                )
                competing_best_error = (
                    area_candidates[0]["area_relative_error"]
                    if area_candidates else math.inf
                )
                branch_not_materially_worse = (
                    branch_candidate["area_relative_error"]
                    - competing_best_error <= 0.10
                )
                distance_is_decisive = (
                    direct_upstream_branch_distance_margin_m
                    >= max(100.0, 0.25 * branch_effective_distance)
                )
                direct_upstream_branch_priority_applied = (
                    branch_not_materially_worse and distance_is_decisive
                )
                direct_upstream_branch_boundary = (
                    branch_not_materially_worse and not distance_is_decisive
                )
                if direct_upstream_branch_priority_applied:
                    area_rule_applied = True
                    confluence_area_increment_applied = False
                    connected_area_rescue_applied = False
                    connected_area_rescue_direction = None
                    connected_area_rescue_tier = None
                    sibling_area_rescue_applied = False
                    upstream_area_rescue_applied = False
                    upstream_area_rescue_tier = None
                    plausible = [branch_candidate]

            # 测站可能代表汇流后的控制断面，但其面积对应汇流点入口，因而
            # 不一定与包含整段局部集水区的下游河段 UPLAND_SKM 一致。此时
            # 检查所有直接汇入河段的面积之和，避免把测站留在最近的小支流。
            if not area_rule_applied and nearest_next is not None:
                incoming_indices = direct_upstream_indices.get(nearest_next, [])
                incoming_rows = [river_gdf.iloc[idx] for idx in incoming_indices]
                incoming_areas = [
                    pd.to_numeric(row.get("UPLAND_SKM"), errors="coerce")
                    for row in incoming_rows
                ]
                downstream_matches = [
                    candidate for candidate in candidates
                    if _normalise_key(
                        river_gdf.iloc[candidate["line_idx"]].get("HYRIV_ID")
                    ) == nearest_next
                    and candidate["distance"] <= short_connected_rescue_limit
                ]
                valid_incoming_areas = (
                    len(incoming_indices) >= 2
                    and nearest_id in {
                        _normalise_key(row.get("HYRIV_ID")) for row in incoming_rows
                    }
                    and all(pd.notna(area) and float(area) > 0 for area in incoming_areas)
                )
                if valid_incoming_areas and len(downstream_matches) == 1:
                    incoming_area_sum = sum(float(area) for area in incoming_areas)
                    incoming_sum_error = abs(
                        incoming_area_sum - station_area
                    ) / station_area
                    individual_errors = [
                        abs(float(area) - station_area) / station_area
                        for area in incoming_areas
                    ]
                    downstream_candidate = downstream_matches[0]
                    downstream_row = river_gdf.iloc[downstream_candidate["line_idx"]]
                    downstream_upland = pd.to_numeric(
                        downstream_row.get("UPLAND_SKM"), errors="coerce"
                    )
                    downstream_order = pd.to_numeric(
                        downstream_row.get("ORD_STRA"), errors="coerce"
                    )
                    incoming_orders = [
                        pd.to_numeric(row.get("ORD_STRA"), errors="coerce")
                        for row in incoming_rows
                    ]
                    order_is_consistent = (
                        pd.notna(downstream_order)
                        and all(pd.notna(order) for order in incoming_orders)
                        and float(downstream_order) >= max(
                            float(order) for order in incoming_orders
                        )
                    )
                    downstream_area_is_consistent = (
                        pd.notna(downstream_upland)
                        and 0.95 * incoming_area_sum <= float(downstream_upland)
                        <= 1.50 * incoming_area_sum
                    )
                    confluence_area_sum_applied = (
                        nearest_at_endpoint
                        and nearest_area_error is not None
                        and nearest_area_error > 0.30
                        and incoming_sum_error <= 0.05
                        and min(individual_errors) > 0.15
                        and order_is_consistent
                        and downstream_area_is_consistent
                    )
                    if confluence_area_sum_applied:
                        downstream_candidate["upstream_area_km2"] = float(
                            downstream_upland
                        )
                        downstream_candidate["area_relative_error"] = abs(
                            float(downstream_upland) - station_area
                        ) / station_area
                        downstream_candidate["area_log_error"] = abs(
                            math.log(float(downstream_upland) / station_area)
                        )
                        confluence_incoming_area_sum_km2 = incoming_area_sum
                        confluence_incoming_area_sum_error = incoming_sum_error
                        area_rule_applied = True
                        plausible = [downstream_candidate]

        # 部分河道站缺少控制面积，但站点元数据明确注明位于“干流”。若其
        # 最近河段只是汇流口旁、相对干流显著较弱的支流时，应允许在有限
        # 距离内救回到同一汇流结构中的优势干流。这里不再用“最近河段等级
        # <= 2”的绝对门槛，而由等级差、累计面积比和直接连通关系共同约束。
        # 该规则仍必须有明确的“干流”语义证据，从而避免把实际代表支流
        # 控制区的普通径流站一律推向干流。
        if node_profile == "channel" and not area_rule_applied:
            nearest_order_value = pd.to_numeric(
                nearest_row.get("ORD_STRA"), errors="coerce"
            )
            nearest_upland_value = pd.to_numeric(
                nearest_row.get("UPLAND_SKM"), errors="coerce"
            )
            dominant_candidates = []
            if (
                mainstem_fields
                and pd.notna(nearest_order_value)
                and pd.notna(nearest_upland_value)
                and float(nearest_upland_value) > 0
            ):
                nearest_next = _normalise_key(nearest_row.get("NEXT_DOWN"))
                for candidate in candidates:
                    if candidate["line_idx"] == candidates[0]["line_idx"]:
                        continue
                    candidate_row = river_gdf.iloc[candidate["line_idx"]]
                    candidate_id = _normalise_key(candidate_row.get("HYRIV_ID"))
                    candidate_next = _normalise_key(candidate_row.get("NEXT_DOWN"))
                    candidate_order = pd.to_numeric(
                        candidate_row.get("ORD_STRA"), errors="coerce"
                    )
                    candidate_upland = pd.to_numeric(
                        candidate_row.get("UPLAND_SKM"), errors="coerce"
                    )
                    is_sibling_mainstem = (
                        nearest_next is not None and candidate_next == nearest_next
                    )
                    is_direct_downstream_mainstem = (
                        nearest_next is not None and candidate_id == nearest_next
                    )
                    same_confluence_structure = (
                        is_sibling_mainstem or is_direct_downstream_mainstem
                    )
                    if (
                        same_confluence_structure
                        and candidate["distance"] <= nearest_distance + 1000.0
                        and candidate["distance"] <= 1500.0
                        and pd.notna(candidate_order)
                        and float(candidate_order) >= float(nearest_order_value) + 3
                        and pd.notna(candidate_upland)
                        and float(candidate_upland)
                        >= 20.0 * float(nearest_upland_value)
                    ):
                        candidate["mainstem_order"] = float(candidate_order)
                        candidate["mainstem_upland_km2"] = float(candidate_upland)
                        candidate["mainstem_relation"] = (
                            "sibling_incoming" if is_sibling_mainstem
                            else "direct_downstream"
                        )
                        dominant_candidates.append(candidate)
            def mainstem_candidate_rank(candidate):
                return (
                -candidate["mainstem_order"],
                candidate["endpoint_penalty"] > 0,
                candidate["distance"] + candidate["endpoint_penalty"],
                -candidate["mainstem_upland_km2"],
                candidate["line_idx"],
                )

            sibling_mainstems = sorted(
                (
                    candidate for candidate in dominant_candidates
                    if candidate["mainstem_relation"] == "sibling_incoming"
                ),
                key=mainstem_candidate_rank,
            )
            downstream_mainstems = sorted(
                (
                    candidate for candidate in dominant_candidates
                    if candidate["mainstem_relation"] == "direct_downstream"
                ),
                key=mainstem_candidate_rank,
            )
            # 默认保留汇流前干流入口；仅当该入口投影落在端点、而直接
            # 下游干流投影位于内部且有效距离至少改善 250 m 时，才认为
            # 测站位于汇流后的干流段。该判别区分同江与佳木斯两类情景。
            if sibling_mainstems:
                dominant_candidates = sibling_mainstems
                best_sibling = sibling_mainstems[0]
                if downstream_mainstems:
                    best_downstream = downstream_mainstems[0]
                    sibling_effective = (
                        best_sibling["distance"] + best_sibling["endpoint_penalty"]
                    )
                    downstream_effective = (
                        best_downstream["distance"]
                        + best_downstream["endpoint_penalty"]
                    )
                    if (
                        best_sibling["endpoint_penalty"] > 0
                        and best_downstream["endpoint_penalty"] == 0
                        and sibling_effective - downstream_effective >= 250.0
                    ):
                        dominant_candidates = downstream_mainstems
            else:
                dominant_candidates = downstream_mainstems
            if dominant_candidates:
                best_mainstem = dominant_candidates[0]
                runner_up = dominant_candidates[1] if len(dominant_candidates) > 1 else None
                best_is_unique = (
                    runner_up is None
                    or best_mainstem["mainstem_order"] > runner_up["mainstem_order"]
                    or best_mainstem["mainstem_upland_km2"]
                    >= 1.10 * runner_up["mainstem_upland_km2"]
                    or (
                        runner_up["distance"] + runner_up["endpoint_penalty"]
                        - best_mainstem["distance"]
                        - best_mainstem["endpoint_penalty"]
                        >= 250.0
                    )
                )
                if best_is_unique:
                    mainstem_metadata_rescue_applied = True
                    mainstem_metadata_field = mainstem_fields[0]
                    mainstem_order_gain = (
                        best_mainstem["mainstem_order"] - float(nearest_order_value)
                    )
                    mainstem_area_ratio = (
                        best_mainstem["mainstem_upland_km2"]
                        / float(nearest_upland_value)
                    )
                    mainstem_rescue_distance_m = best_mainstem["distance"]
                    mainstem_rescue_relation = best_mainstem["mainstem_relation"]
                    plausible = [best_mainstem]

        def rank(candidate):
            line_idx = candidate["line_idx"]
            row = river_gdf.iloc[line_idx]
            strahler = pd.to_numeric(row.get("ORD_STRA"), errors="coerce")
            upland = pd.to_numeric(row.get("UPLAND_SKM"), errors="coerce")
            strahler = float(strahler) if pd.notna(strahler) else -1.0
            upland = float(upland) if pd.notna(upland) else -1.0
            # 河流等级用于区分干支流；同等级时较大的累计汇水面积通常表示
            # 汇流后的干流延续段，可避免把汇流口附近站点放在某一条汇入分支上。
            effective_distance = candidate["distance"] + candidate["endpoint_penalty"]
            return (-strahler, -upland, effective_distance,
                    str(_normalise_key(row["HYRIV_ID"])), line_idx)

        if area_rule_applied:
            selected = min(plausible, key=lambda candidate: (
                candidate["area_relative_error"], candidate["distance"],
                candidate["endpoint_penalty"], candidate["line_idx"],
            ))
        else:
            selected = min(plausible, key=rank)
        line_idx = selected["line_idx"]
        selected_distance = selected["distance"]
        line = river_gdf.geometry.iloc[line_idx]
        projection = selected["projection"]
        selected_upland = pd.to_numeric(
            river_gdf.iloc[line_idx].get("UPLAND_SKM"), errors="coerce"
        )
        selected_upland = float(selected_upland) if pd.notna(selected_upland) else None
        selected_area_error = (
            abs(selected_upland - station_area) / station_area
            if selected_upland is not None and station_area is not None else None
        )
        selected_area_log_error = (
            abs(math.log(selected_upland / station_area))
            if selected_upland is not None and station_area is not None
            and selected_upland > 0 else None
        )

        selected_river_id = _normalise_key(river_gdf.iloc[line_idx]["HYRIV_ID"])
        selected_next = _normalise_key(river_gdf.iloc[line_idx].get("NEXT_DOWN"))
        selected_order = pd.to_numeric(river_gdf.iloc[line_idx].get("ORD_STRA"), errors="coerce")
        used_downstream_rule = selected in downstream_candidates
        if mainstem_metadata_rescue_applied:
            selection_reason = "mainstem_metadata_rescue_rule"
        elif direct_upstream_branch_priority_applied:
            selection_reason = "direct_upstream_branch_priority_rule"
        elif confluence_area_sum_applied:
            selection_reason = "confluence_area_sum_rule"
        elif confluence_area_increment_applied:
            selection_reason = "confluence_area_increment_rule"
        elif sibling_area_rescue_applied:
            selection_reason = "sibling_confluence_area_rescue_rule"
        elif connected_area_rescue_applied:
            selection_reason = "connected_catchment_area_rescue_rule"
        elif area_rule_applied:
            selection_reason = "catchment_area_consistency_rule"
        elif used_downstream_rule:
            selection_reason = "downstream_confluence_rule"
        elif selected_distance > nearest_distance + 1e-8:
            selection_reason = "network_hierarchy_rule"
        elif selected["endpoint_penalty"] > 0:
            selection_reason = "nearest_with_endpoint_penalty"
        else:
            selection_reason = "nearest"

        # 全量干流一致性审计：救回规则未触发时，继续检查 Top-k 中是否
        # 存在与当前河段直接相连、但等级和累计面积均显著占优的干流。
        # 此处只标记而不自动改挂，供人工确认规则边界。
        if (
            node_profile == "channel"
            and mainstem_fields
            and not mainstem_metadata_rescue_applied
            and pd.notna(selected_order)
            and selected_upland is not None
            and selected_upland > 0
        ):
            for candidate in candidates:
                if candidate["line_idx"] == line_idx:
                    continue
                candidate_row = river_gdf.iloc[candidate["line_idx"]]
                candidate_id = _normalise_key(candidate_row.get("HYRIV_ID"))
                candidate_next = _normalise_key(candidate_row.get("NEXT_DOWN"))
                candidate_order = pd.to_numeric(
                    candidate_row.get("ORD_STRA"), errors="coerce"
                )
                candidate_upland = pd.to_numeric(
                    candidate_row.get("UPLAND_SKM"), errors="coerce"
                )
                relation = (
                    "sibling_incoming"
                    if selected_next is not None and candidate_next == selected_next
                    else "direct_downstream"
                    if selected_next is not None and candidate_id == selected_next
                    else "direct_upstream"
                    if candidate_next == selected_river_id
                    else None
                )
                if (
                    relation is not None
                    and pd.notna(candidate_order)
                    and float(candidate_order) >= float(selected_order) + 3
                    and pd.notna(candidate_upland)
                    and float(candidate_upland) >= 20.0 * selected_upland
                ):
                    mainstem_mismatch_candidates.append({
                        "line_idx": candidate["line_idx"],
                        "river_id": candidate_id,
                        "distance_m": candidate["distance"],
                        "effective_distance_m": (
                            candidate["distance"] + candidate["endpoint_penalty"]
                        ),
                        "order": float(candidate_order),
                        "upstream_area_km2": float(candidate_upland),
                        "relation": relation,
                    })
            mainstem_mismatch_candidates.sort(key=lambda candidate: (
                -candidate["order"],
                candidate["effective_distance_m"],
                -candidate["upstream_area_km2"],
                candidate["line_idx"],
            ))
            mainstem_attachment_mismatch = bool(mainstem_mismatch_candidates)

        # 只有规则仍无法消除的歧义才需要人工审核。触发一条明确规则本身
        # 不降低置信度，例如更高河网等级或直接 NEXT_DOWN 汇流延续段。
        review_reasons = []
        if mainstem_attachment_mismatch:
            review_reasons.append("mainstem_attachment_mismatch")
        # 第一阶段仅记录“可能缺支流”：单独的面积不一致不足以确认河网
        # 缺失，后续还要结合附近其他站点对所选河段的占用证据。
        target_branch_missing_candidate = (
            node_profile == "channel"
            and station_area is not None
            and not area_rule_applied
            and not used_downstream_rule
            and selected_area_error is not None
            and selected_area_error > 0.30
            and (best_area_error is None or best_area_error > 0.30)
        )
        if selected_distance > max_distance and len(ambiguous) == 1:
            review_reasons.append("isolated_beyond_preferred_distance")
        if direct_upstream_branch_boundary:
            review_reasons.append("direct_upstream_branch_boundary")
        if (
            not area_rule_applied and best_area_error is not None
            and second_area_error is not None and best_area_error <= 0.15
            and second_area_error - best_area_error < 0.05
            and selected["line_idx"] != area_candidates[0]["line_idx"]
        ):
            review_reasons.append("catchment_area_boundary")

        unresolved_parallel = []
        for candidate in plausible:
            if candidate["line_idx"] == line_idx:
                continue
            row = river_gdf.iloc[candidate["line_idx"]]
            candidate_id = _normalise_key(row["HYRIV_ID"])
            candidate_next = _normalise_key(row.get("NEXT_DOWN"))
            candidate_order = pd.to_numeric(row.get("ORD_STRA"), errors="coerce")
            same_order = (
                pd.isna(selected_order) or pd.isna(candidate_order)
                or float(selected_order) == float(candidate_order)
            )
            directly_connected = (
                candidate_id == selected_next or selected_river_id == candidate_next
            )
            if same_order and not directly_connected:
                unresolved_parallel.append(candidate)
        if unresolved_parallel:
            review_reasons.append("unresolved_parallel_branches")

        # 若所选线仍落在端点且存在同等级、非直接连接的替代线，端点惩罚
        # 没能消除歧义，需要人工核验；单独的端点命中则已由规则处理。
        if selected["endpoint_penalty"] > 0 and unresolved_parallel:
            review_reasons.append("endpoint_conflict")
        confidence = "low" if review_reasons else "high"

        ambiguous_line_indices = [candidate["line_idx"] for candidate in plausible]
        if mainstem_attachment_mismatch:
            ambiguous_line_indices.extend(
                candidate["line_idx"] for candidate in mainstem_mismatch_candidates
            )
            ambiguous_line_indices = list(dict.fromkeys(ambiguous_line_indices))

        assignments[node_idx] = {
            "line_idx": line_idx,
            "river_id": _normalise_key(river_gdf.iloc[line_idx]["HYRIV_ID"]),
            "snap_dist_m": selected_distance,
            "nearest_dist_m": nearest_distance,
            "candidate_count": len(candidates),
            "top_k_candidates": [dict(candidate) for candidate in candidates],
            "top_k_line_indices": [
                candidate["line_idx"] for candidate in candidates
            ],
            "ambiguous_candidate_count": len(plausible),
            "ambiguous_line_indices": ambiguous_line_indices,
            "attachment_profile": node_profile,
            "channel_distance_band_m": channel_band,
            "station_area_km2": station_area,
            "selected_upstream_area_km2": selected_upland,
            "area_relative_error": selected_area_error,
            "area_log_error": selected_area_log_error,
            "best_area_relative_error": best_area_error,
            "area_advantage": (
                second_area_error - best_area_error
                if best_area_error is not None and second_area_error is not None else None
            ),
            "area_candidate_count": len(area_candidates),
            "area_rule_applied": area_rule_applied,
            "confluence_area_increment_applied": confluence_area_increment_applied,
            "confluence_area_sum_applied": confluence_area_sum_applied,
            "confluence_incoming_area_sum_km2": confluence_incoming_area_sum_km2,
            "confluence_incoming_area_sum_error": confluence_incoming_area_sum_error,
            "direct_upstream_branch_priority_applied": (
                direct_upstream_branch_priority_applied
            ),
            "direct_upstream_branch_distance_margin_m": (
                direct_upstream_branch_distance_margin_m
            ),
            "connected_area_rescue_applied": connected_area_rescue_applied,
            "connected_area_rescue_direction": connected_area_rescue_direction,
            "connected_area_rescue_tier": connected_area_rescue_tier,
            "sibling_area_rescue_applied": sibling_area_rescue_applied,
            "mainstem_metadata_rescue_applied": mainstem_metadata_rescue_applied,
            "mainstem_metadata_field": mainstem_metadata_field,
            "mainstem_order_gain": mainstem_order_gain,
            "mainstem_area_ratio": mainstem_area_ratio,
            "mainstem_rescue_distance_m": mainstem_rescue_distance_m,
            "mainstem_rescue_relation": mainstem_rescue_relation,
            "explicit_mainstem_metadata_field": (
                mainstem_fields[0] if mainstem_fields else None
            ),
            "mainstem_attachment_mismatch": mainstem_attachment_mismatch,
            "mainstem_mismatch_candidates": mainstem_mismatch_candidates,
            "upstream_area_rescue_applied": upstream_area_rescue_applied,
            "upstream_area_rescue_tier": upstream_area_rescue_tier,
            "selection_reason": selection_reason,
            "attachment_confidence": confidence,
            "attachment_risk": "|".join(review_reasons),
            "target_branch_missing_candidate": target_branch_missing_candidate,
            "target_branch_missing": False,
            "endpoint_distance_m": selected["endpoint_distance"],
            "endpoint_penalty_m": selected["endpoint_penalty"],
            "projection_m": float(projection),
            "snap_point": line.interpolate(projection),
        }
    return assignments


def _build_river_index(river_gdf):
    by_id, duplicates = {}, set()
    for idx, row in river_gdf.iterrows():
        river_id = _normalise_key(row["HYRIV_ID"])
        if river_id in by_id:
            duplicates.add(river_id)
            if row.geometry.length > river_gdf.geometry.iloc[by_id[river_id]].length:
                by_id[river_id] = idx
        else:
            by_id[river_id] = idx
    if duplicates:
        print(f"警告: {len(duplicates)} 个 HYRIV_ID 重复，主索引保留最长几何")
    next_by_id = {
        river_id: _normalise_key(river_gdf.iloc[idx]["NEXT_DOWN"])
        for river_id, idx in by_id.items()
    }
    return by_id, next_by_id


def _infer_downstream_orientation(river_gdf, by_id, next_by_id):
    downstream_is_last = {}
    for river_id, idx in by_id.items():
        line = river_gdf.geometry.iloc[idx]
        start, end = Point(line.coords[0]), Point(line.coords[-1])
        next_id = next_by_id[river_id]
        if next_id in by_id:
            next_line = river_gdf.geometry.iloc[by_id[next_id]]
            downstream_is_last[river_id] = end.distance(next_line) <= start.distance(next_line)
        else:
            downstream_is_last[river_id] = True
    return downstream_is_last


def _repair_area_monotonicity_conflicts(nodes_gdf, river_gdf, assignments,
                                        by_id, next_by_id, downstream_is_last,
                                        max_snap_distance):
    """利用面积单调性把汇流口冲突站分配到合理的上游支流。"""
    upstream_by_id = {}
    for river_id, next_id in next_by_id.items():
        if next_id is not None:
            upstream_by_id.setdefault(next_id, []).append(river_id)

    station_by_river = {}
    for node_idx, item in assignments.items():
        river_id = item["river_id"]
        line = river_gdf.geometry.iloc[item["line_idx"]]
        measure = (
            item["projection_m"]
            if downstream_is_last.get(river_id, True)
            else line.length - item["projection_m"]
        )
        station_by_river.setdefault(river_id, []).append((float(measure), node_idx))

    search_limit = max(3000.0, 1.5 * max_snap_distance)
    repaired_nodes = set()
    repair_count = 0

    def apply_repair(node_idx, candidate, shared_river_id):
        item = assignments[node_idx]
        line = river_gdf.geometry.iloc[candidate["line_idx"]]
        point = nodes_gdf.geometry.iloc[node_idx]
        projection = float(line.project(point))
        endpoint_distance = min(
            projection, max(0.0, line.length - projection)
        )
        item.update({
            "line_idx": candidate["line_idx"],
            "river_id": candidate["river_id"],
            "snap_dist_m": candidate["distance"],
            "selected_upstream_area_km2": candidate["area"],
            "area_relative_error": candidate["area_error"],
            "area_log_error": abs(math.log(candidate["area"] / item["station_area_km2"])),
            "area_rule_applied": True,
            "confluence_area_sum_applied": False,
            "confluence_incoming_area_sum_km2": None,
            "confluence_incoming_area_sum_error": None,
            "direct_upstream_branch_priority_applied": False,
            "direct_upstream_branch_distance_margin_m": None,
            "connected_area_rescue_applied": False,
            "connected_area_rescue_direction": None,
            "connected_area_rescue_tier": None,
            "upstream_area_rescue_applied": False,
            "upstream_area_rescue_tier": None,
            "monotonicity_repair_applied": True,
            "monotonicity_conflict_river_id": shared_river_id,
            "selection_reason": "area_monotonicity_conflict_repair_rule",
            "attachment_confidence": "high",
            "attachment_risk": "",
            "endpoint_distance_m": endpoint_distance,
            "endpoint_penalty_m": max(0.0, 250.0 - endpoint_distance),
            "projection_m": projection,
            "snap_point": line.interpolate(projection),
        })

    for shared_river_id, measured_nodes in station_by_river.items():
        measured_nodes.sort()
        upstream_ids = upstream_by_id.get(shared_river_id, [])
        if len(upstream_ids) < 2:
            continue

        for (_, source_idx), (_, target_idx) in zip(measured_nodes, measured_nodes[1:]):
            if source_idx in repaired_nodes or target_idx in repaired_nodes:
                continue
            source_area = assignments[source_idx].get("station_area_km2")
            target_area = assignments[target_idx].get("station_area_km2")
            if source_area is None or target_area is None:
                continue
            # 下游控制面积允许 15% 数据误差；更大的反向下降视为物理冲突。
            if target_area >= 0.85 * source_area:
                continue

            pair_candidates = {}
            for node_idx, station_area in (
                (source_idx, source_area), (target_idx, target_area)
            ):
                point = nodes_gdf.geometry.iloc[node_idx]
                candidates = []
                for river_id in upstream_ids:
                    line_idx = by_id.get(river_id)
                    if line_idx is None:
                        continue
                    row = river_gdf.iloc[line_idx]
                    upland = pd.to_numeric(row.get("UPLAND_SKM"), errors="coerce")
                    if pd.isna(upland) or float(upland) <= 0:
                        continue
                    distance = float(row.geometry.distance(point))
                    area_error = abs(float(upland) - station_area) / station_area
                    if distance <= search_limit and area_error <= 0.15:
                        candidates.append({
                            "river_id": river_id,
                            "line_idx": line_idx,
                            "distance": distance,
                            "area": float(upland),
                            "area_error": area_error,
                        })
                pair_candidates[node_idx] = candidates

            options = []
            for source_candidate in pair_candidates[source_idx]:
                for target_candidate in pair_candidates[target_idx]:
                    if source_candidate["river_id"] == target_candidate["river_id"]:
                        continue
                    total_area_error = (
                        source_candidate["area_error"] + target_candidate["area_error"]
                    )
                    total_distance = (
                        source_candidate["distance"] + target_candidate["distance"]
                    )
                    options.append((
                        total_area_error, total_distance,
                        source_candidate, target_candidate,
                    ))
            if not options:
                continue
            options.sort(key=lambda option: (option[0], option[1]))
            best = options[0]
            second_area_error = options[1][0] if len(options) > 1 else math.inf
            current_errors = [
                assignments[node_idx].get("area_relative_error")
                for node_idx in (source_idx, target_idx)
            ]
            if any(error is None for error in current_errors):
                continue
            current_area_error = sum(current_errors)
            # 联合面积误差至少改善 20 个百分点，且最佳支流组合比次优
            # 至少好 5 个百分点，才自动消除倒挂。
            if current_area_error - best[0] < 0.20 or second_area_error - best[0] < 0.05:
                continue

            for node_idx, candidate in (
                (source_idx, best[2]), (target_idx, best[3])
            ):
                apply_repair(node_idx, candidate, shared_river_id)
                repaired_nodes.add(node_idx)
                repair_count += 1

    # 第二类倒挂发生在相邻河段：某站已在一条汇入支流上，而面积更小的
    # “下游站”误挂到公共汇流后河段。此时只需把下游站移到未占用且面积
    # 匹配的兄弟支流。该检查扩展了 same_segment 校正，但仍限制为一跳连通。
    occupied_by_river = {}
    for node_idx, item in assignments.items():
        occupied_by_river.setdefault(item["river_id"], set()).add(node_idx)

    for target_idx, target_item in assignments.items():
        if target_idx in repaired_nodes:
            continue
        shared_river_id = target_item["river_id"]
        upstream_ids = upstream_by_id.get(shared_river_id, [])
        if len(upstream_ids) < 2:
            continue
        target_area = target_item.get("station_area_km2")
        current_error = target_item.get("area_relative_error")
        if target_area is None or current_error is None:
            continue

        repaired = False
        for occupied_upstream_id in upstream_ids:
            source_indexes = occupied_by_river.get(occupied_upstream_id, set())
            for source_idx in source_indexes:
                source_area = assignments[source_idx].get("station_area_km2")
                if source_area is None or target_area >= 0.85 * source_area:
                    continue

                candidates = []
                for candidate_id in upstream_ids:
                    if candidate_id == occupied_upstream_id:
                        continue
                    if occupied_by_river.get(candidate_id):
                        continue
                    line_idx = by_id.get(candidate_id)
                    if line_idx is None:
                        continue
                    row = river_gdf.iloc[line_idx]
                    upland = pd.to_numeric(row.get("UPLAND_SKM"), errors="coerce")
                    if pd.isna(upland) or float(upland) <= 0:
                        continue
                    distance = float(row.geometry.distance(
                        nodes_gdf.geometry.iloc[target_idx]
                    ))
                    area_error = abs(float(upland) - target_area) / target_area
                    if distance <= search_limit and area_error <= 0.15:
                        candidates.append({
                            "river_id": candidate_id,
                            "line_idx": line_idx,
                            "distance": distance,
                            "area": float(upland),
                            "area_error": area_error,
                        })
                if not candidates:
                    continue
                candidates.sort(key=lambda candidate: (
                    candidate["area_error"], candidate["distance"]
                ))
                best = candidates[0]
                second_error = (
                    candidates[1]["area_error"] if len(candidates) > 1 else math.inf
                )
                if current_error - best["area_error"] < 0.20:
                    continue
                if second_error - best["area_error"] < 0.05:
                    continue

                occupied_by_river[shared_river_id].discard(target_idx)
                apply_repair(target_idx, best, shared_river_id)
                occupied_by_river.setdefault(best["river_id"], set()).add(target_idx)
                repaired_nodes.add(target_idx)
                repair_count += 1
                repaired = True
                break
            if repaired:
                break
    return repair_count


def _repair_joint_confluence_station_groups(nodes_gdf, river_gdf, assignments,
                                              by_id, next_by_id):
    """将同一汇流口的多个低置信站点一对一分配到面积匹配的兄弟支流。"""
    upstream_by_id = {}
    for river_id, next_id in next_by_id.items():
        if next_id is not None:
            upstream_by_id.setdefault(next_id, []).append(river_id)

    repaired_nodes = set()
    repaired_groups = []
    for shared_river_id, upstream_ids in upstream_by_id.items():
        if len(upstream_ids) < 2:
            continue
        # 至少一个站当前落在汇流后河段，才存在“上游站被截获”的联合冲突。
        nearby_nodes = [
            node_idx for node_idx, item in assignments.items()
            if node_idx not in repaired_nodes
            and item.get("attachment_profile") == "channel"
            and item.get("attachment_confidence") == "low"
            and item.get("river_id") in set(upstream_ids) | {shared_river_id}
        ]
        if len(nearby_nodes) < 2 or not any(
            assignments[node_idx].get("river_id") == shared_river_id
            for node_idx in nearby_nodes
        ):
            continue

        options_by_node = {}
        for node_idx in nearby_nodes:
            station_area = assignments[node_idx].get("station_area_km2")
            if station_area is None or station_area <= 0:
                continue
            point = nodes_gdf.geometry.iloc[node_idx]
            distance_limit = assignments[node_idx].get("nearest_dist_m", 0.0) + 500.0
            options = []
            for river_id in upstream_ids:
                line_idx = by_id.get(river_id)
                if line_idx is None:
                    continue
                row = river_gdf.iloc[line_idx]
                upland = pd.to_numeric(row.get("UPLAND_SKM"), errors="coerce")
                if pd.isna(upland) or float(upland) <= 0:
                    continue
                line = row.geometry
                distance = float(line.distance(point))
                projection = float(line.project(point))
                endpoint_distance = min(
                    projection, max(0.0, line.length - projection)
                )
                area_error = abs(float(upland) - station_area) / station_area
                if (
                    distance <= distance_limit and endpoint_distance <= 250.0
                    and area_error <= 0.15
                ):
                    options.append({
                        "river_id": river_id,
                        "line_idx": line_idx,
                        "distance": distance,
                        "projection": projection,
                        "endpoint_distance": endpoint_distance,
                        "endpoint_penalty": max(0.0, 250.0 - endpoint_distance),
                        "area": float(upland),
                        "area_error": area_error,
                    })
            if options:
                options_by_node[node_idx] = options

        group_nodes = sorted(options_by_node)
        if len(group_nodes) < 2:
            continue
        if len({
            option["river_id"]
            for options in options_by_node.values() for option in options
        }) < len(group_nodes):
            continue

        matchings = []

        def search(position, used_rivers, chosen):
            if position == len(group_nodes):
                total_area_error = sum(option["area_error"] for option in chosen)
                total_effective_distance = sum(
                    option["distance"] + option["endpoint_penalty"]
                    for option in chosen
                )
                matchings.append((total_area_error, total_effective_distance, list(chosen)))
                return
            node_idx = group_nodes[position]
            for option in options_by_node[node_idx]:
                if option["river_id"] in used_rivers:
                    continue
                used_rivers.add(option["river_id"])
                chosen.append(option)
                search(position + 1, used_rivers, chosen)
                chosen.pop()
                used_rivers.remove(option["river_id"])

        search(0, set(), [])
        if not matchings:
            continue
        matchings.sort(key=lambda value: (value[0], value[1]))
        best = matchings[0]
        second_area_error = matchings[1][0] if len(matchings) > 1 else math.inf
        identity_ambiguous = second_area_error - best[0] < 0.05

        for node_idx, selected_option in zip(group_nodes, best[2]):
            item = assignments[node_idx]
            line = river_gdf.geometry.iloc[selected_option["line_idx"]]
            item.update({
                "line_idx": selected_option["line_idx"],
                "river_id": selected_option["river_id"],
                "snap_dist_m": selected_option["distance"],
                "selected_upstream_area_km2": selected_option["area"],
                "area_relative_error": selected_option["area_error"],
                "area_log_error": abs(math.log(
                    selected_option["area"] / item["station_area_km2"]
                )),
                "area_rule_applied": True,
                "monotonicity_repair_applied": False,
                "joint_confluence_assignment_applied": True,
                "joint_confluence_river_id": shared_river_id,
                "joint_confluence_assignment_ambiguous": identity_ambiguous,
                "selection_reason": "joint_confluence_branch_assignment_rule",
                "attachment_confidence": "low" if identity_ambiguous else "high",
                "attachment_risk": (
                    "joint_branch_identity_ambiguity" if identity_ambiguous else ""
                ),
                "ambiguous_line_indices": [
                    option["line_idx"] for option in options_by_node[node_idx]
                ],
                "ambiguous_candidate_count": len(options_by_node[node_idx]),
                "endpoint_distance_m": selected_option["endpoint_distance"],
                "endpoint_penalty_m": selected_option["endpoint_penalty"],
                "projection_m": selected_option["projection"],
                "snap_point": line.interpolate(selected_option["projection"]),
            })
            repaired_nodes.add(node_idx)
        repaired_groups.append({
            "river_id": shared_river_id,
            "node_indices": group_nodes,
            "identity_ambiguous": identity_ambiguous,
        })

    for item in assignments.values():
        item.setdefault("joint_confluence_assignment_applied", False)
        item.setdefault("joint_confluence_river_id", None)
        item.setdefault("joint_confluence_assignment_ambiguous", False)
    return repaired_groups


def _index_stations_by_river(river_gdf, assignments, downstream_is_last):
    """按沿河下游位置建立站点索引，并同步更新投影量。"""
    station_by_river = {}
    for node_idx, item in assignments.items():
        river_id = item["river_id"]
        line = river_gdf.geometry.iloc[item["line_idx"]]
        measure = (
            item["projection_m"]
            if downstream_is_last.get(river_id, True)
            else line.length - item["projection_m"]
        )
        item["downstream_measure_m"] = float(measure)
        station_by_river.setdefault(river_id, []).append(node_idx)
    for river_id, indexes in station_by_river.items():
        indexes.sort(key=lambda i: (assignments[i]["downstream_measure_m"], i))
    return station_by_river


def _assemble_edges_with_area_monotonicity(
    nodes_gdf, river_gdf, assignments, station_by_river, by_id, next_by_id,
    area_tolerance=0.05, enforce_area_monotonicity=True,
):
    """沿河网构边，并跳过控制面积显著小于源站的伪下游站。"""
    edges = []
    diagnostics = Counter()

    def target_is_area_compatible(source_idx, target_idx):
        if not enforce_area_monotonicity:
            return True
        source_area = assignments[source_idx].get("station_area_km2")
        target_area = assignments[target_idx].get("station_area_km2")
        target_item = assignments[target_idx]
        # 已被既有异常规则确认的“缺失支流”或汇流侧冲突必须保留原边并
        # 降低置信度，不能让面积跳过把真实的数据缺陷伪装成确定结果。
        if (
            target_item.get("target_branch_missing", False)
            or (
                target_item.get("selection_reason") == "downstream_confluence_rule"
                and target_item.get("best_area_relative_error") is not None
                and target_item.get("best_area_relative_error") > 0.30
            )
        ):
            return True
        return (
            source_area is None or target_area is None
            or target_area >= (1.0 - area_tolerance) * source_area
        )

    for item in assignments.values():
        item["edge_area_monotonicity_applied"] = False
        item["edge_area_violation_unresolved"] = False
        item["skipped_area_violation_station_indices"] = []

    for node_idx in range(len(nodes_gdf)):
        if node_idx not in assignments:
            diagnostics["unattached"] += 1
            edges.append((node_idx, None, None, "unattached"))
            continue
        item = assignments[node_idx]
        river_id = item["river_id"]
        measure = item["downstream_measure_m"]
        skipped = []
        fallback = None

        same_line = [
            i for i in station_by_river.get(river_id, [])
            if i != node_idx
            and assignments[i]["downstream_measure_m"] > measure + 1e-8
        ]
        for target in same_line:
            distance_m = assignments[target]["downstream_measure_m"] - measure
            if target_is_area_compatible(node_idx, target):
                item["skipped_area_violation_station_indices"] = skipped
                item["edge_area_monotonicity_applied"] = bool(skipped)
                if skipped:
                    diagnostics["downstream_area_violation_skipped"] += len(skipped)
                edges.append((node_idx, target, distance_m / 1000.0, "same_segment"))
                diagnostics["same_segment"] += 1
                break
            skipped.append(target)
            if fallback is None:
                fallback = (target, distance_m / 1000.0, "same_segment")
        else:
            line = river_gdf.geometry.iloc[by_id[river_id]]
            distance_m = line.length - measure
            visited = {river_id}
            current = next_by_id.get(river_id)
            result = None
            status = "outlet"
            while current is not None:
                if current in visited:
                    status = "cycle"
                    break
                visited.add(current)
                if current not in by_id:
                    status = "broken_next_down"
                    break
                candidates = station_by_river.get(current, [])
                selected_on_segment = None
                for target in candidates:
                    candidate_distance = (
                        distance_m + assignments[target]["downstream_measure_m"]
                    )
                    if target_is_area_compatible(node_idx, target):
                        selected_on_segment = (
                            target, candidate_distance / 1000.0
                        )
                        break
                    skipped.append(target)
                    if fallback is None:
                        fallback = (
                            target, candidate_distance / 1000.0,
                            "downstream_segment",
                        )
                if selected_on_segment is not None:
                    result = selected_on_segment
                    status = "downstream_segment"
                    break
                distance_m += river_gdf.geometry.iloc[by_id[current]].length
                current = next_by_id.get(current)

            if result is not None:
                item["skipped_area_violation_station_indices"] = skipped
                item["edge_area_monotonicity_applied"] = bool(skipped)
                if skipped:
                    diagnostics["downstream_area_violation_skipped"] += len(skipped)
                edges.append((node_idx, result[0], result[1], status))
                diagnostics[status] += 1
            elif fallback is not None:
                # 若整条可达路径都没有面积相容站点，保留最近拓扑边但明确
                # 降低置信度，避免因为单个面积元数据错误而切断整张图。
                item["skipped_area_violation_station_indices"] = skipped
                item["edge_area_violation_unresolved"] = True
                edges.append((node_idx, fallback[0], fallback[1], fallback[2]))
                diagnostics[fallback[2]] += 1
                diagnostics["downstream_area_violation_unresolved"] += 1
            else:
                edges.append((node_idx, None, None, status))
                diagnostics[status] += 1
    return edges, diagnostics


def _repair_skipped_intermediate_station_attachments(
    nodes_gdf, river_gdf, assignments, edges, by_id, next_by_id,
    downstream_is_last,
):
    """用面积递增和 Top-k 路径证据救回被越过的直接下游站。"""
    edge_by_source = {edge[0]: edge for edge in edges}
    proposals = {}

    def path_to_target(source_item, target_item):
        source_river = source_item["river_id"]
        target_river = target_item["river_id"]
        path = []
        current = source_river
        visited = set()
        while current is not None and current not in visited and current in by_id:
            visited.add(current)
            path.append(current)
            if current == target_river:
                return path
            current = next_by_id.get(current)
        return None

    for source_idx, target_idx, distance_km, status in edges:
        if target_idx is None or distance_km is None or distance_km < 1.0:
            continue
        source_area = assignments[source_idx].get("station_area_km2")
        target_area = assignments[target_idx].get("station_area_km2")
        if (
            source_area is None or target_area is None
            or target_area < 0.95 * source_area
        ):
            continue
        path = path_to_target(assignments[source_idx], assignments[target_idx])
        if not path:
            continue
        path_position = {river_id: position for position, river_id in enumerate(path)}
        eligible = []
        for middle_idx, middle_item in assignments.items():
            if middle_idx in {source_idx, target_idx}:
                continue
            if (
                middle_item.get("area_rule_applied", False)
                or middle_item.get("monotonicity_repair_applied", False)
                or middle_item.get("joint_confluence_assignment_applied", False)
            ):
                continue
            middle_edge = edge_by_source.get(middle_idx)
            middle_area = middle_item.get("station_area_km2")
            if (
                middle_edge is None or middle_edge[1] != target_idx
                or middle_area is None
                or middle_area < 0.95 * source_area
                # “被越过的直接下一站”应是源站累计面积的小幅增长；若面积
                # 已跳增数倍，更可能是另一条在汇流口附近的平行支流站。
                or middle_area > 1.75 * source_area
                or target_area < 0.95 * middle_area
            ):
                continue
            options = []
            for candidate in middle_item.get("top_k_candidates", []):
                candidate_line_idx = candidate["line_idx"]
                candidate_river = _normalise_key(
                    river_gdf.iloc[candidate_line_idx]["HYRIV_ID"]
                )
                if candidate_river not in path_position:
                    continue
                if candidate["distance"] > min(
                    500.0, middle_item.get("nearest_dist_m", 0.0) + 150.0
                ):
                    continue
                line = river_gdf.geometry.iloc[candidate_line_idx]
                projection = float(line.project(nodes_gdf.geometry.iloc[middle_idx]))
                measure = (
                    projection if downstream_is_last.get(candidate_river, True)
                    else line.length - projection
                )
                if (
                    candidate_river == assignments[source_idx]["river_id"]
                    and measure <= assignments[source_idx]["downstream_measure_m"] + 1e-8
                ):
                    continue
                if (
                    candidate_river == assignments[target_idx]["river_id"]
                    and measure >= assignments[target_idx]["downstream_measure_m"] - 1e-8
                ):
                    continue
                options.append((
                    path_position[candidate_river], measure,
                    candidate["distance"] + candidate["endpoint_penalty"],
                    candidate, candidate_river,
                ))
            if options:
                options.sort(key=lambda value: (value[0], value[1], value[2]))
                eligible.append((options[0], middle_area, middle_idx))

        # 多个站点同时满足时证据不足，不把参考式层级猜测写成确定规则。
        if len(eligible) != 1:
            continue
        option, middle_area, middle_idx = eligible[0]
        candidate = option[3]
        candidate_river = option[4]
        proposal = (candidate, candidate_river, source_idx, target_idx)
        existing = proposals.get(middle_idx)
        if existing is None or candidate["distance"] < existing[0]["distance"]:
            proposals[middle_idx] = proposal

    for middle_idx, (candidate, candidate_river, source_idx, target_idx) in proposals.items():
        item = assignments[middle_idx]
        line_idx = candidate["line_idx"]
        line = river_gdf.geometry.iloc[line_idx]
        projection = float(line.project(nodes_gdf.geometry.iloc[middle_idx]))
        endpoint_distance = min(
            projection, max(0.0, line.length - projection)
        )
        upland = pd.to_numeric(
            river_gdf.iloc[line_idx].get("UPLAND_SKM"), errors="coerce"
        )
        upland = float(upland) if pd.notna(upland) and float(upland) > 0 else None
        station_area = item.get("station_area_km2")
        item.update({
            "line_idx": line_idx,
            "river_id": candidate_river,
            "snap_dist_m": candidate["distance"],
            "selected_upstream_area_km2": upland,
            "area_relative_error": (
                abs(upland - station_area) / station_area
                if upland is not None and station_area else None
            ),
            "area_log_error": (
                abs(math.log(upland / station_area))
                if upland is not None and station_area else None
            ),
            "skipped_station_rescue_applied": True,
            "skipped_station_rescue_source_index": source_idx,
            "skipped_station_rescue_target_index": target_idx,
            "selection_reason": "skipped_station_route_rescue_rule",
            "attachment_confidence": "high",
            "attachment_risk": "",
            "endpoint_distance_m": endpoint_distance,
            "endpoint_penalty_m": max(0.0, 250.0 - endpoint_distance),
            "projection_m": projection,
            "snap_point": line.interpolate(projection),
        })

    for item in assignments.values():
        item.setdefault("skipped_station_rescue_applied", False)
        item.setdefault("skipped_station_rescue_source_index", None)
        item.setdefault("skipped_station_rescue_target_index", None)
    return len(proposals)


def build_station_topology(river_gdf, nodes_gdf, max_snap_distance=3000,
                           snap_ambiguity_distance=500,
                           attachment_profile="auto",
                           enable_attachment_monotonicity_repair=True,
                           enable_joint_confluence_assignment=True,
                           enable_downstream_area_monotonicity=True,
                           enable_skipped_station_rescue=True):
    """构建最近下游站点边，返回 edges、唯一挂接结果和异常诊断。"""
    river_gdf = _validate_river_network(river_gdf)
    by_id, next_by_id = _build_river_index(river_gdf)
    # 重复编号不能作为两个网络节点；先规范为每个 HYRIV_ID 一条主几何再挂站。
    if len(by_id) != len(river_gdf):
        river_gdf = river_gdf.iloc[sorted(by_id.values())].reset_index(drop=True)
        by_id, next_by_id = _build_river_index(river_gdf)
    if attachment_profile == "euclidean_nearest":
        assignments = _choose_euclidean_nearest_station_segments(
            nodes_gdf, river_gdf
        )
        downstream_is_last = _infer_downstream_orientation(
            river_gdf, by_id, next_by_id
        )
        station_by_river = _index_stations_by_river(
            river_gdf, assignments, downstream_is_last
        )
        edges, edge_diagnostics = _assemble_edges_with_area_monotonicity(
            nodes_gdf, river_gdf, assignments, station_by_river,
            by_id, next_by_id,
        )
        for source_idx, target_idx, _, _ in edges:
            item = assignments.get(source_idx)
            if item is None:
                continue
            item["edge_rule"] = "euclidean_nearest"
            item["edge_confidence"] = "high"
            item["edge_risk"] = ""
        diagnostics = Counter(
            stations_total=len(nodes_gdf),
            stations_attached=len(assignments),
        )
        diagnostics.update(edge_diagnostics)
        return edges, assignments, dict(diagnostics), river_gdf
    assignments = _choose_unique_station_segments(
        nodes_gdf, river_gdf, max_snap_distance, snap_ambiguity_distance,
        attachment_profile=attachment_profile,
    )
    downstream_is_last = _infer_downstream_orientation(river_gdf, by_id, next_by_id)
    monotonicity_repair_count = 0
    if enable_attachment_monotonicity_repair:
        monotonicity_repair_count = _repair_area_monotonicity_conflicts(
            nodes_gdf, river_gdf, assignments, by_id, next_by_id,
            downstream_is_last, max_snap_distance,
        )
    joint_confluence_groups = []
    if enable_joint_confluence_assignment:
        joint_confluence_groups = _repair_joint_confluence_station_groups(
            nodes_gdf, river_gdf, assignments, by_id, next_by_id,
        )
    else:
        for item in assignments.values():
            item.setdefault("joint_confluence_assignment_applied", False)
            item.setdefault("joint_confluence_river_id", None)
            item.setdefault("joint_confluence_assignment_ambiguous", False)

    station_by_river = _index_stations_by_river(
        river_gdf, assignments, downstream_is_last
    )

    # 第二阶段确认“目标支流缺失”。若附近另一个站点在同一算法河段上与
    # 河段面积吻合，而当前站严重不符，则该河段已被另一站合理解释；当前
    # 站更可能属于未被当前河网表达的邻近支流。河名可用时还要求河名不同。
    river_name_field = next(
        (field for field in ("river", "RIVER", "RIV_NAME", "river_name")
         if field in nodes_gdf.columns),
        None,
    )
    target_branch_missing_nodes = set()
    for node_idx, item in assignments.items():
        if (
            not item.get("target_branch_missing_candidate", False)
            or item.get("area_rule_applied", False)
            or (item.get("area_relative_error") or 0.0) <= 0.30
        ):
            continue
        point = nodes_gdf.geometry.iloc[node_idx]
        source_name = (
            str(nodes_gdf.iloc[node_idx].get(river_name_field, "")).strip().casefold()
            if river_name_field else ""
        )
        confirming_station = None
        for other_idx in station_by_river.get(item["river_id"], []):
            if other_idx == node_idx:
                continue
            other = assignments[other_idx]
            other_error = other.get("area_relative_error")
            if other_error is None or other_error > 0.15:
                continue
            if point.distance(nodes_gdf.geometry.iloc[other_idx]) > 2000.0:
                continue
            other_name = (
                str(nodes_gdf.iloc[other_idx].get(river_name_field, "")).strip().casefold()
                if river_name_field else ""
            )
            if source_name and other_name and source_name == other_name:
                continue
            confirming_station = other_idx
            break
        if confirming_station is None:
            continue
        item["target_branch_missing"] = True
        item["target_branch_missing_evidence_station"] = confirming_station
        item["attachment_confidence"] = "low"
        risks = [risk for risk in item.get("attachment_risk", "").split("|") if risk]
        if "target_branch_missing" not in risks:
            risks.append("target_branch_missing")
        item["attachment_risk"] = "|".join(risks)
        target_branch_missing_nodes.add(node_idx)

    diagnostics = Counter(
        stations_total=len(nodes_gdf), stations_attached=len(assignments),
        area_monotonicity_repair=monotonicity_repair_count,
        joint_confluence_assignment_group=len(joint_confluence_groups),
        joint_confluence_assignment_station=sum(
            len(group["node_indices"]) for group in joint_confluence_groups
        ),
        target_branch_missing=len(target_branch_missing_nodes),
    )
    edges, edge_diagnostics = _assemble_edges_with_area_monotonicity(
        nodes_gdf, river_gdf, assignments, station_by_river, by_id, next_by_id,
        enforce_area_monotonicity=enable_downstream_area_monotonicity,
    )
    skipped_station_rescue_count = 0
    if enable_skipped_station_rescue:
        skipped_station_rescue_count = _repair_skipped_intermediate_station_attachments(
            nodes_gdf, river_gdf, assignments, edges, by_id, next_by_id,
            downstream_is_last,
        )
    else:
        for item in assignments.values():
            item.setdefault("skipped_station_rescue_applied", False)
            item.setdefault("skipped_station_rescue_source_index", None)
            item.setdefault("skipped_station_rescue_target_index", None)
    if skipped_station_rescue_count:
        # 救回会改变站点占用河段，必须按新挂接完整重建边，而不能只修改
        # 某一条局部边，否则绘图河段和拓扑表会相互矛盾。
        station_by_river = _index_stations_by_river(
            river_gdf, assignments, downstream_is_last
        )
        edges, edge_diagnostics = _assemble_edges_with_area_monotonicity(
            nodes_gdf, river_gdf, assignments, station_by_river, by_id, next_by_id,
            enforce_area_monotonicity=enable_downstream_area_monotonicity,
        )
        for node_idx, item in assignments.items():
            if item.get("skipped_station_rescue_applied", False):
                item["target_branch_missing"] = False
                target_branch_missing_nodes.discard(node_idx)
    diagnostics.update(edge_diagnostics)
    diagnostics["skipped_station_route_rescue"] = skipped_station_rescue_count
    diagnostics["target_branch_missing"] = len(target_branch_missing_nodes)

    # 有些站点恰好落在汇流点，汇流前主干与汇流后主干同等级且近似等距，
    # 站点面积又与所有 RiverAtlas 候选严重不符。若选择汇流后河段进一步
    # 造成站点控制面积沿下游显著减小，则现有证据不足以确定汇流前/后侧，
    # 保留当前拓扑但必须降低挂接和受影响上游边的置信度。
    upstream_by_id = {}
    for upstream_id, downstream_id in next_by_id.items():
        if downstream_id is not None:
            upstream_by_id.setdefault(downstream_id, []).append(upstream_id)
    unresolved_confluence_edges = set()
    for source_idx, target_idx, _, status in edges:
        if target_idx is None or status != "downstream_segment":
            continue
        source_area = assignments[source_idx].get("station_area_km2")
        target_area = assignments[target_idx].get("station_area_km2")
        target_item = assignments[target_idx]
        if (
            source_area is None or target_area is None
            or target_area >= 0.85 * source_area
            or target_item.get("selection_reason") != "downstream_confluence_rule"
            or target_item.get("endpoint_distance_m", math.inf) > 250.0
            or (target_item.get("best_area_relative_error") or 0.0) <= 0.30
        ):
            continue
        selected_id = target_item["river_id"]
        selected_order = pd.to_numeric(
            river_gdf.iloc[target_item["line_idx"]].get("ORD_STRA"), errors="coerce"
        )
        selected_distance = target_item["snap_dist_m"]
        point = nodes_gdf.geometry.iloc[target_idx]
        alternative_upstream_sides = []
        for upstream_id in upstream_by_id.get(selected_id, []):
            line_idx = by_id.get(upstream_id)
            if line_idx is None:
                continue
            row = river_gdf.iloc[line_idx]
            order = pd.to_numeric(row.get("ORD_STRA"), errors="coerce")
            if (
                pd.isna(selected_order) or pd.isna(order)
                or float(order) != float(selected_order)
            ):
                continue
            distance = float(row.geometry.distance(point))
            projection = float(row.geometry.project(point))
            endpoint_distance = min(
                projection, max(0.0, row.geometry.length - projection)
            )
            if abs(distance - selected_distance) <= 100.0 and endpoint_distance <= 250.0:
                alternative_upstream_sides.append(upstream_id)
        if not alternative_upstream_sides:
            continue
        target_item["unresolved_confluence_side_conflict"] = True
        target_item["confluence_side_alternative_river_ids"] = (
            alternative_upstream_sides
        )
        target_item["attachment_confidence"] = "low"
        existing_risks = [
            risk for risk in target_item.get("attachment_risk", "").split("|") if risk
        ]
        if "unresolved_confluence_side_conflict" not in existing_risks:
            existing_risks.append("unresolved_confluence_side_conflict")
        target_item["attachment_risk"] = "|".join(existing_risks)
        unresolved_confluence_edges.add(source_idx)
        diagnostics["unresolved_confluence_side_conflict_edge"] += 1

    for item in assignments.values():
        item.setdefault("unresolved_confluence_side_conflict", False)
        item.setdefault("confluence_side_alternative_river_ids", [])

    # 一个站点的候选边是否唯一，必须同时考虑其他站点的候选挂接，不能把
    # 其他站点的暂定唯一挂接当作真值固定下来。
    potential_station_by_river = {}
    for possible_node_idx, possible_item in assignments.items():
        if possible_item["attachment_confidence"] == "high":
            possible_line_indices = [possible_item["line_idx"]]
        else:
            possible_line_indices = possible_item.get("ambiguous_line_indices", [])
        for possible_line_idx in possible_line_indices:
            possible_river_id = _normalise_key(
                river_gdf.iloc[possible_line_idx]["HYRIV_ID"]
            )
            potential_station_by_river.setdefault(possible_river_id, set()).add(possible_node_idx)

    def potential_measure(node_idx, river_id):
        line = river_gdf.geometry.iloc[by_id[river_id]]
        projection = float(line.project(nodes_gdf.geometry.iloc[node_idx]))
        return projection if downstream_is_last.get(river_id, True) else line.length - projection

    def downstream_target_from_candidate(node_idx, candidate_line_idx):
        """把同一站点暂挂到候选河段，返回首次命中的下游站点。"""
        row = river_gdf.iloc[candidate_line_idx]
        river_id = _normalise_key(row["HYRIV_ID"])
        point = nodes_gdf.geometry.iloc[node_idx]
        projection = float(row.geometry.project(point))
        measure = projection if downstream_is_last.get(river_id, True) else row.geometry.length - projection

        same_line = [
            i for i in potential_station_by_river.get(river_id, set())
            if i != node_idx and potential_measure(i, river_id) > measure + 1e-8
        ]
        if same_line:
            return min(same_line, key=lambda i: potential_measure(i, river_id))

        visited = {river_id}
        current = next_by_id.get(river_id)
        while current is not None:
            if current in visited or current not in by_id:
                return None
            visited.add(current)
            # 候选路径可能汇入当前站点最终选中的河段；此时必须排除
            # 当前站点自身，否则会把 source 误判成自己的下游目标。
            candidates = [
                i for i in potential_station_by_river.get(current, set()) if i != node_idx
            ]
            if candidates:
                return min(candidates, key=lambda i: potential_measure(i, current))
            current = next_by_id.get(current)
        return None

    # 挂接不确定不等于拓扑边不确定。若所有合理候选最终命中同一站点，
    # 则边由河网汇流关系唯一确定，仍属于高置信连接。
    edge_by_source = {edge[0]: edge for edge in edges}
    for node_idx, item in assignments.items():
        item["edge_confidence"] = item["attachment_confidence"]
        item["edge_rule"] = item["selection_reason"]
        item["edge_risk"] = item["attachment_risk"]
        if item["attachment_confidence"] != "low":
            continue
        # 目标支流缺失意味着真实候选根本不在当前河网中，不能用现有候选
        # 恰好指向同一下游站这一“共识”反向升级为高置信边。
        if item.get("target_branch_missing", False):
            continue
        candidate_targets = [
            downstream_target_from_candidate(node_idx, line_idx)
            for line_idx in item.get("ambiguous_line_indices", [])
        ]
        item["candidate_downstream_targets"] = candidate_targets
        unique_targets = set(candidate_targets)
        chosen_target = edge_by_source[node_idx][1]
        if len(unique_targets) == 1 and None not in unique_targets and chosen_target in unique_targets:
            item["edge_confidence"] = "high"
            item["edge_rule"] = "downstream_target_consensus"
            item["edge_risk"] = ""
    for source_idx in unresolved_confluence_edges:
        item = assignments[source_idx]
        item["edge_confidence"] = "low"
        item["edge_rule"] = "unresolved_confluence_side_conflict"
        existing_risks = [
            risk for risk in item.get("edge_risk", "").split("|") if risk
        ]
        if "unresolved_confluence_side_conflict" not in existing_risks:
            existing_risks.append("unresolved_confluence_side_conflict")
        item["edge_risk"] = "|".join(existing_risks)
    # 若一条边被“疑似属于缺失支流”的中间站截断，该上游边的目标也必须
    # 降低置信度，但不能反过来把上游站本身标为目标支流缺失。
    for source_idx, target_idx, _, _ in edges:
        if target_idx not in target_branch_missing_nodes:
            continue
        item = assignments[source_idx]
        item["edge_confidence"] = "low"
        item["edge_rule"] = "downstream_target_branch_missing"
        risks = [risk for risk in item.get("edge_risk", "").split("|") if risk]
        if "downstream_target_branch_missing" not in risks:
            risks.append("downstream_target_branch_missing")
        item["edge_risk"] = "|".join(risks)
    edge_by_source = {edge[0]: edge for edge in edges}
    for group in joint_confluence_groups:
        group_nodes = set(group["node_indices"])
        targets = {
            edge_by_source[node_idx][1] for node_idx in group_nodes
            if node_idx in edge_by_source
        }
        if len(targets) != 1 or None in targets or targets & group_nodes:
            continue
        for node_idx in group_nodes:
            item = assignments[node_idx]
            item["edge_confidence"] = "high"
            item["edge_rule"] = "joint_confluence_target_consensus"
            item["edge_risk"] = ""
    # 面积单调与中间站救回属于已形式化的确定规则；只有找不到任何面积
    # 相容下游站时才保留低置信边，避免把规则已解决的特殊情景误报审核。
    for source_idx, target_idx, _, _ in edges:
        item = assignments.get(source_idx)
        if item is None:
            continue
        if item.get("edge_area_violation_unresolved", False):
            item["edge_confidence"] = "low"
            item["edge_rule"] = "downstream_area_violation"
            risks = [risk for risk in item.get("edge_risk", "").split("|") if risk]
            if "downstream_area_violation" not in risks:
                risks.append("downstream_area_violation")
            item["edge_risk"] = "|".join(risks)
            continue
        if item.get("edge_area_monotonicity_applied", False):
            item["edge_confidence"] = "high"
            item["edge_rule"] = "downstream_area_monotonicity_rule"
            item["edge_risk"] = ""
        target_item = assignments.get(target_idx)
        if (
            target_item is not None
            and target_item.get("skipped_station_rescue_applied", False)
            and target_item.get("skipped_station_rescue_source_index") == source_idx
        ):
            item["edge_confidence"] = "high"
            item["edge_rule"] = "skipped_station_route_rescue_rule"
            item["edge_risk"] = ""
    return edges, assignments, dict(diagnostics), river_gdf


def build_topology_csv(rows, basin_id, mode="all"):
    topo = pd.DataFrame(rows, columns=[
        "ID", "NEXTDOWNID", "BASIN_ID", "dist_hdn", "status",
        "attachment_rule", "attachment_confidence", "attachment_risk",
        "edge_rule", "edge_confidence", "edge_risk",
    ])
    clean = lambda value: str(value).replace("chn_dllg_", "") if value else ""
    topo["ID"], topo["NEXTDOWNID"] = topo["ID"].map(clean), topo["NEXTDOWNID"].map(clean)
    topo["elev_diff"], topo["strm_slope"] = "", ""
    match = re.search(r"songliao_(\d+)", basin_id or "")
    basin_number = match.group(1) if match else basin_id
    out_dir = os.path.join(os.path.dirname(__file__), "result", "adjacency")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"adjacency_{basin_number}_{mode}.csv")
    topo[[
        "ID", "NEXTDOWNID", "dist_hdn", "elev_diff", "strm_slope", "status",
        "attachment_rule", "attachment_confidence", "attachment_risk",
        "edge_rule", "edge_confidence", "edge_risk",
    ]].to_csv(path, index=False)
    print(f"站点拓扑关系已保存到: {path}")
    return path


def save_topology_figure(ax, basin_id, mode="all", title="River-station topology",
                         publication=False):
    ax.set_title(title, fontsize=18 if publication else None, pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(x=0.015, y=0.025)
    plt.tight_layout(pad=0.35 if publication else 1.08)
    out_dir = os.path.join(os.path.dirname(__file__), "result", "topo")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"river_topology_{basin_id}_{mode}.png")
    plt.savefig(
        path,
        dpi=300 if publication else 180,
        bbox_inches="tight" if publication else None,
        pad_inches=0.04 if publication else 0.1,
        facecolor=ax.get_facecolor() if publication else "white",
    )
    if publication:
        pdf_path = os.path.splitext(path)[0] + ".pdf"
        plt.savefig(
            pdf_path, bbox_inches="tight", pad_inches=0.04,
            facecolor=ax.get_facecolor(),
        )
        print(f"论文矢量图已保存到: {pdf_path}")
    plt.close()
    print(f"河网与站点拓扑图已保存到: {path}")
    return path


def save_topology_review_figure(river_lines, nodes_gdf, node_ids, edges,
                                assignments, basin_id, mode="all"):
    """为低置信边和断链节点生成局部审核图集。"""
    review_edges = [
        edge for edge in edges
        if (
            assignments.get(edge[0], {}).get("edge_confidence", "low") == "low"
            or assignments.get(edge[0], {}).get(
                "mainstem_attachment_mismatch", False
            )
            or edge[1] is None
            or edge[3] not in {"same_segment", "downstream_segment"}
        )
    ]
    if not review_edges:
        return None

    column_count = 3
    row_count = (len(review_edges) + column_count - 1) // column_count
    fig, axes = plt.subplots(
        row_count, column_count, figsize=(18, 5 * row_count), squeeze=False
    )
    for ax, (source, target, distance_km, status) in zip(axes.flat, review_edges):
        source_point = nodes_gdf.geometry.iloc[source]
        points = [source_point]
        if target is not None:
            points.append(nodes_gdf.geometry.iloc[target])
        x_values = [point.x for point in points]
        y_values = [point.y for point in points]
        span = max(max(x_values) - min(x_values), max(y_values) - min(y_values))
        padding = max(15_000, span * 0.18)
        bounds = (
            min(x_values) - padding, min(y_values) - padding,
            max(x_values) + padding, max(y_values) + padding,
        )
        local_rivers = river_lines.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        local_nodes = nodes_gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        if not local_rivers.empty:
            local_rivers.plot(ax=ax, color="#69b3d0", linewidth=0.55, alpha=0.7)
        if not local_nodes.empty:
            local_nodes.plot(ax=ax, color="#777777", markersize=8, alpha=0.55)
        ax.scatter(source_point.x, source_point.y, color="crimson", s=34, zorder=5)
        ax.annotate(
            node_ids[source], (source_point.x, source_point.y), xytext=(4, 4),
            textcoords="offset points", color="crimson", fontsize=9,
            fontweight="bold", zorder=6,
        )
        target_id = "None"
        if target is not None:
            target_point = nodes_gdf.geometry.iloc[target]
            target_id = node_ids[target]
            ax.scatter(target_point.x, target_point.y, color="#175a9c", s=34, zorder=5)
            ax.annotate(
                target_id, (target_point.x, target_point.y), xytext=(4, 4),
                textcoords="offset points", color="#175a9c", fontsize=9,
                fontweight="bold", zorder=6,
            )
            ax.add_patch(FancyArrowPatch(
                (source_point.x, source_point.y), (target_point.x, target_point.y),
                arrowstyle="->", color="crimson", linestyle="--", linewidth=2,
                mutation_scale=13, zorder=4,
            ))
        source_item = assignments.get(source, {})
        risk = (
            source_item.get("edge_risk")
            or source_item.get("attachment_risk")
            or "unattached"
        )
        distance_text = "" if distance_km is None else f", {distance_km:.1f} km"
        ax.set_title(
            f"{node_ids[source]} → {target_id}{distance_text}\n{status} | {risk}",
            fontsize=10,
        )
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

    for ax in axes.flat[len(review_edges):]:
        ax.set_visible(False)
    fig.suptitle(f"Topology review atlas | {len(review_edges)} cases", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    out_dir = os.path.join(os.path.dirname(__file__), "result", "topo")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"river_topology_{basin_id}_{mode}_review.png")
    plt.savefig(path, dpi=180)
    plt.close(fig)
    print(f"拓扑人工审核图集已保存到: {path}")
    return path


def plot_river_and_topology(river_path, nodes_path, buffer_dist=50, clip=True,
                            river_buffer_dist=3000, mode="all", choices=None,
                            label_mode="auto", large_network_threshold=150,
                            work_crs=None, reference_downstream_field=None,
                            attachment_profile="auto", outlet_node_ids=None,
                            legend_location="upper left", legend_anchor=(0.015, 0.98),
                            legend_columns=1):
    river_gdf, nodes_gdf, total_buffer, basin_id = read_and_prepare_data(
        river_path, nodes_path, buffer_dist, mode=mode, choices=choices,
        work_crs=work_crs,
    )
    edges, assignments, diagnostics, river_lines = build_station_topology(
        river_gdf, nodes_gdf, max_snap_distance=river_buffer_dist,
        attachment_profile=attachment_profile,
    )
    id_series = _get_node_id_series(nodes_gdf)
    node_ids = id_series.tolist() if id_series is not None else [str(i) for i in range(len(nodes_gdf))]
    plot_rivers = clip_river(river_lines, total_buffer) if clip else river_lines
    is_large_network = len(nodes_gdf) > large_network_threshold
    if label_mode == "auto":
        label_mode = "review" if is_large_network else "all"
    if label_mode not in {"all", "review", "none"}:
        raise ValueError("label_mode 必须是 auto、all、review 或 none")

    if is_large_network:
        bounds_source = plot_rivers if not plot_rivers.empty else nodes_gdf
        min_x, min_y, max_x, max_y = bounds_source.total_bounds
        spatial_aspect = max((max_x - min_x) / max(max_y - min_y, 1), 0.25)
        figure_width = 15.5
        figure_height = min(11.0, max(6.2, figure_width / spatial_aspect + 0.8))
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        fig.patch.set_facecolor("#f3f5f6")
        ax.set_facecolor("#f3f5f6")
    else:
        fig, ax = plt.subplots(figsize=(14, 10))
    if not plot_rivers.empty:
        if is_large_network and "ORD_STRA" in plot_rivers.columns:
            river_order = pd.to_numeric(
                plot_rivers["ORD_STRA"], errors="coerce"
            ).fillna(1).clip(lower=1)
            river_width = 0.18 + 0.17 * river_order
        else:
            river_width = 0.35 if is_large_network else 1
        plot_rivers.plot(
            ax=ax, color="#4f9fbd",
            linewidth=river_width,
            alpha=0.82 if is_large_network else 1, zorder=1,
        )
    nodes_gdf.plot(
        ax=ax, color="#17252b" if is_large_network else "red",
        edgecolor="white" if is_large_network else None,
        linewidth=0.24 if is_large_network else 0,
        markersize=15 if is_large_network else 60,
        alpha=0.96 if is_large_network else 1, zorder=3,
    )
    mainstem_audit_indexes = {
        idx for idx, item in assignments.items()
        if item.get("mainstem_attachment_mismatch", False)
    }
    if mainstem_audit_indexes:
        audit_points = nodes_gdf.iloc[sorted(mainstem_audit_indexes)]
        ax.scatter(
            audit_points.geometry.x, audit_points.geometry.y,
            marker="s", s=72 if is_large_network else 105,
            facecolors="none", edgecolors="#d81b60", linewidths=1.8,
            zorder=8,
        )

    rows = []
    reference_mismatch_count = 0
    reference_mismatch_indexes = set()
    for source, target, distance_km, status in edges:
        target_id = node_ids[target] if target is not None else ""
        attachment = assignments.get(source, {})
        attachment_confidence = attachment.get("attachment_confidence", "low")
        edge_confidence = attachment.get("edge_confidence", attachment_confidence)
        rows.append((
            node_ids[source], target_id, basin_id, distance_km or "", status,
            attachment.get("selection_reason", "unattached"),
            attachment_confidence,
            attachment.get("attachment_risk", "unattached"),
            attachment.get("edge_rule", "unattached"),
            edge_confidence,
            attachment.get("edge_risk", "unattached"),
        ))
        if target is not None:
            p0, p1 = nodes_gdf.geometry.iloc[source], nodes_gdf.geometry.iloc[target]
            is_low = edge_confidence == "low"
            reference_target = None
            if (
                reference_downstream_field
                and reference_downstream_field in nodes_gdf.columns
            ):
                reference_target = _normalise_key(
                    nodes_gdf.iloc[source][reference_downstream_field]
                )
            predicted_target = _normalise_key(node_ids[target])
            is_reference_mismatch = (
                reference_target is not None and predicted_target != reference_target
            )
            if is_reference_mismatch:
                reference_mismatch_count += 1
                reference_mismatch_indexes.add(source)
            if is_reference_mismatch:
                edge_color, edge_linestyle = "#7b2cbf", "-."
            elif is_low:
                edge_color, edge_linestyle = "crimson", "--"
            else:
                edge_color, edge_linestyle = "orange", "-"
            ax.add_patch(FancyArrowPatch(
                (p0.x, p0.y), (p1.x, p1.y), arrowstyle="->",
                color=edge_color, linestyle=edge_linestyle,
                linewidth=(2.15 if is_reference_mismatch else (2.4 if is_low else 0.82))
                          if is_large_network else (2.4 if (is_low or is_reference_mismatch) else 2),
                alpha=(0.98 if is_reference_mismatch else (1.0 if is_low else 0.66))
                      if is_large_network else 1,
                mutation_scale=(10 if is_reference_mismatch else (11 if is_low else 7))
                               if is_large_network else 18,
                zorder=5 if (is_low or is_reference_mismatch) else 2,
            ))
            if is_low and not is_large_network:
                mid_x, mid_y = (p0.x + p1.x) / 2, (p0.y + p1.y) / 2
                ax.annotate(
                    "LOW", (mid_x, mid_y), xytext=(4, 4),
                    textcoords="offset points", color="crimson", fontsize=8,
                    fontweight="bold", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1),
                )

    terminal_indexes = {
        source for source, target, _, _ in edges if target is None
    }
    explicit_outlet_ids = {
        _normalise_key(value) for value in (outlet_node_ids or [])
    }
    explicit_outlet_ids.discard(None)
    explicit_outlets = {
        idx for idx, value in enumerate(node_ids)
        if _normalise_key(value) in explicit_outlet_ids
    }
    has_reference_outlet = bool(
        explicit_outlets
        or (
            reference_downstream_field
            and reference_downstream_field in nodes_gdf.columns
        )
    )
    if explicit_outlets:
        terminal_indexes = explicit_outlets
    elif has_reference_outlet:
        reference_outlets = {
            idx for idx, value in enumerate(nodes_gdf[reference_downstream_field])
            if _normalise_key(value) is None
        }
        if reference_outlets:
            terminal_indexes = reference_outlets

    review_indexes = {
        source for source, target, _, status in edges
        if (
            assignments.get(source, {}).get("edge_confidence", "low") == "low"
            or target is None
            or status not in {"same_segment", "downstream_segment"}
        )
    }
    # 大型流域概览仅标注需要读者关注的节点；参考不一致边即使为
    # 高置信，也要在总图中定位，以便与对应的局部审核图交叉阅读。
    review_indexes.update(reference_mismatch_indexes)
    review_indexes.update(mainstem_audit_indexes)
    # 终端节点使用独立符号表达，不再与审核节点文字重复标注。
    review_indexes.difference_update(terminal_indexes)
    if label_mode == "all":
        label_indexes = range(len(nodes_gdf))
    elif label_mode == "review":
        label_indexes = sorted(review_indexes)
    else:
        label_indexes = []
    for idx in label_indexes:
        point = nodes_gdf.geometry.iloc[idx]
        needs_review = idx in review_indexes
        ax.annotate(
            node_ids[idx], (point.x, point.y), xytext=(3, 3),
            textcoords="offset points", fontsize=8.5 if is_large_network else 9,
            color="crimson" if needs_review else "#202020",
            fontweight="bold" if needs_review else "normal", zorder=7,
            bbox=(dict(facecolor="#f3f5f6", edgecolor="none", alpha=0.92, pad=0.8)
                  if needs_review else None),
        )

    legend_handles = [
        Line2D([0], [0], color="orange", lw=2, linestyle="-",
               label="High-confidence edge"),
        Line2D([0], [0], color="crimson", lw=2.4, linestyle="--",
               label="Low-confidence edge"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="#202020" if is_large_network else "red",
               markeredgecolor="#202020" if is_large_network else "red",
               markersize=7, label="Station"),
    ]
    if reference_downstream_field:
        legend_handles.insert(2, Line2D(
            [0], [0], color="#7b2cbf", lw=2.4, linestyle="-.",
            label="Reference mismatch",
        ))
    if mainstem_audit_indexes:
        legend_handles.append(Line2D(
            [0], [0], marker="s", color="none", markerfacecolor="none",
            markeredgecolor="#d81b60", markeredgewidth=1.6, markersize=7,
            label="Mainstem attachment audit",
        ))
    if is_large_network:
        legend_handles.append(Line2D(
            [0], [0], marker="*" if has_reference_outlet else "D", color="none",
            markerfacecolor="#ffd43b", markeredgecolor="#7a5d00",
            markersize=9 if has_reference_outlet else 6.5,
            label="Outlet" if has_reference_outlet else "Network terminal",
        ))
        legend_handles.insert(0, Line2D(
            [0], [0], color="#4f9fbd", lw=2, label="River network",
        ))
        ax.legend(
            handles=legend_handles,
            loc=legend_location,
            bbox_to_anchor=legend_anchor,
            ncol=legend_columns,
            frameon=True,
            facecolor="#f3f5f6",
            edgecolor="none",
            framealpha=0.97,
            fontsize=22,
            markerscale=1.5,
            handlelength=2.7,
            labelspacing=0.55,
            columnspacing=1.3,
        )
        ax.set_axis_off()
        # 有参考拓扑时突出真实出口；无参考拓扑时只能表达网络终端，
        # 不能把 broken_next_down 节点直接解释成流域出口。
        for idx in sorted(terminal_indexes):
            point = nodes_gdf.geometry.iloc[idx]
            ax.scatter(
                point.x, point.y, marker="*" if has_reference_outlet else "D",
                s=82 if has_reference_outlet else 38, color="#ffd43b",
                edgecolor="#7a5d00", linewidth=0.8, zorder=8,
            )
            if has_reference_outlet:
                ax.annotate(
                    f"Outlet {node_ids[idx]}", (point.x, point.y), xytext=(5, 4),
                    textcoords="offset points", fontsize=12, color="#7a5d00",
                    fontweight="bold", zorder=9,
                    bbox=dict(
                        facecolor="#f3f5f6", edgecolor="none", alpha=0.9, pad=0.8
                    ),
                )
    else:
        ax.legend(handles=legend_handles, loc="best", framealpha=0.9)
    print("拓扑诊断:", diagnostics)
    build_topology_csv(rows, basin_id, mode=mode)
    low_edge_count = sum(
        assignments.get(source, {}).get("edge_confidence", "low") == "low"
        for source, target, _, _ in edges if target is not None
    )
    title = (
        f"River-station topology | stations={len(nodes_gdf)}, "
        f"valid edges={sum(target is not None for _, target, _, _ in edges)}, "
        f"review edges={low_edge_count}"
    )
    if reference_downstream_field:
        title += f", reference mismatches={reference_mismatch_count}"
    elif has_reference_outlet and terminal_indexes:
        title += f", outlets={len(terminal_indexes)}"
    elif terminal_indexes:
        title += f", network terminals={len(terminal_indexes)}"
    if mainstem_audit_indexes:
        title += f", mainstem audits={len(mainstem_audit_indexes)}"
    save_topology_figure(
        ax, basin_id, mode=mode, title=title, publication=is_large_network
    )
    if is_large_network:
        save_topology_review_figure(
            river_lines, nodes_gdf, node_ids, edges, assignments, basin_id, mode=mode
        )
    return edges, assignments, diagnostics


if __name__ == "__main__":
    plot_river_and_topology(
        "data_songliao/processed/northeast_rivers/northeast_rivers.shp",
        "data_songliao/processed/target_stations/songliao_21401550/target_stations.shp",
        buffer_dist=20000, clip=True, river_buffer_dist=2000, mode="all",
    )
