# src/moshousapient/utils/behavior_analysis_utils.py
"""
提供與高階行為分析相關的輔助函式。

這些函式是無狀態的，接收追蹤數據和設定作為輸入，並返回分析結果。
"""

# 1. 標準庫導入
from typing import Dict, Any, List, Tuple, Union, cast

# 2. 第三方庫導入
import numpy as np
from shapely.geometry import Point, Polygon, LineString

# 3. 本專案相對導入
from .geometry_utils import calculate_anchor_points, get_point_side_of_line


def analyze_roi_status(
    tracks: np.ndarray,
    roi_enabled: bool,
    roi_polygon: Union[Polygon, None],
    roi_settings: Dict[str, Any],
    global_anchor_points: Union[str, List[str]],
) -> Dict[int, bool]:
    """
    根據傳入的 ROI 設定，計算每個追蹤目標是否在感興趣區域內。

    :param tracks: Ultralytics 追蹤器輸出的 Numpy 陣列。
    :param roi_enabled: ROI 功能是否啟用的布林值。
    :param roi_polygon: 代表 ROI 區域的 Shapely Polygon 物件。
    :param roi_settings: 包含 ROI 特定設定的字典 (如 'anchor_points')。
    :param global_anchor_points: 全域預設的錨點策略。
    :return: 一個字典，key 為 track_id，value 為布林值 (True 表示在 ROI 內)。
    """
    if not roi_enabled or not roi_polygon:
        return {}

    anchor_strategy = roi_settings.get("anchor_points", global_anchor_points)
    track_roi_status = {}

    for track in tracks:
        track_id = int(track[4])
        bbox = track[:4]
        is_in_roi = False

        bbox_tuple = cast(Tuple[float, float, float, float], tuple(bbox))
        anchors = calculate_anchor_points(bbox_tuple, anchor_strategy)

        for anchor in anchors:
            if isinstance(anchor, Point):
                if roi_polygon.contains(anchor):
                    is_in_roi = True
                    break
            elif isinstance(anchor, Polygon):
                if roi_polygon.intersects(anchor):
                    is_in_roi = True
                    break
        track_roi_status[track_id] = is_in_roi
    return track_roi_status


def analyze_tripwire_crossings(
    tracks: np.ndarray,
    track_last_positions: Dict[Tuple[int, int], Point],
    tripwires_enabled: bool,
    tripwire_line_objects: List[Dict[str, Any]],
    global_anchor_points: Union[str, List[str]],
) -> Tuple[Dict[int, bool], Dict[Tuple[int, int], Point]]:
    """
    根據傳入的 Tripwire 設定，分析每個追蹤目標是否穿越了警戒線。

    :param tracks: Ultralytics 追蹤器輸出的 Numpy 陣列。
    :param track_last_positions: 一個包含追蹤目標上一次位置的字典。
    :param tripwires_enabled: Tripwire 功能是否啟用的布林值。
    :param tripwire_line_objects: 包含已解析的 Shapely 物件和設定的警戒線列表。
    :param global_anchor_points: 全域預設的錨點策略。
    :return: 一個元組，包含:
             - crossed_ids: 一個字典，key 為 track_id，value 為 True 表示發生穿越。
             - updated_positions: 更新後的位置字典。
    """
    if not tripwires_enabled or not tripwire_line_objects:
        return {}, track_last_positions

    crossed_ids = {}
    updated_positions = track_last_positions.copy()

    for track in tracks:
        track_id = int(track[4])
        bbox = track[:4]

        for tripwire_obj in tripwire_line_objects:
            tripwire_line = tripwire_obj["line"]
            alert_direction = tripwire_obj["direction"]
            tripwire_config = tripwire_obj["config"]

            anchor_strategy = tripwire_config.get("anchor_points", global_anchor_points)
            bbox_tuple = cast(Tuple[float, float, float, float], tuple(bbox))
            current_anchors = calculate_anchor_points(bbox_tuple, anchor_strategy)

            for i, current_anchor in enumerate(current_anchors):
                if not isinstance(current_anchor, Point):
                    continue

                anchor_key = (track_id, i)
                last_position = updated_positions.get(anchor_key)

                if last_position and last_position != current_anchor:
                    movement_line = LineString([last_position, current_anchor])
                    if movement_line.intersects(tripwire_line):
                        p1, p2 = tripwire_line.coords
                        side_before = get_point_side_of_line(last_position, Point(p1), Point(p2))
                        side_after = get_point_side_of_line(current_anchor, Point(p1), Point(p2))

                        if side_before != 0 and side_after != 0 and side_before != side_after:
                            crossed_to_right = side_before == 1 and side_after == -1
                            crossed_to_left = side_before == -1 and side_after == 1
                            should_alert = (
                                alert_direction == "both"
                                or (alert_direction == "cross_to_right" and crossed_to_right)
                                or (alert_direction == "cross_to_left" and crossed_to_left)
                            )
                            if should_alert:
                                crossed_ids[track_id] = True
                                break

                updated_positions[anchor_key] = current_anchor

            if track_id in crossed_ids:
                break

    return crossed_ids, updated_positions
