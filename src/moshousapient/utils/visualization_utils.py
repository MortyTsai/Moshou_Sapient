# src/moshousapient/utils/visualization_utils.py
import cv2
import numpy as np
from typing import Dict, Any, List, Set, Union
from shapely.geometry import Polygon


def draw_static_overlays(frame: np.ndarray,
                         scale_x: float,
                         scale_y: float,
                         roi_enabled: bool,
                         roi_polygon: Union[Polygon, None],
                         tripwires_enabled: bool,
                         tripwire_line_objects: List[Dict[str, Any]],
                         tripwire_line_thickness: int,
                         tripwire_tip_length: float
                         ) -> np.ndarray:
    """
    在影像幀上繪製不會隨追蹤目標變化的靜態視覺元素（如 ROI 區域和警戒線）。
    此函式已解耦，不依賴任何全域 Config/Settings 模組。
    """
    overlay = frame.copy()

    # 繪製 ROI 區域
    if roi_enabled and roi_polygon:
        roi_points = np.array(roi_polygon.exterior.coords, dtype=np.int32)
        roi_points_scaled = (roi_points * np.array([scale_x, scale_y])).astype(np.int32)
        roi_overlay = overlay.copy()
        cv2.fillPoly(roi_overlay, [roi_points_scaled], color=(255, 255, 0))
        overlay = cv2.addWeighted(roi_overlay, 0.3, overlay, 0.7, 0)
        cv2.polylines(overlay, [roi_points_scaled], isClosed=True, color=(255, 255, 0), thickness=4)

    # 繪製 Tripwire 警戒線
    if tripwires_enabled and tripwire_line_objects:
        for tripwire_obj in tripwire_line_objects:
            line, direction = tripwire_obj['line'], tripwire_obj["direction"]
            p1, p2 = np.array(line.coords[0]), np.array(line.coords[1])
            p1_s = tuple((p1 * np.array([scale_x, scale_y])).astype(np.int32))
            p2_s = tuple((p2 * np.array([scale_x, scale_y])).astype(np.int32))
            if direction == "cross_to_right":
                cv2.arrowedLine(overlay, p1_s, p2_s, (0, 0, 255), tripwire_line_thickness, tipLength=tripwire_tip_length)
            elif direction == "cross_to_left":
                cv2.arrowedLine(overlay, p2_s, p1_s, (0, 0, 255), tripwire_line_thickness, tipLength=tripwire_tip_length)
            else:
                cv2.line(overlay, p1_s, p2_s, (0, 0, 255), tripwire_line_thickness)

    return overlay


def draw_dynamic_overlays(frame: np.ndarray,
                          tracks: List[Dict[str, Any]],
                          active_alert_ids: Set[int],
                          active_roi_ids: Set[int],
                          scale_x: float,
                          scale_y: float
                          ) -> np.ndarray:
    """
    在影像幀上繪製隨追蹤目標動態變化的視覺元素（如追蹤框、ID、錨點）。
    此函式已解耦，不依賴任何全域 Config/Settings 模組。
    """
    overlay = frame.copy()
    active_tracks = [t for t in tracks if t['track_id'] in active_alert_ids or t['track_id'] in active_roi_ids]
    inactive_tracks = [t for t in tracks if t not in active_tracks]

    # 繪製半透明的非活躍目標
    temp_overlay = overlay.copy()
    for track in inactive_tracks:
        box = track['box_xyxy']
        x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y])
        cv2.rectangle(temp_overlay, (x1, y1), (x2, y2), (128, 128, 128), 2)
    overlay = cv2.addWeighted(temp_overlay, 0.5, overlay, 0.5, 0)

    # 繪製高亮的活躍目標
    for track in active_tracks:
        track_id = track['track_id']
        box = track['box_xyxy']
        x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y])
        box_color = (0, 0, 255) if track_id in active_alert_ids else (0, 255, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
        anchor_coords = track.get('anchors', [])
        for anchor in anchor_coords:
            center_point = (int(anchor[0] * scale_x), int(anchor[1] * scale_y))
            cv2.circle(overlay, center_point, 5, box_color, -1)
        cv2.putText(overlay, f'ID:{track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    return overlay


def draw_info_panel(frame: np.ndarray, event_type: str, elapsed_time: float) -> np.ndarray:
    """
    在影像幀的頂部繪製一個包含事件類型和持續時間的資訊面板。
    此函式已解耦，不依賴任何全域 Config/Settings 模組。
    """
    height, width, _ = frame.shape
    info_panel = np.zeros((80, width, 3), dtype=np.uint8)
    frame[0:80, 0:width] = cv2.addWeighted(frame[0:80, 0:width], 0.5, info_panel, 0.5, 0)

    event_text = f"EVENT: {event_type.upper()}"
    cv2.putText(frame, event_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if elapsed_time >= 0:
        duration_text = f"DURATION: {elapsed_time:.1f}s"
        cv2.putText(frame, duration_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame