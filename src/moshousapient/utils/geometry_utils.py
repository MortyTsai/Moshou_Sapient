# src/moshousapient/utils/geometry_utils.py
"""
提供與幾何計算相關的、無狀態的輔助函式。

這些函式主要基於 Shapely 庫，用於處理點、線、多邊形等幾何物件的計算，
例如計算錨點、判斷點與線的相對位置等。
"""

# 1. 標準庫導入
from typing import List, Union, Tuple

# 2. 第三方庫導入
from shapely.geometry import Point, box, Polygon

# 為了類型提示，預先定義 Bbox 類型
Bbox = Tuple[float, float, float, float]


def get_point_side_of_line(p: Point, line_p1: Point, line_p2: Point) -> int:
    """
    使用向量叉積計算點 p 在有向線段 (p1 -> p2) 的哪一側。
    (已針對螢幕座標系 Y 軸向下的情況進行校正)

    :param p: 要判斷的點。
    :param line_p1: 線段的起點。
    :param line_p2: 線段的終點。
    :return: 1 表示在左側, -1 表示在右側, 0 表示在線上。
    """
    tolerance = 1e-9
    val = (line_p2.x - line_p1.x) * (p.y - line_p1.y) - \
          (line_p2.y - line_p1.y) * (p.x - line_p1.x)

    if val > tolerance:
        return 1  # 右側
    elif val < -tolerance:
        return -1  # 左側
    else:
        return 0  # 在線上


def calculate_anchor_points(bbox: Bbox, strategies: Union[str, List[str]]) -> List[Union[Point, Polygon]]:
    """
    根據指定的策略，從一個辨識框計算出一個或多個錨點/區域。

    :param bbox: 辨識框，格式為 (x1, y1, x2, y2)。
    :param strategies: 一個策略名稱或策略名稱列表。
    :return: 一個包含 Shapely Point 或 Polygon 物件的列表。
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return []

    x1, y1, x2, y2 = bbox
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)

    if isinstance(strategies, str):
        strategies = [strategies]

    anchor_map = {
        'bottom_center': Point((min_x + max_x) / 2, max_y),
        'centroid': Point((min_x + max_x) / 2, (min_y + max_y) / 2),
        'bottom_left': Point(min_x, max_y),
        'bottom_right': Point(max_x, max_y),
        'top_left': Point(min_x, min_y),
        'top_right': Point(max_x, min_y),
        'top_center': Point((min_x + max_x) / 2, min_y),
        'left_center': Point(min_x, (min_y + max_y) / 2),
        'right_center': Point(max_x, (min_y + max_y) / 2),
        'full_bbox': box(min_x, min_y, max_x, max_y)
    }

    results = []
    for strategy in strategies:
        if strategy in anchor_map:
            results.append(anchor_map[strategy])
    return results