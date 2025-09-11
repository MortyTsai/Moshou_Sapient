# src/moshousapient/utils/geometry_utils.py

from shapely.geometry import Point


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
        return -1  # 右側
    elif val < -tolerance:
        return 1  # 左側
    else:
        return 0  # 在線上或非常接近線