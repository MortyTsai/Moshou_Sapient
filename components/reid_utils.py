# components/reid_utils.py
import numpy as np
from numpy.typing import NDArray


def cosine_similarity(feature1: NDArray, feature2: NDArray) -> float:
    """
    計算兩個 NumPy 特徵向量之間的餘弦相似度。

    Args:
        feature1: 第一個特徵向量 (1D NumPy array)。
        feature2: 第二個特徵向量 (1D NumPy array)。

    Returns:
        兩個向量之間的餘弦相似度 (float, 範圍在 -1 到 1 之間)。
    """
    # 確保輸入是 NumPy 陣列
    feature1 = np.asarray(feature1)
    feature2 = np.asarray(feature2)

    # 計算點積
    dot_product = np.dot(feature1, feature2)

    # 計算範數 (模)
    norm_feature1 = np.linalg.norm(feature1)
    norm_feature2 = np.linalg.norm(feature2)

    # 避免除以零的錯誤
    if norm_feature1 == 0 or norm_feature2 == 0:
        return 0.0

    # 計算並返回餘弦相似度
    similarity = dot_product / (norm_feature1 * norm_feature2)

    return float(similarity)