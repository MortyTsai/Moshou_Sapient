# src/moshousapient/utils/reid_utils.py
import pickle
import numpy as np
from typing import Optional
from numpy.typing import NDArray
from ..models import Person

def cosine_similarity(feature1: NDArray, feature2: NDArray) -> float:
    """計算兩個 NumPy 特徵向量之間的餘弦相似度。"""
    feature1 = np.asarray(feature1)
    feature2 = np.asarray(feature2)

    dot_product = np.dot(feature1, feature2)
    norm_feature1 = np.linalg.norm(feature1)
    norm_feature2 = np.linalg.norm(feature2)

    if norm_feature1 == 0 or norm_feature2 == 0:
        return 0.0

    similarity = (dot_product / (norm_feature1 * norm_feature2)).item()  # type: ignore
    return similarity


def find_best_match_in_gallery(new_feature: NDArray,
                               gallery: list[Person],
                               match_threshold: float
                               ) -> Optional[Person]:
    """
    在給定的畫廊（候選人列表）中，為新特徵尋找超過指定閾值的最佳匹配。
    此函式已解耦，不依賴任何全域 Config 模組。

    :param new_feature: 新的 Re-ID 特徵向量。
    :param gallery: 包含已知人物及其特徵的列表。
    :param match_threshold: 判定為同一個人的相似度閾值。
    :return: 匹配成功則返回 Person 物件，否則返回 None。
    """
    best_match_person = None
    highest_overall_similarity = -1.0

    for person in gallery:
        if not person.features:
            continue

        max_similarity_for_this_person = -1.0
        for existing_feature_obj in person.features:
            existing_feature = pickle.loads(existing_feature_obj.feature)
            similarity = cosine_similarity(new_feature, existing_feature)
            if similarity > max_similarity_for_this_person:
                max_similarity_for_this_person = similarity

        if max_similarity_for_this_person > highest_overall_similarity:
            highest_overall_similarity = max_similarity_for_this_person
            best_match_person = person

    if highest_overall_similarity >= match_threshold and best_match_person:
        return best_match_person

    return None