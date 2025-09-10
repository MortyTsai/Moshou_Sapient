# utils/reid_utils.py
import numpy as np
from numpy.typing import NDArray
import pickle
from models import Person
from typing import Optional
from config import Config

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


def find_best_match_in_gallery(new_feature: NDArray, gallery: list[Person]) -> Optional[Person]:
    """
    在給定的畫廊(候選人列表)中，為新特徵尋找最佳匹配。
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

    if highest_overall_similarity >= Config.PERSON_MATCH_THRESHOLD and best_match_person:
        return best_match_person

    return None