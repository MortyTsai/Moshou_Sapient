# utils/reid_utils.py
import numpy as np
from numpy.typing import NDArray
import pickle
from sqlalchemy.orm import Session
from models import Person

PERSON_MATCH_THRESHOLD = 0.65
FEATURE_UPDATE_ALPHA = 0.2

def cosine_similarity(feature1: NDArray, feature2: NDArray) -> float:
    """
    計算兩個 NumPy 特徵向量之間的餘弦相似度。

    Args:
        feature1: 第一個特徵向量 (1D NumPy array)。
        feature2: 第二個特徵向量 (1D NumPy array)。

    Returns:
        兩個向量之間的餘弦相似度 (float, 範圍在 -1 到 1 之間)。
    """
    feature1 = np.asarray(feature1)
    feature2 = np.asarray(feature2)
    dot_product = np.dot(feature1, feature2)
    norm_feature1 = np.linalg.norm(feature1)
    norm_feature2 = np.linalg.norm(feature2)

    if norm_feature1 == 0 or norm_feature2 == 0:
        return 0.0

    similarity = dot_product / (norm_feature1 * norm_feature2)

    return float(similarity)

def find_or_create_person(db: Session, new_feature: NDArray) -> tuple[int, bool]:
    """
    在人物畫廊中尋找匹配的個體，如果找不到則建立一個新的。

    Args:
        db (Session): SQLAlchemy 的資料庫會話。
        new_feature (NDArray): 新偵測到的人物的 Re-ID 特徵向量。

    Returns:
        tuple[int, bool]: 一個包含 (人物的永久 ID, 是否為新建立的人物) 的元組。
    """
    all_persons = db.query(Person).all()
    best_match_id = -1
    highest_similarity = -1.0

    for person in all_persons:
        existing_feature = pickle.loads(person.representative_feature)
        similarity = cosine_similarity(new_feature, existing_feature)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_id = person.id

    if highest_similarity >= PERSON_MATCH_THRESHOLD:
        matched_person = db.query(Person).filter_by(id=best_match_id).one()
        existing_feature = pickle.loads(matched_person.representative_feature)
        updated_feature = (1 - FEATURE_UPDATE_ALPHA) * existing_feature + FEATURE_UPDATE_ALPHA * new_feature
        updated_feature /= np.linalg.norm(updated_feature)
        matched_person.representative_feature = pickle.dumps(updated_feature)
        matched_person.sighting_count += 1
        db.commit()
        return matched_person.id, False
    else:
        serialized_feature = pickle.dumps(new_feature)
        new_person = Person(
            representative_feature=serialized_feature,
            sighting_count=1
        )
        db.add(new_person)
        db.commit()
        db.refresh(new_person)
        return new_person.id, True