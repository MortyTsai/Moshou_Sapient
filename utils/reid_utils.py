# utils/reid_utils.py
import numpy as np
from numpy.typing import NDArray
import pickle
from sqlalchemy.orm import Session, selectinload
from models import Person, PersonFeature

PERSON_MATCH_THRESHOLD = 0.65


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


def find_or_create_person(db: Session, new_feature: NDArray) -> tuple[int, bool]:
    """
    在人物畫廊中尋找匹配的個體。
    比對邏輯升級為：將新特徵與每個 Person 的特徵集 (gallery) 進行一對多比對。
    如果找到匹配，則將新特徵加入該 Person 的畫廊中。
    如果找不到，則建立一個新的 Person。
    """
    all_persons = db.query(Person).options(selectinload(Person.features)).all()

    best_match_person_id = -1
    highest_similarity = -1.0

    for person in all_persons:
        if not person.features:
            continue

        person_id = person.id
        for existing_feature_obj in person.features:
            existing_feature = pickle.loads(existing_feature_obj.feature)
            similarity = cosine_similarity(new_feature, existing_feature)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_person_id = person_id

    if highest_similarity >= PERSON_MATCH_THRESHOLD:
        matched_person = db.query(Person).filter(Person.id == best_match_person_id).one()

        new_feature_to_add = PersonFeature(
            feature=pickle.dumps(new_feature),
            person_id=matched_person.id
        )
        db.add(new_feature_to_add)

        matched_person.sighting_count += 1
        db.flush()
        db.commit()

        return matched_person.id, False  # type: ignore

    else:
        serialized_feature = pickle.dumps(new_feature)

        new_person = Person()
        new_person.features.append(PersonFeature(feature=serialized_feature))

        db.add(new_person)
        db.commit()
        db.refresh(new_person)

        return new_person.id, True  # type: ignore