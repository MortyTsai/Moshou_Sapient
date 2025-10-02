# src/moshousapient/services/database_service.py
import logging
import os
import pickle
from typing import List
import numpy as np

from ..database import SessionLocal
from ..models import Event, Person, PersonFeature
from ..utils.reid_utils import find_best_match_in_gallery, cosine_similarity
from ..config import Config


def process_reid_and_identify_person(reid_features_list: List[np.ndarray]) -> int | None:
    """
    處理 Re-ID 特徵聚類、資料庫比對，並返回主要人物的 ID。
    如果建立了新人物，也會返回其 ID。
    """
    if not reid_features_list:
        return None

    # 去重，確保每個獨立的特徵只處理一次
    unique_features = list({feat.tobytes(): feat for feat in reid_features_list}.values())
    logging.info(f"[特徵處理] 原始特徵數: {len(reid_features_list)}, 去重後: {len(unique_features)}")

    db = SessionLocal()
    try:
        # 1. 事件內部聚類：將本次事件中相似的特徵歸為一類
        event_clusters = []
        for feature in unique_features:
            best_match_cluster, highest_sim = None, -1.0
            for cluster in event_clusters:
                # 使用聚類的第一個特徵作為代表
                rep_feature = pickle.loads(cluster.features[0].feature)
                sim = cosine_similarity(feature, rep_feature)
                if sim > highest_sim:
                    highest_sim, best_match_cluster = sim, cluster

            # 使用與 Re-ID 匹配閾值略低的內部聚類閾值
            if highest_sim >= (Config.PERSON_MATCH_THRESHOLD - 0.06) and best_match_cluster:
                best_match_cluster.features.append(PersonFeature(feature=pickle.dumps(feature)))
            else:
                new_cluster = Person()
                new_cluster.features.append(PersonFeature(feature=pickle.dumps(feature)))
                event_clusters.append(new_cluster)

        logging.info(f"[特徵處理] 事件內聚類完成，發現 {len(event_clusters)} 個潛在獨立人物。")
        if not event_clusters:
            return None

        # 2. 與資料庫畫廊比對
        static_persons_gallery = db.query(Person).all()
        initial_db_persons = set(static_persons_gallery)
        final_person_map = {}

        for cluster in event_clusters:
            rep_feature = pickle.loads(cluster.features[0].feature)
            db_match = find_best_match_in_gallery(
                new_feature=rep_feature,
                gallery=static_persons_gallery,
                match_threshold=Config.PERSON_MATCH_THRESHOLD
            )

            if db_match:
                final_person_map[cluster] = db_match
            else:
                # 如果在資料庫中找不到匹配，則視為新人物
                db.add(cluster)
                final_person_map[cluster] = cluster
                static_persons_gallery.append(cluster)  # 加入畫廊以供後續聚類比對

        # 3. 合併特徵並更新資料庫
        unique_persons_in_event = set(final_person_map.values())
        for cluster, final_person in final_person_map.items():
            if final_person != cluster:  # 如果聚類被合併到一個已存在的人物
                for feature_obj in cluster.features:
                    final_person.features.append(PersonFeature(feature=feature_obj.feature))

            if final_person in initial_db_persons:
                final_person.sighting_count += 1

        # 假設事件中最主要的 Person 是第一個聚類的最終歸屬
        main_person_obj = final_person_map[event_clusters[0]]
        db.commit()

        logging.info(
            f"[資料庫] Re-ID 處理完成。涉及 {len(unique_persons_in_event)} 人。主要人物 ID: {main_person_obj.id}")
        return main_person_obj.id

    except Exception as e:
        logging.error(f"[特徵處理] 處理 Re-ID 時發生錯誤，交易已回滾: {e}", exc_info=True)
        db.rollback()
        return None
    finally:
        db.close()

def save_event(video_path: str, event_type: str, person_id: int | None) -> None:
    """將單個事件記錄儲存到資料庫。"""
    db = SessionLocal()
    try:
        new_event = Event(
            video_path=video_path,
            event_type=event_type,
            status="unreviewed",
            person_id=person_id
        )
        db.add(new_event)
        db.commit()
        logging.info(f"[資料庫] 已成功將事件紀錄 (影片: {os.path.basename(video_path)}) 寫入資料庫。")
    except Exception as e:
        logging.error(f"[資料庫] 寫入事件紀錄時發生錯誤: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()