# src/moshousapient/processors/nfc_processor.py

import torch
import logging
from typing import Dict
import numpy as np


# ==============================================================================
#  原始 Pose2ID NFC 邏輯
#  來源: Pose2ID 官方儲存庫, NFC.py
#  我們將其封裝以便在 MoshouSapient 專案中安全地使用。
# ==============================================================================

def _pairwise_distance(query_features: torch.Tensor, gallery_features: torch.Tensor) -> torch.Tensor:
    """計算兩組特徵向量之間的歐氏距離平方。"""
    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    # 計算 (x-y)^2 = x^2 + y^2 - 2xy
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)

    return dist_mat


class NFCProcessor:
    """
    鄰近特徵中心化 (Neighbor Feature Centralization) 處理器。
    這是一個即時特徵後處理模組，旨在增強 Re-ID 特徵的魯棒性。
    """

    def __init__(self, k1: int = 2, k2: int = 2, device: str = 'cuda'):
        """
        初始化 NFC 處理器。
        :param k1: 在尋找鄰居時，考慮的最近鄰數量。
        :param k2: 在驗證互為鄰居關係時，對方列表中的考慮範圍。
        :param device: 計算所使用的設備 ('cuda' or 'cpu')。
        """
        self.k1 = k1
        self.k2 = k2
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logging.info(f"[NFCProcessor] 初始化完成. k1={self.k1}, k2={self.k2}, device='{self.device}'")

    def process_features(self, reid_features_map: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        對一組 Re-ID 特徵進行 NFC 處理。
        :param reid_features_map: 一個字典，key 為 track_id，value 為對應的 numpy 特徵向量。
        :return: 處理後的新特徵字典。
        """
        # logging.info(f"[NFCProcessor] 正在處理 {len(reid_features_map)} 個軌跡的 Re-ID 特徵...")

        if not reid_features_map or len(reid_features_map) <= self.k1:
            # 如果特徵數量不足，無法進行有意義的鄰居查找，直接返回原特徵
            # 在高頻調用下，此警告也可能產生過多日誌，故暫時註解
            # if reid_features_map:
            #     logging.debug(f"[NFCProcessor] 特徵數量 ({len(reid_features_map)}) 過少，跳過 NFC 處理。")
            return reid_features_map

        # 1. 將特徵字典轉換為 Tensor，並記錄 track_id 的順序
        track_ids = list(reid_features_map.keys())
        original_features = list(reid_features_map.values())

        try:
            feat_tensor = torch.from_numpy(np.array(original_features)).to(self.device)
        except (TypeError, ValueError) as e:
            logging.error(f"[NFCProcessor] 特徵轉換為 Tensor 失敗: {e}")
            return reid_features_map

        # 2. 計算成對距離
        dist_mat = _pairwise_distance(feat_tensor, feat_tensor)

        # 3. 尋找 k1 個最近鄰
        # 將對角線（自身與自身的距離）設為極大值，以排除自身
        eye = torch.eye(dist_mat.size(0), device=self.device)
        dist_mat[eye == 1] = float('inf')

        _, top_k_indices = dist_mat.topk(self.k1, largest=False, dim=1)

        # 4. 尋找互為最近鄰的樣本
        mutual_topk_list = []
        for i in range(top_k_indices.size(0)):
            mutual_list = []
            for j_idx in top_k_indices[i]:
                # 檢查 i 是否在 j 的 top_k2 列表中
                if i in top_k_indices[j_idx][:self.k2]:
                    mutual_list.append(j_idx.item())
            mutual_topk_list.append(mutual_list)

        # 5. 特徵中心化：將互為鄰居的特徵進行聚合
        enhanced_features = feat_tensor.clone()
        for i in range(len(mutual_topk_list)):
            if mutual_topk_list[i]:
                neighbor_indices = torch.tensor(mutual_topk_list[i], device=self.device)
                # 將找到的互為鄰居的特徵加到當前特徵上
                enhanced_features[i] += torch.sum(feat_tensor[neighbor_indices], dim=0)

        # 注意：原始 Pose2ID 實現沒有重新進行 L2 標準化，這可能會改變特徵向量的模長。
        # 在 Re-ID 中，特徵通常是單位向量。這裡我們選擇遵循原始實現，
        # 但在未來優化時，可以考慮是否需要重新標準化。
        # enhanced_features = torch.nn.functional.normalize(enhanced_features, p=2, dim=1)

        # 6. 將結果轉換回字典格式
        processed_features_np = enhanced_features.cpu().numpy()
        new_reid_features_map = {track_ids[i]: processed_features_np[i] for i in range(len(track_ids))}

        return new_reid_features_map