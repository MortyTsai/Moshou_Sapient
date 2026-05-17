# MoshouSapient: 個人化 AI 智慧保全系統

![Python Version](https://img.shields.io/badge/python-3.11-blue) ![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue) ![Status: Phase Archived](https://img.shields.io/badge/status-phase_archived-purple)

MoshouSapient 是一個探索如何將個人電腦轉化為 AI 保全節點的實驗性專案。本專案旨在驗證消費級硬體透過 NVIDIA GPU 加速，執行即時物件偵測、行為分析與場景異常告警的可行性。


https://github.com/user-attachments/assets/61f11b20-0a7d-4697-843a-d4f4d00c1345


> **專案定位與階段性總結**
> 本專案為我申請大學的個人作品集。目前地端單機版（Standalone）的核心功能與真人概念驗證（PoC）均已在 Windows 環境下順利跑通。這是我探索「AI 協同開發」與「系統架構設計」的重要里程碑。基於單機測試中所觀察到的硬體瓶頸，本專案已完成階段性任務並予以封存，為未來進入大學後結合大數據與分散式架構的升級做準備（詳見文末「未來展望」）。

---

## 核心架構與功能驗證
*   **硬體加速推論**：整合 **YOLO** 與 **NVIDIA TensorRT**，驗證消費級顯卡在影像辨識上的加速效果。
*   **非同步任務架構**：採用「生產者-消費者」模型，並以 **SQLite (WAL 模式)** 實作基礎的持久化任務佇列，探索系統解耦與穩定性。
*   **事件驅動排程**：實作基礎的搶佔式排程器，嘗試在有限資源下，優先保障即時監控任務的硬體資源。
*   **複合式警報機制**：結合 AI 行為分析（ROI、警戒線）與傳統影像演算法（防篡改、畫面凍結偵測），驗證多維度異常偵測。

---

## 技術細節與系統架構

<details>
<summary><strong>► 點此展開：詳細技術棧與檔案結構</strong></summary>

### 技術棧 (Tech Stack)
- **核心框架**: Python 3.11
- **AI / CV**: PyTorch, TensorRT, Ultralytics YOLO, BoT-SORT, Shapely (幾何分析)
- **資料庫**: SQLite, SQLAlchemy (ORM)
- **影像處理**: FFmpeg, OpenCV-Python
- **設定管理**: Pydantic-Settings, PyYAML

### 系統檔案結構
```text
MoshouSapient/
├── configs/                # 行為分析規則與追蹤器設定
├── data/                   # 執行時生成的資料 (錄影、日誌、資料庫)
├── models/                 # AI 模型資產 (.pt, .engine)
├── scripts/                # 輔助腳本 (模型轉換、快照工具)
└── src/moshousapient/      # 專案原始碼
    ├── core/               # 應用程式協調器與排程器
    ├── processors/         # 影像分析管線 (AI & Non-AI)
    ├── services/           # 任務佇列、資料庫與通知服務
    ├── workers/            # 背景影片編碼程序
    └── web/                # Flask 儀表板
```
</details>

---

## 安裝與執行指南 (重現步驟)

<details>
<summary><strong>► 點此展開：環境準備與安裝步驟</strong></summary>

### 硬體需求
- **OS**: Windows 10 / 11
- **GPU**: 支援 NVENC 的 NVIDIA 顯示卡 (建議 RTX 系列)

### 安裝步驟
1. **安裝 NVIDIA 工具鏈**: 確保已安裝 CUDA Toolkit, cuDNN 與 TensorRT。
2. **設定虛擬環境**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **安裝依賴**:
   ```bash
   pip install -e .
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu12x
   pip install -r requirements.txt
   ```
4. **準備模型**:
   將 `yolo11s.pt` 放入 `models/` 並執行轉換：
   ```bash
   python scripts/export_tensorrt.py
   ```
</details>

<details>
<summary><strong>► 點此展開：專案設定與執行說明</strong></summary>

### 1. 設定環境變數
複製 `.env.example` 為 `.env`，並填入您的 `RTSP_URL` 與 Discord Token (選填)。

### 2. 自定義行為規則
編輯 `configs/behavior_analysis.yaml` 來劃定您的 ROI 區域或虛擬警戒線座標。

### 3. 啟動系統
```bash
python -m moshousapient
```
啟動後可訪問 `http://127.0.0.1:5000` 查看 Web 儀表板。
</details>

---

## 開發回顧與挑戰 (Retrospective)

### 1. 環境選擇與技術取捨 (Design Trade-offs)
本專案定位為「大眾個人化智慧保全」，目標受眾為一般家庭或小型辦公室用戶。我深知在軟體工程中，Linux 伺服器在多程序處理（Multi-processing）上具備原生優勢，但考量到一般用戶的部署門檻與日常使用習慣，我刻意選擇在 Windows 10/11 環境下進行開發與驗證。

然而，Python 在 Windows 下的並發機制（Spawn）啟動成本較高，且容易遇到跨進程序列化（Pickle）的限制。在開發初期，我因為不熟悉這些作業系統底層的差異而踩了不少坑。但在 AI 的協作與引導下，我逐步調整系統架構，成功在 Windows 環境下跑通了非同步的任務佇列。這次的經驗讓我深刻體會到，架構設計不僅是追求極致的效能，更需要依據「用戶實際情境」與「部署成本」進行務實的權衡與取捨。

### 2. AI 驅動的開發模式 (AI-Centric Workflow)
本專案是我與大型語言模型 (LLM) 深度協作的成果。在這次開發中，我嘗試使用一種現代化的軟體開發流程：
*   **技術選型與諮詢**：我大量利用 AI 進行技術方案的評估（例如選擇 SQLite 作為任務佇列而非 Redis），AI 提供選項，而我負責做最終的技術決策。
*   **重大架構轉彎**：開發過程中經歷了幾次核心架構的調整。例如，初期嘗試同步處理導致畫面卡頓，在我的主導下，系統轉向了現在看到的「非同步任務佇列」與「搶佔式排程」架構。
*   **決策者角色**：AI 負責生成程式碼片段與優化細節，而我則負責定義整體架構、並在 AI 邏輯出現偏差時進行修正與除錯。

---

## 未來展望 (Future Work)

MoshouSapient 目前的地端單機版（Standalone）已成功完成概念驗證（PoC）與真人測試。在單機運行過程中，我清楚觀察到單一消費級硬體在面對多路資料流併發處理時的算力瓶頸與儲存限制。因此，我決定將此單機版本「階段性封存」。

這並非專案的終點，而是下一個階段的起點。最初會選擇開發影像辨識系統，僅是因為手邊剛好有網路攝影機可供實驗；但我真正感興趣的，並不侷限於單一的視覺傳感器，而是更廣泛的數據收集與整合。我非常期待未來進入大學就讀後，能透過系統化的學術訓練，將這套系統的概念延伸為「**跨感測器的大數據智慧物聯網（IoT）系統**」：

*   **多源感測器整合 (Multi-sensor Integration)**：打破僅依賴視覺（Camera）的限制，未來希望能學習接入更多元的物聯網設備（如溫濕度、紅外線、聲音等各類傳感器），實現多維度的環境感知與數據交叉驗證。
*   **分散式架構與微服務 (Distributed Architecture)**：學習如何評估與導入專業的分散式訊息佇列與微服務架構，取代目前的單機 SQLite 佇列。期望能將資料擷取、分析與儲存模組化，徹底擺脫單一作業系統的環境限制，實現跨節點的任務分發與高可用性。
*   **雲端與大數據分析 (Cloud & Big Data Analytics)**：將邊緣端（Edge）的輕量化即時反應，與雲端（Cloud）的大數據運算結合。學習如何處理海量感測器數據，進而實現更複雜的長期行為模式預測與資料探勘。

這份作品集記錄了我目前的技術邊界。未來的求學階段，我期望能學習如何正確地進行「技術選型」，並具備建構大型數據系統架構的能力，這也是我重返校園持續進修的最大動力。

---

## ⚖️ 法律與授權聲明 (Legal & License)

### 1. 專案授權
本專案的原始碼採用 [AGPL-3.0-or-later](https://www.gnu.org/licenses/agpl-3.0.html) 授權。

### 2. 第三方軟體與 TensorRT 聲明
本專案整合了 NVIDIA TensorRT 以實現高效能推論。請注意：
- **非散佈性質**：本專案**不散佈**任何由 NVIDIA 提供的二進位檔案、庫檔案 (.so/.dll) 或預編譯的 TensorRT 引擎檔案 (.engine)。
- **使用者責任**：使用者在執行 `scripts/export_tensorrt.py` 產生引擎檔案時，即代表使用者已同意並遵守 [NVIDIA TensorRT 授權協議 (EULA)](https://developer.nvidia.com/)。
- **授權衝突說明**：本專案僅提供「實作指南（原始碼）」。由於 TensorRT 為專有軟體，其授權條款與 AGPL-3.0 不同。使用者在本地端自行編譯與使用之行為，不構成對 TensorRT 專有軟體的非法散佈。

### 3. 關鍵應用免責聲明 (Critical Application Disclaimer)
**警告**：本系統僅供個人學習、研究及一般安防參考。
- 本系統**未經過** NVIDIA 或任何認證機構的安全性測試，**絕對禁止**將其用於任何涉及人類生命安全、醫療維生、軍事防禦或自動駕駛等「關鍵應用 (Critical Applications)」場景。
- 開發者對因使用本系統而導致的任何直接或間接損失（包括但不限於財產損失、漏報或誤報）不承擔任何法律責任。
