# MoshouSapient: 個人化 AI 智慧保全系統

![Python Version](https://img.shields.io/badge/python-3.11-blue) ![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue) ![Status: Archived](https://img.shields.io/badge/status-archived-orange)

MoshouSapient 是一個將普通電腦轉變為進階 AI 保全主機的開源專案。它能在低硬體負載下，透過 NVIDIA GPU 加速實現即時的物件偵測、行為分析與場景異常告警。

https://github.com/user-attachments/assets/dfd1b02a-1547-458b-8680-2b774ba843d6

> **專案定位與維護說明**
> 本專案為我申請大學的個人作品集。目前已達成核心功能驗證與架構優化目標，**現已停止主動維護**。這是我探索「AI 協同開發」與「系統架構設計」的階段性成果。

---

## 核心設計理念
*   **高效能推論**：整合 **YOLO** 與 **NVIDIA TensorRT**，在消費級顯卡上實現高速物件偵測。
*   **工業級架構**：採用「生產者-消費者」模型，利用 **SQLite (WAL 模式)** 實作持久化任務佇列，確保系統穩定性。
*   **智慧型排程**：內建**搶佔式排程器**，確保即時監控任務永遠擁有最高硬體優先權。
*   **多層次警報**：結合 AI 行為分析（ROI、警戒線）與底層影像演算法（防篡改、畫面凍結偵測）。

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

### 1. 為什麼只有 Windows？
本專案開發時專為 **Windows 10/11** 與 **NVIDIA GPU** 生態系設計。主因是我個人主要使用 Windows 系統，對其開發環境與驅動配置最為熟悉。為了確保系統能在我的硬體（NVIDIA GPU）上達到最佳效能，我選擇專注於此平台的優化，暫未考慮跨平台相容性。

### 2. AI 驅動的開發模式 (AI-Centric Workflow)
本專案是我與大型語言模型 (LLM) 深度協作的成果。在這次開發中，我嘗試使用一種現代化的軟體開發流程：
*   **技術選型與諮詢**：我大量利用 AI 進行技術方案的評估（例如選擇 SQLite 作為任務佇列而非 Redis），AI 提供選項，而我負責做最終的技術決策。
*   **重大架構轉彎**：開發過程中經歷了幾次核心架構的重大調整。例如，初期嘗試同步處理導致畫面卡頓，在我的主導下，系統轉向了現在看到的「非同步任務佇列」與「搶佔式排程」架構。
*   **決策者角色**：AI 負責生成程式碼片段與優化細節，而我則負責定義整體架構、並在 AI 邏輯出現偏差時進行修正與除錯。

---

## 授權
本專案採用 [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html) 授權。
.

