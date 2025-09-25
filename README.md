# MoshouSapient: AI 智慧影像分析平台

![Project Status: Active Dev](https://img.shields.io/badge/status-active%20development-green) ![Python Version](https://img.shields.io/badge/python-3.11-blue) ![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)

MoshouSapient 是一個基於 Python 與 NVIDIA TensorRT 技術棧所建構的高效能智慧影像分析平台。系統採用了穩健的程序級隔離架構，能夠穩定處理 RTSP 即時影像流或本地影片檔案，執行物件偵測、追蹤與高階行為分析。當觸發特定規則時，系統會將結構化的分析結果與事件影片進行持久化儲存，為後續的數據分析和安全審計提供支持。

<p align="center">
  <video src="https://github.com/MortyTsai/Moshou_Sapient/raw/master/docs/assets/demo.mp4" autoplay loop muted playsinline width="80%"></video>
</p>

## 專案狀態

:construction: **本專案為學習與實踐導向，仍在持續開發中。**

目前已完成核心功能的開發，具備穩定的基礎架構與可靠的行為分析能力。系統架構已為未來的功能擴展 (如白名單系統、多攝影機管理) 預留了清晰的介面。歡迎任何形式的建議與討論。

## 核心特性

-   **高效能推論管線 (High-Performance Inference Pipeline)**: 整合 **YOLO** 物件偵測模型與 **NVIDIA TensorRT** 引擎進行加速，並利用 **NVENC** 硬體編碼器生成事件影片，實現高效的影像處理能力。
-   **程序級隔離架構 (Process-Level Isolation)**: 在處理本地影片檔案時，將資源密集型的 AI 推論任務封裝在一個獨立的作業系統子程序中執行。這種設計從根本上保證了主應用程式的穩定性和響應能力，杜絕了因底層 AI 函式庫狀態洩漏或記憶體問題導致的處理阻塞。
-   **魯棒的物件追蹤 (Robust Object Tracking)**: 採用 **BOTSORT** 演算法進行多物件追蹤。透過特徵提取輔助，能夠在一定程度上應對短暫遮擋，為行為分析提供穩定的目標軌跡。追蹤器在事件邊界會被重新實例化，確保了長時間運行的穩定性。
-   **高階行為分析 (Advanced Behavioral Analysis)**:
    -   **全模式支援**: 所有高階行為分析功能均**同時支援 RTSP 即時串流與 FILE 本地影片**兩種模式。
    -   **區域入侵與停留偵測 (ROI Dwell Time)**: 支援使用者自訂多邊形感興趣區域 (ROI)，能夠偵測目標是否進入特定區域，並在停留時間超過預設閾值時觸發警報。
    -   **方向性虛擬警戒線 (Directional Tripwire)**: 支援使用者定義帶有方向的虛擬線段。系統利用向量叉積判斷目標的移動軌跡，僅在符合預設方向的跨越發生時觸發警報，有效過濾無關行為。
    -   **動態視覺化回饋**: 自動在生成的事件影片中繪製半透明的 ROI 區域、帶方向的警戒線箭頭，並根據目標行為（進入ROI、穿越警戒線）即時改變其追蹤框顏色，提供極佳的可視化分析體驗。
-   **事件驅動的持久化 (Event-Driven Persistence)**: 系統能在偵測到異常行為時觸發事件，並使用 SQLAlchemy ORM 將事件元數據高效存入 SQLite 資料庫 (WAL 模式)，便於後續查詢與管理。
-   **靈活的影片輸出設定 (Flexible Video Output Configuration)**:
    -   **幀率控制**: 可選擇保留來源影片的原始幀率 (`SOURCE` 模式)，或將輸出影片降採樣至指定的目標幀率 (`TARGET` 模式)，以在保真度與檔案大小間取得平衡。
    -   **編碼策略**: 提供「品質」(`QUALITY`) 模式與「均衡」(`BALANCED`) 模式，後者可將影片控制在指定的平均位元率，實現可預測的檔案大小。
-   **分層與模組化架構 (Layered & Modular Architecture)**: 採用標準化的 `src` 專案佈局，並將應用程式邏輯清晰地劃分為 `core`, `streams`, `processors`, `services` 等多個職責明確的子套件，實現了高度的內聚與解耦。
-   **遠端存取與可選通知 (Remote Access & Optional Notifications)**:
    -   內建基於 **Flask** 的輕量級 Web 儀表板，用於遠端查看事件紀錄與回放。
    -   可選整合 **Discord Bot**，以非同步方式推送即時警報。

## 技術棧

-   **核心框架**: Python 3.11
-   **AI / CV**: PyTorch, TensorRT, Ultralytics YOLO, BOTSORT, Shapely (幾何分析)
-   **資料庫**: SQLite, SQLAlchemy (ORM)
-   **Web 後端**: Flask
-   **影像處理**: FFmpeg, OpenCV-Python
-   **設定管理**: Pydantic-Settings, PyYAML
-   **其他**: python-dotenv

## 系統檔案結構
```
MoshouSapient/                          # 專案根目錄
│
├── .env.example                        # 環境變數設定檔範本
├── .gitignore                          # Git 版本控制忽略清單
├── README.md                           # 專案說明文件
├── requirements.txt                    # Python 依賴套件列表
│
├── configs/                            # 存放所有靜態設定檔
│   ├── behavior_analysis.yaml          # 行為分析規則 (ROI, 警戒線)
│   └── custom_botsort.yaml             # BoT-SORT 追蹤器客製化參數
│
├── data/                               # 存放專案資料 (執行時生成)
│   ├── captures/                       # 儲存事件錄影
│   ├── security_events.db              # SQLite 資料庫檔案
│   └── video_samples/                  # 存放 FILE 模式的範例影片
│
├── docs/                               # 存放所有文件與相關資源
│   └── assets/
│       └── demo.mp4                    # README 中使用的演示影片
│
├── models/                             # 存放所有 AI 模型資產
│
├── scripts/                            # 存放輔助開發腳本
│   └── export_tensorrt.py              # 模型轉換為 TensorRT 引擎的腳本
│
└── src/                                # 存放所有專案原始碼
    └── moshouSapient/                  # 專案主 Python 套件
        ├── __main__.py                 # 套件執行入口
        │
        ├── core/                       # 核心業務邏輯與協調器
        │   ├── camera_worker.py        # (RTSP) 管理單一攝影機管線
        │   ├── main.py                 # 應用程式主邏輯
        │   └── runners.py              # 執行策略模組 (RTSP/File)
        │
        ├── processors/                 # 資料流處理單元
        │   ├── base_processor.py       # 處理器抽象基礎類別
        │   ├── event_processor.py      # (RTSP) 事件偵測與狀態管理
        │   ├── file_result_processor.py# (FILE) 處理 JSON 結果與事件生成
        │   └── inference_processor.py  # (RTSP) AI 推論處理器
        │
        ├── services/                   # 外部服務與獨立邏輯單元
        │   ├── database_service.py     # 資料庫互動服務
        │   ├── discord_notifier.py     # Discord Bot 通知服務
        │   ├── isolated_inference_service.py # (FILE) 獨立的 AI 推論子程序
        │   └── video_recorder.py       # (RTSP) 影片錄製服務
        │
        ├── streams/                    # 資料來源讀取模組
        │
        ├── utils/                      # 通用工具函式
        │
        ├── web/                        # Web 儀表板
        │
        ├── config.py                   # 應用程式組態初始化
        ├── database.py                 # 資料庫設定與 Session 管理
        ├── logging_setup.py            # 全域日誌設定
        ├── models.py                   # 資料庫 ORM 模型定義
        └── settings.py                 # Pydantic 靜態設定管理```
```
## 環境準備

### 硬體與軟體需求
-   **作業系統**: Windows 10 / 11
-   **GPU**: 支援 NVENC 硬體編碼的 NVIDIA 顯示卡 (建議 GeForce RTX 系列)
-   **Python**: 3.11

### 安裝步驟

1.  **安裝 NVIDIA 工具鏈**:
    -   NVIDIA 驅動程式
    -   CUDA Toolkit (建議版本 12.x 或更高)
    -   cuDNN (需對應 CUDA 版本)
    -   TensorRT (需對應 CUDA 版本)

2.  **安裝核心工具**:
    -   Python 3.11 (安裝時建議勾選 "Add Python to PATH")
    -   FFmpeg (需將其 `bin` 目錄加入系統環境變數 PATH)

3.  **設定 Python 虛擬環境**:
    ```bash
    # 建立虛擬環境
    python -m venv venv
    # 啟用虛擬環境
    .\venv\Scripts\activate
    ```

4.  **安裝 Python 依賴**:
    ```bash
    # 1. 根據您的 CUDA 版本，從 PyTorch 官網安裝對應的 GPU 版本
    # 例如 CUDA 12.x:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu12x

    # 2. 安裝其餘依賴
    pip install -r requirements.txt
    ```

5.  **準備 AI 模型**:
    -   從指定來源下載 `yolo11s.pt` (物件偵測) 和 `yolo11s-cls.pt` (特徵提取) 模型檔案，並放置在 `models/` 資料夾中。
    -   執行轉換腳本，將**偵測模型**生成為 TensorRT 引擎：
        ```bash
        python scripts/export_tensorrt.py
        ```
    -   成功後會在 `models/` 資料夾下生成 `yolo11s.engine` 檔案。

## 專案設定與執行

1.  **設定環境變數**:
    在專案根目錄下，將 `.env.example` 複製一份並重新命名為 `.env`。此檔案用於集中管理所有可變設定。

    ```env
    # .env 範例

    # --- 影像來源設定 (必要) ---
    # 可選值: "RTSP" 或 "FILE"
    VIDEO_SOURCE_TYPE="FILE"
    
    # 【FILE 模式專用】影片檔案的路徑 (相對於專案根目錄)
    VIDEO_FILE_PATH="data/video_samples/input.mp4"
    
    # 【RTSP 模式專用】攝影機的 RTSP 串流網址
    RTSP_URL=""
    ```

2.  **設定行為分析規則 (重要)**:
    打開 `configs/behavior_analysis.yaml` 檔案，根據您的場景需求，設定感興趣區域 (ROI) 和虛擬警戒線 (Tripwire) 的座標與規則。檔案內有詳細的註解說明。**此設定對 RTSP 和 FILE 模式同時生效。**

3.  **啟動系統**:
    在專案**根目錄**下，執行以下指令：
    ```bash
    python -m moshousapient
    ```

4.  **驗證**:
    -   打開瀏覽器，訪問 Web 儀表板： `http://127.0.0.1:5000`
    -   觸發事件（例如，讓人物出現在攝影機畫面中，或使用包含人物的影片檔案）。
    -   檢查 `data/captures` 目錄是否生成了帶有視覺化標記的事件影片。
    -   檢查 Web 儀表板是否出現新的事件紀錄。

## 發展藍圖

-   **[首要] 白名單系統 (Whitelist System)**: 開發一套白名單機制。當事件觸發時，可將偵測到的人物特徵與預先註冊的白名單特徵庫進行比對，若匹配成功則抑制警報通知，以過濾授權人員的正常活動。
-   **進階資料庫查詢與關聯**: 探索基於人物特徵的相似度搜尋，以實現特定人物的歷史事件檢索（例如「顯示這個人今天所有出現過的片段」）。
-   **性能優化**: 針對高幀率、高解析度的影像來源，持續優化系統的處理吞吐量與資源利用率。
-   **前端介面強化**: 擴充 Web 儀表板功能，例如增加事件篩選、排序，或引入更豐富的數據視覺化圖表，並提供 ROI/警戒線的視覺化設定介面。
-   **多攝影機協同**: 將現有的單攝影機架構擴展，使其能夠由一個主程序同時管理多個獨立的攝影機，並實現跨攝影機的目標追蹤。

## 開發模式說明

本專案的開發過程，是一次探索人類開發者與大型語言模型 (LLM) 協同作業的實踐。在此模式中，人類開發者的角色聚焦於定義高階目標、提供精確技術上下文、以及進行迭代式驗證與除錯，旨在將人類的策略性思考與 LLM 的高效程式碼生成能力相結合，探索一種現代化的軟體開發工作流程。

## License

本專案採用 [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html) 授權。