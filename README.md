# MoshouSapient: AI 智慧影像分析平台

![Project Status: Active Dev](https://img.shields.io/badge/status-active%20development-green) ![Python Version](https://img.shields.io/badge/python-3.11-blue) ![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)

MoshouSapient 是一個基於 Python 與 NVIDIA TensorRT 技術棧所建構的高效能智慧影像分析平台。系統採用了穩健的**非同步任務佇列架構**，能夠穩定處理 RTSP 即時影像流，並透過**事件驅動的搶佔式排程**高效處理背景影片檔案分析任務。當觸發特定規則時，系統會將結構化的分析結果與事件影片進行持久化儲存，為後續的數據分析和安全審計提供支持。

https://github.com/user-attachments/assets/dfd1b02a-1547-458b-8680-2b774ba843d6

## 專案狀態

:construction: **本專案為學習與實踐導向，仍在持續開發中。**

目前已完成核心功能的開發與一次重大的架構升級。系統現在基於一個更可靠、更具擴展性的非同步任務處理模型，為未來的功能擴展 (如多攝影機管理、進階警報功能) 奠定了堅實的基礎。歡迎任何形式的建議與討論。

## 核心特性

-   **高效能推論與編碼**:
    -   **AI 推論**: 整合 **YOLO** 物件偵測模型與 **NVIDIA TensorRT** 引擎進行加速，實現高速物件偵測與特徵提取。
    -   **影片編碼**: 利用 **NVENC** 硬體編碼器生成事件影片。

-   **工業級非同步任務架構**:
    -   **生產者-消費者模型**: 系統被解耦為「生產者」（負責即時分析與事件偵測）和「消費者」（負責高負載的影片生成），兩者通過任務佇列進行非同步通信。
    -   **零依賴任務佇列**: 利用 **SQLite (WAL 模式)** 實現了一個持久化、執行緒安全的任務佇列，在不增加任何外部服務依賴（如 Redis, RabbitMQ）的前提下，實現了工業級的可靠性。
    -   **並行處理**: 可透過設定啟動多個並行的影片處理 Worker，充分利用多核 CPU 與 GPU 的硬體編碼能力，實現了單機內的水平擴展。

-   **智慧型非同步檔案處理**:
    -   **事件驅動的搶佔式排程**: 系統內建一個智慧排程器 (`Scheduler`)，能夠在處理 `RTSP` 即時影像流的同時，接收背景檔案分析任務。
    -   **RTSP 優先原則**: 排程器以 `RTSP` 事件的活躍狀態作為決策依據。只有在 `RTSP` 模式閒置時，才會啟動獨立的 `Job` 程序處理背景任務，確保即時分析的絕對優先權。
    -   **搶佔與恢復**: 當背景任務正在執行時，一旦 `RTSP` 偵測到新事件，排程器會**立即終止**背景任務以釋放資源。事件結束後，被中斷的任務會被**自動恢復**並重新排入佇列，等待下一個閒置窗口。
    -   **自動化攝取**: 內建 `IngestionService`，可監控指定資料夾，任何被放入的影片檔案都將被自動轉換為背景分析任務，實現了「熱資料夾」(Hot Folder) 功能。

-   **穩健的物件追蹤**: 採用 **BOTSORT** 演算法與 **Re-ID** 特徵提取進行多物件追蹤，能夠在一定程度上應對短暫遮擋，為行為分析提供穩定的目標軌跡。

-   **高階行為分析與動態視覺化**:
    -   **區域入侵與停留偵測 (ROI Dwell Time)**: 支援自訂多邊形感興趣區域 (ROI)，能夠偵測目標是否進入特定區域，並在停留時間超過預設閾值時觸發警報。
    -   **方向性虛擬警戒線 (Directional Tripwire)**: 支援定義帶有方向的虛擬線段，僅在符合預設方向的跨越發生時觸發警報。
    -   **精細化錨點策略 (Granular Anchor Strategy)**: 除了可設定全域錨點外，**每一條** ROI 或 Tripwire 規則都可以獨立覆寫 `anchor_points` 參數，提升了複雜場景下的分析準確性。
    -   **專業級視覺化回饋**: 自動在生成的事件影片中繪製半透明的 ROI 區域、帶方向的警戒線箭頭，並具備主目標凸顯、狀態疊加層、錨點繪製等進階視覺化能力。

-   **事件驅動的持久化**: 系統能在偵測到異常行為時觸發事件，並使用 SQLAlchemy ORM 將事件元數據高效存入 SQLite 資料庫 (WAL 模式)，便於後續查詢與管理。

-   **專業級日誌系統 (Professional Logging System)**:
    -   **關注點分離**: 實現了「使用者日誌」（主控台）與「開發者日誌」（檔案）的徹底分離。
    -   **使用者日誌 (主控台)**: 主控台輸出極度簡潔，只報告使用者關心的關鍵事件與系統狀態。
    -   **開發者日誌 (檔案)**: 所有程序的詳細 `DEBUG` 級日誌，包括第三方套件的輸出，都會被即時、同步地捕獲並寫入輪替日誌檔案中。

-   **靈活的影片輸出設定**:
    -   **事件分段**: 自動將長時間的連續事件分割成多個較短的影片檔案。
    -   **幀率與編碼控制**: 提供多種幀率與編碼策略，以在保真度與檔案大小間取得平衡。

-   **分層與模組化架構**: 採用標準化的 `src` 專案佈局與 `pyproject.toml` 設定，將專案封裝為一個**可安裝套件**。應用程式邏輯清晰地劃分為 `configs`, `core`, `jobs`, `processors`, `services`, `streams`, `utils`, `workers` 等多個職責明確的子套件。

-   **遠端存取與可選通知**:
    -   內建基於 **Flask** 的輕量級 Web 儀表板，用於遠端查看事件紀錄與回放。
    -   可選整合 **Discord Bot**，以非同步方式推送即時警報。

## 技術棧

-   **核心框架**: Python 3.11
-   **AI / CV**: PyTorch, TensorRT, Ultralytics YOLO, BOTSSORT, Shapely (幾何分析)
-   **資料庫**: SQLite, SQLAlchemy (ORM)
-   **Web 後端**: Flask
-   **影像處理**: FFmpeg, OpenCV-Python
-   **設定管理**: Pydantic-Settings, PyYAML
-   **其他**: python-dotenv, watchdog

## 系統檔案結構
```
MoshouSapient/                                  # 專案根目錄
│
├── .env.example                                # 環境變數設定檔範本
├── .gitignore                                  # Git 版本控制忽略清單
├── pyproject.toml                              # 專案建置與結構設定檔 (PEP 518/621)
├── README.md                                   # 專案說明文件
├── requirements.txt                            # Python 依賴套件列表
│
├── configs/                                    # 存放所有使用者可直接編輯的設定檔
│   ├── behavior_analysis.yaml                  # 高階行為分析規則 (ROI, 警戒線, 錨點)
│   └── custom_botsort.yaml                     # BoT-SORT 追蹤器客製化參數
│
├── data/                                       # 存放所有執行時生成的資料 (被 .gitignore 忽略)
│   ├── captures/                               # 儲存最終生成的事件錄影 (.mp4)
│   ├── logs/                                   # 儲存詳細的開發者日誌檔案 (.log)
│   ├── security_events.db                      # SQLite 事件資料庫檔案
│   ├── tasks.db                                # SQLite 任務佇列資料庫檔案
│   ├── uploads/                                # 檔案攝取服務的監控目錄 ("熱資料夾")
│   └── video_samples/                          # 存放用於測試的範例影片
│
├── models/                                     # 存放所有 AI 模型資產 (.pt, .engine)
│
├── scripts/                                    # 存放與專案核心邏輯無關的輔助開發腳本
│   └── export_tensorrt.py                      # 將 .pt 模型轉換為 TensorRT 引擎的腳本
│
└── src/                                        # 存放所有專案原始碼
    └── moshousapient/                          # 專案主 Python 套件
        ├── __init__.py
        ├── __main__.py                         # `python -m moshousapient` 的主入口點
        │
        ├── configs/                            # 應用程式配置層 (負責加載與解析設定)
        │   ├── __init__.py
        │   ├── behavior_config.py              # 載入並解析 behavior_analysis.yaml 的高階規則
        │   ├── logging_config.py               # 設定全域日誌系統 (佇列、格式化、分離)
        │   └── settings_config.py              # 使用 Pydantic 從 .env 載入基礎靜態設定
        │
        ├── core/                               # 應用程式的骨架：協調與生命週期管理
        │   ├── __init__.py
        │   ├── app_orchestrator.py             # 應用程式主協調器 (初始化、啟動與關閉所有服務)
        │   ├── producer_runners.py             # 執行策略模組 (現僅包含 RTSP 模式)
        │   ├── scheduler.py                    # 事件驅動的搶佔式排程器
        │   └── worker_manager.py               # VideoConsumerWorker 程序池的生命週期管理器
        │
        ├── jobs/                               # 獨立的、由 subprocess 呼叫的一次性背景作業
        │   ├── __init__.py
        │   └── queue_inference_job.py          # 處理佇列中檔案推論任務的作業腳本
        │
        ├── processors/                         # (RTSP) 即時數據流的處理單元 (生產者的一部分)
        │   ├── __init__.py
        │   ├── base_processor.py               # 處理器執行緒的抽象基礎類別
        │   ├── file_event_producer.py          # 現為通用的推論結果處理器
        │   ├── inference_processor.py          # (RTSP) 執行 AI 推論與追蹤
        │   ├── rtsp_event_producer.py          # (RTSP) 進行事件判斷與任務生產
        │   └── rtsp_processing_pipeline.py     # (RTSP) 串聯處理流程的管線
        │
        ├── services/                           # 可重用的、與外部設施互動的基礎服務
        │   ├── __init__.py
        │   ├── database_models.py              # 資料庫 ORM 模型定義 (SQLAlchemy)
        │   ├── database_service.py             # 資料庫連接與會話管理
        │   ├── ingestion_service.py            # 監控 "熱資料夾" 的檔案攝取服務
        │   ├── notification_service.py         # Discord Bot 通知服務
        │   └── task_queue_service.py           # 基於 SQLite 的持久化任務佇列服務
        │
        ├── streams/                            # (RTSP) 原始數據流獲取層
        │   ├── __init__.py
        │   └── video_streamer.py               # 使用 FFmpeg 讀取 RTSP 影像串流
        │
        ├── utils/                              # 無狀態的、純粹的通用工具函式
        │   ├── __init__.py
        │   ├── behavior_analysis_utils.py      # 行為分析演算法 (ROI, Tripwire)
        │   ├── geometry_utils.py               # 通用幾何計算工具 (Shapely)
        │   ├── logging_utils.py                # 日誌重定向輔助工具
        │   ├── reid_matching_utils.py          # Re-ID 特徵匹配演算法
        │   ├── video_io_utils.py               # 通用影片 I/O 工具 (ffprobe)
        │   └── visualization_utils.py          # 視覺化繪圖工具 (OpenCV)
        │
        ├── workers/                            # 由 multiprocessing 管理的常駐背景工作程序 (消費者)
        │   ├── __init__.py
        │   └── video_consumer_worker.py        # 統一的影片分段、繪圖與編碼 Worker
        │
        └── web/                                # Web 儀表板相關模組
            ├── __init__.py
            ├── app.py                          # Flask 應用與路由定義
            └── templates/                      # HTML 樣板
                └── index.html
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
    # 進入專案根目錄
    cd path/to/MoshouSapient
    # 建立虛擬環境
    python -m venv venv
    # 啟用虛擬環境
    .\venv\Scripts\activate
    ```

4.  **安裝專案與依賴**:
    ```bash
    # 1. 將專案本身以「可編輯模式」安裝到虛擬環境中
    pip install -e .

    # 2. 根據您的 CUDA 版本，從 PyTorch 官網安裝對應的 GPU 版本
    #    例如 CUDA 12.x:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu12x

    # 3. 安裝其餘依賴
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

    # --- 系統運行模式 ---
    # "RTSP": 啟用即時攝影機監控。
    # 任何其他值 (或留空): 系統進入閒置模式，僅運行背景服務。
    VIDEO_SOURCE_TYPE="RTSP"
    RTSP_URL="rtsp://your_camera_stream"

    # --- 非同步檔案處理 (可與 RTSP 模式同時啟用) ---
    # 啟用「熱資料夾」功能
    INGESTION_ENABLED=True
    # 啟用智慧排程器以處理背景任務
    SCHEDULER_ENABLED=True
    ```

2.  **設定行為分析規則 (重要)**:
    打開 `configs/behavior_analysis.yaml` 檔案，根據您的場景需求，設定感興趣區域 (ROI)、虛擬警戒線 (Tripwire) 等。檔案內有詳細的註解說明。

3.  **啟動系統**:
    在**啟用虛擬環境**後，您可以在專案的**任何目錄**下，執行以下指令：
    ```bash
    python -m moshousapient
    ```

4.  **驗證**:
    -   **RTSP 模式**: 觸發攝影機前的事件，檢查 `data/captures` 是否生成影片。
    -   **檔案處理模式**: 將一個影片檔案複製到 `data/uploads` 資料夾，等待系統在閒置時自動處理並生成影片。
    -   **通用**: 打開瀏覽器訪問 Web 儀表板 `http://127.0.0.1:5000` 查看事件紀錄。
    -   **(除錯)** 檢查 `data/logs/` 目錄下的日誌檔案。

## 發展藍圖

-   **[已完成] 架構演進 (Architecture Evolution)**:
    -   **升級至任務佇列架構**: 成功將專案重構為基於 SQLite 的、可靠的非同步任務佇列架構。
    -   **實現搶佔式排程**: 引入了事件驅動的智慧排程器，實現了 `RTSP` 即時任務對背景檔案處理任務的自動搶佔與恢復，確保了核心業務的絕對優先權。
-   **進階警報 (Advanced Alerts)**:
    -   **遮蔽警報 (Occlusion Alert)**: 開發基於 AI 偵測的遮蔽警報。
    -   **畫面異常警報 (Scene Anomaly Alert)**: 開發不依賴 AI 的底層影像分析警報（如畫面凍結、亮度劇變）。
-   **白名單系統 (Whitelist System)**: 開發一套白名單機制，以過濾授權人員的正常活動。
-   **前端介面強化**: 擴充 Web 儀表板功能，例如增加事件篩選、排序，或引入 ROI/警戒線的視覺化設定介面。
-   **技術債**:
    -   **搶佔恢復的原子性**: 優化任務恢復機制，以避免在極端情況下（搶佔發生在 `Job` 創建下游任務之後、但在自身完成之前）可能導致的重複處理問題。

## 開發模式說明

本專案的開發過程，是一次探索人類開發者與大型語言模型 (LLM) 協同作業的實踐。在此模式中，人類開發者的角色聚焦於定義高階目標、提供精確技術上下文、以及進行迭代式驗證與除錯，旨在將人類的策略性思考與 LLM 的高效程式碼生成能力相結合，探索一種現代化的軟體開發工作流程。

## License

本專案採用 [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html) 授權。