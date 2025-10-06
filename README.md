### **MoshouSapient: AI 智慧影像分析平台**

# MoshouSapient: AI 智慧影像分析平台

![Project Status: Active Dev](https://img.shields.io/badge/status-active%20development-green) ![Python Version](https://img.shields.io/badge/python-3.11-blue) ![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)

MoshouSapient 是一個基於 Python 與 NVIDIA TensorRT 技術棧所建構的高效能智慧影像分析平台。系統採用了穩健的**非同步任務佇列架構**，能夠穩定處理 RTSP 即時影像流或本地影片檔案，執行物件偵測、追蹤與高階行為分析。當觸發特定規則時，系統會將結構化的分析結果與事件影片進行持久化儲存，為後續的數據分析和安全審計提供支持。

https://github.com/user-attachments/assets/dfd1b02a-1547-458b-8680-2b774ba843d6

## 專案狀態

:construction: **本專案為學習與實踐導向，仍在持續開發中。**

目前已完成核心功能的開發與一次重大的架構升級。系統現在基於一個更可靠、更具擴展性的非同步任務處理模型，為未來的功能擴展 (如多攝影機管理、進階警報功能) 奠定了堅實的基礎。歡迎任何形式的建議與討論。

## 核心特性

-   **高效能推論與編碼**:
    -   **AI 推論**: 整合 **YOLO** 物件偵測模型與 **NVIDIA TensorRT** 引擎進行加速，實現高速物件偵測與特徵提取。
    -   **影片編碼**: 利用 **NVENC** 硬體編碼器生成事件影片。

-   **工業級非同步任務佇列架構**:
    -   **生產者-消費者模型**: 系統被解耦為「生產者」（負責即時分析與事件偵測）和「消費者」（負責高負載的影片生成），兩者通過任務佇列進行非同步通信。
    -   **零依賴任務佇列**: 利用 **SQLite (WAL 模式)** 實現了一個持久化、執行緒安全的任務佇列，在不增加任何外部服務依賴（如 Redis, RabbitMQ）的前提下，實現了工業級的可靠性。
    -   **負載平滑與可靠性**: 即使瞬間觸發大量事件，任務也只是在佇列中有序排隊，由背景的 Worker 程序池穩定處理，避免了資源競爭，並確保即使程序崩潰，任務也不會遺失。
    -   **並行處理**: 可透過設定啟動多個並行的影片處理 Worker，充分利用多核 CPU 與 GPU 的硬體編碼能力，實現了單機內的水平擴展。

-   **穩健的物件追蹤**: 採用 **BOTSORT** 演算法與 **Re-ID** 特徵提取進行多物件追蹤，能夠在一定程度上應對短暫遮擋，為行為分析提供穩定的目標軌跡。

-   **高階行為分析與動態視覺化**:
    -   **全模式功能對等**: 所有高階行為分析與視覺化功能均**同時支援 RTSP 與 FILE 兩種模式**，確保了行為的一致性。
    -   **區域入侵與停留偵測 (ROI Dwell Time)**: 支援自訂多邊形感興趣區域 (ROI)，能夠偵測目標是否進入特定區域，並在停留時間超過預設閾值時觸發警報。
    -   **方向性虛擬警戒線 (Directional Tripwire)**: 支援定義帶有方向的虛擬線段。系統利用向量叉積判斷目標的移動軌跡，僅在符合預設方向的跨越發生時觸-發警報。
    -   **精細化錨點策略 (Granular Anchor Strategy)**: 除了可設定全域錨點外，**每一條** ROI 或 Tripwire 規則都可以獨立覆寫 `anchor_points` 參數。這使得系統能夠在同一個畫面中，為不同形態的目標（如遠處的全身人像與近處的半身人像）應用最合適的判斷基準點，提升了複雜場景下的分析準確性。
    -   **專業級視覺化回饋**: 自動在生成的事件影片中繪製半透明的 ROI 區域、帶方向的警戒線箭頭，並具備以下進階視覺化能力：
        -   **主目標凸顯**: 在多目標場景中，以鮮明色彩**凸顯觸發事件的主要目標**，並將次要目標以半透明灰色弱化顯示，讓影片證據一目了然。
        -   **狀態疊加層**: 在影片左上角清晰地疊加**事件類型**與**動態持續時間**，為影片證據提供關鍵的上下文資訊。
        -   **錨點繪製**: 可在影片中繪製出用於行為判斷的**錨點**，增強了系統的可除錯性與演算法透明度。

-   **事件驅動的持久化**: 系統能在偵測到異常行為時觸發事件，並使用 SQLAlchemy ORM 將事件元數據高效存入 SQLite 資料庫 (WAL 模式)，便於後續查詢與管理。

-   **專業級日誌系統 (Professional Logging System)**:
    -   **關注點分離**: 系統實現了「使用者日誌」與「開發者日誌」的徹底分離。
    -   **使用者日誌 (主控台)**: 主控台輸出極度簡潔，只報告使用者關心的關鍵事件與系統狀態，在正常運行時保持靜默，確保了出色的可讀性。
    -   **開發者日誌 (檔案)**: 所有程序的詳細 `DEBUG` 級日誌，包括第三方套件的輸出，都會被即時、同步地捕獲並寫入 `data/logs/` 下的輪替日誌檔案中，為系統除錯和事後分析提供了完整且可靠的依據。
    -   **動態級別控制**: 可透過 `.env` 檔案中的 `LOG_LEVEL` 參數，在不修改程式碼的情況下，動態調整主控台的日誌詳細程度。

-   **靈活的影片輸出設定**:
    -   **事件分段**: 內建 `MAX_EVENT_DURATION` 機制，能自動將長時間的連續事件分割成多個較短的影片檔案，便於管理和傳輸。
    -   **幀率控制**: 可選擇保留來源影片的原始幀率 (`SOURCE` 模式)，或將輸出影片轉換至指定的目標幀率 (`TARGET` 模式)，以在保真度與檔案大小間取得平衡。
    -   **編碼策略**: 提供「品質」(`QUALITY`) 模式與「均衡」(`BALANCED`) 模式，後者可將影片控制在指定的平均位元率，實現可預測的檔案大小。

-   **分層與模組化架構**: 採用標準化的 `src` 專案佈局，並將應用程式邏輯清晰地劃分為 `configs`, `core`, `processors`, `services`, `streams`, `utils`, `workers` 等多個職責明確的子套件，實現了高度的內聚與解耦。

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
-   **其他**: python-dotenv

## 系統檔案結構
```
MoshouSapient/                                  # 專案根目錄
│
├── .env.example                                # 環境變數設定檔範本
├── .gitignore                                  # Git 版本控制忽略清單
├── README.md                                   # 專案說明文件
├── requirements.txt                            # Python 依賴套件列表
│
├── configs/                                    # 存放所有使用者設定檔
│   ├── behavior_analysis.yaml                  # 行為分析規則 (ROI, 警戒線, 錨點)
│   └── custom_botsort.yaml                     # BoT-SORT 追蹤器客製化參數
│
├── data/                                       # 存放專案資料 (執行時生成)
│   ├── captures/                               # 儲存事件錄影
│   ├── logs/                                   # 儲存詳細的開發者日誌檔案
│   ├── security_events.db                      # SQLite 事件資料庫檔案
│   ├── tasks.db                                # SQLite 任務佇列資料庫檔案
│   └── video_samples/                          # 存放 FILE 模式的範例影片
│
├── models/                                     # 存放所有 AI 模型資產
│
├── scripts/                                    # 存放輔助開發腳本
│   └── export_tensorrt.py                      # 模型轉換為 TensorRT 引擎的腳本
│
└── src/                                        # 存放所有專案原始碼
    └── moshousapient/                          # 專案主 Python 套件
        ├── __init__.py
        ├── __main__.py
        │
        ├── configs/                            # 應用程式配置層 (加載與解析)
        │   ├── __init__.py
        │   ├── behavior_config.py              # 載入 YAML 行為規則
        │   ├── logging_config.py               # 全域日誌設定
        │   └── settings_config.py              # 載入 .env 靜態設定
        │
        ├── core/                               # 應用程式協調與管理
        │   ├── __init__.py
        │   ├── app_orchestrator.py             # 應用程式主協調器 (啟動與關閉)
        │   ├── producer_runners.py             # 執行策略模組 (RTSP/FILE)
        │   └── worker_manager.py               # Worker 程序池生命週期管理器
        │
        ├── processors/                         # 數據分析與任務生產 (生產者)
        │   ├── __init__.py
        │   ├── base_processor.py               # 處理器執行緒的抽象基礎類別
        │   ├── file_event_producer.py          # (FILE) 處理 JSON 結果並生產任務
        │   ├── inference_processor.py          # (RTSP) 執行 AI 推論與追蹤
        │   ├── rtsp_event_producer.py          # (RTSP) 進行事件判斷與任務生產
        │   └── rtsp_processing_pipeline.py     # (RTSP) 串聯處理流程的管線
        │
        ├── services/                           # 共享的基礎設施服務
        │   ├── __init__.py
        │   ├── database_models.py              # 資料庫 ORM 模型定義
        │   ├── database_service.py             # 資料庫連接與會話管理
        │   ├── isolated_inference_service.py   # (FILE) 獨立的 AI 推論子程序
        │   ├── notification_service.py         # Discord Bot 通知服務
        │   └── task_queue_service.py           # 基於 SQLite 的持久化任務佇列服務
        │
        ├── streams/                            # 原始數據流獲取
        │   ├── __init__.py
        │   └── video_streamer.py               # (RTSP) 使用 FFmpeg 讀取影像串流
        │
        ├── utils/                              # 無狀態的通用工具函式
        │   ├── __init__.py
        │   ├── behavior_analysis_utils.py      # 行為分析演算法 (ROI, Tripwire)
        │   ├── geometry_utils.py               # 通用幾何計算工具
        │   ├── logging_utils.py                # 日誌重定向輔助工具
        │   ├── reid_matching_utils.py          # Re-ID 特徵匹配演算法
        │   ├── video_io_utils.py               # 通用影片 I/O 工具
        │   └── visualization_utils.py          # 視覺化繪圖工具
        │
        ├── workers/                            # 背景工作程序 (消費者)
        │   ├── __init__.py
        │   └── video_consumer_worker.py        # 統一的影片分段、繪圖與編碼 Worker
        │
        └── web/                                # Web 儀表板
            ├── __init__.py
            ├── app.py                          # Flask 應用與路由
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
        python scripts/export_tensor_rt.py
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

    # --- 系統效能設定 (可選) ---
    # 背景影片處理程序的數量。建議值: 2
    VIDEO_PROCESSING_WORKERS=2

    # --- 日誌系統設定 (可選) ---
    # 主控台的日誌詳細程度。可選值: "INFO", "DEBUG"
    LOG_LEVEL="INFO"
    ```

2.  **設定行為分析規則 (重要)**:
    打開 `configs/behavior_analysis.yaml` 檔案，根據您的場景需求，設定感興趣區域 (ROI)、虛擬警戒線 (Tripwire) 以及全域/局部錨點策略。檔案內有詳細的註解說明。**此設定對 RTSP 和 FILE 模式同時生效。**

3.  **啟動系統**:
    在專案**根目錄**下，執行以下指令：
    ```bash
    cd src
    python -m moshousapient
    ```

4.  **驗證**:
    -   打開瀏覽器，訪問 Web 儀表板： `http://127.0.0.1:5000`
    -   觸發事件（例如，讓人物出現在攝影機畫面中，或使用包含人物的影片檔案）。
    -   檢查 `data/captures` 目錄是否生成了帶有完整視覺化標記的事件影片。
    -   檢查 Web 儀表板是否出現新的事件紀錄。
    -   **(除錯)** 檢查 `data/logs/moshousapient.log` 檔案，確認所有程序的詳細日誌均被正確記錄。

## 發展藍圖

-   **[已完成] 架構演進 (Architecture Evolution)**:
    -   **升級至任務佇列架構**: 已成功將專案重構為基於 SQLite 的、可靠的非同步任務佇列架構。此舉解耦了即時分析（生產者）與高負載處理（消費者），實現了系統穩定性、處理流程的可靠性與資源使用的可控性。
-   **進階警報 (Advanced Alerts)**:
    -   **遮蔽警報 (Occlusion Alert)**: 開發基於 AI 偵測的遮蔽警報，當單一可識別物件（如人）的邊界框佔據畫面過大比例時觸發，以防止鏡頭被惡意遮擋。
    -   **畫面異常警報 (Scene Anomaly Alert)**: 開發不依賴 AI 物件偵測的底層影像分析警報，例如畫面凍結、亮度劇變（全黑/全白）等，以增強系統對攝影機本身故障的感知能力。
-   **白名單系統 (Whitelist System)**: 開發一套白名單機制。當事件觸發時，可將偵測到的人物特徵與預先註冊的白名單特徵庫進行比對，若匹配成功則抑制警報通知，以過濾授權人員的正常活動。
-   **前端介面強化**: 擴充 Web 儀表板功能，例如增加事件篩選、排序，或引入更豐富的數據視覺化圖表，並提供 ROI/警戒線的視覺化設定介面。

## 開發模式說明

本專案的開發過程，是一次探索人類開發者與大型語言模型 (LLM) 協同作業的實踐。在此模式中，人類開發者的角色聚焦於定義高階目標、提供精確技術上下文、以及進行迭代式驗證與除錯，旨在將人類的策略性思考與 LLM 的高效程式碼生成能力相結合，探索一種現代化的軟體開發工作流程。

## License

本專案採用 [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html) 授權。
