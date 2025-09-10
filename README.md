# MoshouSapient: AI 即時影像監控與分析系統

![alt text](https://img.shields.io/badge/status-work%20in%20progress-yellow) ![Python Version](https://img.shields.io/badge/python-3.11-blue)    ![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)

本專案是一個基於 Python 與 NVIDIA TensorRT 的高效能智慧保全系統，旨在探索即時影像處理、物件追蹤與人物重識別 (Re-ID) 技術的整合應用。系統能夠處理 RTSP 影像流，進行即時物件偵測，並在觸發特定事件時，提取人物外觀特徵向量進行持久化儲存，為後續的跨攝影機追蹤與資料檢索奠定基礎。

![](assets/demo_1.gif)   ![](assets/demo_3.gif)

## 專案狀態

:construction: **本專案為學習與實踐導向，仍在持續開發中。**

目前已完成核心的 Re-ID 功能與基礎架構重構，並成功實作了第一個高階行為分析模組。系統架構已為未來的功能擴展 (如多攝影機支援) 預留了空間。歡迎任何形式的建議與討論。

## 核心特性

-   **高效能推論管線**: 整合 **YOLO11** 與 **NVIDIA TensorRT** 引擎進行物件偵測，並利用 **NVENC** 硬體編碼加速影片處理，確保低延遲。
-   **穩定的物件追蹤**: 採用 **BOTSORT** 追蹤器，並將偵測與特徵提取流程解耦，手動管理追蹤器生命週期，提升管線的穩定性與可控性。
-   **長時人物重識別 (Long-Term Re-ID)**:
    -   **特徵集畫廊 (Gallery of Feature Sets)**: 建立「全域人物畫廊」資料庫，為每個人物維護一個由多個特徵向量組成的集合，而非單一代表性特徵。
    -   **魯棒的匹配邏輯**: 採用「穩定代表元聚類」演算法，能準確地在單一複雜事件中區分多個獨立人物，並透過「全域校準」將其與歷史資料庫進行比對，極大提升了在姿態、光照變化下的識別準確率。
-   **高階行為分析**:
    -   **區域闖入與停留偵測 (ROI Dwell Time)**: 支援自訂多邊形感興趣區域 (ROI)，能夠即時偵測目標是否進入特定區域，並在停留時間超過預設閾值時觸發獨立的 `'dwell_alert'` 事件。
    -   **事件切分機制**: 透過設定最大錄影時長，系統能夠在持續活動的場景中自動切分事件，確保長時間的活動能被記錄為多個獨立、可管理的事件片段。
-   **事件驅動的特徵持久化**: 系統能在偵測到人物或異常行為時觸發事件，並提取 Re-ID 特徵向量，使用 SQLAlchemy ORM 與 SQLite (WAL 模式) 將其與事件元數據一同高效存入資料庫。
-   **模組化與可擴展架構**: 採用標準化的 `src` 佈局，將所有原始碼封裝在一個可安裝的套件中。以 `CameraWorker` 類別封裝單一攝影機的處理邏輯，具備良好的可擴展性。
-   **遠端存取與可選通知**:
    -   內建基於 **Flask** 的輕量級 Web 儀表板，用於遠端查看事件紀錄與回放。
    -   透過 **Discord Bot** 非同步推送即時警報，此功能可透過設定檔完全啟用或禁用。

## 技術棧

-   **核心框架**: Python 3.11
-   **AI / CV**: PyTorch, TensorRT, Ultralytics YOLO, BOTSORT, Shapely (幾何分析)
-   **資料庫**: SQLite, SQLAlchemy (ORM)
-   **Web 後端**: Flask
-   **影像處理**: FFmpeg, OpenCV-Python
-   **其他**: python-dotenv, PyYAML

## 系統檔案結構
```
MoshouSapient/ # 專案根目錄
│
├── .env.example # 環境變數設定檔範本
├── .gitignore # Git 版本控制忽略清單
├── README.md # 專案說明文件
├── requirements.txt # Python 依賴套件列表
│
├── configs/ # 存放所有靜態設定檔
│ └── custom_botsort.yaml # BoT-SORT 追蹤器客製化參數
│
├── data/ # (動態生成) 存放執行時資料
│ ├── captures/ # (動態生成) 存放事件錄影
│ └── security_events.db # (動態生成) SQLite 資料庫檔案
│
├── models/ # 存放所有 AI 模型資產
│ ├── yolo11s.pt # (需下載) PyTorch 格式的偵測模型
│ ├── yolo11s-cls.pt # (需下載) PyTorch 格式的 Re-ID 模型
│ └── yolo11s.engine # (動態生成) TensorRT 格式的偵測模型
│
├── scripts/ # 存放輔助開發腳本
│ └── export_tensorrt.py # 模型轉換為 TensorRT 引擎的腳本
│
└── src/ # 存放所有專案原始碼
└── moshousapient/ # 專案主 Python 套件
├── init.py # 將目錄標記為 Python 套件
├── main.py # 套件執行入口 (python -m moshousapient)
│
├── components/ # 核心功能元件子套件
│ ├── init.py # 將目錄標記為 Python 子套件
│ ├── camera_worker.py # 管理單一攝影機的核心類別
│ ├── discord_notifier.py # Discord Bot 通知模組
│ ├── event_processor.py # 核心 AI 處理管線 (偵測、追蹤、Re-ID、行為分析)
│ ├── runners.py # 執行策略模組 (RTSP/File 模式)
│ └── video_streamer.py # FFmpeg 影像流讀取模組
│
├── utils/ # 通用工具函式子套件
│ ├── init.py # 將目錄標記為 Python 子套件
│ ├── reid_utils.py # Re-ID 相關工具函式
│ └── video_utils.py # 影片元數據讀取工具
│
├── web/ # Web 儀表板子套件
│ ├── init.py # 將目錄標記為 Python 子套件
│ ├── app.py # Flask 應用程式與路由定義
│ └── templates/ # Web 儀表板的 HTML 樣板
│ └── index.html # 儀表板主頁面樣板
│
├── config.py # 中央設定類別與路徑管理
├── database.py # SQLAlchemy 資料庫初始化與 Session 管理
├── logging_setup.py # 全域日誌 (Logging) 設定模組
├── main.py # 專案主程式邏輯
└── models.py # 資料庫 ORM 模型定義
```

## 環境準備

### 硬體與軟體需求
-   **作業系統**: Windows 10 / 11
-   **GPU**: 支援 NVENC 硬體編碼的 NVIDIA 顯示卡 (建議 GeForce RTX 系列)
-   **攝影機**: 支援 RTSP 協定的網路攝影機
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
    -   將 `yolo11s.pt` (物件偵測) 和 `yolo11s-cls.pt` (Re-ID) 模型檔案放置在 `models/` 資料夾中。
    -   執行轉換腳本，將**偵測模型**生成為 TensorRT 引擎：
        ```bash
        python scripts/export_tensorrt.py
        ```
    -   成功後會在 `models/` 資料夾下生成 `yolo11s.engine` 檔案。

## 專案設定與執行

1.  **設定環境變數**:
    在專案根目錄下，將 `.env.example` 複製一份並重新命名為 `.env`。然後根據您的需求填寫設定：

    ```env
    # .env

    # Discord Bot 功能總開關 (True/False)
    DISCORD_ENABLED=False

    # Discord Bot Credentials (僅在 DISCORD_ENABLED=True 時需要)
    DISCORD_TOKEN="YourDiscordBotTokenHere"
    DISCORD_CHANNEL_ID="YourChannelIDHere"

    # 影像來源類型: "RTSP" 或 "FILE"
    VIDEO_SOURCE_TYPE="RTSP"

    # RTSP 模式所需憑證
    # 請在此填入您攝影機的完整 RTSP URL。
    RTSP_URL="rtsp://YourCameraUsername:YourCameraPassword@YourCameraIPAddress:554/stream1"

    # FILE 模式所需路徑 (相對於專案根目錄)
    VIDEO_FILE_PATH="videos/input.mp4"
    ```

2.  **微調追蹤器 (可選)**:
    您可以在 `configs/custom_botsort.yaml` 檔案中微調追蹤演算法的相關參數。

3.  **啟動系統**:
    在專案**根目錄**下，執行以下指令：
    ```bash
    python -m moshousapient
    ```

4.  **驗證**:
    -   打開瀏覽器，訪問 Web 儀表板： `http://127.0.0.1:5000`
    -   觸發事件（例如，讓人物出現在攝影機畫面中，或使用包含人物的影片檔案）。
    -   如果啟用了 Discord，檢查是否收到通知。
    -   檢查 Web 儀表板是否出現新的事件紀錄。

## 未來可能的發展方向

作為一個學習與探索性質的專案，以下是一些未來可能的研究與開發方向。這些並非確定的開發計畫，而是基於現有架構的潛在擴展思路：

-   **行為分析與異常偵測**: 在已實現的 ROI 停留偵測基礎上，繼續開發如「虛擬警戒線跨越偵測」、「方向判斷」等更複雜的分析模組。
-   **進階資料庫查詢**: 探索基於 Re-ID 特徵向量的相似度搜尋，以實現特定人物的歷史事件檢索（例如「顯示這個人今天所有出現過的片段」）。
-   **前端介面強化**: 擴充 Web 儀表板功能，例如增加事件篩選、排序，或引入更豐富的數據視覺化圖表。
-   **多攝影機協同**: 將現有的單攝影機 `CameraWorker` 架構擴展，使其能夠由一個主程序同時管理多個獨立的攝影機，並利用特徵集畫廊實現跨攝影機的目標重識別。
-   **模型管理與抽象化**: 將模型載入與設定的邏輯抽象化，讓使用者可以更容易地透過設定檔替換不同的偵測或 Re-ID 模型。

## 開發模式說明

本專案的開發過程，是一次探索人類開發者與大型語言模型 (LLM, 如 Google Gemini Pro) 協同作業的實踐。

在此模式中，人類開發者的角色聚焦於：
-   **定義高階目標與架構設計**: 提出專案的整體目標、功能規格與模組劃分。
-   **提供精確的技術上下文**: 蒐集並提供關鍵技術的官方文件或 API 規格，作為 LLM 生成程式碼的依據。
-   **進行迭代式驗證與除錯**: 在真實環境中測試程式碼，並將錯誤訊息、堆疊追蹤和非預期行為，以結構化的方式反饋給 LLM，引導其進行修正與優化。

這個流程旨在將人類的策略性思考、領域知識與 LLM 的高效程式碼生成能力相結合，探索一種現代化的軟體開發工作流程。

## License

本專案採用 [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html) 授權。```
