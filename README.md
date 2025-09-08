# MoshouSapient: AI 即時影像監控與分析系統

![Project Status: WIP](https://img.shields.io/badge/status-work%20in%20progress-yellow) ![Python Version](https://img.shields.io/badge/python-3.11-blue)    ![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)

本專案是一個基於 Python 與 NVIDIA TensorRT 的高效能智慧保全系統，旨在探索即時影像處理、物件追蹤與人物重識別 (Re-ID) 技術的整合應用。系統能夠處理 RTSP 影像流，進行即時物件偵測，並在觸發特定事件時，提取人物外觀特徵向量進行持久化儲存，為後續的跨攝影機追蹤與資料檢索奠定基礎。

## 專案狀態

:construction: **本專案為學習與實踐導向，仍在持續開發中。**

目前已完成核心功能的 PoC (概念驗證)，包括單一攝影機的影像處理管線、事件偵測、特徵提取與資料庫儲存。系統架構已為未來的功能擴展 (如多攝影機支援) 預留了空間。歡迎任何形式的建議與討論。

## 核心特性

-   **高效能推論管線**: 整合 **YOLO11s** 與 **NVIDIA TensorRT** 引擎進行物件偵測，並利用 **NVENC** 硬體編碼加速影片處理，確保低延遲。
-   **穩定的物件追蹤**: 採用 **BOTSORT** 追蹤器，並將偵測與特徵提取流程解耦，手動管理追蹤器生命週期，提升管線的穩定性與可控性。
-   **事件驅動的特徵持久化**: 系統能在偵測到人物時觸發事件，並提取 Re-ID 特徵向量，使用 SQLAlchemy ORM 與 SQLite (WAL 模式) 將其與事件元數據一同高效存入資料庫。
-   **即時性能調優**: 透過對處理管線的延遲分析，實施了 **Re-ID 節流 (Throttling)** 策略，有效將事件處理的端到端延遲控制在即時預算內，確保追蹤器穩定運作。
-   **模組化架構**: 以 `CameraWorker` 類別封裝單一攝影機的處理邏輯 (影像擷取、AI處理、事件錄影)，具備良好的可擴展性。
-   **遠端存取與通知**:
    -   內建基於 **Flask** 的輕量級 Web 儀表板，用於遠端查看事件紀錄與回放。
    -   透過 **Discord Bot** 非同步推送即時警報訊息與事件影片。

## 技術棧

-   **核心框架**: Python 3.11
-   **AI / CV**: PyTorch, TensorRT, Ultralytics YOLO, BOTSORT
-   **資料庫**: SQLite, SQLAlchemy (ORM)
-   **Web 後端**: Flask
-   **影像處理**: FFmpeg, OpenCV-Python
-   **其他**: aiohttp (非同步網路請求)

## 系統檔案結構

```
MoshouSapient/
│
├── .env                    # 環境變數設定檔 (需手動建立)
├── .gitignore              # Git 版本控制忽略清單
├── config.py               # 中央設定檔，管理所有參數與環境變數
├── custom_botsort.yaml     # BoT-SORT 追蹤器客製化參數
├── database.py             # SQLAlchemy 資料庫初始化與 Session 管理
├── export_tensorrt.py      # YOLO 模型轉換為 TensorRT 引擎的腳本
├── logging_setup.py        # 全域日誌 (Logging) 設定模組
├── main.py                 # 專案主程式入口
├── models.py               # 資料庫 ORM 模型定義 (Event 表)
├── requirements.txt        # Python 依賴套件列表
├── web_dashboard.py        # Flask Web 應用程式與路由定義
├── yolo11s.engine          # (生成) TensorRT 格式的偵測模型
├── yolo11s.pt              # (需下載) PyTorch 格式的偵測模型
├── yolo11s-cls.pt          # (需下載) PyTorch 格式的 Re-ID 模型
│
├── components/             # 核心功能元件
│   ├── camera_worker.py    # 核心類別，管理單一攝影機的所有執行緒與資源
│   ├── discord_notifier.py # Discord Bot 通知模組
│   ├── event_processor.py  # 核心 AI 處理管線 (偵測、追蹤、Re-ID)
│   ├── reid_utils.py       # Re-ID 相關工具函式 (例如餘弦相似度計算)
│   └── video_streamer.py   # 使用 FFmpeg 進行 RTSP 影像流讀取的生產者模組
│
├── templates/              # Web 儀表板的 HTML 樣板
│   └── index.html
│
├── captures/               # (動態生成) 儲存事件錄影的資料夾
│
└── venv/                   # Python 虛擬環境
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
    -   CUDA Toolkit (建議版本 12.9 或更高)
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
    # 例如 CUDA 12.9:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

    # 2. 安裝 TensorRT Python wheel 檔案 (請務必替換為您自己的實際路徑)
    pip install "C:\Path\To\Your\TensorRT-version\python\tensorrt-version.whl"

    # 3. 安裝其餘依賴
    pip install -r requirements.txt
    ```

5.  **準備 AI 模型**:
    -   下載 `yolo11s.pt` (物件偵測) 和 `yolo11s-cls.pt` (Re-ID) 模型檔案至專案根目錄。
    -   執行轉換腳本，將**偵測模型**生成為 TensorRT 引擎：
        ```bash
        python export_tensorrt.py
        ```
    -   成功後會生成 `yolo11s.engine` 檔案。Re-ID 模型將維持以 PyTorch 格式載入。

## 專案設定與執行

1.  **設定環境變數**:
    在專案根目錄下建立一個 `.env` 檔案，並填入您的憑證：
    ```env
    # Discord Bot Credentials
    DISCORD_TOKEN="YourDiscordBotTokenHere"
    DISCORD_CHANNEL_ID="YourChannelIDHere"

    # Camera Credentials
    TAPO_IP="YourCameraIPAddress"
    TAPO_USER="YourCameraUsername"
    TAPO_PASS="YourCameraPassword"
    ```

2.  **微調追蹤器 (可選)**:
    專案包含 `custom_botsort.yaml` 檔案，您可以在其中微調追蹤演算法的相關參數。

3.  **啟動系統**:
    ```bash
    python main.py
    ```

4.  **驗證**:
    -   打開瀏覽器，訪問 Web 儀表板： `http://127.0.0.1:5000`
    -   觸發事件（例如，讓人物出現在攝影機畫面中）。
    -   檢查 Discord 是否收到通知，以及 Web 儀表板是否出現新的事件紀錄。

## 未來可能的發展方向

作為一個學習與探索性質的專案，以下是一些未來可能的研究與開發方向。這些並非確定的開發計畫，而是基於現有架構的潛在擴展思路：

-   **進階資料庫查詢**: 探索基於 Re-ID 特徵向量的相似度搜尋，以實現特定人物的歷史事件檢索。
-   **前端介面強化**: 擴充 Web 儀表板功能，例如增加事件篩選、排序，或引入更豐富的數據視覺化圖表。
-   **多攝影機支援**: 將現有的單攝影機 `CameraWorker` 架構擴展，使其能夠由一個主程序同時管理多個獨立的攝影機影像來源。
-   **模型管理與抽象化**: 將模型載入與設定的邏輯抽象化，讓使用者可以更容易地透過設定檔替換不同的偵測或 Re-ID 模型。
-   **系統健壯性提升**: 引入更結構化的日誌系統，並針對長時間運行的穩定性進行優化。

## 開發模式說明

本專案的開發過程，是一次探索人類開發者與大型語言模型 (LLM, 如 Google Gemini Pro) 協同作業的實踐。

在此模式中，人類開發者的角色聚焦於：
-   **定義高階目標與架構設計**: 提出專案的整體目標、功能規格與模組劃分。
-   **提供精確的技術上下文**: 蒐集並提供關鍵技術的官方文件或 API 規格，作為 LLM 生成程式碼的依據。
-   **進行迭代式驗證與除錯**: 在真實環境中測試程式碼，並將錯誤訊息、堆疊追蹤和非預期行為，以結構化的方式反饋給 LLM，引導其進行修正與優化。

這個流程旨在將人類的策略性思考、領域知識與 LLM 的高效程式碼生成能力相結合，探索一種現代化的軟體開發工作流程。

## License

本專案採用 [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html) 授權。