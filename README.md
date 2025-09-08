# MoshouSapient v7.4 - AI 智慧保全系統 (Re-ID 持久化與性能優化版)

本專案旨在建構一套基於 Windows 11 平台、利用 NVIDIA GPU 硬體加速能力的高效能智慧保全系統。系統整合了 YOLO11s 物件辨識模型與 TensorRT 推論引擎，實現了具備事件上下文感知能力（事前錄影）的高效能、高畫質、高響應的即時監控與警報功能，並進一步整合了人物重識別 (Re-ID) 特徵提取與持久化功能，為實現長時距追蹤奠定基礎。

## 核心特性

-   **模組化與可擴展架構**: 透過 `CameraWorker` 類別封裝單一攝影機的處理邏輯，為未來擴展多攝影機協同作業奠定基礎。
-   **高效能管線**: 採用「生產者-多消費者」的多執行緒模型，並透過 NVIDIA NVENC 硬體編碼實現高速影片處理。
-   **解耦式物件追蹤**: 整合了 **BOTSORT** 追蹤器。系統採用**偵測與特徵提取分離**的架構，手動管理追蹤器生命週期，確保了處理管線的穩定性與可控性。
-   **事件與特徵持久化**: 採用 SQLAlchemy ORM 與 SQLite 資料庫 (啟用 WAL 模式)，並擴展了資料庫結構，能夠將事件觸發時捕獲的人物**外觀特徵向量 (Re-ID Features)** 以 `BLOB` 格式進行持久化儲存。
-   **即時性能優化**：透過對處理管線進行細粒度延遲分析，定位了性能瓶頸，並實施了 **Re-ID 節流 (Throttling)** 策略，將事件處理期間的端到端延遲穩定在即時預算內，顯著提升了追蹤器的穩定性。
-   **Web 遠端儀表板**: 內建一個基於 Flask 的輕量級 Web 伺服器，用於遠端查看事件紀錄與回放影片。
-   **遠程事件通報**: 透過非同步的 Discord Bot 推送警報訊息與事件影片。

## 系統檔案結構

```
MoshouSapient/
│
├── .env                    # 環境變數設定檔 (需手動建立)
├── custom_botsort.yaml     # BoT-SORT 追蹤器客製化參數
├── database.py             # SQLAlchemy 資料庫初始化與 Session 管理
├── export_tensorrt.py      # YOLO 模型轉換為 TensorRT 引擎的腳本
├── main.py                 # 專案主程式入口
├── models.py               # 資料庫 ORM 模型定義 (Event 表)
├── requirements.txt        # Python 依賴套件列表
├── yolo11s.engine          # (生成) TensorRT 格式的偵測模型
├── yolo11s.pt              # (需下載) PyTorch 格式的偵測模型
├── yolo11s-cls.pt          # (需下載) PyTorch 格式的 Re-ID 模型
│
├── components/
│   ├── __init__.py
│   ├── camera_worker.py    # 核心類別，管理單一攝影機的所有執行緒與資源
│   ├── discord_notifier.py # Discord Bot 通知模組
│   ├── event_processor.py  # 核心 AI 處理管線 (偵測、追蹤、Re-ID)
│   ├── reid_utils.py       # Re-ID 相關工具函式 (例如餘弦相似度計算)
│   └── video_streamer.py   # 使用 FFmpeg 進行 RTSP 影像流讀取的生產者模組
│
├── templates/
│   └── index.html          # Flask Web 儀表板的 HTML 樣板
│
└── venv/                   # Python 虛擬環境
```

## 硬體與軟體需求

-   **主機**: 一台運行 Windows 10/11 的個人電腦。
-   **GPU**: 一張支援 NVENC 硬體編碼的 NVIDIA 顯示卡 (建議使用 GeForce RTX 系列)。
-   **網路攝影機**: 一台支援 RTSP 協定的網路攝影機 (例如 Tapo C210)。
-   **Python**: Python 3.11

## 環境設定與部署步驟

1.  **安裝 NVIDIA 工具鏈**:
    -   NVIDIA 驅動程式
    -   CUDA Toolkit (建議版本 12.9 或更高)
    -   cuDNN (需對應 CUDA 版本)
    -   TensorRT (需對應 CUDA 版本)

2.  **安裝核心工具**:
    -   Python 3.11 (安裝時建議勾選 "Add Python to PATH")
    -   FFmpeg

3.  **建立虛擬環境**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **安裝 Python 依賴**:
    ```bash
    # 1. 安裝 PyTorch (GPU 版本，請根據您的 CUDA 版本從官網選擇對應指令)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

    # 2. 安裝 TensorRT Python 模組 (請務必替換為您自己的實際路徑)
    pip install "C:\Path\To\Your\TensorRT\python\tensorrt-version.whl"

    # 3. 安裝其餘依賴
    pip install -r requirements.txt
    ```

5.  **準備 AI 模型**:
    -   下載 `yolo11s.pt` (物件偵測) 和 `yolo11s-cls.pt` (Re-ID) 兩個模型檔案至專案根目錄。
    -   執行轉換腳本，僅將**偵測模型**生成為 TensorRT 引擎：
        ```bash
        python export_tensorrt.py
        ```
    -   成功後會生成 `yolo11s.engine` 檔案。Re-ID 模型將直接以 PyTorch 格式載入。

## 專案設定

1.  在專案根目錄下建立一個 `.env` 檔案，並填入您的憑證：
    ```env
    # Discord Bot Credentials
    DISCORD_TOKEN=YourDiscordBotTokenHere
    DISCORD_CHANNEL_ID=YourChannelIDHere

    # Tapo Camera Credentials
    TAPO_IP=YourCameraIPAddress
    TAPO_USER=YourCameraUsername
    TAPO_PASS=YourCameraPassword
    
2.  專案包含一個 `custom_botsort.yaml` 檔案，用於初始化手動管理的追蹤器實例。您可以在其中微調追蹤演算法的相關參數。

## 執行與驗證

1.  啟動主程式：
    ```bash
    python main.py
    ```
2.  打開瀏覽器，訪問 Web 儀表板： `http://127.0.0.1:5000`
3.  觸發事件（例如，讓人物出現在攝影機畫面中），並檢查 Discord 是否收到通知，以及 Web 儀表板是否出現新的事件紀錄。

## 開發模式說明

本專案採用了以 Google Gemini Pro 為核心的 AI 輔助開發模式。在此模式中，開發流程由人類專家設定目標與方向，並由 AI 負責具體的程式碼實現與分析。

開發者的主要職責是：
-   **定義需求與設計架構**：提出專案的整體目標、功能規格與模組劃分。
-   **提供關鍵上下文**：蒐集並提供相關技術的官方文件，作為 AI 生成程式碼的依據。
-   **迭代式除錯與驗證**：在真實環境中測試程式碼，並將遇到的錯誤訊息、堆疊追蹤和非預期行為反饋給 AI，引導其進行修正。

這個專案的開發過程，是探索人類策略性指導與 AI 高效能程式碼生成能力相結合的一個實踐案例。