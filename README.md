# MoshouSapient v7.2 - AI 智慧保全系統

本專案旨在建構一套基於 Windows 11 平台、利用 NVIDIA GPU 硬體加速能力的高效能智慧保全系統。系統整合了 YOLO11 物件辨識模型與 TensorRT 推論引擎，實現了具備事件上下文感知能力（事前錄影）的高效能、高畫質、高響應的即時監控與警報功能。

## 核心特性

- **模組化與可擴展架構**: 透過 `CameraWorker` 類別封裝單一攝影機的處理邏輯，為未來擴展奠定基礎。
- **高效能管線**: 採用「生產者-多消費者」的多執行緒模型，並透過 NVIDIA NVENC 硬體編碼實現高速影片處理。
- **即時物件追蹤**: 整合了 **BoT-SORT** 演算法，並透過客製化設定檔啟用 Re-ID 功能，提升了在單一連續畫面中對短時遮擋目標的追蹤穩定性。
- **事件持久化儲存**: 採用 SQLAlchemy ORM 與 SQLite 資料庫 (並啟用 WAL 模式以支援併發讀寫)。
- **Web 遠端儀表板**: 內建一個基於 Flask 的輕量級 Web 伺服器，用於遠端查看事件紀錄。
- **遠程事件通報**: 透過非同步的 Discord Bot 推送警報訊息與事件影片。

## 硬體與軟體需求

- **主機**: 一台運行 Windows 10/11 的個人電腦。
- **GPU**: 一張支援 NVENC 硬體編碼的 NVIDIA 顯示卡 (建議使用 GeForce RTX 系列)。
- **網路攝影機**: 一台支援 RTSP 協定的網路攝影機 (例如 Tapo C210)。
- **Python**: Python 3.11

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
    -   下載 `yolo11n.pt` 模型檔案。
    -   執行轉換腳本，生成 TensorRT 引擎：
    ```bash
    python export_tensorrt.py
    ```
    成功後會生成 `yolo11n.engine` 檔案。

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
    ```
2.  (可選) 專案包含一個 `custom_botsort.yaml` 檔案，您可以在其中微調追蹤演算法的相關參數。

## 執行與驗證

1.  啟動主程式：
    ```bash
    python main.py
    ```
2.  打開瀏覽器，訪問 Web 儀表板： `http://127.0.0.1:5000`
3.  觸發事件（例如，讓人物出現在攝影機畫面中），並檢查 Discord 是否收到通知。

## 開發模式說明

本專案採用了以 Google Gemini Pro 為核心的 AI 輔助開發模式。在此模式中，開發流程由人類專家設定目標與方向，並由 AI 負責具體的程式碼實現與分析。

開發者的主要職責是：
- **定義需求與設計架構**：提出專案的整體目標、功能規格與模組劃分。
- **提供關鍵上下文**：蒐集並提供相關技術的官方文件，作為 AI 生成程式碼的依據。
- **迭代式除錯與驗證**：在真實環境中測試程式碼，並將遇到的錯誤訊息、堆疊追蹤和非預期行為反饋給 AI，引導其進行修正。

這個專案的開發過程，是探索人類策略性指導與 AI 高效能程式碼生成能力相結合的一個實踐案例。