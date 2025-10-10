# src/moshousapient/services/notification_service.py
"""
提供通知服務，目前透過 Discord Bot 實現。
此模組封裝了與 Discord API 的所有互動，提供了一個簡單的介面來發送訊息和檔案。
"""

import discord
import asyncio
import threading
import os
import logging
from typing import List, Optional
from concurrent.futures import Future


class NotificationService:
    """
    負責與 Discord Bot 互動，發送通知與影片檔案。
    """

    def __init__(self, token: str, channel_id: int):
        """
        初始化通知服務。

        :param token: Discord Bot 的 token。
        :param channel_id: 要發送通知的目標頻道 ID。
        """
        self.token = token
        self.channel_id = channel_id
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.client = discord.Client(intents=self.intents)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.channel: Optional[discord.TextChannel] = None
        self._pending_tasks: List[Future] = []
        self._lock = threading.Lock()
        self._is_stopping = False

        @self.client.event
        async def on_ready():
            logging.debug(f"[Discord] 已登入為 {self.client.user}")
            self.channel = self.client.get_channel(self.channel_id)
            if self.channel:
                # 將此處的 INFO 降級為 DEBUG
                logging.debug(f"[Discord] 已成功連接至頻道: {self.channel.name}")
            else:
                logging.error(f"[Discord] 錯誤: 找不到頻道 ID: {self.channel_id}")

    def start(self):
        """在一個獨立的執行緒中啟動 Discord Bot。"""
        discord_logger = logging.getLogger("discord")
        discord_logger.setLevel(logging.WARNING)
        self.thread = threading.Thread(target=self._run_bot, name="DiscordBotThread", daemon=True)
        self.thread.start()
        logging.debug("Discord Bot: 執行緒已啟動。")

    def _run_bot(self):
        """Bot 的主事件迴圈。"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.client.start(self.token))
        except Exception as e:
            if "Login failure" not in str(e):
                logging.error(f"[Discord] Bot 執行時發生錯誤: {e}", exc_info=True)
        finally:
            logging.debug("[Discord] Bot 事件迴圈已停止。")

    async def _send_notification(self, message: str, file_path: str = None):
        """【協程】實際發送通知到 Discord。"""
        if not self.channel:
            logging.error("[Discord] 錯誤: 頻道尚未準備就緒。")
            return
        try:
            dfile = discord.File(file_path) if file_path and os.path.exists(file_path) else None
            await self.channel.send(message, file=dfile)
            logging.info(f"[Discord] 已成功將通知發送至 {self.channel.name}")
        except Exception as e:
            logging.error(f"[Discord] 錯誤: 發送通知時發生錯誤: {e}", exc_info=True)

    def schedule_notification(self, message: str, file_path: str = None):
        """從任何執行緒安全地排程一個通知發送任務。"""
        if self._is_stopping:
            logging.warning("[Discord] Bot 正在關閉，已拒絕新的通知任務。")
            return

        if self.client.is_ready() and self.loop and self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._send_notification(message, file_path), self.loop)
            with self._lock:
                self._pending_tasks.append(future)
        else:
            logging.warning("[Discord] Bot 尚未就緒或事件迴圈未運行，無法發送通知。")

    def get_pending_tasks_count(self) -> int:
        """
        安全地獲取待處理的通知任務數量。
        :return: 待處理任務的數量。
        """
        with self._lock:
            return len(self._pending_tasks)

    def stop(self):
        """優雅地關閉 Discord Bot 連線。"""
        if self._is_stopping:
            return
        logging.debug("[Discord] 正在優雅地關閉 Bot...")
        self._is_stopping = True

        tasks_to_wait = []
        with self._lock:
            tasks_to_wait = list(self._pending_tasks)

        if tasks_to_wait:
            logging.debug(f"[Discord] 等待 {len(tasks_to_wait)} 個待發送的通知完成...")
            for future in tasks_to_wait:
                try:
                    future.result(timeout=10)
                except Exception as e:
                    logging.error(f"[Discord] 等待通知完成時發生錯誤: {e}")
            logging.debug("[Discord] 所有待發送通知已處理完畢。")

        if self.client.is_ready() and self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.client.close(), self.loop)
            logging.debug("[Discord] 已發送登出請求。")

        if self.thread and self.thread.is_alive():
            logging.debug("[Discord] 等待 Bot 執行緒完全終止...")
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                logging.warning("[Discord] Bot 執行緒關閉超時。")
            else:
                logging.debug("[Discord] Bot 執行緒已成功終止。")
