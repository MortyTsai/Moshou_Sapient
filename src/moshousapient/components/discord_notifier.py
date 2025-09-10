# discord_notifier.py
import discord
import asyncio
import threading
import os
import logging

class DiscordNotifier:
    """
    負責與 Discord Bot 互動, 發送通知與影片檔案。
    (版本 4.4 - 最終執行緒同步修正)
    """

    def __init__(self, token, channel_id):
        self.token = token
        self.channel_id = channel_id
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.client = discord.Client(intents=self.intents)
        self.loop = None
        self.thread = None
        self.channel = None
        self._pending_tasks = []
        self._lock = threading.Lock()
        self._is_stopping = False

        @self.client.event
        async def on_ready():
            logging.info(f'[Discord] 已登入為 {self.client.user}')
            self.channel = self.client.get_channel(self.channel_id)
            if self.channel:
                logging.info(f'[Discord] 已成功連接至頻道: {self.channel.name}')
            else:
                logging.error(f'[Discord] 錯誤: 找不到頻道 ID: {self.channel_id}')

    def start(self):
        discord_logger = logging.getLogger('discord')
        discord_logger.handlers.clear()
        discord_logger.propagate = False
        self.thread = threading.Thread(target=self.run_bot, name="DiscordBotThread", daemon=True)
        self.thread.start()
        logging.info("Discord Bot: 執行緒已啟動。")

    def run_bot(self):
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.client.start(self.token))
        except Exception as e:
            if "Login failure" not in str(e):
                logging.error(f"[Discord] Bot 執行時發生錯誤: {e}", exc_info=True)
        finally:
            logging.info("[Discord] Bot 事件迴圈已停止。")

    async def _send_notification(self, message, file_path=None):
        if not self.channel:
            logging.error("[Discord] 錯誤: 頻道尚未準備就緒。")
            return
        try:
            dfile = discord.File(file_path) if file_path and os.path.exists(file_path) else None
            await self.channel.send(message, file=dfile)
            logging.info(f"[Discord] 已將通知發送至 {self.channel.name}")
        except Exception as e:
            logging.error(f"[Discord] 錯誤: 發送通知時發生錯誤: {e}", exc_info=True)

    def schedule_notification(self, message, file_path=None):
        if self._is_stopping:
            logging.warning("[Discord] Bot 正在關閉, 已拒絕新的通知任務。")
            return
        if self.client.is_ready() and self.loop and self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._send_notification(message, file_path), self.loop)
            with self._lock:
                self._pending_tasks.append(future)
        else:
            logging.warning("[Discord] Bot 尚未就緒或事件迴圈未執行, 無法發送通知。")

    def stop(self):
        logging.info("[Discord] 正在優雅地關閉 Bot...")
        self._is_stopping = True
        try:
            with self._lock:
                tasks_to_wait = list(self._pending_tasks)

            if tasks_to_wait:
                logging.info(f"[Discord] 等待 {len(tasks_to_wait)} 個待發送的通知完成...")
                for future in tasks_to_wait:
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"[Discord] 等待通知完成時發生錯誤: {e}")
                logging.info("[Discord] 所有待發送通知已處理完畢。")

            if self.client.is_ready() and self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.client.close(), self.loop)
                logging.info("[Discord] 已發送登出請求。")

            if self.thread and self.thread.is_alive():
                logging.info("[Discord] 等待 Bot 執行緒完全終止...")
                self.thread.join(timeout=30)
                if self.thread.is_alive():
                    logging.warning("[Discord] Bot 執行緒關閉超時。")
                else:
                    logging.info("[Discord] Bot 執行緒已成功終止。")

        except Exception as e:
            logging.error(f"[Discord] 關閉 Bot 時發生未預期的錯誤: {e}", exc_info=True)