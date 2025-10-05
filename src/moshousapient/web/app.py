# src/moshousapient/web/app.py
"""
提供 Flask Web 應用程式，用於展示系統儀表板。
"""

# 1. 標準庫導入
import os
import logging

# 2. 第三方庫導入
from flask import Flask, render_template, send_from_directory
from sqlalchemy import desc, exc

# 3. 本專案相對導入
from ..services.database_service import SessionLocal
from ..services.database_models import Event
from ..configs.behavior_config import Config


def create_flask_app():
    """
    創建並配置 Flask 應用實例。
    """
    app = Flask(__name__)

    # 抑制 Werkzeug 的標準日誌輸出，以保持終端乾淨
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route('/captures/<path:filename>')
    def serve_capture(filename):
        """提供對錄製影片檔案的訪問。"""
        return send_from_directory(Config.CAPTURES_DIR, filename)

    @app.route('/')
    def index():
        """渲染主儀表板頁面。"""
        db = SessionLocal()
        events = []
        try:
            events = db.query(Event).order_by(desc(Event.timestamp)).all()
            for event in events:
                event.video_filename = os.path.basename(event.video_path)
        except exc.SQLAlchemyError as e:
            logging.error(f"從資料庫讀取事件時發生錯誤: {e}")
        finally:
            db.close()
        return render_template('index.html', events=events)

    return app