# web_dashboard.py
import os
import logging
from flask import Flask, render_template, send_from_directory
from sqlalchemy import desc, exc

from database import SessionLocal
from models import Event
from config import Config


def create_flask_app():
    app = Flask(__name__)

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route('/captures/<path:filename>')
    def serve_capture(filename):
        return send_from_directory(Config.CAPTURES_DIR, filename)

    @app.route('/')
    def index():
        db = SessionLocal()
        try:
            events = db.query(Event).order_by(desc(Event.timestamp)).all()
            for event in events:
                event.video_filename = os.path.basename(event.video_path)
        except exc.SQLAlchemyError as e:
            logging.error(f"從資料庫讀取事件時發生錯誤: {e}")
            events = []
        finally:
            db.close()

        return render_template('index.html', events=events)

    return app