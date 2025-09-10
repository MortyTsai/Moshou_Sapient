# database.py
import logging
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# 建立指向專案根目錄的路徑
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 定義資料庫檔案的完整路徑
DB_FILE = os.path.join(PROJECT_ROOT, "data", "security_events.db")

# 確保 data 資料夾存在
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_FILE}?check_same_thread=False"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"timeout": 15}
)

@event.listens_for(engine, "connect")
def set_wal_pragma_on_connect(dbapi_connection, _connection_record):
    """啟用 SQLite WAL 模式以支援高併發讀寫"""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL;")
        logging.info("資料庫連線已成功啟用 WAL 模式。")
    finally:
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    try:
        logging.info("正在初始化資料庫, 建立資料表...")
        Base.metadata.create_all(bind=engine)
        logging.info("資料庫資料表建立完成 (如果尚未存在)。")
    except Exception as e:
        logging.error(f"建立資料庫資料表時發生錯誤: {e}", exc_info=True)
        raise