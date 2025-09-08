# models.py
from sqlalchemy import Column, Integer, String, DateTime, func, LargeBinary
from database import Base

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    event_type = Column(String, default="person_detected")
    video_path = Column(String, index=True)
    status = Column(String, default="unreviewed")
    reid_features = Column(LargeBinary, nullable=True)

    def __repr__(self):
        return f"<Event(id={self.id}, time='{self.timestamp}', path='{self.video_path}')>"