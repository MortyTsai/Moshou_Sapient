# models.py
from sqlalchemy import Column, Integer, String, DateTime, func, LargeBinary, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    event_type = Column(String, default="person_detected")
    video_path = Column(String, index=True)
    status = Column(String, default="unreviewed")
    reid_features = Column(LargeBinary, nullable=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True, index=True)
    person = relationship("Person")

    def __repr__(self):
        return f"<Event(id={self.id}, person_id={self.person_id}, time='{self.timestamp}', path='{self.video_path}')>"

class Person(Base):
    """
    代表一個被系統獨立識別的個體 (全域人物畫廊)。
    """
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True)
    first_seen = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_seen = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    representative_feature = Column(LargeBinary, nullable=False)
    sighting_count = Column(Integer, default=1, nullable=False)

    def __repr__(self):
        return f"<Person(id={self.id}, last_seen='{self.last_seen}', sightings={self.sighting_count})>"