# models.py
from __future__ import annotations
from typing import List
from sqlalchemy import DateTime, func, LargeBinary, ForeignKey, Integer, String
from sqlalchemy.orm import relationship, Mapped, mapped_column
from .database import Base

class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    event_type: Mapped[str] = mapped_column(String, default="person_detected")
    video_path: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, default="unreviewed")

    reid_features: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    person_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("persons.id"), nullable=True, index=True)
    person: Mapped[Person] = relationship(back_populates="events")

    def __repr__(self) -> str:
        return f"<Event(id={self.id}, person_id={self.person_id}, time='{self.timestamp}', path='{self.video_path}')>"


class PersonFeature(Base):
    __tablename__ = "person_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feature: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    person_id: Mapped[int] = mapped_column(Integer, ForeignKey("persons.id"), nullable=False, index=True)

    person: Mapped[Person] = relationship(back_populates="features")

    def __repr__(self) -> str:
        return f"<PersonFeature(id={self.id}, person_id={self.person_id})>"


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    first_seen: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_seen: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
                                                nullable=False)
    sighting_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    events: Mapped[List[Event]] = relationship(back_populates="person")
    features: Mapped[List[PersonFeature]] = relationship(back_populates="person", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Person(id={self.id}, last_seen='{self.last_seen}', sightings={self.sighting_count})>"