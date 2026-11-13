"""SQL metadata store (SQLite by default, any SQLAlchemy URL works)."""
import uuid
from collections.abc import Iterator
import datetime as dt

from sqlalchemy import Date, DateTime, String, Time, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import get_settings


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int]
    chunking_strategy: Mapped[str] = mapped_column(String(20))
    num_chunks: Mapped[int]
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Booking(Base):
    __tablename__ = "bookings"
    # The same slot cannot be booked twice.
    __table_args__ = (UniqueConstraint("date", "time", name="uq_booking_slot"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(320))
    date: Mapped[dt.date] = mapped_column(Date)
    time: Mapped[dt.time] = mapped_column(Time)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


_url = get_settings().database_url
engine = create_engine(_url, connect_args={"check_same_thread": False} if _url.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session
