"""Interview booking validation and persistence."""
import datetime as dt

from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.models import Booking


class BookingRequest(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    email: EmailStr
    date: dt.date
    time: dt.time

    @field_validator("date")
    @classmethod
    def not_in_past(cls, v: dt.date) -> dt.date:
        if v < dt.date.today():
            raise ValueError("interview date cannot be in the past")
        return v


class SlotTakenError(Exception):
    pass


def create_booking(db: Session, session_id: str, req: BookingRequest) -> Booking:
    booking = Booking(session_id=session_id, **req.model_dump())
    db.add(booking)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise SlotTakenError(f"{req.date} {req.time:%H:%M} is already booked") from e
    return booking
