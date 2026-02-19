from pydantic import BaseModel, EmailStr, Field
from uuid import UUID
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    email: str
    display_name: str
    age: int = Field(ge=18, le=100)
    gender: str
    location: Optional[str] = None

class UserResponse(BaseModel):
    id: UUID
    email: str
    display_name: str
    age: int
    gender: str
    location: Optional[str]
    photos: list[str] = []
    created_at: datetime
    is_active: bool

    model_config = {"from_attributes": True}

class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    age: Optional[int] = Field(None, ge=18, le=100)
    gender: Optional[str] = None
    location: Optional[str] = None
