from pydantic import BaseModel
from typing import List, Optional
from .common import ValidationResult


class GroupCreate(BaseModel):
    name: str
    member_ids: List[str] = []


class GroupUpdate(BaseModel):
    name: Optional[str] = None
    member_ids: Optional[List[str]] = None


class GroupData(BaseModel):
    id: str
    name: str
    member_ids: List[str] = []


class GroupResponse(BaseModel):
    group: GroupData
    validation: ValidationResult
