from pydantic import BaseModel
from typing import Optional, List
from .common import TensorShape, ValidationResult


class TensorCreate(BaseModel):
    shape: TensorShape
    dtype: str = "float32"


class TensorUpdate(BaseModel):
    shape: Optional[TensorShape] = None
    dtype: Optional[str] = None


class TensorData(BaseModel):
    id: str
    shape: TensorShape
    dtype: str


class TensorResponse(BaseModel):
    tensor: TensorData
    updated_tensors: List["TensorData"] = []
    validation: ValidationResult
