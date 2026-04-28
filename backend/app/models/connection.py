from pydantic import BaseModel
from typing import Optional, List
from .common import ValidationResult
from .tensor import TensorData


class ConnectionCreate(BaseModel):
    source_id: str
    target_id: str
    target_handle: Optional[str] = None


class ConnectionData(BaseModel):
    id: str
    source_id: str
    target_id: str
    target_handle: Optional[str] = None
    output_tensor_id: Optional[str] = None
    output_tensor_ids: List[str] = []  # multi-output modules (e.g. topk)


class ConnectionResponse(BaseModel):
    connection: ConnectionData
    output_tensor: Optional[TensorData] = None
    output_tensors: List[TensorData] = []  # multi-output modules (e.g. topk)
    validation: ValidationResult
