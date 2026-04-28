from pydantic import BaseModel
from typing import Union, List, TYPE_CHECKING
from enum import Enum
from .common import ValidationResult
if TYPE_CHECKING:
    from .tensor import TensorData


class ModuleType(str, Enum):
    linear = "linear"
    softmax = "softmax"
    identity = "identity"
    silu = "silu"
    sigmoid = "sigmoid"
    topk = "topk"
    mul = "mul"
    view = "view"
    flatten = "flatten"
    bincount = "bincount"
    unsqueeze = "unsqueeze"
    where = "where"
    index_add = "index_add"
    indexing = "indexing"
    sum = "sum"


class LinearParams(BaseModel):
    n_out: Union[int, str]


class ModuleCreate(BaseModel):
    type: ModuleType
    params: dict
    input_ids: List[str] = []


class ModuleUpdate(BaseModel):
    params: dict


class ModuleData(BaseModel):
    id: str
    type: ModuleType
    params: dict
    input_ids: List[str] = []


class ModuleResponse(BaseModel):
    module: ModuleData
    updated_tensors: List = []
    validation: ValidationResult
