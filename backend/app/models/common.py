from pydantic import BaseModel
from typing import List, Union


class TensorShape(BaseModel):
    dims: List[Union[int, str]]

    @property
    def rank(self) -> int:
        return len(self.dims)


class ValidationError(BaseModel):
    code: str
    message: str
    offending_ids: List[str] = []


class ValidationResult(BaseModel):
    ok: bool
    errors: List[ValidationError] = []
