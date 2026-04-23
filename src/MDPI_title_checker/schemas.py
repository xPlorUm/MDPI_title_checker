from pydantic import BaseModel
from typing import List


class TitleRequest(BaseModel):
    reference: str
    other: list[str]

class TitleResponse(BaseModel):
    top_result: str
        