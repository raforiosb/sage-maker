from typing import Optional, List
from pydantic import BaseModel

class InvocationsRequest(BaseModel):
    slug: str
    slug_blocked: Optional[List[str]] = None
    topn: Optional[int] = 3
    lang: Optional[str] = "en"

class InvocationsResponse(BaseModel):
    # relates: List[tuple] = []
    relates: List[str] = []
