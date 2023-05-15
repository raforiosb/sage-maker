from pydantic import BaseModel
from typing import Optional

class Blog(BaseModel):
    ID: int
    post_title:  str
    post_content : str
    post_name: str
    lang : Optional[str] = None

    # Activate orm model (object read model)
    class Config:
        orm_mode = True

class BasePost(BaseModel):
    post_name: str
    post_title: str
    post_excerpt: str
    class Config:
        orm_mode = True
        
class Service(BasePost):
    pass

class Industrie(BasePost):
    pass