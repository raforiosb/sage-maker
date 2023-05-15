from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class Post(BaseModel):
    id: int
    slug: str
    link: str
    title: str
    post_modified: str
    post_date: str
    author: str
    industry: str
    content_type: str
    image_alt: str
    image: str


class Paging(BaseModel):
    total_count: int
    total_pages: int
    current_page: int
    per_page: int


class SearchResponseModel(BaseModel):
    paging: Paging
    posts: List[Post] = []


class SearchRequestModel(BaseModel):
    s: str
    per_page: Optional[int] = 1
    page: Optional[int] = 1
    lang: Optional[str] = "en"
    content_type: Optional[List[str]] = [""]
    term: Optional[List[str]] = [""]
    industry: Optional[str] = ""
