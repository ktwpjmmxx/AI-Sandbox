from pydantic import BaseModel

class ArticleRequest(BaseModel):
    topic: str
    level: str  # 例: "初心者", "中級者" など