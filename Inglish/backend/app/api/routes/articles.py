from fastapi import APIRouter
from app.schemas.articles import ArticleRequest
from app.services.rag_service import generate_personalized_article

router = APIRouter()

@router.post("/generate")
async def generate_article(request: ArticleRequest):
    # RAGサービスを呼び出して記事を生成
    article_text = await generate_personalized_article(request.topic, request.level)
    return {"content": article_text}