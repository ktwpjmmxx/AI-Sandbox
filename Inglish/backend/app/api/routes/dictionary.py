from fastapi import APIRouter
from app.schemas.dictionary import TranslationRequest
from app.services.dict_service import translate_text

router = APIRouter()

@router.post("/translate")
async def translate(request: TranslationRequest):
    # サービス層に翻訳処理を依頼する
    translated_text = await translate_text(request.text)
    # フロントエンドが受け取りやすいJSON形式で返す
    return {"translation": translated_text}