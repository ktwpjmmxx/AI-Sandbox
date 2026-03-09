import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

async def translate_text(text: str) -> str:
    # APIキーが未設定の場合のフェイルセーフ
    if not GEMINI_API_KEY or GEMINI_API_KEY == "ここにGeminiのAPIキーを貼り付けます":
        return f"【バックエンド貫通テスト成功！】\nAPIキーが設定されれば、「{text}」の翻訳がここに表示されます。"

    try:
        # Gemini APIの初期設定
        genai.configure(api_key=GEMINI_API_KEY)
        
        # 応答速度が速い軽量モデルを指定
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 辞書として機能させるためのプロンプト
        prompt = f"次の英語を自然な日本語に翻訳してください。解説や余計な言葉は省き、翻訳結果のみをシンプルに出力してください。\n\n対象テキスト: {text}"
        
        # 非同期でAPIを呼び出す
        response = await model.generate_content_async(prompt)
        
        return response.text.strip()
        
    except Exception as e:
        return f"翻訳エラーが発生しました: {str(e)}"