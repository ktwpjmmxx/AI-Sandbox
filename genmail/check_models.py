import google.generativeai as genai
import os
from dotenv import load_dotenv

# 環境変数からAPIキーを読み込み
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("=== 使用可能なモデル一覧（文章生成対応） ===")
# 利用可能な全モデルを取得し、文章生成（generateContent）に対応しているかチェック
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)