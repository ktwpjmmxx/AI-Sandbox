import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# データベースの場所を絶対パスで指定
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../vector_db"))

async def generate_personalized_article(topic: str, level: str) -> str:
    # 1. ChromaDBから、テーマに関連する文脈を検索（Retrieval）
    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection(name="english_materials")
        results = collection.query(
            query_texts=[topic],
            n_results=2  # 関連度の高い2つの段落を取得
        )
        # 検索結果を1つの文字列にまとめる
        context = "\n".join(results['documents'][0]) if results['documents'] else "一般的な知識として回答してください。"
    except Exception as e:
        print(f"DB検索エラー: {e}")
        context = "一般的な知識として回答してください。"

    # 2. Geminiに渡すプロンプトを作成（Augmented Generation）
    prompt = f"""
    あなたはプロの英語教師です。以下の【参考情報】を元に、英語学習者向けのコラムを作成してください。
    
    【学習者のレベル】: {level}
    【テーマ】: {topic}
    
    【参考情報】:
    {context}
    
    【条件】:
    - 魅力的な英語のタイトル（# 見出し）をつけてください。
    - 英語の本文のみを出力してください（日本語訳は不要です）。
    - {level}の学習者が理解できる単語と文法を使用してください。
    - 2〜3段落程度の長さにしてください。
    """

    # 3. Geminiで記事を生成
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = await model.generate_content_async(prompt)
    
    return response.text