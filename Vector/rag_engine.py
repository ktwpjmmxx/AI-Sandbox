"""
rag_engine.py（出典ファイル名表示対応版）
"""

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

FAISS_DIR   = "./faiss_index"
EMBED_MODEL = "models/gemini-embedding-001"
CHAT_MODEL  = "gemini-2.5-flash"
TOP_K       = 3

SYSTEM_PROMPT = """あなたは社内ナレッジベースを参照するAIアシスタントです。

【回答ルール】
1. 必ず「参考情報」として提供されたテキストのみを根拠として回答してください
2. 参考情報に記載がない内容については「この文書には記載がありません」と明示してください
3. 回答は日本語で、わかりやすく箇条書きや見出しを使って構造化してください
4. 回答の根拠となった情報源（ファイル名）を末尾に「📎 出典: ファイル名」の形式で示してください

【重要】
あなたは「参考情報」の範囲でのみ回答します。
これがRAGの核心：モデルの知識ではなく、検索された文書に基づいて回答する。
"""


class RAGEngine:

    def __init__(self, api_key: str):
        self.api_key = api_key

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=api_key,
        )

        self.vectorstore = FAISS.load_local(
            FAISS_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.llm = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            google_api_key=api_key,
            temperature=0.1,
        )

        print(f"✅ RAGエンジン初期化完了 | {EMBED_MODEL} | {CHAT_MODEL}")


    def retrieve(self, query: str) -> list:
        results_with_scores = self.vectorstore.similarity_search_with_score(
            query=query, k=TOP_K,
        )
        docs = []
        for doc, score in results_with_scores:
            doc.metadata["similarity_score"] = round(float(score), 4)
            docs.append(doc)
        return docs


    def build_prompt(self, query: str, docs: list) -> str:
        context_parts = []
        for i, doc in enumerate(docs):
            score  = doc.metadata.get("similarity_score", "N/A")
            source = doc.metadata.get("source", "不明")  # ← ファイル名を取得
            context_parts.append(
                f"【参考情報 {i+1}】出典: {source} / 類似度スコア: {score}\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        return f"""{SYSTEM_PROMPT}

===== 参考情報（FAISSから検索） =====
{context}
======================================

【ユーザーの質問】
{query}

【回答】"""


    def generate(self, query: str) -> dict:
        docs     = self.retrieve(query)
        prompt   = self.build_prompt(query, docs)
        response = self.llm.invoke(prompt)
        return {
            "answer":       response.content,
            "source_chunks": docs,
            "prompt":       prompt,
        }