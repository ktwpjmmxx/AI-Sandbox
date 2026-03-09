import os
import chromadb

def main():
    # 1. データベースの保存先を指定（バックエンドから読み込める場所に作成）
    db_path = os.path.join(os.path.dirname(__file__), "../backend/vector_db")
    client = chromadb.PersistentClient(path=db_path)

    # 2. コレクションを作成
    collection_name = "english_materials"
    collection = client.get_or_create_collection(name=collection_name)

    # 3. テキストデータの読み込み
    data_file = os.path.join(os.path.dirname(__file__), "raw_data/sample_articles.txt")
    with open(data_file, "r", encoding="utf-8") as f:
        text_data = f.read()

    # 4. 空行（\n\n）でテキストを段落ごとに分割（チャンキング）
    chunks = [chunk.strip() for chunk in text_data.split("\n\n") if chunk.strip()]

    # 5. ChromaDBに保存
    for i, chunk in enumerate(chunks):
        collection.upsert(
            documents=[chunk],
            metadatas=[{"source": "sample_articles.txt", "chunk_index": i}],
            ids=[f"doc_{i}"]
        )
        print(f"チャンク {i+1}/{len(chunks)} を保存しました: {chunk[:30]}...")

    print("✅ データベースへの読み込み（Ingest）が完了しました！")

if __name__ == "__main__":
    main()