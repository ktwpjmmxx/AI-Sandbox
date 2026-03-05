import os
import tempfile
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# さきほど作成した「裏のシェフ（AI処理クラス）」を呼び出す
from core.processor import VoxProcessor

app = FastAPI(
    title="VoxNote API",
    description="音声認識とLLM要約を行うバックエンドAPI",
    version="1.0.0"
)

# CORS設定（Reactからの通信を許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 【重要ポイント①】「いつ」呼び出すのか？
# ==========================================
# サーバー（main.py）が起動した瞬間に、1回だけ呼び出してスタンバイさせます。
print("VoxNoteサーバーを起動中... AIモデルをメモリに読み込んでいます...")
processor = VoxProcessor()
print("準備完了！リクエストの受付を開始します。")


@app.get("/")
def read_root():
    return {"message": "VoxNote API is running smoothly!"}


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Reactから送られてきた音声ファイルを受け取り、AIで処理する窓口
    """
    # ==========================================
    # 【重要ポイント②】「どうやって」処理するのか？
    # ==========================================
    
    # 1. FastAPIはファイルの中身（データ）だけを受け取るので、
    # Whisperが読み込めるように「一時的なファイル」としてPCに保存します。
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        content = await file.read()
        temp_audio.write(content)
        temp_audio_path = temp_audio.name

    try:
        # 2. 保存したファイルのパスをシェフ（processor）に渡して、仕事を丸投げします。
        # ※ここでは一旦、ジャンルを "it_dev" に固定してテストします。
        print(f"AI処理を開始します: {file.filename}")
        result = processor.process(temp_audio_path, genre="it_dev")
        
        # 3. シェフから返ってきた「文字起こし」と「要約」の結果をReactに返します。
        return {
            "status": "success",
            "filename": file.filename,
            "transcription": result["transcription"],
            "summary": result["summary"]
        }
    
    finally:
        # 4. 【後片付け】処理が終わったら、エラーが起きた場合でも必ず一時ファイルを削除します。
        # これをやらないと、使えば使うほどサーバーの容量がパンクしてしまいます。
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
if __name__ == "__main__":
    print("サーバーをポート8000で起動します...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)