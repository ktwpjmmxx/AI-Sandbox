from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import dictionary
from app.api.routes import articles

app = FastAPI(title="Inglish API")

# React(Vite)フロントエンドからのアクセスを許可する設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 辞書機能のルーティングを登録
app.include_router(dictionary.router, prefix="/api")
app.include_router(articles.router, prefix="/api/articles")

@app.get("/")
def read_root():
    return {"message": "Inglish Backend is running!"}