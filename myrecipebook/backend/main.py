"""
MyRecipeBook - FastAPI バックエンド
起動: uvicorn main:app --reload
"""
from __future__ import annotations
import os, uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Float, Boolean, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── 設定 ────────────────────────────────────
DATABASE_URL   = os.getenv("DATABASE_URL", "sqlite:///./recipes.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
UPLOAD_DIR     = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── DB ──────────────────────────────────────
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase): pass

class RecipeORM(Base):
    __tablename__ = "recipes"
    id            = Column(Integer, primary_key=True, index=True)
    title         = Column(String(255), nullable=False, index=True)
    category      = Column(String(100), nullable=False, index=True)
    description   = Column(Text, default="")
    base_servings = Column(Float, default=2.0)   # 登録時の基準人数
    prep_time     = Column(Integer, default=0)
    cook_time     = Column(Integer, default=0)
    image_url     = Column(String(512), nullable=True)
    is_favorite   = Column(Boolean, default=False)
    ingredients   = Column(JSON, default=list)   # [{name, amount(float), unit}]
    steps         = Column(JSON, default=list)   # [{order, description, tip}]
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:    yield db
    finally: db.close()

# ── Pydantic ─────────────────────────────────
class IngredientIn(BaseModel):
    name:   str
    amount: float
    unit:   str

class StepIn(BaseModel):
    order:       int
    description: str
    tip:         Optional[str] = None

class RecipeCreate(BaseModel):
    title:         str   = Field(..., min_length=1)
    category:      str   = Field(..., min_length=1)
    description:   str   = ""
    base_servings: float = Field(2.0, gt=0)
    prep_time:     int   = Field(0, ge=0)
    cook_time:     int   = Field(0, ge=0)
    ingredients:   list[IngredientIn] = []
    steps:         list[StepIn]       = []

class RecipeUpdate(BaseModel):
    title:         Optional[str]   = None
    category:      Optional[str]   = None
    description:   Optional[str]   = None
    base_servings: Optional[float] = None
    prep_time:     Optional[int]   = None
    cook_time:     Optional[int]   = None
    is_favorite:   Optional[bool]  = None
    ingredients:   Optional[list[IngredientIn]] = None
    steps:         Optional[list[StepIn]]       = None

class RecipeOut(BaseModel):
    id:            int
    title:         str
    category:      str
    description:   str
    base_servings: float
    prep_time:     int
    cook_time:     int
    image_url:     Optional[str] = None
    is_favorite:   bool
    ingredients:   list[dict]
    steps:         list[dict]
    created_at:    datetime
    updated_at:    datetime
    class Config: from_attributes = True

class AIRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

class AIResponse(BaseModel):
    answer:  str
    is_mock: bool = True

# ── App ──────────────────────────────────────
app = FastAPI(title="MyRecipeBook API", version="1.0.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:3000"],
    allow_methods=["*"], allow_headers=["*"])
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

def to_out(r: RecipeORM) -> RecipeOut:
    return RecipeOut(
        id=r.id, title=r.title, category=r.category, description=r.description,
        base_servings=r.base_servings, prep_time=r.prep_time, cook_time=r.cook_time,
        image_url=r.image_url, is_favorite=r.is_favorite or False,
        ingredients=r.ingredients or [], steps=r.steps or [],
        created_at=r.created_at, updated_at=r.updated_at)

def not_found(): raise HTTPException(404, "レシピが見つかりません")

# ── CRUD ─────────────────────────────────────
@app.get("/api/recipes", response_model=list[RecipeOut])
def list_recipes(category: Optional[str]=None, sort: str="updated_at",
                 order: str="desc", favorites_only: bool=False, db: Session=Depends(get_db)):
    stmt = select(RecipeORM)
    if category:        stmt = stmt.where(RecipeORM.category == category)
    if favorites_only:  stmt = stmt.where(RecipeORM.is_favorite == True)
    col  = {"title": RecipeORM.title, "cook_time": RecipeORM.cook_time}.get(sort, RecipeORM.updated_at)
    stmt = stmt.order_by(col.desc() if order == "desc" else col.asc())
    return [to_out(r) for r in db.execute(stmt).scalars().all()]

@app.get("/api/recipes/{rid}", response_model=RecipeOut)
def get_recipe(rid: int, db: Session=Depends(get_db)):
    r = db.get(RecipeORM, rid)
    if not r: not_found()
    return to_out(r)

@app.post("/api/recipes", response_model=RecipeOut, status_code=201)
def create_recipe(body: RecipeCreate, db: Session=Depends(get_db)):
    now = datetime.utcnow()
    r   = RecipeORM(title=body.title, category=body.category, description=body.description,
                    base_servings=body.base_servings, prep_time=body.prep_time, cook_time=body.cook_time,
                    ingredients=[i.model_dump() for i in body.ingredients],
                    steps=[s.model_dump() for s in body.steps],
                    created_at=now, updated_at=now)
    db.add(r); db.commit(); db.refresh(r)
    _index_vector(r)
    return to_out(r)

@app.patch("/api/recipes/{rid}", response_model=RecipeOut)
def update_recipe(rid: int, body: RecipeUpdate, db: Session=Depends(get_db)):
    r = db.get(RecipeORM, rid)
    if not r: not_found()
    for k, v in body.model_dump(exclude_unset=True).items():
        if k in ("ingredients","steps") and v is not None:
            setattr(r, k, [i.model_dump() if hasattr(i,"model_dump") else i for i in v])
        elif v is not None:
            setattr(r, k, v)
    r.updated_at = datetime.utcnow()
    db.commit(); db.refresh(r)
    _index_vector(r)
    return to_out(r)

@app.delete("/api/recipes/{rid}", status_code=204)
def delete_recipe(rid: int, db: Session=Depends(get_db)):
    r = db.get(RecipeORM, rid)
    if not r: not_found()
    db.delete(r); db.commit()

@app.post("/api/recipes/{rid}/image", response_model=RecipeOut)
async def upload_image(rid: int, file: UploadFile=File(...), db: Session=Depends(get_db)):
    r = db.get(RecipeORM, rid)
    if not r: not_found()
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".jpg",".jpeg",".png",".webp"}:
        raise HTTPException(422, "jpg/png/webp のみ対応")
    path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    path.write_bytes(await file.read())
    r.image_url  = f"/uploads/{path.name}"
    r.updated_at = datetime.utcnow()
    db.commit(); db.refresh(r)
    return to_out(r)

@app.patch("/api/recipes/{rid}/favorite", response_model=RecipeOut)
def toggle_favorite(rid: int, db: Session=Depends(get_db)):
    r = db.get(RecipeORM, rid)
    if not r: not_found()
    r.is_favorite = not (r.is_favorite or False)
    r.updated_at  = datetime.utcnow()
    db.commit(); db.refresh(r)
    return to_out(r)

@app.get("/api/categories", response_model=list[str])
def list_categories(db: Session=Depends(get_db)):
    rows = db.execute(select(RecipeORM.category).distinct()).scalars().all()
    return sorted(rows)

# ── AI (mock / real) ─────────────────────────
@app.post("/api/recipes/{rid}/ai-assist", response_model=AIResponse)
def ai_assist(rid: int, body: AIRequest, db: Session=Depends(get_db)):
    r = db.get(RecipeORM, rid)
    if not r: not_found()
    if OPENAI_API_KEY:
        return AIResponse(answer=_llm(r, body.question), is_mock=False)
    return AIResponse(answer=_mock_answer(r, body.question))

@app.post("/api/ai/suggest-menu", response_model=AIResponse)
def suggest_menu(body: AIRequest, db: Session=Depends(get_db)):
    recipes = db.execute(select(RecipeORM)).scalars().all()
    titles  = [r.title for r in recipes[:5]]
    mock    = (f"保存中のレシピ（{len(recipes)}件）から提案します。\n\n"
               f"📋 本日のおすすめ\n・メイン: {titles[0] if titles else '未登録'}\n"
               f"・副菜: {titles[1] if len(titles)>1 else 'サラダ'}\n"
               f"※ OPENAI_API_KEY を設定するとAIが本格回答します。")
    return AIResponse(answer=mock)

def _index_vector(r: RecipeORM):
    try:
        import chromadb
        c  = chromadb.PersistentClient(path="./chroma_data")
        co = c.get_or_create_collection("recipes")
        ings  = ", ".join(f"{i['name']} {i['amount']}{i['unit']}" for i in (r.ingredients or []))
        steps = " ".join(f"工程{s['order']}: {s['description']}" for s in (r.steps or []))
        co.upsert(ids=[str(r.id)],
                  documents=[f"レシピ:{r.title} カテゴリ:{r.category} 材料:{ings} 手順:{steps}"],
                  metadatas=[{"title":r.title,"category":r.category}])
    except Exception: pass

def _mock_answer(r, q):
    if "時短" in q: return "圧力鍋で煮込み時間を1/3に短縮できます。電子レンジ下茹でも有効です。"
    if "代用" in q: return "みりん→砂糖小さじ1＋酒大さじ1。醤油→めんつゆ2倍濃縮で代用可能です。"
    return f"「{r.title}」へのご質問ありがとうございます。「時短」「代用」などのキーワードでお試しください。"

def _llm(r, q):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    ings   = ", ".join(f"{i['name']} {i['amount']}{i['unit']}" for i in (r.ingredients or []))
    resp   = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":
            f"レシピ「{r.title}」（材料: {ings}）について日本語で答えてください。\n質問: {q}"}],
        max_tokens=400)
    return resp.choices[0].message.content
