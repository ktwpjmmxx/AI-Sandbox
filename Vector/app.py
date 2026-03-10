"""
app.py - 法務RAGチャットボット
"""

import os
import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(
    page_title="法務 RAG チャットボット",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Noto Sans JP', sans-serif !important;
    background: #f7f8fc !important;
    color: #1a1a2e !important;
}

/* ヘッダー */
.app-header {
    background: white;
    border-bottom: 2px solid #e8eaf0;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
    border-radius: 0 0 12px 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.app-header-icon { font-size: 1.8rem; }
.app-header-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1a1a2e;
}
.app-header-sub {
    font-size: 0.78rem;
    color: #888;
    margin-top: 2px;
}
.status-dot {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    color: #888;
}
.dot-on  { width:10px; height:10px; border-radius:50%; background:#22c55e; display:inline-block; }
.dot-off { width:10px; height:10px; border-radius:50%; background:#ef4444; display:inline-block; }

/* APIキーカード */
.setup-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 28px 32px;
    max-width: 560px;
    margin: 0 auto 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.setup-card h3 {
    font-size: 1rem;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 16px;
}

/* チャットエリア */
.chat-wrap {
    max-width: 720px;
    margin: 0 auto;
}

/* ユーザーメッセージ */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
}
.msg-user-bubble {
    background: #3b5bdb;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 75%;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(59,91,219,0.25);
}

/* AIメッセージ */
.msg-ai {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
    gap: 10px;
    align-items: flex-start;
}
.msg-ai-avatar {
    width: 36px;
    height: 36px;
    background: #f1f3f9;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    border: 1px solid #e2e8f0;
}
.msg-ai-bubble {
    background: white;
    border: 1px solid #e8eaf0;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px;
    max-width: 80%;
    font-size: 0.93rem;
    line-height: 1.75;
    color: #2d3748;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* ソースチャンク */
.chunk-section {
    background: #f8faff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 8px 0 4px 46px;
}
.chunk-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: #3b5bdb;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.chunk-item {
    background: white;
    border: 1px solid #e8eaf0;
    border-left: 3px solid #3b5bdb;
    border-radius: 4px 8px 8px 4px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    color: #555;
    line-height: 1.6;
}
.chunk-score {
    display: inline-block;
    background: #eef2ff;
    color: #3b5bdb;
    border-radius: 20px;
    padding: 1px 8px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 5px;
}

/* ウェルカムカード */
.welcome-card {
    background: white;
    border: 1px solid #e8eaf0;
    border-radius: 14px;
    padding: 32px;
    text-align: center;
    max-width: 600px;
    margin: 0 auto;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
.welcome-icon { font-size: 3rem; margin-bottom: 12px; }
.welcome-title { font-size: 1.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 8px; }
.welcome-desc  { font-size: 0.88rem; color: #666; line-height: 1.7; }

/* クイック質問 */
.quick-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    margin-top: 20px;
}

/* 入力エリア */
.input-area {
    background: white;
    border: 2px solid #e2e8f0;
    border-radius: 14px;
    padding: 4px 4px 4px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-top: 16px;
    transition: border-color 0.2s;
}
.input-area:focus-within { border-color: #3b5bdb; }

/* Streamlitデフォルトスタイルの上書き */
.stTextInput > div > div > input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #1a1a2e !important;
    font-family: 'Noto Sans JP', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 10px 0 !important;
}
.stTextInput > div { border: none !important; }

.stButton > button {
    background: #3b5bdb !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Noto Sans JP', sans-serif !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #2f4ac4 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(59,91,219,0.3) !important;
}
.stButton > button[kind="secondary"] {
    background: white !important;
    color: #555 !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 0.82rem !important;
    padding: 6px 12px !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #f8faff !important;
    border-color: #3b5bdb !important;
    color: #3b5bdb !important;
}

/* スピナー */
.stSpinner { color: #3b5bdb !important; }

/* 区切り線 */
hr { border-color: #e8eaf0 !important; margin: 16px 0 !important; }

/* サイドバー非表示 */
[data-testid="stSidebar"] { display: none; }
section[data-testid="stSidebarContent"] { display: none; }

/* エラー・警告 */
.stAlert { border-radius: 10px !important; font-family: 'Noto Sans JP', sans-serif !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #f8faff !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    color: #3b5bdb !important;
}
</style>
""", unsafe_allow_html=True)


# ===================== セッション状態 =====================
if "messages"    not in st.session_state: st.session_state.messages    = []
if "engine"      not in st.session_state: st.session_state.engine      = None
if "show_chunks" not in st.session_state: st.session_state.show_chunks = True
if "api_key"     not in st.session_state: st.session_state.api_key     = os.environ.get("GEMINI_API_KEY", "")


# ===================== ヘッダー =====================
status_html = (
    '<span class="dot-on"></span> エンジン稼働中'
    if st.session_state.engine else
    '<span class="dot-off"></span> エンジン未起動'
)
st.markdown(f"""
<div class="app-header">
    <div class="app-header-icon">⚖️</div>
    <div>
        <div class="app-header-title">法務ナレッジ RAG</div>
        <div class="app-header-sub">FAISS × Gemini による法務文書Q&Aシステム</div>
    </div>
    <div class="status-dot">{status_html}</div>
</div>
""", unsafe_allow_html=True)


# ===================== APIキー設定エリア =====================
if not st.session_state.engine:
    st.markdown('<div class="setup-card"><h3>🔑 APIキーを入力してエンジンを起動</h3>', unsafe_allow_html=True)

    col_key, col_btn = st.columns([3, 1])
    with col_key:
        key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="AIza...",
            label_visibility="collapsed",
        )
    with col_btn:
        start_btn = st.button("起動 →", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if start_btn:
        if key_input:
            with st.spinner("FAISSインデックスを読み込み中..."):
                try:
                    st.session_state.engine  = RAGEngine(key_input)
                    st.session_state.api_key = key_input
                    st.rerun()
                except Exception as e:
                    st.error(f"エラー: {e}")
        else:
            st.warning("APIキーを入力してください")


# ===================== チャットエリア =====================
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">📚</div>
        <div class="welcome-title">法務ドキュメントに質問してみましょう</div>
        <div class="welcome-desc">
            PDFから生成したナレッジベースをもとに、<br>
            GeminiがRAGで回答を生成します。<br><br>
            エンジン起動後、下の質問ボタンまたは自由入力で質問できます。
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-user-bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # マークダウンをそのままstreamlitに渡すため分割表示
        st.markdown(f"""
        <div class="msg-ai">
            <div class="msg-ai-avatar">⚖️</div>
            <div class="msg-ai-bubble">{msg["content"].replace(chr(10), "<br>")}</div>
        </div>
        """, unsafe_allow_html=True)

        # ソースチャンク
        if st.session_state.show_chunks and "chunks" in msg:
            with st.expander(f"📎 参照ソース {len(msg['chunks'])}件（クリックで展開）"):
                for i, chunk in enumerate(msg["chunks"]):
                    score = chunk.metadata.get("similarity_score", "N/A")
                    text  = chunk.page_content.replace('\n', ' ')
                    st.markdown(f"""
                    <div class="chunk-item">
                        <span class="chunk-score">チャンク {i+1}　距離スコア: {score}</span><br>
                        {text}
                    </div>
                    """, unsafe_allow_html=True)

        # プロンプト確認（学習用）
        if "prompt" in msg:
            with st.expander("🔍 Geminiに送ったプロンプトを確認（学習用）"):
                st.code(msg["prompt"], language="text")

st.markdown('</div>', unsafe_allow_html=True)


# ===================== 入力エリア =====================
st.markdown('<div class="quick-label">💬 クイック質問</div>', unsafe_allow_html=True)

quick_questions = [
    "時間外労働の割増賃金率は？",
    "有給休暇の付与日数は？",
    "個人情報漏洩時の義務は？",
    "NDAに記載すべき内容は？",
    "契約解除の方法を教えて",
    "著作権の保護期間は？",
]

selected_quick = None
q_cols = st.columns(3)
for i, q in enumerate(quick_questions):
    if q_cols[i % 3].button(q, key=f"q{i}", type="secondary"):
        selected_quick = q

st.markdown("<br>", unsafe_allow_html=True)

col_input, col_send = st.columns([6, 1])
with col_input:
    user_input = st.text_input(
        "質問",
        placeholder="例：解雇するためには何が必要ですか？",
        key="user_input_field",
        label_visibility="collapsed",
    )
with col_send:
    send_btn = st.button("送信", use_container_width=True)

# 設定トグル（折りたたみ）
with st.expander("⚙️ 設定"):
    st.session_state.show_chunks = st.toggle("参照チャンクを表示", value=st.session_state.show_chunks)
    if st.button("🗑️ 会話をリセット"):
        st.session_state.messages = []
        st.rerun()


# ===================== 送信処理 =====================
query = selected_quick or (user_input if send_btn else None)

if query:
    if not st.session_state.engine:
        st.error("APIキーを入力してエンジンを起動してください")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("検索中 → 回答生成中..."):
            try:
                result = st.session_state.engine.generate(query)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result["answer"],
                    "chunks":  result["source_chunks"],
                    "prompt":  result["prompt"],
                })
            except Exception as e:
                st.error(f"エラー: {e}")
        st.rerun()