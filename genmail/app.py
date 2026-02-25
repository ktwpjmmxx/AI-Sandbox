# app.py
import streamlit as st
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import prompts

# 環境変数の読み込み
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("エラー: .env ファイルに GEMINI_API_KEY が設定されていません。")
    st.stop()

# APIキーの設定
genai.configure(api_key=GEMINI_API_KEY)

# モデル初期化関数
def get_generative_model(system_prompt):
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt
    )

# 設定データの保存・読み込み機能
SETTINGS_FILE = "settings.json"

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"to_company": "", "to_name": "", "from_name": "", "signature": ""}

def save_settings(data):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

settings = load_settings()

# UI設定 (サイドバーをデフォルトで開く設定を追加)
st.set_page_config(
    page_title="AIメールアシスタント",
    layout="wide",
    initial_sidebar_state="expanded"
)

# サイドバー (設定画面)
with st.sidebar:
    st.header("宛名・署名設定")
    st.write("設定を保存すると生成時に反映されます。空欄の場合はプレースホルダーで出力されます。")
    
    to_company = st.text_input("宛先企業名", settings.get("to_company", ""))
    to_name = st.text_input("宛先担当者名", settings.get("to_name", ""))
    from_name = st.text_input("ご自身の名前", settings.get("from_name", ""))
    signature = st.text_area("ご自身の署名", settings.get("signature", ""), height=150)
    
    if st.button("設定を保存", key="btn_save"):
        save_settings({
            "to_company": to_company,
            "to_name": to_name,
            "from_name": from_name,
            "signature": signature
        })
        st.success("設定を保存しました。")

# メイン画面
st.title("AIメールアシスタント")
st.write("用途に合わせてタブを選択してください。")

tab1, tab2 = st.tabs(["1. メール自動添削", "2. メール自動生成"])

# タブ1: メール自動添削
with tab1:
    st.header("メール自動添削")
    input_text = st.text_area("元の文章を入力してください", height=200, 
                              placeholder="例：明日の会議、15時からに変更してほしい。場所は第2会議室で。")
    
    if st.button("添削する", key="btn_proofread"):
        if input_text:
            with st.spinner("AIが添削中..."):
                try:
                    model = get_generative_model(prompts.PROOFREAD_SYSTEM_PROMPT)
                    
                    settings_text = f"""
                    【設定情報（反映用）】
                    宛先企業名: {to_company if to_company else '[宛先企業名]'}
                    宛先担当者名: {to_name if to_name else '[宛先担当者名]'}
                    ご自身の名前: {from_name if from_name else '[ご自身の名前]'}
                    ご自身の署名:
                    {signature if signature else '[ご自身の署名]'}
                    """
                    user_prompt = f"{input_text}\n\n{settings_text}"
                    
                    response = model.generate_content(user_prompt)
                    st.success("添削完了！枠内をクリックしてすべて選択（Ctrl+A または Cmd+A）でコピーできます。")
                    
                    # テキスト折り返し表示に対応
                    st.text_area("添削結果", value=response.text, height=300, key="result_proofread")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
        else:
            st.warning("文章を入力してください。")

# タブ2: メール自動生成
with tab2:
    st.header("メール自動生成")
    context_input = st.text_area("状況を教えてください", height=100, 
                                 placeholder="例：先方が提示した日程がすべて合わないため、来週の火曜か水曜の午後で再調整をお願いしたい。")
    nuance_input = st.text_input("ニュアンスを指定してください", 
                                 placeholder="例：謝罪強めで、少し丁寧めに")
    
    if st.button("生成する", key="btn_generate"):
        if context_input and nuance_input:
            with st.spinner("AIが生成中..."):
                try:
                    model = get_generative_model(prompts.GENERATE_SYSTEM_PROMPT)
                    
                    settings_text = f"""
                    【設定情報（反映用）】
                    宛先企業名: {to_company if to_company else '[宛先企業名]'}
                    宛先担当者名: {to_name if to_name else '[宛先担当者名]'}
                    ご自身の名前: {from_name if from_name else '[ご自身の名前]'}
                    ご自身の署名:
                    {signature if signature else '[ご自身の署名]'}
                    """
                    
                    user_prompt = f"【状況】\n{context_input}\n\n【ニュアンス】\n{nuance_input}\n\n{settings_text}"
                    
                    response = model.generate_content(user_prompt)
                    st.success("生成完了！枠内をクリックしてすべて選択（Ctrl+A または Cmd+A）でコピーできます。")
                    
                    # テキスト折り返し表示に対応
                    st.text_area("生成結果", value=response.text, height=300, key="result_generate")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
        else:
            st.warning("状況とニュアンスの両方を入力してください。")