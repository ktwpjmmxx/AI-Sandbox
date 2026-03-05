import os
from faster_whisper import WhisperModel
from google import genai
from dotenv import load_dotenv

# .envファイルから環境変数（APIキーなど）を読み込む
load_dotenv()

class VoxProcessor:
    def __init__(self):
        # 1. 音声認識モデルの読み込み（専門家A：Whisper）
        # ※起動時に一度だけ読み込むようにして効率化します
        print("AIモデルを準備しています...")
        self.whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        
        # 2. Geminiクライアントの初期化（専門家B：Gemini）
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def process(self, audio_path: str, genre: str = "general"):
        """
        文字起こしから要約までを一気に実行する
        """
        # --- STEP 1: 文字起こし ---
        # ジャンルに応じたヒント（initial_prompt）を設定
        prompts = {
            "it_dev": "IT・SaaS開発の会議。用語: React, FastAPI, UI/UX, 議事録SaaS",
            "general": "一般的なビジネス会議の議事録です。"
        }
        
        segments, _ = self.whisper_model.transcribe(
            audio_path, 
            beam_size=5, 
            initial_prompt=prompts.get(genre, prompts["general"])
        )
        
        # 文字起こし結果を一つの文章にまとめる
        full_text = " ".join([segment.text for segment in segments])

        # --- STEP 2: 要約 ---
        summary_prompt = f"""
        以下の会議の文字起こしデータを、適切な言葉に補正した上で、
        「会議の目的」「決定事項」「Next Action」の形式で構造化してください。
        
        【重要】
        出力結果はマークダウン記号（# や ** など）を一切使用せず、
        コピー＆ペーストしやすいプレーンなテキスト形式（通常の文章と改行のみ）で出力してください。
        
        【対象テキスト】
        {full_text}
        """
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=summary_prompt
        )

        return {
            "transcription": full_text,
            "summary": response.text
        }