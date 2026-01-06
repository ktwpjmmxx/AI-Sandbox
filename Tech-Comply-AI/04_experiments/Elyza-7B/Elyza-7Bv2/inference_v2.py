import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import argparse
import re

class ComplianceCheckerV2:
    def __init__(self, base_model_name, adapter_path):
        print("=" * 70)
        print("モデルを読み込んでいます...")
        
        # トークナイザーの読み込み (use_fast=False を追加して日本語を安定化)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
        # 【重要】学習時と完全に一致させる
        self.system_message = (
            "あなたは日本の法律とコンプライアンスに精通した専門アシスタントです。"
            "景品表示法、個人情報保護法、著作権法などに基づいて、"
            "正確で実用的なアドバイスを提供してください。"
            "回答は必ず指定されたJSON形式で、それ以外の文章は含めないでください。"
        )
        
        print("✓ モデルの読み込み完了")

    def generate_response(self, instruction, input_text="", max_new_tokens=512):
        # 学習時と同じ形式に構築 (\n\n回答(JSON形式のみ): は削除)
        if input_text:
            full_instruction = f"{instruction}\n\n入力:\n{input_text}"
        else:
            full_instruction = instruction
        
        # テンプレートから余計な改行やスペースを排除
        prompt = (
            f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n"
            f"{full_instruction} [/INST]"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # JSON出力にはGreedy Searchが安定
                repetition_penalty=1.1,    # 繰り返し抑制は控えめに
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # [/INST] 以降を抽出
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response

    def extract_json(self, text):
        """テキストからJSON部分を抽出してパースを試みる"""
        try:
            # 最初の { から 最後の } までを抽出
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                # 引用符の揺れ（“”や‘’）を正規化
                json_str = json_str.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
                return json_str
        except:
            pass
        return text

    def check_advertisement(self, ad_text):
        instruction = (
            "あなたはWebサービス開発を支援する法務アシスタントです。"
            "入力された広告文に対し、景品表示法に基づくリスク判定、理由、"
            "および修正案をJSON形式で回答してください。"
        )
        return self.generate_response(instruction, ad_text)

    def validate_json(self, text):
        """JSON形式の検証"""
        json_text = self.extract_json(text)
        try:
            parsed = json.loads(json_text)
            return True, parsed, None
        except Exception as e:
            return False, None, str(e)

# (以下、main関数などは元のスクリプトと同様)