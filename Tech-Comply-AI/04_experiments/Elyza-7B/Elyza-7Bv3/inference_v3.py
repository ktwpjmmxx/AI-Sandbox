import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

class ComplianceCheckerV3:
    def __init__(self, base_model_name, adapter_path):
        print("=" * 70)
        print("Tech-Comply-AI: コンプライアンス・エンジン起動中...")
        print("=" * 70)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        # ポートフォリオ用に「専門家」としての振る舞いを定義
        self.system_message = (
            "あなたはIT法務およびコンプライアンスの専門コンサルタントです。"
            "ユーザーから提示された契約書やサービス内容の条項に対し、法的リスクを分析してください。"
            "回答は以下のフォーマットに従い、専門的かつ分かりやすい日本語で記述してください。\n\n"
            "【リスクレベル】\n"
            "（低・中・高とその判定理由）\n\n"
            "【該当法】\n"
            "（関連する法律名や条文）\n\n"
            "【法的根拠・理由】\n"
            "（なぜ違反の可能性があるのか詳細に解説）\n\n"
            "【修正案】\n"
            "（具体的な条文の書き換え例や、サービス仕様の変更提案）"
        )
        print("✓ エンジンの準備が完了しました")

    def generate_response(self, instruction, input_text, max_new_tokens=512):
        # 学習データに近いフォーマットでプロンプトを作成
        prompt = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{instruction}\n\n相談内容:\n{input_text} [/INST] "
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        return response

def run_portfolio_demo(checker):
    """ポートフォリオで見栄えのするデモ出力"""
    test_input = "ECサイトで「今すぐ買って、支払いは来月でOK」という独自の後払い決済を導入したい。クレカ決済ではないし、分割手数料も取らないから、割賦販売法の規制は受けないという理解で合ってる？"
    
    print("\n" + "【 AIコンプライアンス診断レポート 】".center(60))
    print("-" * 70)
    print(f"■ ご相談内容:\n{test_input}")
    print("-" * 70)
    print("■ 診断結果:")
    
    result = checker.generate_response("IT法務コンサルタントとして、BNPL（後払い）機能の導入を評価してください。", test_input)
    
    # 結果をきれいに表示
    print(result)
    print("-" * 70)
    print("以上".rjust(68))

def main():
    # パラメータ設定（適宜書き換えてください）
    base_model = "elyza/ELYZA-japanese-Llama-2-7b"
    adapter_path = "./results_v3/final_adapter"
    
    checker = ComplianceCheckerV3(base_model, adapter_path)
    run_portfolio_demo(checker)

if __name__ == "__main__":
    main()