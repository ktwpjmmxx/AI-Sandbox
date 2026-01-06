import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

class ComplianceCheckerV4:
    def __init__(self, base_model_name, adapter_path, config_path="./config_v4.yaml"):
        print("=" * 70)
        print("⚖️  Tech-Comply-AI: コンプライアンス・エンジン起動中...")
        print("=" * 70)
        
        # 設定ファイル読み込み
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                self.inference_config = self.config.get("inference", {})
        except:
            print("⚠️  config_v4.yamlが見つかりません。デフォルト設定を使用します。")
            self.inference_config = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.15,
                "do_sample": True
            }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            trust_remote_code=True
        )
        
        # システムメッセージを学習データのフォーマットと完全一致
        self.system_message = (
            "あなたはIT法務およびコンプライアンスの専門コンサルタントです。\n"
            "以下のフォーマットに厳密に従って回答してください。\n\n"
            "リスクレベル：（低・中・高のいずれかを明記）\n"
            "該当法：（関連する法律名と条文番号）\n"
            "理由：（法的根拠と違反の可能性について詳細に解説）\n"
            "修正案：（具体的な改善提案。必ず記述すること）\n\n"
            "上記4項目を必ず含め、修正案は省略せずに記述してください。"
        )
        
        print("✓ エンジンの準備が完了しました")
        print(f"   推論設定: temp={self.inference_config['temperature']}, "
              f"top_p={self.inference_config['top_p']}, "
              f"rep_penalty={self.inference_config['repetition_penalty']}")

    def generate_response(self, instruction, input_text):
        """
        学習時のフォーマットと完全一致させる
        """
        # 学習時と同じキーワード「入力:」を使用
        prompt = (
            f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n"
            f"{instruction}\n\n入力:\n{input_text} [/INST]"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 設定ファイルから読み込んだパラメータを使用
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.inference_config["max_new_tokens"],
                do_sample=self.inference_config["do_sample"],
                temperature=self.inference_config["temperature"],
                top_p=self.inference_config["top_p"],
                top_k=self.inference_config.get("top_k", 50),
                repetition_penalty=self.inference_config["repetition_penalty"],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # [/INST]以降を抽出
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response

def run_portfolio_demo(checker):
    """ポートフォリオ用デモ"""
    test_cases = [
        {
            "instruction": "以下のサービス条項について法的リスクを評価してください。",
            "input": "ECサイトで「今すぐ買って、支払いは来月でOK」という独自の後払い決済を導入したい。クレカ決済ではないし、分割手数料も取らないから、割賦販売法の規制は受けないという理解で合ってる？"
        },
        {
            "instruction": "以下の利用規約の条項について法的問題を指摘してください。",
            "input": "「当社は、ユーザーの投稿コンテンツを無償で自由に利用・改変・二次利用できるものとします。著作権はユーザーに帰属しますが、当社への利用許諾は取り消せません。」"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print("\n" + "=" * 70)
        print(f"【 テストケース {i} 】".center(60))
        print("=" * 70)
        print(f"\n■ 質問:\n{case['input']}\n")
        print("-" * 70)
        print("■ 診断結果:\n")
        
        result = checker.generate_response(case['instruction'], case['input'])
        print(result)
        print("\n" + "-" * 70)

def main():
    parser = argparse.ArgumentParser(description="Tech-Comply-AI v4 推論エンジン")
    parser.add_argument("--base_model", type=str, 
                       default="elyza/ELYZA-japanese-Llama-2-7b")
    parser.add_argument("--adapter_path", type=str, 
                       default="./results_v4/final_adapter")
    parser.add_argument("--config_path", type=str, 
                       default="./config_v4.yaml")
    args = parser.parse_args()
    
    checker = ComplianceCheckerV4(
        args.base_model, 
        args.adapter_path,
        args.config_path
    )
    
    run_portfolio_demo(checker)
    
    print("\n" + "=" * 70)
    print("✓ すべてのテストが完了しました")
    print("=" * 70)

if __name__ == "__main__":
    main()
