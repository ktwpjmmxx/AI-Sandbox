"""
============================================================
Tech-Comply-AI 推論スクリプト
ファインチューニング済みモデルを使用したコンプライアンスチェック
============================================================
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import argparse

class ComplianceChecker:
    """コンプライアンスチェッカークラス"""
    
    def __init__(self, base_model_name, adapter_path):
        """
        初期化
        
        Args:
            base_model_name: ベースモデルのHugging Face ID
            adapter_path: ファインチューニング済みアダプターのパス
        """
        print("=" * 70)
        print("モデルを読み込んでいます...")
        print("=" * 70)
        
        # ベースモデルの読み込み
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # アダプターの読み込み
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
        # トークナイザーの読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        
        # システムメッセージ
        self.system_message = (
            "あなたは日本の法律とコンプライアンスに精通した専門アシスタントです。"
            "景品表示法、個人情報保護法、著作権法などに基づいて、"
            "正確で実用的なアドバイスを提供してください。"
        )
        
        print("✓ モデルの読み込み完了")
        print("=" * 70)
    
    def generate_response(self, instruction, input_text="", 
                         max_new_tokens=512, temperature=0.7, top_p=0.9):
        """
        プロンプトに対する応答を生成
        
        Args:
            instruction: 指示文
            input_text: 入力テキスト(オプション)
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_p: Top-pサンプリングの閾値
        
        Returns:
            生成されたテキスト
        """
        # プロンプトの構築
        if input_text:
            full_instruction = f"{instruction}\n\n入力:\n{input_text}"
        else:
            full_instruction = instruction
        
        prompt = f"""<s>[INST] <<SYS>>
{self.system_message}
<</SYS>>

{full_instruction} [/INST] """
        
        # トークン化
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 推論実行
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 結果のデコード
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # [/INST]以降を抽出
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response
    
    def check_advertisement(self, ad_text):
        """広告文のコンプライアンスチェック(景品表示法)"""
        instruction = (
            "あなたはWebサービス開発を支援する法務アシスタントです。"
            "入力された広告文に対し、景品表示法に基づくリスク判定、理由、"
            "および修正案をJSON形式で回答してください。"
        )
        return self.generate_response(instruction, ad_text)
    
    def check_privacy_policy(self, service_description):
        """サービス仕様に対するプライバシーポリシー要件チェック"""
        instruction = (
            "あなたはWebサービス開発を支援する法務アシスタントです。"
            "入力された仕様に対し、個人情報保護法に基づくプライバシーポリシーの"
            "記述が必要か判断してください。"
        )
        return self.generate_response(instruction, service_description)
    
    def check_copyright(self, feature_description):
        """機能実装に対する著作権法チェック"""
        instruction = (
            "あなたはWebサービス開発を支援する法務アシスタントです。"
            "著作権法の観点から機能実装を評価してください。"
        )
        return self.generate_response(instruction, feature_description)


def run_demo(checker):
    """デモモード: サンプルケースの実行"""
    print("\n" + "=" * 70)
    print("デモモード: サンプルケースの実行")
    print("=" * 70)
    
    # テストケース1: 景品表示法
    print("\n[テストケース1] 景品表示法チェック")
    print("-" * 70)
    ad_text = "「業界No.1の実績!」という広告を出したいが、特に調査データはない。"
    print(f"入力: {ad_text}")
    print("\n結果:")
    result = checker.check_advertisement(ad_text)
    print(result)
    
    # テストケース2: 個人情報保護法
    print("\n" + "=" * 70)
    print("[テストケース2] プライバシーポリシーチェック")
    print("-" * 70)
    service_desc = "ユーザーの位置情報を常時取得して、近くの店舗をレコメンドする機能"
    print(f"入力: {service_desc}")
    print("\n結果:")
    result = checker.check_privacy_policy(service_desc)
    print(result)
    
    # テストケース3: 著作権法
    print("\n" + "=" * 70)
    print("[テストケース3] 著作権法チェック")
    print("-" * 70)
    feature_desc = "ユーザーがアップロードした音楽ファイルを自動で解析して、似た曲をレコメンドする機能"
    print(f"入力: {feature_desc}")
    print("\n結果:")
    result = checker.check_copyright(feature_desc)
    print(result)
    
    print("\n" + "=" * 70)


def run_interactive(checker):
    """インタラクティブモード: 対話型チェック"""
    print("\n" + "=" * 70)
    print("インタラクティブモード")
    print("=" * 70)
    print("終了するには 'quit' または 'exit' と入力してください\n")
    
    while True:
        print("\nチェックタイプを選択:")
        print("1. 広告文チェック(景品表示法)")
        print("2. プライバシーポリシーチェック(個人情報保護法)")
        print("3. 著作権チェック")
        print("4. カスタム質問")
        
        choice = input("\n選択 (1-4): ").strip()
        
        if choice.lower() in ['quit', 'exit']:
            print("終了します")
            break
        
        if choice == '1':
            text = input("\n広告文を入力: ")
            result = checker.check_advertisement(text)
        elif choice == '2':
            text = input("\nサービス仕様を入力: ")
            result = checker.check_privacy_policy(text)
        elif choice == '3':
            text = input("\n機能説明を入力: ")
            result = checker.check_copyright(text)
        elif choice == '4':
            instruction = input("\n指示文を入力: ")
            text = input("入力テキストを入力: ")
            result = checker.generate_response(instruction, text)
        else:
            print("無効な選択です")
            continue
        
        print("\n" + "=" * 70)
        print("結果:")
        print("=" * 70)
        print(result)
        print("=" * 70)


def run_batch(checker, test_file, output_file=None):
    """バッチモード: テストケースファイルを使用した評価"""
    print("\n" + "=" * 70)
    print("バッチモード")
    print("=" * 70)
    
    results = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            test_case = json.loads(line)
            
            print(f"\n[{line_num}] 処理中...")
            
            response = checker.generate_response(
                test_case['instruction'],
                test_case['input']
            )
            
            result = {
                'test_case_id': line_num,
                'instruction': test_case['instruction'],
                'input': test_case['input'],
                'expected_output': test_case.get('output', ''),
                'generated_output': response
            }
            
            results.append(result)
    
    print(f"\n✓ {len(results)}件の処理が完了しました")
    
    # 結果の保存
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ 結果を保存しました: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Tech-Comply-AI 推論スクリプト"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="elyza/ELYZA-japanese-Llama-2-7b",
        help="ベースモデルのHugging Face ID"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./results/final_adapter",
        help="アダプターのパス"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['demo', 'interactive', 'batch'],
        default='demo',
        help="実行モード"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="バッチモード用のテストファイル"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="結果の出力ファイル"
    )
    
    args = parser.parse_args()
    
    # チェッカーの初期化
    checker = ComplianceChecker(args.base_model, args.adapter_path)
    
    # モード別実行
    if args.mode == 'demo':
        run_demo(checker)
    elif args.mode == 'interactive':
        run_interactive(checker)
    elif args.mode == 'batch':
        if not args.test_file:
            print("エラー: --test_file を指定してください")
            return
        run_batch(checker, args.test_file, args.output_file)
    
    print("\n" + "=" * 70)
    print("処理完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
