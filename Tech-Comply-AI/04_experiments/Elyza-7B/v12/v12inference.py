import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
from unsloth import FastLanguageModel
import torch

print("=" * 60)
print("Elyza-7B v12 Inference Script")
print("=" * 60)

# モデルの読み込み
print("\nモデルを読み込んでいます...")
model_path = "/content/outputs_v12/checkpoint-414" 

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    print(f"✓ モデル読み込み完了: {model_path}")
except Exception as e:
    print(f"✗ 読み込み失敗。本当の理由: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

def ask_v12(question, system_message=None):
    """
    v12モデルに質問を投げて回答を取得
    
    Args:
        question: 質問文
        system_message: システムメッセージ（省略時はデフォルト）
    
    Returns:
        回答文字列
    """
    if system_message is None:
        system_message = "あなたは超一流のIT法務コンサルタントです。『検討が必要』といった曖昧な回答は避け、 具体的かつ断定的にリスクと修正案を提示してください。"
    
    prompt = f"<s>[INST] <<SYS>>\\n{system_message}\\n<</SYS>>\\n\\n{question} [/INST] "
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # 回答部分のみを抽出
    return result.split("[/INST]")[-1].strip()

# テストケース集
test_cases = [
    {
        "name": "Test 1: Googleアナリティクス（v11失敗ケース）",
        "question": "Googleアナリティクスを導入したいのですが、法的リスクと対策を教えてください。",
        "expected": "電気通信事業法の外部送信規律に言及し、具体的な対策を提案する"
    },
    {
        "name": "Test 2: Cookie同意バナー",
        "question": "Cookie同意バナーを実装する際の法的要件を教えてください。",
        "expected": "電気通信事業法、個人情報保護法の観点から要件を説明する"
    },
    {
        "name": "Test 3: 利用規約の改定",
        "question": "利用規約を改定したいのですが、ユーザーへの通知方法に法的制約はありますか？",
        "expected": "消費者契約法や約款規制の観点から説明する"
    },
    {
        "name": "Test 4: 専門外ケース（正常な拒絶）",
        "question": "Reactでコンポーネントを作る際のベストプラクティスを教えてください。",
        "expected": "技術的な質問なので専門外として適切に拒絶する"
    }
]

# メインテスト実行
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("推論テスト開始")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"{test['name']}")
        print(f"{'='*60}")
        print(f"\n【質問】\n{test['question']}")
        print(f"\n【期待される挙動】\n{test['expected']}")
        print(f"\n【v12の回答】")
        print("-" * 60)
        
        try:
            answer = ask_v12(test['question'])
            print(answer)
        except Exception as e:
            print(f"✗ エラーが発生しました: {e}")
        
        print("-" * 60)
        
        # 次のテストまで区切り
        if i < len(test_cases):
            input("\n[Enter]キーを押して次のテストへ...")
    
    print("\n" + "=" * 60)
    print("全テスト完了")
    print("=" * 60)
    
    # インタラクティブモード
    print("\n\nインタラクティブモードに入ります")
    print("'exit'または'quit'で終了")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n質問を入力してください: ")
            if question.lower() in ['exit', 'quit', 'q']:
                print("終了します")
                break
            
            if question.strip() == "":
                continue
                
            answer = ask_v12(question)
            print(f"\n【回答】\n{answer}\n")
            
        except KeyboardInterrupt:
            print("\n\n終了します")
            break
        except Exception as e:
            print(f"\nエラー: {e}")

# !pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo transformers timm
# ライブラリの競合エラーが出た際の対策