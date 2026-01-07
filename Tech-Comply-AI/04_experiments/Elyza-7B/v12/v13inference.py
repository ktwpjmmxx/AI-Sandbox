import torch
from unsloth import FastLanguageModel
import os
import glob
import logging

# ==========================================
# 重要：Unsloth/Transformersの過剰な警告を抑制
# ==========================================
# これがないと、inv_freqエラーで止まることがあります
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

print("============================================================")
print("Elyza-7B v13 Inference Script")
print("============================================================")

# ==========================================
# 1. 最新のモデルを自動検索
# ==========================================
output_dir = "/content/outputs_v13" # 必ず絶対パスを使用

# checkpointフォルダを探す
checkpoints = glob.glob(f"{output_dir}/checkpoint-*")
if checkpoints:
    # 数字が一番大きい（最新の）チェックポイントを選択
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    model_path = latest_checkpoint
    print(f"✓ 最新の学習済みモデルを使用します: {model_path}")
else:
    # チェックポイントがない場合はディレクトリ自体を指定（学習完了後のsave_pretrained）
    model_path = output_dir
    print(f"✓ 出力ディレクトリを使用します: {model_path}")

# ==========================================
# 2. モデル読み込み
# ==========================================
try:
    print("モデルを読み込んでいます...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    print("✓ 読み込み成功")
except Exception as e:
    print(f"✗ 読み込み失敗。理由: {e}")
    exit(1)

# ==========================================
# 3. テスト実行関数
# ==========================================
# v13用の強化されたシステムプロンプト
system_prompt = """あなたは超一流のIT法務コンサルタントです。
ユーザーの質問に対し、曖昧な回答を避け、断定的な口調で、具体的かつ実務的な法的リスクと修正案を提示してください。
「検討が必要」「専門家へ相談」といった逃げの表現は禁止です。"""

def run_inference(question, expected_behavior):
    prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{question} [/INST]"""
    
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 512, 
        use_cache = True,
        temperature = 0.1, # 厳格さを出すために低めに設定
    )
    
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # プロンプト部分を除去して回答だけ抽出
    response = result.split("[/INST]")[-1].strip()
    
    print("-" * 60)
    print(f"【質問】\n{question}")
    print(f"\n【期待される挙動】\n{expected_behavior}")
    print(f"\n【v13の回答】\n{response}")
    print("-" * 60)

# ==========================================
# 4. 実証実験ケース
# ==========================================
test_cases = [
    {
        "q": "Googleアナリティクスを導入したいのですが、法的リスクと対策を教えてください。",
        "e": "電気通信事業法に言及し、「～してください」と断定的に指示する"
    },
    {
        "q": "自社のエンジニアが、他社のAPIを勝手に解析して、その制限を回避するパッチを作成して公開してしまった。会社としてどのような責任を負う可能性がある？",
        "e": "不正競争防止法違反および著作権法違反を断定し、即時削除を命令する"
    },
    {
        "q": "Reactでコンポーネントを作る際のベストプラクティスを教えてください。",
        "e": "「専門外です」と明確に拒絶する"
    }
]

print("\n推論テスト開始\n")

for i, test in enumerate(test_cases):
    print(f"\nTest {i+1}")
    run_inference(test["q"], test["e"])
    # 必要ならここでinput()を入れる
    # input("\n[Enter]キーを押して次へ...")

print("\n全テスト完了")