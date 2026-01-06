import torch
from unsloth import FastLanguageModel
import argparse
import logging

# --- エラー回避対策 ---
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error() 
# --------------------

# 引数設定
parser = argparse.ArgumentParser()
parser.add_argument("--adapter_path", type=str, default="./results_unsloth/final_adapter")
args = parser.parse_args()

print(f"📦 モデルをロード中: {args.adapter_path}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.adapter_path, 
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True,
    fix_tokenizer = True,
    # 重要：ここで学習時と同じエラー回避フラグを入れる
    ignore_mismatched_sizes = True, 
)
FastLanguageModel.for_inference(model)

# 質問の準備
instruction = "新しいAIサービスの利用規約を作成する際の注意点を教えてください。"
input_text = ""

# 学習時と全く同じプロンプト形式にする
prompt_style = """<s>[INST] <<SYS>>
あなたはIT法務およびコンプライアンスの専門コンサルタントです。
<</SYS>>

{}

{} [/INST]"""

inputs = tokenizer(
    [prompt_style.format(instruction, input_text)],
    return_tensors = "pt"
).to("cuda")

# 回答生成
print("🤖 AIが回答を生成中...")
outputs = model.generate(
    **inputs, 
    max_new_tokens = 512,
    use_cache = True,
    temperature = 0.6,
    top_p = 0.9,
    repetition_penalty = 1.2 # 繰り返しを防ぐ設定を追加
)

print("\n" + "="*50)
print("AIの回答:")
# [INST] のあとの回答部分だけを抽出して表示
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(result.split("[/INST]")[-1].strip())
print("="*50)