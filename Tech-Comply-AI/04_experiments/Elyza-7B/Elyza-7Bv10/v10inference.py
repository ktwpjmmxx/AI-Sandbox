import os
# Transformersの「rotary_emb.inv_freq」未初期化によるRuntimeErrorを回避
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

from unsloth import FastLanguageModel
import torch

# 最新のチェックポイント、またはLoRAフォルダを指定
model_path = "outputs_v10/checkpoint-210" 

# 1. モデルの読み込み
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 2. 推論実行関数
def generate_response(question, system_mode="legal"):
    if system_mode == "legal":
        sys_msg = "IT法務コンサルタントとして、リスク判定と修正案を提示してください。専門外の質問には適切に回答を拒絶してください。"
    else:
        sys_msg = "あなたはITに詳しい優秀なアシスタントです。質問に対して柔軟に回答してください。"

    prompt = f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{question} [/INST] "
    
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    # v10はガードレールが強いため、自由度を少し上げる(0.5-0.7)のがコツ
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 512, 
        temperature = 0.5, 
        top_p = 0.9,
        repetition_penalty = 1.1 # v7で起きたループを防止
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1]

# 3. テスト
if __name__ == "__main__":
    print("⚖️ 法務テスト:", generate_response("SaaSでの個人情報保護対応について"))
    print("☕ 汎用テスト:", generate_response("おすすめの休憩法は？", system_mode="soft"))