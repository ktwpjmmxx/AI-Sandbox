from unsloth import FastLanguageModel
import torch

# v11の成果物を指定
model_path = "outputs_v11" 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

def ask_v11(question):
    # デトックスデータセットと同じ指示文を使用
    sys_msg = "IT法務コンサルタントとして、プロダクト仕様に関連する法的リスクを判定し、実務的な修正案を提示してください。"
    
    prompt = f"<s>[INST] <<SYS>>\\n{sys_msg}\\n<</SYS>>\\n\\n{question} [/INST] "
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # 回答部分のみを抽出
    return result.split("[/INST]")[-1].strip()

# テスト実行
if __name__ == "__main__":
    q = "Googleアナリティクスを導入したいのですが、法的リスクと対策を教えてください。"
    print(f"Q: {q}\nA: {ask_v11(q)}")