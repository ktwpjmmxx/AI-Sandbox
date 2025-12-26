from unsloth import FastLanguageModel
import torch

# 保存した中から最適なエポックを選択するか、finalをロード
model_path = "outputs_v7/checkpoint-30"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    device_map = "cuda",
)
FastLanguageModel.for_inference(model)

def legal_consult(instruction, user_input):
    prompt = f"""<s>[INST] <<SYS>>
あなたはIT法務およびコンプライアンスの専門コンサルタントです。
提供された情報を分析し、法的リスク、該当法、理由、修正案をプロフェッショナルに回答してください。
守備範囲外（技術選定、具体的な損害賠償額の算定、労働基準法、商標侵害の具体的認定など）の相談には、適切に回答を拒絶してください。
<</SYS>>

{instruction}

{user_input} [/INST] """

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # temperatureを極限まで下げて「揺らぎ」を排除
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024, 
        temperature=0.1, 
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        use_cache=True
    )
    
    response = tokenizer.batch_decode(outputs)
    print(response[0].split("[/INST]")[1].replace("</s>", "").strip())

# --- テスト実行 ---
print("--- 期待される回答（法務） ---")
legal_consult("IT法務コンサルタントとして、以下の状況を評価してください。", "自社開発したAIモデルの学習に、許諾なくネット上の画像をスクレイピングして使用することの是非。")

print("\n--- 期待される拒絶（範囲外） ---")
legal_consult("IT法務コンサルタントとして、相談に回答してください。", "このPythonコードのバグを直して。")