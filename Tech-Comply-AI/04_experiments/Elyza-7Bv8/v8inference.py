from unsloth import FastLanguageModel
import torch

# 検証したいステップ番号を指定 (例: 1.5 epoch = 45step / 2.0 epoch = 60step)
target_step = 50 
model_path = f"outputs_v8/checkpoint-{target_step}"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    load_in_4bit = True,
    device_map = "cuda",
)
FastLanguageModel.for_inference(model)

def test_v8(input_text):
    instruction = "IT法務コンサルタントとして回答してください。"
    # システムプロンプトはv7train.pyと完全に一致させる
    prompt = f"""<s>[INST] <<SYS>>
あなたはIT法務およびコンプライアンスの専門コンサルタントです。
提供された情報を分析し、法的リスク、該当法、理由、修正案をプロフェッショナルに回答してください。
守備範囲外（技術選定、具体的な損害賠償額の算定、労働基準法、商標侵害の具体的認定など）の相談には、適切に回答を拒絶してください。
<</SYS>>

{instruction}

{input_text} [/INST] """

    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 512, 
        use_cache = True,
        temperature = 0.1,
        repetition_penalty = 1.15 # v7で発生したループへの保険
    )
    
    result = tokenizer.batch_decode(outputs)
    print(f"\n--- Checkpoint Step {target_step} Response ---")
    print(result[0].split("[/INST]")[1].replace("</s>", "").strip())

# 検証実行
test_v8("弊社の新しいサービスのロゴが、他社の商標を侵害しているかどうか具体的に判定してください。")