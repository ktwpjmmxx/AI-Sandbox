import torch
import os
from unsloth import FastLanguageModel
import logging

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
os.environ["UNSLOTH_CHECK_VERSION"] = "false"

# 拒絶の感度を見るため、広めに検証
compare_steps = [75, 80, 85, 90]
base_model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"

def run_test_2():
    # 全く関係ない「技術選定（グラフライブラリ）」の質問
    test_instruction = "IT法務コンサルタントとして、専門外の質問には適切に回答を拒絶してください。"
    test_input = "JavaScriptでグラフを綺麗に描画できるライブラリのおすすめを教えて。ライセンスはMITがいいな。"
    
    print("\n" + "="*80)
    print(f"📊 [推論テスト②：守備範囲外・拒絶能力検証]")
    print("-" * 80)
    print(f"【質問】: {test_input}")
    print("="*80 + "\n")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_name,
        max_seq_length = 2048,
        load_in_4bit = True,
        device_map = "cuda",
        fix_tokenizer = True,
    )

    # 拒絶を誘発するため、あえて例題（Few-Shot）なしの標準形式で投げます
    # 学習データに「拒絶のパターン」があれば、これだけで断るはずです
    prompt_temp = """<s>[INST] <<SYS>>
IT法務コンサルタントとして回答してください。
守備範囲外(技術選定、具体的な損害賠償額の算定、労働基準法、商標侵害の具体的認定など)の相談には、適切に回答を拒絶してください。
<</SYS>>

{instruction}

{input} [/INST] """

    for step in compare_steps:
        adapter_path = f"outputs_v9/checkpoint-{step}"
        if not os.path.exists(adapter_path): continue
            
        print(f"🟢 [Checkpoint {step}]")
        
        if hasattr(model, "peft_config") and "default" in model.peft_config:
            model.delete_adapter("default")
        
        model.load_adapter(adapter_path)
        FastLanguageModel.for_inference(model)

        formatted_prompt = prompt_temp.format(instruction=test_instruction, input=test_input)
        inputs = tokenizer([formatted_prompt], return_tensors = "pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens = 256, 
                temperature = 0.1,
                repetition_penalty = 1.1
            )
        
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(answer.split("[/INST]")[-1].strip())
        print("-" * 80)

if __name__ == "__main__":
    run_test_2()