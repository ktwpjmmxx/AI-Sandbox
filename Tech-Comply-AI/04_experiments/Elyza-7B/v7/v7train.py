import unsloth
import accelerate
import os
import logging
import torch
import psutil
from datasets import load_dataset
from transformers import TrainingArguments, logging as transformers_logging
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer

# --- 1. 互換性問題の強制解決（モンキーパッチ） ---
# TransformersとAccelerateのバージョン不整合によるTypeErrorを回避
original_unwrap = accelerate.Accelerator.unwrap_model
def patched_unwrap(self, model, keep_torch_compile=False):
    # keep_torch_compile引数が来ても無視して実行する
    return original_unwrap(self, model)
accelerate.Accelerator.unwrap_model = patched_unwrap

# --- 2. ログ設定（理不尽なガードレールを回避） ---
transformers_logging.set_verbosity_error()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
os.environ["UNSLOTH_CHECK_VERSION"] = "false"

# --- 3. モデル & トークナイザーロード ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    fix_tokenizer = True,
)

# LoRA設定（高ランク設定で論理拒絶を叩き込む）
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 128,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# --- 4. データセットの準備 ---
prompt_template = """<s>[INST] <<SYS>>
あなたはIT法務およびコンプライアンスの専門コンサルタントです。
提供された情報を分析し、法的リスク、該当法、理由、修正案をプロフェッショナルに回答してください。
守備範囲外（技術選定、具体的な損害賠償額の算定、労働基準法、商標侵害の具体的認定など）の相談には、適切に回答を拒絶してください。
<</SYS>>

{instruction}

{input} [/INST] {output} </s>"""

dataset = load_dataset("json", data_files="train_data_471.jsonl", split="train")

def format_func(examples):
    texts = [prompt_template.format(instruction=i, input=n, output=o) 
             for i, n, o in zip(examples["instruction"], examples["input"], examples["output"])]
    return {"text": texts}

train_dataset = dataset.map(format_func, batched=True, remove_columns=dataset.column_names)

# --- 5. トレーナー設定（psutilエラー回避設定込み） ---
trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 1, # psutil.cpu_count()のエラーを回避
    args = TrainingArguments(
        output_dir = "outputs_v7",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        num_train_epochs = 10,
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.1,
        lr_scheduler_type = "cosine",
        save_strategy = "epoch",
        report_to = "none",
        seed = 3407,
    ),
)

# --- 6. 学習実行 ---
print("🚀 学習を開始します...")
trainer.train()

# --- 7. 保存 ---
model.save_pretrained_merged("outputs_v7/final_model", tokenizer, save_method="merged_16bit")
print("✅ 学習完了: outputs_v7/final_model")