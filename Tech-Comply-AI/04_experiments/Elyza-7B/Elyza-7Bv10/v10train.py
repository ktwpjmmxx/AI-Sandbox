from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. モデル読み込み
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. LoRA設定適用
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

# 3. データセット準備
dataset = load_dataset("json", data_files="traindata_v10.jsonl", split="train")

# 4. トレーナー設定
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_v10",
        save_strategy = "steps",
        save_steps = 30, # 頻繁に保存して最良地点を探れるようにする
    ),
)

# 5. 学習開始
trainer.train()

# 6. 保存（LoRAアダプタのみ）
model.save_pretrained("Elyza-7B-IT-Legal-v10-lora")
tokenizer.save_pretrained("Elyza-7B-IT-Legal-v10-lora")