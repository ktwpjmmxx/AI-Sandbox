import os
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
import yaml
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. 設定の読み込み
with open("config_v11.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 2. モデルとトークナイザーの準備
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config['model_config']['base_model'],
    max_seq_length = config['model_config']['max_seq_length'],
    load_in_4bit = config['model_config']['load_in_4bit'],
)

# 3. LoRA設定の適用
model = FastLanguageModel.get_peft_model(
    model,
    r = config['lora_config']['r'],
    target_modules = config['lora_config']['target_modules'],
    lora_alpha = config['lora_config']['lora_alpha'],
    lora_dropout = config['lora_config']['lora_dropout'],
    bias = config['lora_config']['bias'],
    use_gradient_checkpointing = config['lora_config']['use_gradient_checkpointing'],
    random_state = config['lora_config']['random_state'],
)

# 4. データセットの読み込みと整形
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # ELYZA/Llama-2 のプロンプト形式に合わせる
        text = f"<s>[INST] <<SYS>>\\n{instruction}\\n<</SYS>>\\n\\n{input} [/INST] {output} </s>"
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files=config['training_config']['dataset_path'], split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. トレーナーの設定
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = config['model_config']['max_seq_length'],
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = config['training_config']['per_device_train_batch_size'],
        gradient_accumulation_steps = config['training_config']['gradient_accumulation_steps'],
        warmup_steps = config['training_config']['warmup_steps'],
        num_train_epochs = config['training_config']['num_train_epochs'],
        learning_rate = float(config['training_config']['learning_rate']),
        fp16 = config['training_config']['fp16'],
        logging_steps = config['training_config']['logging_steps'],
        optim = config['training_config']['optim'],
        weight_decay = config['training_config']['weight_decay'],
        lr_scheduler_type = config['training_config']['lr_scheduler_type'],
        seed = config['training_config']['seed'],
        output_dir = config['training_config']['output_dir'],
    ),
)

# 6. 学習実行
trainer_stats = trainer.train()

# 7. モデルの保存
model.save_pretrained(config['training_config']['output_dir'])
tokenizer.save_pretrained(config['training_config']['output_dir'])
print(f"v11学習完了: {config['training_config']['output_dir']} に保存されました")