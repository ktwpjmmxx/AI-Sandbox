import os
import yaml
import torch
import argparse
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
# transformersの警告レベルを上げて、Unslothがエラーを検知できないようにする
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
from unsloth import FastLanguageModel

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="./config_v5.yaml") # デフォルト名を修正
args = parser.parse_args()
config = load_config(args.config_path)

# --- 1. モデルロード ---
max_seq_length = config["training_arguments"].get("max_seq_length", 2048)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config["model_name"],
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = config["peft_config"]["r"],
    target_modules = config["peft_config"]["target_modules"],
    lora_alpha = config["peft_config"]["lora_alpha"],
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# --- 2. データ読み込みとフォーマット変換 ---
dataset = load_dataset("json", data_files=config["train_file_path"], split="train")

# ELYZA/Llama2形式のプロンプト作成
prompt_style = """<s>[INST] <<SYS>>
あなたはIT法務およびコンプライアンスの専門コンサルタントです。
<</SYS>>

{}

{} [/INST] {} </s>"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = prompt_style.format(instruction, input_text, output)
        texts.append(text)
    return { "text" : texts, }

train_dataset = dataset.map(formatting_prompts_func, batched = True)

# --- 3. トレーニング設定 ---
training_args = TrainingArguments(
    output_dir = config["training_arguments"]["output_dir"],
    per_device_train_batch_size = config["training_arguments"]["per_device_train_batch_size"],
    gradient_accumulation_steps = config["training_arguments"]["gradient_accumulation_steps"],
    warmup_steps = config["training_arguments"]["warmup_steps"],
    num_train_epochs = config["training_arguments"]["num_train_epochs"],
    learning_rate = float(config["training_arguments"]["learning_rate"]),
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    save_strategy = "steps",
    save_steps = 50,
    save_total_limit = 2,
    report_to = "none",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text", # map関数で作ったカラムを指定
    max_seq_length = max_seq_length,
    dataset_num_proc = 1, # 安定性のため1に
    packing = False,
    args = training_args,
)

trainer.train()

# --- 4. 保存 ---
model.save_pretrained(os.path.join(config["training_arguments"]["output_dir"], "final_adapter"))
tokenizer.save_pretrained(os.path.join(config["training_arguments"]["output_dir"], "final_adapter"))
print("✅ 学習が完了し、アダプターが保存されました。")