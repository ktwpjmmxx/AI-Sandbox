"""
Elyza-7B v3 ファインチューニングスクリプト

"""

import os
import yaml
import torch
import argparse
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

parser = argparse.ArgumentParser(description="Fine-tune ELYZA with QLoRA (v3)")
parser.add_argument("--config_path", type=str, default="./config_v3.yaml")
args = parser.parse_args()

config = load_config(args.config_path)

print("=" * 70)
print("Elyza-7B v3 ファインチューニング")
print("=" * 70)
print(f"Model: {config['model_name']}")
print(f"Output: {config['new_model_name']}")
print("=" * 70)

# モデルとトークナイザー
print("\n[1/7] モデルとトークナイザーを読み込んでいます...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=config["bnb_config"]["load_in_4bit"],
    bnb_4bit_quant_type=config["bnb_config"]["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, config["bnb_config"]["bnb_4bit_compute_dtype"]),
    bnb_4bit_use_double_quant=config["bnb_config"]["bnb_4bit_use_double_quant"],
)

model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False,
    trust_remote_code=True,
)

model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(
    config["model_name"],
    trust_remote_code=True,
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✓ 完了")

# LoRA設定
print("\n[2/7] LoRA設定を適用しています...")

peft_config = LoraConfig(
    r=config["peft_config"]["r"],
    lora_alpha=config["peft_config"]["lora_alpha"],
    lora_dropout=config["peft_config"]["lora_dropout"],
    bias=config["peft_config"]["bias"],
    task_type=config["peft_config"]["task_type"],
    target_modules=config["peft_config"]["target_modules"],
)

model = get_peft_model(model, peft_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

print(f"学習対象: {trainable_params:,} パラメータ")
print(f"全体: {all_params:,} パラメータ")
print(f"LoRAランク: {config['peft_config']['r']}")

# データセット
print("\n[3/7] データセットを読み込んでいます...")

dataset = load_dataset("json", data_files=config["train_file_path"], split="train")
print(f"✓ {len(dataset)}件のデータを読み込みました")

split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"  - 訓練: {len(train_dataset)}件")
print(f"  - 評価: {len(eval_dataset)}件")

SYSTEM_MESSAGE = (
    "あなたはIT法務およびコンプライアンスの専門コンサルタントです。"
    "ユーザーから提示された契約書やサービス内容の条項に対し、法的リスクを分析してください。"
    "回答は以下のフォーマットに従い、専門的かつ分かりやすい日本語で記述してください。\n\n"
    "【リスクレベル】\n"
    "（低・中・高とその判定理由）\n\n"
    "【該当法】\n"
    "（関連する法律名や条文）\n\n"
    "【法的根拠・理由】\n"
    "（なぜ違反の可能性があるのか詳細に解説）\n\n"
    "【修正案】\n"
    "（具体的な条文の書き換え例や、サービス仕様の変更提案）"
)

def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i] if examples['input'][i] else ""
        output_text = examples['output'][i]
        
        if input_text:
            full_instruction = f"{instruction}\\n\\n入力:\\n{input_text}"
        else:
            full_instruction = instruction
        
        text = (
            f"<s>[INST] <<SYS>>\\n{SYSTEM_MESSAGE}\\n<</SYS>>\\n\\n"
            f"{full_instruction} [/INST] {output_text} </s>"
        )
        output_texts.append(text)
    return output_texts

# 学習設定
print("\n[4/7] 学習の準備をしています...")

training_args = TrainingArguments(
    output_dir=config["training_arguments"]["output_dir"],
    num_train_epochs=config["training_arguments"]["num_train_epochs"],
    per_device_train_batch_size=config["training_arguments"]["per_device_train_batch_size"],
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=config["training_arguments"]["gradient_accumulation_steps"],
    learning_rate=float(config["training_arguments"]["learning_rate"]),
    weight_decay=config["training_arguments"]["weight_decay"],
    fp16=config["training_arguments"]["fp16"],
    logging_steps=config["training_arguments"]["logging_steps"],
    save_strategy=config["training_arguments"]["save_strategy"],
    save_steps=config["training_arguments"]["save_steps"],
    evaluation_strategy=config["training_arguments"]["eval_strategy"],
    eval_steps=config["training_arguments"]["eval_steps"],
    load_best_model_at_end=config["training_arguments"]["load_best_model_at_end"],
    metric_for_best_model=config["training_arguments"]["metric_for_best_model"],
    max_grad_norm=config["training_arguments"]["max_grad_norm"],
    warmup_ratio=config["training_arguments"]["warmup_ratio"],
    group_by_length=config["training_arguments"]["group_by_length"],
    lr_scheduler_type=config["training_arguments"]["lr_scheduler_type"],
    save_total_limit=config["training_arguments"]["save_total_limit"],
    report_to="none",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=training_args,
    max_seq_length=512,
    packing=False,
)

print("✓ 準備完了")
print("\n--- v3の改善点 ---")
print("1. JSON形式の厳密化")
print("2. エポック数: 7（v2: 5）")
print("3. 学習率: 8e-5（より安定）")
print("4. システムメッセージ強化")

# 学習実行
print("\n[5/7] 学習を開始します...")
print("=" * 70)
print("推定時間: T4 8-12時間, A100 3-5時間")
print("=" * 70)

try:
    trainer.train()
    print("\n✓ 学習完了!")
except Exception as e:
    print(f"\n❌ エラー: {e}")
    exit(1)

# モデル保存
print("\n[6/7] モデルを保存しています...")

save_path = os.path.join(config["training_arguments"]["output_dir"], "final_adapter")
os.makedirs(save_path, exist_ok=True)

trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✓ 保存完了: {save_path}")

# Google Drive保存
try:
    if os.path.exists('/content/drive/MyDrive'):
        drive_save_path = f"/content/drive/MyDrive/{config['new_model_name']}"
        os.makedirs(drive_save_path, exist_ok=True)
        
        import shutil
        shutil.copytree(save_path, os.path.join(drive_save_path, "final_adapter"), dirs_exist_ok=True)
        print(f"✓ Google Driveにも保存: {drive_save_path}")
except:
    pass

print("\n[7/7] 完了!")
print("\n" + "=" * 70)
print("Elyza-7B v3 学習完了!")
print("=" * 70)
print(f"\n保存先: {save_path}")
print(f"エポック数: {config['training_arguments']['num_train_epochs']}")
print("\n期待される改善:")
print("- JSON形式遵守率: 90%以上")
print("- 余計なテキスト: なし")
print("\n次: python inference_v3.py --adapter_path {save_path}")
print("=" * 70)
