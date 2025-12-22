"""
============================================================
Tech-Comply-AI v2 ファインチューニングスクリプト (修正版)
ELYZA-japanese-Llama-2-7b + QLoRA
============================================================
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
    """設定ファイル(YAML)を読み込む"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

parser = argparse.ArgumentParser(description="Fine-tune ELYZA with QLoRA (v2)")
parser.add_argument("--config_path", type=str, default="./config_v2.yaml")
args = parser.parse_args()

config = load_config(args.config_path)

print("=" * 70)
print("Tech-Comply-AI v2 ファインチューニング")
print("=" * 70)
print(f"Config: {args.config_path}")
print(f"Model: {config['model_name']}")
print(f"Output: {config['new_model_name']}")
print("=" * 70)

# モデルとトークナイザーの準備
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

print("✓ モデルとトークナイザーの読み込み完了")

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
trainable_percent = 100 * trainable_params / all_params

print("\n--- 学習可能パラメータ (v2) ---")
print(f"学習対象: {trainable_params:,} パラメータ")
print(f"全体:     {all_params:,} パラメータ")
print(f"学習率:   {trainable_percent:.4f}%")
print(f"LoRAランク: {config['peft_config']['r']} (v1: 16)")
print(f"対象モジュール数: {len(config['peft_config']['target_modules'])} (v1: 4)")
print("-" * 40)

# データセットの準備
print("\n[3/7] データセットを読み込んでいます...")

dataset = load_dataset("json", data_files=config["train_file_path"], split="train")
print(f"✓ {len(dataset)}件のデータを読み込みました (v1: 35件)")

# 訓練・評価データに分割 (90/10)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"  - 訓練データ: {len(train_dataset)}件")
print(f"  - 評価データ: {len(eval_dataset)}件")

SYSTEM_MESSAGE = (
    "あなたは日本の法律とコンプライアンスに精通した専門アシスタントです。"
    "景品表示法、個人情報保護法、著作権法などに基づいて、"
    "正確で実用的なアドバイスを提供してください。"
    "回答は必ず指定されたJSON形式で、それ以外の文章は含めないでください。"
)

def formatting_prompts_func(examples):
    """プロンプトフォーマット関数"""
    output_texts = []
    
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i] if examples['input'][i] else ""
        output_text = examples['output'][i]
        
        if input_text:
            full_instruction = f"{instruction}\n\n入力:\n{input_text}"
        else:
            full_instruction = instruction
        
        text = (
            f"<s>[INST] <<SYS>>\n{SYSTEM_MESSAGE}\n<</SYS>>\n\n"
            f"{full_instruction} [/INST] {output_text} </s>"
        )
        output_texts.append(text)
    
    return output_texts

print("\n--- フォーマット済みプロンプトのサンプル ---")
sample_formatted = formatting_prompts_func({
    'instruction': [train_dataset[0]['instruction']],
    'input': [train_dataset[0]['input']],
    'output': [train_dataset[0]['output']]
})
sample_text = sample_formatted[0]
print(sample_text[:400] + "..." if len(sample_text) > 400 else sample_text)
print("-" * 70)

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
    evaluation_strategy=config["training_arguments"]["eval_strategy"],  # eval_strategy → evaluation_strategy
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
    dataloader_num_workers=0,
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

print("✓ 学習の準備完了")
print("\n--- v2の改善点 ---")
print(f"1. データ量: 35件 → 150件 (4.3倍)")
print(f"2. LoRAランク: 16 → 32 (2倍)")
print(f"3. 学習モジュール: 4つ → 7つ")
print(f"4. エポック数: 3 → 5")
print(f"5. 学習率: 2e-4 → 1e-4 (より安定)")
print(f"6. 評価機能: なし → あり")
print("-" * 40)

# 学習実行
print("\n[5/7] 学習を開始します...")
print("=" * 70)
print("ℹ️  推定学習時間:")
print("   - T4 GPU: 6-10時間 (v1: 3-5時間)")
print("   - A100 GPU: 2-4時間 (v1: 1-2時間)")
print("ℹ️  50ステップごとに評価を実施")
print("=" * 70 + "\n")

try:
    trainer.train()
    
    print("\n" + "=" * 70)
    print("✓ 学習が正常に完了しました!")
    print("=" * 70)
    
except KeyboardInterrupt:
    print("\n⚠️ ユーザーによって学習が中断されました")
    print("チェックポイントは保存されています")
    
except Exception as e:
    print(f"\n❌ 学習中にエラーが発生しました: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# モデル保存
print("\n[6/7] モデルを保存しています...")

save_path = os.path.join(config["training_arguments"]["output_dir"], "final_adapter")
os.makedirs(save_path, exist_ok=True)

trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✓ モデルを保存しました: {save_path}")

# Google Driveへの保存
try:
    if os.path.exists('/content/drive/MyDrive'):
        drive_save_path = f"/content/drive/MyDrive/{config['new_model_name']}"
        os.makedirs(drive_save_path, exist_ok=True)
        
        import shutil
        shutil.copytree(
            save_path, 
            os.path.join(drive_save_path, "final_adapter"),
            dirs_exist_ok=True
        )
        
        print(f"✓ Google Driveにも保存しました: {drive_save_path}")
except Exception as e:
    print(f"ℹ️ Google Driveへの保存をスキップしました: {e}")

# 学習結果の要約
print("\n[7/7] 学習結果の要約...")
print("\n" + "=" * 70)
print("🎉 Tech-Comply-AI v2 学習完了!")
print("=" * 70)
print(f"\n📁 保存先: {save_path}")
print(f"📝 モデル名: {config['new_model_name']}")
print(f"📊 学習データ: {len(train_dataset)}件 (v1: 35件)")
print(f"📈 エポック数: {config['training_arguments']['num_train_epochs']} (v1: 3)")
print(f"🎯 LoRAランク: {config['peft_config']['r']} (v1: 16)")

print("\n--- v1からの改善期待値 ---")
print("JSON形式遵守率: 40% → 85%以上")
print("回答完結性: 50% → 90%以上")
print("法律知識精度: 良好 → 優秀")

print("\n次のステップ:")
print("1. inference_v2.py で推論テスト")
print("2. v1との比較評価")
print("3. モデルをZIP化してダウンロード")
print("=" * 70)
