"""
============================================================
Tech-Comply-AI ファインチューニングスクリプト (修正版)
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

# ==========================================
# 1. 設定の読み込み
# ==========================================
def load_config(config_path):
    """設定ファイル(YAML)を読み込む"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# コマンドライン引数のパース
parser = argparse.ArgumentParser(description="Fine-tune ELYZA with QLoRA")
parser.add_argument(
    "--config_path", 
    type=str, 
    default="./config_base.yaml", 
    help="Path to the config file"
)
parser.add_argument(
    "--use_wandb", 
    action="store_true", 
    help="Enable Weights & Biases logging"
)
args = parser.parse_args()

# 設定読み込み
config = load_config(args.config_path)

print("=" * 70)
print(f"Config: {args.config_path}")
print(f"Model: {config['model_name']}")
print(f"Output: {config['new_model_name']}")
print("=" * 70)

# ==========================================
# 2. モデルとトークナイザーの準備
# ==========================================
print("\n[1/6] モデルとトークナイザーを読み込んでいます...")

# 4-bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=config["bnb_config"]["load_in_4bit"],
    bnb_4bit_quant_type=config["bnb_config"]["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, config["bnb_config"]["bnb_4bit_compute_dtype"]),
    bnb_4bit_use_double_quant=config["bnb_config"]["bnb_4bit_use_double_quant"],
)

# モデル読み込み
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False,
    trust_remote_code=True,
)

# 学習用の前処理
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

# トークナイザー読み込み
tokenizer = AutoTokenizer.from_pretrained(
    config["model_name"],
    trust_remote_code=True,
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✓ モデルとトークナイザーの読み込み完了")

# ==========================================
# 3. LoRA設定
# ==========================================
print("\n[2/6] LoRA設定を適用しています...")

peft_config = LoraConfig(
    r=config["peft_config"]["r"],
    lora_alpha=config["peft_config"]["lora_alpha"],
    lora_dropout=config["peft_config"]["lora_dropout"],
    bias=config["peft_config"]["bias"],
    task_type=config["peft_config"]["task_type"],
    target_modules=config["peft_config"]["target_modules"],
)

model = get_peft_model(model, peft_config)

# 学習可能パラメータの表示
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
trainable_percent = 100 * trainable_params / all_params

print("\n--- 学習可能パラメータ ---")
print(f"学習対象: {trainable_params:,} パラメータ")
print(f"全体:     {all_params:,} パラメータ")
print(f"学習率:   {trainable_percent:.4f}%")
print("-" * 40)

# ==========================================
# 4. データセットの準備
# ==========================================
print("\n[3/6] データセットを読み込んでいます...")

dataset = load_dataset("json", data_files=config["train_file_path"], split="train")
print(f"✓ {len(dataset)}件のデータを読み込みました")

# システムメッセージ(全データ共通)
SYSTEM_MESSAGE = (
    "あなたは日本の法律とコンプライアンスに精通した専門アシスタントです。"
    "景品表示法、個人情報保護法、著作権法などに基づいて、"
    "正確で実用的なアドバイスを提供してください。"
)

# プロンプトフォーマット関数（修正版：リストを返す）
def formatting_prompts_func(examples):
    """
    バッチのexamplesを受け取り、フォーマット済みの文字列のリストを返す
    
    重要: TRL 0.8.1以降では、リストを返す必要があります
    """
    output_texts = []
    
    # バッチ処理
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i] if examples['input'][i] else ""
        output_text = examples['output'][i]
        
        # instructionとinputを結合
        if input_text:
            full_instruction = f"{instruction}\n\n入力:\n{input_text}"
        else:
            full_instruction = instruction
        
        # ELYZAの標準フォーマット
        text = (
            f"<s>[INST] <<SYS>>\n{SYSTEM_MESSAGE}\n<</SYS>>\n\n"
            f"{full_instruction} [/INST] {output_text} </s>"
        )
        output_texts.append(text)
    
    return output_texts

# サンプルデータの確認
print("\n--- フォーマット済みプロンプトのサンプル ---")
sample_formatted = formatting_prompts_func({
    'instruction': [dataset[0]['instruction']],
    'input': [dataset[0]['input']],
    'output': [dataset[0]['output']]
})
sample_text = sample_formatted[0]
if len(sample_text) > 500:
    print(sample_text[:500] + "...")
else:
    print(sample_text)
print("-" * 70)

# ==========================================
# 5. 学習設定
# ==========================================
print("\n[4/6] 学習の準備をしています...")

training_args = TrainingArguments(
    output_dir=config["training_arguments"]["output_dir"],
    num_train_epochs=config["training_arguments"]["num_train_epochs"],
    per_device_train_batch_size=config["training_arguments"]["per_device_train_batch_size"],
    gradient_accumulation_steps=config["training_arguments"]["gradient_accumulation_steps"],
    learning_rate=float(config["training_arguments"]["learning_rate"]),
    weight_decay=config["training_arguments"]["weight_decay"],
    fp16=config["training_arguments"]["fp16"],
    logging_steps=config["training_arguments"]["logging_steps"],
    save_strategy=config["training_arguments"]["save_strategy"],
    save_steps=config["training_arguments"]["save_steps"],
    max_grad_norm=config["training_arguments"]["max_grad_norm"],
    warmup_ratio=config["training_arguments"]["warmup_ratio"],
    group_by_length=config["training_arguments"]["group_by_length"],
    lr_scheduler_type=config["training_arguments"]["lr_scheduler_type"],
    report_to="none",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    dataloader_num_workers=0,
)

# SFTTrainer (修正版)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=training_args,
    max_seq_length=512,
    packing=False,
)

print("✓ 学習の準備完了")

# ==========================================
# 6. 学習実行
# ==========================================
print("\n[5/6] 学習を開始します...")
print("=" * 70)
print("ℹ️  学習には数時間かかる場合があります")
print("ℹ️  進捗はログで確認できます")
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

# ==========================================
# 7. モデル保存
# ==========================================
print("\n[6/6] モデルを保存しています...")

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

# ==========================================
# 完了
# ==========================================
print("\n" + "=" * 70)
print("🎉 すべての処理が完了しました!")
print("=" * 70)
print(f"\n📁 保存先: {save_path}")
print(f"📝 モデル名: {config['new_model_name']}")
print(f"📊 学習データ: {len(dataset)}件")
print(f"📈 エポック数: {config['training_arguments']['num_train_epochs']}")
print("\n次のステップ:")
print("1. inference.py で推論テスト")
print("2. モデルをZIP化してダウンロード")
print("3. 本番環境へデプロイ")
print("=" * 70)
