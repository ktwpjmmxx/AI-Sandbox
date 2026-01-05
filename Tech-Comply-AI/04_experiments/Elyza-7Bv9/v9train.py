#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elyza-7B IT法務ファインチューニング v9
成功構成を再現 (v8ベース)

学習成功日: 2025年12月26日
環境: Google Colab (Tesla T4)
構成: psutilエラー完全回避版
"""

import os
import sys
import yaml

# ===== CRITICAL: psutilを最初にbuiltinsに登録 =====
import psutil
import builtins
builtins.psutil = psutil

# 環境変数設定
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['UNSLOTH_CHECK_VERSION'] = 'false'

# 通常のインポート
import torch
import logging
from datasets import load_dataset
from transformers import TrainingArguments, logging as transformers_logging

# Unslothのインポート
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
import accelerate
from trl import SFTTrainer

# Acceleratorの互換性パッチ
try:
    original_unwrap = accelerate.Accelerator.unwrap_model
    def patched_unwrap(self, model, keep_torch_compile=False):
        return original_unwrap(self, model)
    accelerate.Accelerator.unwrap_model = patched_unwrap
except Exception as e:
    print(f"⚠️ Acceleratorパッチ適用時の警告: {e}")

# ログ設定
transformers_logging.set_verbosity_error()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

print("=" * 60)
print("🦥 Elyza-7B IT法務ファインチューニング v9")
print("=" * 60)

# ===== 設定ファイルの読み込み =====
config_file = "config_v9.yaml"
if os.path.exists(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"✅ 設定ファイル読み込み: {config_file}")
else:
    print(f"⚠️ {config_file} が見つかりません。デフォルト設定を使用します。")
    config = {
        'model_name': 'elyza/ELYZA-japanese-Llama-2-7b-instruct',
        'new_model_name': 'Elyza-7B-IT-Legal-v9',
        'train_file_path': './train_data.jsonl',
        'peft_config': {
            'r': 64,
            'lora_alpha': 128,
            'lora_dropout': 0,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        },
        'training_arguments': {
            'output_dir': './outputs_v9',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 4,
            'learning_rate': 1e-5,
            'warmup_steps': 10,
            'logging_steps': 1,
            'save_strategy': 'steps',
            'save_steps': 5,
            'max_seq_length': 2048,
            'weight_decay': 0.1,
            'lr_scheduler_type': 'cosine',
            'optim': 'adamw_8bit',
            'report_to': 'none',
            'seed': 3407
        }
    }

# 設定の展開
model_name = config['model_name']
new_model_name = config['new_model_name']
train_file_path = config['train_file_path']
peft_config = config['peft_config']
train_args = config['training_arguments']

print(f"\n📋 設定:")
print(f"  - モデル: {model_name}")
print(f"  - 出力名: {new_model_name}")
print(f"  - データ: {train_file_path}")
print(f"  - LoRA rank: {peft_config['r']}")
print(f"  - 学習率: {train_args['learning_rate']}")
print(f"  - Epoch数: {train_args['num_train_epochs']}")

# ===== モデルロード =====
print("\n📦 モデルをロード中...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=train_args['max_seq_length'],
    dtype=None,
    load_in_4bit=True,
    fix_tokenizer=True,
)
print("✅ モデルロード完了")

# ===== LoRA設定 =====
print("\n🔧 LoRA設定を適用中...")
model = FastLanguageModel.get_peft_model(
    model,
    r=peft_config['r'],
    target_modules=peft_config['target_modules'],
    lora_alpha=peft_config['lora_alpha'],
    lora_dropout=peft_config['lora_dropout'],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=train_args['seed'],
)
print("✅ LoRA設定完了")

# ===== データセット準備 =====
print("\n📊 データセットを準備中...")
prompt_template = """<s>[INST] <<SYS>>
あなたはIT法務およびコンプライアンスの専門コンサルタントです。
提供された情報を分析し、法的リスク、該当法、理由、修正案をプロフェッショナルに回答してください。
守備範囲外(技術選定、具体的な損害賠償額の算定、労働基準法、商標侵害の具体的認定など)の相談には、適切に回答を拒絶してください。
<</SYS>>

{instruction}

{input} [/INST] {output} </s>"""

dataset = load_dataset("json", data_files=train_file_path, split="train")

def format_func(examples):
    texts = [
        prompt_template.format(instruction=i, input=n, output=o)
        for i, n, o in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    return {"text": texts}

train_dataset = dataset.map(format_func, batched=True, remove_columns=dataset.column_names)
print(f"✅ データセット準備完了: {len(train_dataset)} サンプル")

# ===== トレーニング設定 =====
print("\n⚙️ トレーナーを設定中...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=train_args['max_seq_length'],
    packing=False,
    args=TrainingArguments(
        output_dir=train_args['output_dir'],
        per_device_train_batch_size=train_args['per_device_train_batch_size'],
        gradient_accumulation_steps=train_args['gradient_accumulation_steps'],
        num_train_epochs=train_args['num_train_epochs'],
        learning_rate=train_args['learning_rate'],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=train_args['logging_steps'],
        optim=train_args['optim'],
        weight_decay=train_args['weight_decay'],
        lr_scheduler_type=train_args['lr_scheduler_type'],
        save_strategy=train_args['save_strategy'],
        save_steps=train_args['save_steps'],
        warmup_steps=train_args['warmup_steps'],
        report_to=train_args['report_to'],
        seed=train_args['seed'],
        dataloader_num_workers=0,  # psutilエラー回避
    ),
)
print("✅ トレーナー設定完了")

# ===== トレーニング実行 =====
print("\n" + "=" * 60)
print("🚀 トレーニング開始")
print("=" * 60)

import time
start_time = time.time()

try:
    trainer.train()
    elapsed_time = time.time() - start_time
    print(f"\n✅ トレーニング完了! (所要時間: {elapsed_time/60:.1f}分)")
except Exception as e:
    print(f"\n❌ エラーが発生しました: {e}")
    raise

# ===== モデル保存 =====
print("\n💾 モデルを保存中...")
output_model_dir = f"{train_args['output_dir']}/final_model"

try:
    model.save_pretrained_merged(
        output_model_dir,
        tokenizer,
        save_method="merged_16bit"
    )
    print(f"✅ モデル保存完了: {output_model_dir}")
except Exception as e:
    print(f"⚠️ merged保存時の警告: {e}")
    # 代替: LoRA形式で保存
    lora_output_dir = f"{train_args['output_dir']}/final_model_lora"
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    print(f"✅ LoRA形式で保存完了: {lora_output_dir}")

# ===== 学習結果のサマリー =====
print("\n" + "=" * 60)
print("📊 学習結果サマリー")
print("=" * 60)
print(f"モデル名: {new_model_name}")
print(f"学習データ: {len(train_dataset)} サンプル")
print(f"Epoch数: {train_args['num_train_epochs']}")
print(f"所要時間: {elapsed_time/60:.1f}分")
print(f"保存先: {train_args['output_dir']}")
print("=" * 60)
print("🎉 すべての処理が完了しました!")
print("=" * 60)