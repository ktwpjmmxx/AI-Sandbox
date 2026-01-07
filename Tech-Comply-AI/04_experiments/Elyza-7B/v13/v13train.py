import os
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
import yaml
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from datetime import datetime

print("=" * 60)
print("Elyza-7B v13 Training Script")
print("=" * 60)

# 1. 設定の読み込み
print("\n[1/7] 設定ファイルの読み込み...")
with open("config_v13.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print(f"  - LoRA r: {config['lora_config']['r']}")
print(f"  - LoRA alpha: {config['lora_config']['lora_alpha']}")
print(f"  - Learning rate: {config['training_config']['learning_rate']}")
print(f"  - Epochs: {config['training_config']['num_train_epochs']}")
print(f"  - Dropout: {config['lora_config']['lora_dropout']}")

# 2. モデルとトークナイザーの準備
print("\n[2/7] ベースモデルの読み込み...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config['model_config']['base_model'],
    max_seq_length = config['model_config']['max_seq_length'],
    load_in_4bit = config['model_config']['load_in_4bit'],
)
print("  ✓ モデル読み込み完了")

# 3. LoRA設定の適用
print("\n[3/7] LoRAアダプタの設定...")
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
print("  ✓ LoRAアダプタ適用完了")

# 4. データセットの読み込みと整形
print("\n[4/7] データセットの読み込み...")
def formatting_prompts_func(examples):
    """ELYZA/Llama-2形式のプロンプトに整形"""
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # ELYZA/Llama-2 のプロンプト形式
        text = f"<s>[INST] <<SYS>>\\n{instruction}\\n<</SYS>>\\n\\n{input} [/INST] {output} </s>"
        texts.append(text)
    return { "text" : texts, }

try:
    dataset = load_dataset("json", data_files=config['training_config']['dataset_path'], split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)
    print(f"  ✓ データセット読み込み完了: {len(dataset)} samples")
except Exception as e:
    print(f"  ✗ エラー: データセットが見つかりません")
    print(f"  期待されるパス: {config['training_config']['dataset_path']}")
    print(f"  エラー詳細: {e}")
    exit(1)

# 5. トレーナーの設定
print("\n[5/7] トレーナーの設定...")
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
        save_strategy = config['training_config']['save_strategy'],
        save_total_limit = config['training_config']['save_total_limit'],
        load_best_model_at_end = config['training_config']['load_best_model_at_end'],
        metric_for_best_model = config['training_config']['metric_for_best_model'],
        greater_is_better = config['training_config']['greater_is_better'],
    ),
)
print("  ✓ トレーナー設定完了")

# 6. 学習実行
print("\n[6/7] 学習開始...")
print("-" * 60)
start_time = datetime.now()
trainer_stats = trainer.train()
end_time = datetime.now()
training_duration = end_time - start_time

print("-" * 60)
print(f"\n✓ 学習完了!")
print(f"  - 所要時間: {training_duration}")
print(f"  - 最終Loss: {trainer_stats.training_loss:.4f}")

# 7. モデルの保存
print("\n[7/7] モデルの保存...")
model.save_pretrained(config['training_config']['output_dir'])
tokenizer.save_pretrained(config['training_config']['output_dir'])
print(f"  ✓ モデル保存完了: {config['training_config']['output_dir']}")

print("\n" + "=" * 60)
print("v13 学習完了!")
print("=" * 60)
print("\n次のステップ:")
print("  1. inference_v13.py でテスト実行")
print("  2. Googleアナリティクス質問で効果検証")
print("  3. 複数のテストケースで汎用性確認")
