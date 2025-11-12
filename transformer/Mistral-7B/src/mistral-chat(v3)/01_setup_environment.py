!nvidia-smi

# バージョンを固定してインストール
!pip install -q peft==0.10.0
!pip install -q accelerate==0.30.1
!pip install -q transformers==4.38.0
!pip install -q datasets==2.18.0
!pip install -q bitsandbytes==0.43.0
!pip install -q trl==0.7.11
!pip install -q gradio

import os
import json
import torch
from datetime import datetime

# ディレクトリ作成
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/training", exist_ok=True)
os.makedirs("results/evaluation", exist_ok=True)

# 実験設定の保存
experiment_config = {
    "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": "mistralai/Mistral-7B-v0.1",
    "training_data_size": 100,
    "framework": "PEFT + LoRA",
    "gpu": "T4",
}

with open("results/training/experiment_config.json", "w") as f:
    json.dump(experiment_config, f, indent=2)

print("✅ Setup complete!")
print(f"📊 Experiment ID: {experiment_config['experiment_date']}")