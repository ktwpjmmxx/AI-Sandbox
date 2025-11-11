# Mistral-7B Fine-tuned Chatbot Experiments

このリポジトリは、オープンソース大規模言語モデル **Mistral-7B** を用いた
LoRAによる軽量ファインチューニング済みチャットボットの実験プロジェクトです。  
Google Colabやローカル環境での学習・推論・Web UIでの対話までを含みます。

---

## Overview
- **モデル:** [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B)
- **目的:** LoRAによる軽量ファインチューニングの検証およびチャット応答生成
- **技術:** Python, Hugging Face Transformers, PEFT, BitsAndBytes, Gradio
- **開発環境:** Google Colab / ローカルGPU

---

## Repository Structure

Mistral-7B/
├── colab/
│ ├── chatbot_experiment.ipynb # Colabでの実験用Notebook
│ └── chat_log.txt # テスト対話ログ（任意）
├── src/
│ ├── setup_environment.py # ディレクトリ作成・環境初期化
│ ├── model_loader.py # モデル・トークナイザーのロード
│ ├── data_loader.py # JSON形式データの読み込み
│ ├── preprocessing.py # データの前処理・トークナイズ
│ ├── lora_setup.py # LoRA設定・適用
│ ├── train.py # Hugging Face Trainerでの学習
│ ├── save_model.py # Fine-tunedモデルの保存
│ ├── chatbot.py # 推論パイプライン・チャット関数
│ └── gradio_app.py # Web UI（Gradio）でのチャットボット
├── data/ # 学習・評価用データ
│ └── training_data.json
├── results/ # 実験結果・生成サンプル・評価ログ
│ └── chat_log.txt
├── README.md
└── requirements.txt


### 各フォルダ・ファイルの概要

| フォルダ/ファイル | 役割 |
|------------------|------|
| `colab/`         | Notebookでの実験・開発ログの記録。テスト対話もここに保存 |
| `src/`           | 再利用可能なコード一式。学習・推論・Web UIを含む |
| `data/`          | 学習用・評価用データを格納 |
| `results/`       | ポートフォリオ提出用の整形済み生成ログ・サンプル |
| `requirements.txt` | 必要ライブラリ・バージョン管理 |
| `README.md`      | プロジェクト概要、実行手順、成果物の説明 |

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Mistral-7B.git
cd Mistral-7B

2. Install dependencies

pip install -r requirements.txt

3. Prepare directories

from src.setup_environment import prepare_directories
prepare_directories()

4. Load dataset

from src.data_loader import load_json_dataset
dataset = load_json_dataset("data/training_data.json")

5. Preprocess dataset

from src.preprocessing import preprocess_dataset
from src.model_loader import load_mistral_model

tokenizer, _ = load_mistral_model()
processed_dataset = preprocess_dataset(dataset, tokenizer)

6. Apply LoRA

from src.lora_setup import apply_lora
from src.model_loader import load_mistral_model

tokenizer, model = load_mistral_model()
model = apply_lora(model)

7. Train model

from src.train import train_model
trainer = train_model(model, tokenizer, processed_dataset, output_dir="outputs")

8. Save model

from src.save_model import save_model_and_tokenizer
save_model_and_tokenizer(model, tokenizer, output_dir="mistral7b_finetuned")

9. Run Gradio Web UI
python src/gradio_app.py
 - ブラウザでチャットボットと対話可能
 - LoRA Fine-tunedモデルを即時使用

### License
MIT License © 2025 小山



