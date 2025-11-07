# Mistral 7B Chatbot Project

This repository contains experiments and code for developing a chatbot using **Mistral 7B**, with Google Colab as the main training and inference environment.
The goal is to explore fine-tuning techniques, analyze conversational performance, and compare results with other models such as GPT-2.

---

## 📁 Directory Overview

```
mistral7b/
├── colab/
│   ├── mistral7b_finetune.ipynb     # Fine-tuning notebook
│   ├── mistral7b_inference.ipynb     # Inference and testing notebook
│   └── mistral7b_chat_demo.ipynb     # Interactive chatbot demo (Gradio)
│
├── src/
│   ├── train.py                      # Fine-tuning script
│   ├── inference.py                  # Standalone inference script
│   ├── chatbot.py                    # Dialogue management logic
│   └── utils/                        # Data loading, metrics, and logging utilities
│
├── data/
│   ├── training_data.json            # 100-sample conversation dataset
│   ├── eval_data.json                # Validation data
│   └── data_statistics.md            # Dataset analysis and insights
│
├── results/
│   ├── checkpoints/                  # Fine-tuned model checkpoints
│   ├── training_logs/                # Training logs from Colab
│   └── inference_outputs/            # Output samples and observations
│
├── experiments/
│   ├── experiment_log.md             # Parameter tuning and experiment notes
│   └── colab_sessions.md             # Session summaries
│
├── requirements.txt                  # Dependency list
└── README.md                         # This file
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>/models/mistral7b
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open Google Colab

You can start fine-tuning or testing the model using the provided notebooks:

* **Fine-tuning** → `colab/mistral7b_finetune.ipynb`
* **Inference** → `colab/mistral7b_inference.ipynb`
* **Chat demo** → `colab/mistral7b_chat_demo.ipynb`

Each notebook includes detailed steps for setup, training, and evaluation.

---

## Training Configuration

Key training parameters (configurable in `train.py` or the Colab notebook):

| Parameter     | Default                     | Description               |
| ------------- | --------------------------- | ------------------------- |
| Model         | `mistralai/Mistral-7B-v0.1` | Base pretrained model     |
| Epochs        | 3                           | Number of training epochs |
| Batch Size    | 4                           | Per-device batch size     |
| Learning Rate | 2e-5                        | Default optimizer rate    |
| Max Length    | 512                         | Tokenization limit        |

---

## Results & Analysis

Training progress, loss curves, and model checkpoints are automatically stored in `results/`.
Each experiment should also be logged in `experiments/experiment_log.md` for reproducibility.

---

## Note
