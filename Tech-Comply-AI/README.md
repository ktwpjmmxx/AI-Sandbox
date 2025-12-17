# Tech-Comply AI: Web開発者のための法令遵守・リスク検知支援AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/Model-ELYZA--japanese--Llama--2--7b-green)
![License](https://img.shields.io/badge/License-Apache_2.0-orange)

## 概要 (Overview)
**Tech-Comply AI** は、Webサービス開発やマーケティング業務において発生する「法的リスク」を、開発段階で早期に検知・修正支援するための文書修正システムです。

汎用LLM（Large Language Models）の課題である「ハルシネーション（嘘）」を抑制し、**「インターネット・テクノロジー関連法（景品表示法、著作権法、個人情報保護法）」**に特化してファインチューニングを行いました。回答には必ず**法的根拠（条文・判例）**を提示する「グラウンディング機能」を実装しています。

## 解決する課題 (Problem Solving)
Web開発や運営の現場では、以下の課題がボトルネックとなっています。
* **コンプライアンスリスク:** 広告表現や利用規約の不備による炎上・法的制裁のリスク。
* **確認コスト:** 些細な文言確認でも法務部や弁護士への相談が必要で、リードタイムが発生する。

**Tech-Comply AI** は、これらのチェックを自動化・一次請けすることで、**法務確認工数の削減**と**リスクの早期発見**を実現します。

## 主な機能 (Key Features)
1.  **根拠付きリスク判定 (Grounding):**
    * 入力されたテキスト（広告文、規約案）に対し、関連する法律の**条文番号**を引用して回答します。
2.  **リスクスコアリング:**
    * コンプライアンス違反のリスクを「高・中・低」で判定し、優先的に修正すべき箇所を提示します。
3.  **構造化された修正提案:**
    * 法的要件を満たすための修正案を、実務で使いやすい**Markdown/JSON形式**で出力します。

## 技術スタック (Tech Stack)
* **Base Model:** `elyza/ELYZA-japanese-Llama-2-7b`
* **Fine-tuning:** QLoRA (Quantized Low-Rank Adaptation)
* **Frameworks:** PyTorch, Hugging Face Transformers, PEFT, Bitsandbytes
* **Experiment Tracking:** Weights & Biases (W&B)

## 📂 ディレクトリ構成 (Directory Structure)

Tech-Comply-AI/
├── 01_data/            # 学習用データセット（Q&Aペア、条文データ）
├── 02_config/          # モデル設定ファイル (yaml)
├── 03_scripts/         # 前処理、学習、評価用スクリプト
├── 04_experiments/     # 実験ログ、比較レポート、失敗からの知見
├── 05_deployment/      # デモアプリケーション (Gradio/Streamlit)
└── README.md

## 評価と成果 (Evaluation)
本モデルは、汎用モデルと比較して以下の指標で性能向上を確認しました。

| Metric | Baseline (Mistral-7B) | **Tech-Comply AI (Ours)** |
| :--- | :--- | :--- |
| **Accuracy (正答率)** | [TBD]% | **[TBD]%** |
| **Grounding Rate (引用成功率)** | [TBD]% | **[TBD]%** |
| **Hallucination Rate (誤答率)** | [TBD]% | **[TBD]%** |

> ※ 詳細な実験プロセスと考察は `04_experiments/comparison_report.ipynb` を参照してください。

## 免責事項 (Disclaimer)
本ツールは法的な助言を提供するものではなく、開発・運用の補助ツールとして設計されています。最終的な法的判断は必ず専門家（弁護士等）に相談してください。

## Author
* Name: [Tatsuya Koyama]
* Target Role: AI Engineer / AI Product Manager