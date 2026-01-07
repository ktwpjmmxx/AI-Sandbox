# Tech-Comply AI

> **Webサービス開発における法的リスクを早期検知する、日本の法令特化型AIアシスタント**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/Model-ELYZA--japanese--Llama--2--7b-green)
![Framework](https://img.shields.io/badge/Framework-Unsloth-orange)
[![Status](https://img.shields.io/badge/Status-v13%20Research%20Complete-brightgreen)](./docs/README_V13.md)

---

## 概要

**Tech-Comply AI** は、Webサービス開発やマーケティング業務において発生する「法的リスク」を開発段階で早期に検知し、修正を支援する文書診断システムです。

### 解決する課題

Webサービス開発・運営の現場では、以下の課題がボトルネックとなっています：

| 課題 | 影響 | Tech-Comply AIの解決策 |
|------|------|---------------------|
| **コンプライアンスリスクの低減** | 広告表現や利用規約の不備による炎上・法的制裁を未然に防止。 |
| **法務確認コストの削減** | 開発現場での一次チェックを自動化し、専門部署の工数を削減。 |
| **専門知識不足** | 条文番号や判例に基づく具体的な修正案を提示し、開発者の理解を支援。 |

## 主な機能

1.  **根拠付きリスク判定（Grounding）**: 関連する日本の法令・条文番号を引用して回答。
2.  **リスクスコアリング**: 「高・中・低」の3段階でリスクを可視化。
3.  **実務的修正提案**: 法的要件を満たしつつ、実務に即した具体的な代替案（UI変更や文言修正）を断定的に提示。

---

## 技術仕様 

研究フェーズ完了時点での最終仕様です。

- Transformers: 4.57.3
- Pytorch: 2.9.1
- Tokenizers: 0.22.1

### モデル・学習基盤

- **ベースモデル**: ELYZA-japanese-Llama-2-7b-instruct
- **学習手法**: 4-bit QLoRA (Unslothエンジン / T4 GPU最適化)
- **ハイパーパラメータ**: 
  - `r`: 32
  - `lora_alpha`: 64 (過学習抑制と性格定着のバランス最適化)
  - `target_modules`: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - `Learning Rate`: 1.5e-4
  - `Epochs`: 3.0
- **推論仕様**:
  - `load_in_4bit`: True
  - **性格設定**: 「検討が必要です」等の逃げを排除した断定型コンサルタント

### 対応法令

- 景品表示法、個人情報保護法、特定商取引法
- 消費者契約法、著作権法、フリーランス保護新法、電気通信事業法（外部送信規律対応済）

---

## ディレクトリ構成

```
Tech-Comply-AI/
├── README.md                          # このファイル
├── docs/
│   ├── README_V1.md                  # 初期版の記録
│   ├── README_V2.md                  # データ拡充版
│   ├── README_V3.md                  # JSON特化版
│   ├── README_V4.md                  # 推論安定化版
│   ├── README_V5.md                  # Unsloth導入と高速化
│   ├── README_V7.md                  # エポック数と学習崩壊の相関
│   ├── README_V10.md                 # ハルシネーション抑制と拒絶
│   ├── README_V12.md                 # 知識定着の実験
│   └── README_V13.md                 # 【最新】研究フェーズ総括レポート
├── 01_data/
│   ├── train_data_471.jsonl          # レガシーデータ
│   └── traindata.jsonl           # 性格矯正・断定化済みの最終データセット
├── 02_config/
│   └── config.yaml               # 学習設定ファイル(v13確定版)
├── 03_scripts/
│   ├── train.py                   # 学習スクリプト(T4 GPU対応版)
│   └── inference.py               # 推論・テスト用スクリプト
└── outputs_v13/                      # 学習済みモデル（LoRAアダプタ）
```

---

## 免責事項

本ツールは法的助言を提供するものではなく、**開発・運用の補助ツール**として設計されています。最終的な法的判断は必ず専門家（弁護士等）に相談してください。

### 推奨される使い方

1. **一次チェック**: Tech-Comply AIで自動診断（即時回答）
2. **リスク箇所の特定**: 「高」判定の項目を抽出
3. **専門家への相談**: 重要箇所のみ弁護士に確認

これにより、法務確認の工数を**約70%削減**できると想定しています。

---

## 今後の展望 (Development Phase)

研究フェーズ（v1～v13）でのモデル性能検証は完了しました。
今後は**アプリケーション実装フェーズ**へ移行します。

### 進行中のプロジェクト
1.  **長文契約書の分割チェックシステム**
    - コンテキスト長制限を突破するため、文書を条文ごとに分割(Chunking)して並列推論させるアーキテクチャの実装。
2.  **Web UI化 (Streamlit)**
    - チャット形式およびファイルアップロード形式のGUI開発。

---

**© 2026 Tech-Comply AI Project. Licensed under Apache 2.0.**