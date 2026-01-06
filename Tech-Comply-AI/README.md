# Tech-Comply AI

> **Webサービス開発における法的リスクを早期検知する、日本の法令特化型AIアシスタント**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/Model-ELYZA--japanese--Llama--2--7b-green)
![Framework](https://img.shields.io/badge/Framework-Unsloth-orange)
[![Status](https://img.shields.io/badge/Status-v10%20Research%20Phase-blue)](./docs/README_V10.md)

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
3.  **実務的修正提案**: 法的要件を満たしつつ、実務に即した具体的な代替案を提示。

---

## 技術仕様

### モデル・学習基盤

- **ベースモデル**: ELYZA-japanese-Llama-2-7b-instruct
- **学習手法**: 4-bit QLoRA (Unslothエンジン)
- **ハイパーパラメータ**: 
  - `r=16〜64` (タスクの複雑度に応じて調整)
  - `target_modules`: 全線形層
  - `Learning Rate`: 1e-4〜2e-4
- **推論パラメータ (v4推奨値)**:
  - `temperature`: 0.6 (法律相談の正確性を重視)
  - `top_p`: 0.85
  - `repetition_penalty`: 1.15

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
│   ├── README_V4.md                  # 推論安定化版（最新）
│   ├── README_V5.md                  # Unsloth導入と高速化
│   ├── README_V7.md                  # エポック数と学習崩壊の相関
│   └── README_V10.md                 # 最新：ハルシネーション抑制と拒絶
├── 01_data/
│   ├── train_data_471.jsonl          # 元データ
│   └── traindata_v10.jsonl           # 精査済み471件＋ガードレール用データ
├── 02_config/
│   └── config.yaml                　 # config設定(完成版)
├── 03_scripts/
│   ├── train_finetuning.py           # 学習スクリプト(完成版)
│   └── inference.py                  # 推論スクリプト(完成版)
└── 04_experiments/
    └── Elyza-7B/
        └── version1~n/               # 各バージョンにおける検証結果のログ
```

---

## 免責事項

本ツールは法的助言を提供するものではなく、**開発・運用の補助ツール**として設計されています。最終的な法的判断は必ず専門家（弁護士等）に相談してください。

### 推奨される使い方

1. **一次チェック**: Tech-Comply AIで自動診断
2. **リスク箇所の特定**: 「高」判定の項目を抽出
3. **専門家への相談**: 重要箇所のみ弁護士に確認

これにより、法務確認の工数を**約70%削減**できると想定しています。

---

## 今後の展望

### 知識の解放
- 強固なフォーマットを維持しつつ、専門領域での回答許容範囲を再定義。

### 多角的評価の導入
- Loss値だけでなく、法的根拠の正確性とハルシネーション率を数値化して評価

### RAGとのハイブリッド化
- モデル内の知識に頼りすぎず、最新の条文を外部から参照する仕組みの検討。

---

**© 2025 Tech-Comply AI Project. Licensed under Apache 2.0.**
