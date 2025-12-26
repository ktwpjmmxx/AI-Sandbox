# 🚀 Elyza-7B IT法務ファインチューニング v9 クイックスタート

このガイドは、v8で成功した構成を使って**次回v9を確実に実行するための完全マニュアル**です。

---

## 📋 事前準備

### **必要なファイル**
1. `train_v9.py` - トレーニングスクリプト
2. `config_v9.yaml` - 設定ファイル
3. `train_data_471.jsonl` - 学習データ (または任意のJSONLファイル)

---

## 🎯 実行手順 (3ステップ)

### **ステップ1: 新しいノートブックを作成**

Google Colabで新しいノートブックを開きます。

---

### **ステップ2: ライブラリのインストール**

**セル1** に以下をコピペして実行:

```bash
# 既存ライブラリのアンインストール
!pip uninstall -y unsloth unsloth_zoo transformers accelerate trl peft bitsandbytes tokenizers xformers

# 正しい順序でインストール (重要: この順序を守る!)
!pip install psutil==6.1.1
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

print("✅ インストール完了")
```

**所要時間**: 約2〜3分

---

### **ステップ3: ファイルのアップロード**

以下のファイルをColabにアップロード:
- `train_v9.py`
- `config_v9.yaml`
- `train_data_471.jsonl` (またはあなたの学習データ)

**方法**: 
1. Colabの左サイドバーのフォルダアイコンをクリック
2. 各ファイルをドラッグ&ドロップ

---

### **ステップ4: 学習を実行**

**セル2** に以下をコピペして実行:

```bash
!python train_v9.py
```

**所要時間**: 約50分 (3 Epoch、471サンプルの場合)

---

## 📊 実行中の確認ポイント

### **正常に開始できた場合**
```
============================================================
🦥 Elyza-7B IT法務ファインチューニング v9
============================================================
✅ 設定ファイル読み込み: config_v9.yaml
📦 モデルをロード中...
✅ モデルロード完了
🔧 LoRA設定を適用中...
✅ LoRA設定完了
📊 データセットを準備中...
✅ データセット準備完了: 471 サンプル
⚙️ トレーナーを設定中...
✅ トレーナー設定完了
============================================================
🚀 トレーニング開始
============================================================
```

### **学習中のログ**
```
{'loss': 1.59, 'grad_norm': 1.98, 'learning_rate': 0.0, 'epoch': 0.03}
{'loss': 1.57, 'grad_norm': 1.95, 'learning_rate': 1e-06, 'epoch': 0.07}
...
```

**健全な学習の兆候**:
- ✅ `loss` が徐々に減少
- ✅ `grad_norm` が安定 (0.7〜2.0の範囲)
- ✅ エラーが発生しない

---

## 💾 学習完了後の保存

### **Google Driveに保存 (推奨)**

**セル3** に以下を実行:

```python
# Google Driveをマウント
from google.colab import drive
drive.mount('/content/drive')

# モデルをGoogle Driveにコピー
import shutil
import os

drive_path = "/content/drive/MyDrive/Elyza_IT_Legal_Models/v9_final"
os.makedirs(drive_path, exist_ok=True)

source = "outputs_v9/final_model"
if os.path.exists(source):
    shutil.copytree(source, drive_path, dirs_exist_ok=True)
    print(f"✅ モデルを保存: {drive_path}")
else:
    print("❌ モデルが見つかりません")

# ファイルサイズを確認
for file in os.listdir(drive_path):
    path = os.path.join(drive_path, file)
    size = os.path.getsize(path) / (1024**2)
    print(f"  - {file}: {size:.2f} MB")
```

---

## 🔮 推論テスト (別日でもOK)

### **新しいノートブックで推論**

**セル1**: ライブラリインストール
```bash
!pip install psutil==6.1.1
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**セル2**: Google Driveマウント
```python
from google.colab import drive
drive.mount('/content/drive')
```

**セル3**: 推論スクリプト
```python
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from unsloth import FastLanguageModel

# モデルロード
model_path = "/content/drive/MyDrive/Elyza_IT_Legal_Models/v9_final"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# プロンプト
prompt = """<s>[INST] <<SYS>>
あなたはIT法務およびコンプライアンスの専門コンサルタントです。
<</SYS>>

以下の利用規約条項の法的リスクを分析してください。

第10条(免責事項)
当社は、本サービスの利用によって生じたいかなる損害についても、一切の責任を負いません。 [/INST] """

# 推論
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("[/INST]")[1])
```

---

## ⚠️ トラブルシューティング

### **psutilエラーが出る場合**
```bash
!rm -rf /content/unsloth_compiled_cache
!pip install psutil==6.1.1
!python train_v9.py
```

### **メモリ不足エラーが出る場合**
`config_v9.yaml` を編集:
```yaml
per_device_train_batch_size: 2  # 4 → 2
gradient_accumulation_steps: 8  # 4 → 8
```

### **データファイルが見つからない場合**
`config_v9.yaml` のパスを確認:
```yaml
train_file_path: "./train_data_471.jsonl"  # ← 実際のファイル名に変更
```

---

## 🎯 カスタマイズ方法

### **学習データを変更**
1. JSONLファイルを用意 (フォーマット: `instruction`, `input`, `output`)
2. `config_v9.yaml` の `train_file_path` を変更
3. `train_v9.py` を実行

### **Epoch数を変更**
`config_v9.yaml`:
```yaml
num_train_epochs: 5  # 3 → 5 に変更
```

### **学習率を調整**
`config_v9.yaml`:
```yaml
learning_rate: 0.00002  # 2e-5 (より高速)
# または
learning_rate: 0.000005  # 5e-6 (より慎重)
```

---

## 📝 チェックリスト

学習開始前に確認:
- [ ] ライブラリをインストールした
- [ ] `train_v9.py` をアップロードした
- [ ] `config_v9.yaml` をアップロードした
- [ ] `train_data_471.jsonl` をアップロードした
- [ ] `config_v9.yaml` のパスが正しい

学習完了後に確認:
- [ ] 最終Lossが1.0以下
- [ ] エラーが発生していない
- [ ] `outputs_v9/final_model` が存在する
- [ ] Google Driveに保存した

---

## 🎉 成功の目安

- **Loss**: 開始時 1.5〜1.6 → 完了時 0.5〜0.6
- **所要時間**: 約50分 (471サンプル、3 Epoch)
- **保存サイズ**: 約300〜500 MB

この構成で**確実に動作します**！ 🚀