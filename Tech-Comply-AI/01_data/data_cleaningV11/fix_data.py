import json
import pandas as pd

csv_fixed_file = "fixed.csv"     # 修正後のCSV
remaining_file = "stay.jsonl"    # 手順1で分けたデータ
output_v11_file = "traindata_v11.jsonl"

# 1. 修正済みCSVの読み込み
df_fixed = pd.read_csv(csv_fixed_file, encoding="utf-16") # 手順1に合わせる
fixed_data = df_fixed.to_dict(orient="records")

# 2. 統合して書き出し
with open(output_v11_file, "w", encoding="utf-8") as f:
    # 修正済みデータを書き込む
    for item in fixed_data:
        # CSV化でNone(NaN)になった項目を空文字にする等の処理
        clean_item = {k: (v if pd.notna(v) else "") for k, v in item.items()}
        f.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
    
    # 手順1で分けておいた正常データを追加する
    with open(remaining_file, "r", encoding="utf-8") as rf:
        for line in rf:
            f.write(line)

print(f"v11データセット作成完了！ -> {output_v11_file}")