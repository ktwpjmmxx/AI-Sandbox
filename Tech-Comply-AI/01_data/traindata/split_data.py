import json

# 入力ファイル名
input_file = 'traindata_v11.jsonl'
# 出力ファイル名
surgery_needed_file = 'surgery_needed_662.jsonl'
no_fix_needed_file = 'no_fix_needed_362.jsonl'

# 弱気な表現（逃げ）のキーワードリスト
weak_keywords = [
    "慎重な検討が必要",
    "専門家にご相談",
    "弁護士にご相談",
    "判断は法的な観点から",
    "個別具体的な判断については",
    "一概に判断することは困難",
    "確認を推奨します",
    "専門的な知見が必要"
]

def split_jsonl():
    surgery_count = 0
    no_fix_count = 0

    with open(input_file, 'r', encoding='utf-8') as f,\
         open(surgery_needed_file, 'w', encoding='utf-8') as f_surgery,\
         open(no_fix_needed_file, 'w', encoding='utf-8') as f_no_fix:
        
        for line in f:
            data = json.loads(line)
            output_text = data.get('output', '')
            
            # キーワードが含まれているかチェック
            is_surgery_needed = any(kw in output_text for kw in weak_keywords)
            
            if is_surgery_needed:
                f_surgery.write(json.dumps(data, ensure_ascii=False) + '\n')
                surgery_count += 1
            else:
                f_no_fix.write(json.dumps(data, ensure_ascii=False) + '\n')
                no_fix_count += 1

    print(f"分類完了！")
    print(f"・手術が必要なデータ（{surgery_needed_file}）: {surgery_count}件")
    print(f"・修正不要なデータ（{no_fix_needed_file}）: {no_fix_count}件")

if __name__ == "__main__":
    split_jsonl()