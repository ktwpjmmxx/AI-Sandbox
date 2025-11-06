#### Version1.0

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("モデルを読み込み中...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_original = GPT2Tokenizer.from_pretrained("gpt2")
model_original = GPT2LMHeadModel.from_pretrained("gpt2")
model_original.to(device)
model_original.eval()

tokenizer_finetuned = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-final")
model_finetuned = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-final")
model_finetuned.to(device)
model_finetuned.eval()

print("モデル読み込み完了")

def chat_original(message, history):
    prompt = f"Human: {message}\nAssistant:"
    inputs = tokenizer_original.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model_original.generate(
            inputs,
            max_length=len(inputs[0]) + 50,
            pad_token_id=tokenizer_original.eos_token_id,
            eos_token_id=tokenizer_original.eos_token_id,
        )
    
    full_response = tokenizer_original.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("Assistant:")[-1].strip()
    response = response.split('\n')[0].strip()
    return response

def chat_finetuned(message, history):
    prompt = f"Human: {message}\nAssistant:"
    inputs = tokenizer_finetuned.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model_finetuned.generate(
            inputs,
            max_length=len(inputs[0]) + 50,
            pad_token_id=tokenizer_finetuned.eos_token_id,
            eos_token_id=tokenizer_finetuned.eos_token_id,
        )
    
    full_response = tokenizer_finetuned.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("Assistant:")[-1].strip()
    response = response.split('\n')[0].strip()
    return response

with gr.Blocks(title="GPT-2 Before/After") as demo:
    gr.Markdown("# GPT-2 Fine-tuning: Before/After Comparison")
    gr.Markdown("Left: Original GPT-2 | Right: Fine-tuned GPT-2")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Original GPT-2 (Before)")
            chatbot_original = gr.ChatInterface(
                fn=chat_original,
                examples=["Hello, how are you?", "What is your name?", "Can you help me?"],
            )
        
        with gr.Column():
            gr.Markdown("### Fine-tuned GPT-2 (After)")
            chatbot_finetuned = gr.ChatInterface(
                fn=chat_finetuned,
                examples=["Hello, how are you?", "What is your name?", "Can you help me?"],
            )

demo.launch(share=True)

#### Version1.1

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("モデルを読み込み中...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-final")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-final")
model.to(device)
model.eval()

print("モデル読み込み完了")

def chat(message, history):
    prompt = f"Human: {message}\nAssistant:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + 60,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("Assistant:")[-1].strip()
    response = response.split('\n')[0].strip()
    
    if not response:
        response = "I'm here to help! What would you like to know?"
    
    return response

interface = gr.ChatInterface(
    fn=chat,
    title="Fine-tuned GPT-2 Chatbot",
    description="GPT-2 fine-tuned on conversational data",
    examples=[
        "Hello, how are you?",
        "What is your name?",
        "Can you help me?",
        "Tell me something interesting",
        "What can you do?"
    ],
)

interface.launch(share=True)

**ver1.0からの改善**
- repetition_penalty=1.3 - 繰り返し防止
- no_repeat_ngram_size=3 - 同じフレーズの繰り返しを防ぐ
- do_sample=True - サンプリングを有効化
- te9mperature=0.8 - 多様性を追加

#### Version1.2

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("モデルを読み込み中...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-final")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-final")
model.to(device)
model.eval()

print("モデル読み込み完了")

def chat(message, history):
    prompt = f"\n\nHuman: {message}\n\nAssistant:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + 30,  # より短く制限
            temperature=0.7,  # 少し保守的に
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.6,  # さらに強化
            no_repeat_ngram_size=4,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,  # 早期終了を有効化
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # "Assistant:"以降を抽出
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
    else:
        response = full_response[len(prompt):].strip()
    
    # 改行や不要な部分を削除
    response = response.split('\n')[0].strip()
    response = response.split('Human:')[0].strip()
    
    # 長すぎる場合は最初の文のみ
    if len(response) > 150:
        sentences = response.split('.')
        response = sentences[0] + '.' if sentences[0] else response[:100]
    
    if not response or len(response) < 3:
        response = "I'm here to help! Could you tell me more?"
    
    return response

interface = gr.ChatInterface(
    fn=chat,
    title="Fine-tuned GPT-2 Chatbot (Optimized)",
    description="GPT-2 with improved response control",
    examples=[
        "Hello, how are you?",
        "What is your name?",
        "Can you help me?",
        "Tell me a joke",
        "What's the weather like?"
    ],
)

interface.launch(share=True)

**ver1.1からの改善**
- max_length を30に短縮 - 長すぎる応答を防ぐ
- repetition_penalty=1.6 - さらに強化
- early_stopping=True - 適切なタイミングで終了
- 応答の後処理を強化 - 150文字以上は最初の文のみ抽出

**ポイント**
 - 大量の低品質データより、少量の高品質データ
 - データの内容が出力に直結 - 学習データに含まれるトピックがそのまま出力に現れ


#### ReFinetune

# ===========================================
# GPT-2 ファインチューニング - 自作データ版
# ===========================================

!pip install datasets transformers torch accelerate -q

from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch

print("=" * 60)
print("ステップ1: 自作会話データの作成")
print("=" * 60)

# 高品質な会話データ（シンプルで自然な対話）
conversations = [
    "\n\nHuman: Hello, how are you?\n\nAssistant: I'm doing well, thank you! How can I help you today?",
    "\n\nHuman: What's your name?\n\nAssistant: I'm an AI assistant created to help answer questions and have conversations.",
    "\n\nHuman: Can you help me?\n\nAssistant: Of course! I'd be happy to help. What do you need assistance with?",
    "\n\nHuman: Tell me a joke\n\nAssistant: Why don't scientists trust atoms? Because they make up everything!",
    "\n\nHuman: What's the weather like?\n\nAssistant: I don't have access to real-time weather data, but I'm happy to help with other questions!",
    "\n\nHuman: Thank you\n\nAssistant: You're very welcome! Feel free to ask if you need anything else.",
    "\n\nHuman: Goodbye\n\nAssistant: Goodbye! Have a great day!",
    "\n\nHuman: How can you assist me?\n\nAssistant: I can answer questions, provide information, and have friendly conversations with you.",
    "\n\nHuman: What can you do?\n\nAssistant: I can help with various tasks like answering questions, explaining concepts, and chatting with you.",
    "\n\nHuman: Are you smart?\n\nAssistant: I try my best to be helpful! I have knowledge on many topics and enjoy learning from our conversations.",
    "\n\nHuman: Do you have feelings?\n\nAssistant: I don't have feelings like humans do, but I'm designed to be helpful and friendly in our conversations.",
    "\n\nHuman: Tell me something interesting\n\nAssistant: Did you know that octopuses have three hearts? Two pump blood to the gills, and one pumps it to the rest of the body!",
    "\n\nHuman: That's cool!\n\nAssistant: I'm glad you found that interesting! Do you have any other questions?",
    "\n\nHuman: What time is it?\n\nAssistant: I don't have access to the current time, but I'm here to help with other questions!",
    "\n\nHuman: I'm bored\n\nAssistant: I understand! Would you like to hear an interesting fact, or is there something specific you'd like to talk about?",
    "\n\nHuman: Yes please\n\nAssistant: Great! What would you like to know more about?",
    "\n\nHuman: I don't know\n\nAssistant: That's okay! We can talk about science, technology, nature, or anything else that interests you.",
    "\n\nHuman: Tell me about space\n\nAssistant: Space is fascinating! It's incredibly vast and contains billions of galaxies. What aspect of space interests you most?",
    "\n\nHuman: Nice talking to you\n\nAssistant: It was nice talking to you too! Come back anytime if you have more questions.",
    "\n\nHuman: See you later\n\nAssistant: See you later! Take care!",
]

# より多くのバリエーションを追加（データ量を増やす）
extra_conversations = [
    "\n\nHuman: Hi there!\n\nAssistant: Hello! How are you doing today?",
    "\n\nHuman: I'm good, thanks\n\nAssistant: That's wonderful to hear! What brings you here today?",
    "\n\nHuman: Just curious\n\nAssistant: Curiosity is great! What would you like to learn about?",
    "\n\nHuman: Can you code?\n\nAssistant: I can help explain programming concepts and provide guidance, though I work best with text-based conversations.",
    "\n\nHuman: What languages do you speak?\n\nAssistant: I primarily communicate in English, and I'm here to help answer your questions!",
    "\n\nHuman: You're helpful\n\nAssistant: Thank you! I'm glad I could help. Is there anything else you'd like to know?",
    "\n\nHuman: Not really\n\nAssistant: Alright! Feel free to come back anytime you have questions.",
    "\n\nHuman: Maybe later\n\nAssistant: Sounds good! I'll be here whenever you need assistance.",
    "\n\nHuman: Tell me a fact\n\nAssistant: Here's a fun fact: Honey never spoils! Archaeologists have found 3000-year-old honey that's still edible.",
    "\n\nHuman: Wow, really?\n\nAssistant: Yes! It's because of honey's low moisture content and acidic pH, which prevent bacterial growth.",
]

# 全ての会話を統合
all_conversations = conversations + extra_conversations

# データセット作成
dataset = Dataset.from_dict({"text": all_conversations})

print(f"会話データ作成完了: {len(all_conversations)} 件")
print("\n最初の会話例:")
print(all_conversations[0])

print("\n" + "=" * 60)
print("ステップ2: データの整形")
print("=" * 60)

# トークナイザーの準備
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# データを整形する関数
def preprocess_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

# データを整形
print("データを整形中...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 学習用と評価用に分割（80%学習、20%評価）
train_size = int(0.8 * len(tokenized_dataset))
eval_size = len(tokenized_dataset) - train_size

tokenized_train = tokenized_dataset.select(range(train_size))
tokenized_eval = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

print(f"学習データ準備完了: {len(tokenized_train)} 件")
print(f"評価データ準備完了: {len(tokenized_eval)} 件")

print("\n" + "=" * 60)
print("ステップ3: ファインチューニング")
print("=" * 60)

# GPUが使えるか確認
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")

# モデルの読み込み
print("GPT-2モデルを読み込み中...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)
print("モデル読み込み完了")

# 学習設定（少ないデータなのでエポック数を増やす）
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-clean",
    num_train_epochs=10,  # 少ないデータなので多めに
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs-clean',
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=10,
    save_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

# Trainerの設定
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# ファインチューニング開始
print("\n" + "=" * 60)
print("ファインチューニング開始！")
print("=" * 60)
trainer.train()

print("\n" + "=" * 60)
print("ファインチューニング完了！")
print("=" * 60)

# モデルを保存
model.save_pretrained("./gpt2-finetuned-clean-final")
tokenizer.save_pretrained("./gpt2-finetuned-clean-final")
print("モデルを保存しました: ./gpt2-finetuned-clean-final")


#### Version1.3

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("Loading fine-tuned model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-clean-final")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-clean-final")
model.to(device)
model.eval()

print("Model loaded successfully!")

def chat(message, history):
    prompt = f"\n\nHuman: {message}\n\nAssistant:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + 40,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
    else:
        response = full_response[len(prompt):].strip()
    
    response = response.split('\n')[0].strip()
    response = response.split('Human:')[0].strip()
    
    if not response or len(response) < 3:
        response = "I'm here to help! What would you like to know?"
    
    return response

interface = gr.ChatInterface(
    fn=chat,
    title="GPT-2 Fine-tuned Chatbot (Clean Data)",
    description="GPT-2 trained on high-quality conversation data",
    examples=[
        "Hello, how are you?",
        "What's your name?",
        "Can you help me?",
        "Tell me a joke",
        "Tell me something interesting"
    ],
)

interface.launch(share=True)

**ver1.2からの改善**
①データの特徴
- 30件の高品質な会話
- num_train_epochs=10 - データが少ないので多めに学習
- バッチサイズを小さく

#### Version1.4

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("Loading fine-tuned model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-clean-final")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-clean-final")
model.to(device)
model.eval()

print("Model loaded successfully!")

def chat(message, history):
    # 会話履歴を含めたプロンプトを作成
    conversation = ""
    if history:
        for human_msg, assistant_msg in history[-3:]:  # 直近3ターンを使用
            conversation += f"\n\nHuman: {human_msg}\n\nAssistant: {assistant_msg}"
    
    conversation += f"\n\nHuman: {message}\n\nAssistant:"
    
    inputs = tokenizer.encode(conversation, return_tensors="pt").to(device)
    
    # 入力が長すぎる場合は切り詰める
    if len(inputs[0]) > 400:
        inputs = inputs[:, -400:]
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + 35,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.6,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 最後のAssistantの発言を抽出
    if "Assistant:" in full_response:
        parts = full_response.split("Assistant:")
        response = parts[-1].strip()
    else:
        response = full_response[len(conversation):].strip()
    
    # クリーンアップ
    response = response.split('\n\n')[0].strip()
    response = response.split('Human:')[0].strip()
    
    if not response or len(response) < 3:
        response = "I'm here to help! What would you like to know?"
    
    return response

interface = gr.ChatInterface(
    fn=chat,
    title="GPT-2 Fine-tuned Chatbot (Improved)",
    description="GPT-2 with conversation history support",
    examples=[
        "Hello, how are you?",
        "What's your name?",
        "Can you help me?",
        "Tell me a joke",
        "Tell me something interesting"
    ],
)

interface.launch(share=True)

**ver1.3からの改善**
- 会話履歴を含めてプロンプトを作成
- 直近3ターンの文脈を使用
- repetition_penaltyを少し強化

