import gradio as gr
import torch

def chat_with_model(message, history, temperature, max_tokens, repetition_penalty):
    """
    改善版：パラメータを動的に調整可能
    """
    # 会話履歴を含むプロンプト作成
    chat_prompt = ""
    for user_msg, bot_msg in history:
        chat_prompt += f"User: {user_msg}\nBot: {bot_msg}\n"
    chat_prompt += f"User: {message}\nBot:"
    
    # トークン化
    inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # デコード
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_response = result.split("Bot:")[-1].strip()
    
    # 次のユーザー入力があれば削除
    if "User:" in bot_response:
        bot_response = bot_response.split("User:")[0].strip()
    
    return bot_response

# Gradio UI
with gr.Blocks(title="Mistral-7B Fine-tuned Chatbot") as demo:
    gr.Markdown("# 🤖 Mistral-7B ファインチューニング済みチャットボット")
    gr.Markdown("100件の会話データで学習したMistral-7Bモデル")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, label="会話履歴")
            
            with gr.Row():
                msg = gr.Textbox(
                    label="メッセージ",
                    placeholder="メッセージを入力してください...",
                    lines=2,
                    scale=4
                )
                submit = gr.Button("送信", variant="primary", scale=1)
            
            clear = gr.Button("会話をクリア")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 生成パラメータ")
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature（創造性）",
                info="高いほど創造的、低いほど保守的"
            )
            
            max_tokens = gr.Slider(
                minimum=50,
                maximum=500,
                value=150,
                step=50,
                label="最大トークン数",
                info="応答の最大長"
            )
            
            repetition_penalty = gr.Slider(
                minimum=1.0,
                maximum=2.0,
                value=1.2,
                step=0.1,
                label="繰り返しペナルティ",
                info="高いほど繰り返しを避ける"
            )
            
            gr.Markdown("### 📊 テスト例")
            examples = gr.Examples(
                examples=[
                    "Hello, how are you?",
                    "Tell me a joke",
                    "What is artificial intelligence?",
                    "Tell me something interesting",
                    "Can you help me?"
                ],
                inputs=msg
            )
    
    def respond(message, chat_history, temp, max_tok, rep_pen):
        bot_message = chat_with_model(message, chat_history, temp, max_tok, rep_pen)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot, temperature, max_tokens, repetition_penalty], [msg, chatbot])
    submit.click(respond, [msg, chatbot, temperature, max_tokens, repetition_penalty], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=True, debug=True)