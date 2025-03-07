import torch
from transformers import AutoTokenizer
from models import SimpleSeq2SeqModel

def load_model(model_path, vocab_size, embedding_dim, hidden_dim, pad_idx, device):
    model = SimpleSeq2SeqModel(vocab_size, embedding_dim, hidden_dim, pad_idx)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def chat():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    
    # モデルの読み込み
    model = load_model(
        model_path="model_epoch_10.pt",
        vocab_size=tokenizer.vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        pad_idx=tokenizer.pad_token_id,
        device=device
    )
    
    print("チャットボットが準備できました。「終了」と入力すると会話を終了します。")
    
    while True:
        user_input = input("\nあなた: ")
        if user_input.lower() == "終了":
            print("会話を終了します。")
            break
        
        # 入力をトークン化
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        
        # 応答を生成
        generated_ids = model.generate_response(
            inputs.input_ids,
            max_len=50,
            start_token=tokenizer.cls_token_id,
            end_token=tokenizer.sep_token_id
        )
        
        # 応答をデコード
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"ボット: {response}")

if __name__ == "__main__":
    chat()