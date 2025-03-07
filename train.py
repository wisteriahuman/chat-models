import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from models import AttentionSeq2SeqModel
from data_utils import ConversationDataset, prepare_conversation_data

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 5e-5
EMBEDDING_DIM = 256
HIDDEN_DIM = 512

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking",
        word_tokenizer_type="mecab",
        mecab_dic="ipadic"
    )

    conversations = prepare_conversation_data("data/conversations.json")
    dataset = ConversationDataset(conversations, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    vocab_size = tokenizer.vocab_size
    model = AttentionSeq2SeqModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        pad_idx=tokenizer.pad_token_id
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            target_input = labels[:, :-1]
            target_output = labels[:, 1:]

            outputs = model(input_ids, target_input)

            outputs = outputs.reshape(-1, outputs.shape[-1])
            target_output = target_output.reshape(-1)
            
            loss = criterion(outputs, target_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")
    
    print("Training completed!")

if __name__ == "__main__":
    train()