from torch.utils.data import Dataset

class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.conversations = conversations
        self.max_length = max_length
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        input_text = conversation["input"]
        output_text = conversation["output"]
        
        inputs = self.tokenizer(input_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        outputs = self.tokenizer(output_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": outputs["input_ids"].squeeze()
        }

def prepare_conversation_data(file_path):
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    return conversations