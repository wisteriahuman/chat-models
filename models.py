import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class AttentionSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2 + embedding_dim, vocab_size)
    
    def forward(self, source, target):
        
        batch_size = source.shape[0]
        
        embedded_source = self.embedding(source)
        encoder_outputs, (hidden, cell) = self.encoder(embedded_source)
        
        embedded_target = self.embedding(target)
        
        outputs = []
        
        for t in range(embedded_target.shape[1]):
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            attn_weights = attn_weights.unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)
            rnn_input = torch.cat((embedded_target[:, t:t+1], context.unsqueeze(1)), dim=2)
            output, (hidden, cell) = self.decoder(rnn_input, (hidden, cell))
            output = output.squeeze(1)
            context = context
            embedded = embedded_target[:, t]
            
            output = self.fc_out(torch.cat((output, context, embedded), dim=1))
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        
        return outputs
    
    def generate_response(self, source, max_len, start_token, end_token):
        self.eval()
        with torch.no_grad():
            embedded_source = self.embedding(source)
            encoder_outputs, (hidden, cell) = self.encoder(embedded_source)

            current_token = torch.tensor([[start_token]], device=source.device)
            tokens = [start_token]
            
            for _ in range(max_len):
                embedded_token = self.embedding(current_token)
                attn_weights = self.attention(hidden[-1], encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
                rnn_input = torch.cat((embedded_token, context.unsqueeze(1)), dim=2)
                output, (hidden, cell) = self.decoder(rnn_input, (hidden, cell))
                output = output.squeeze(1)
                prediction = self.fc_out(torch.cat((output, context, embedded_token.squeeze(1)), dim=1))
                
                current_token = prediction.argmax(1).unsqueeze(1)
                token = current_token.item()
                
                tokens.append(token)
                
                if token == end_token:
                    break
                    
            return tokens