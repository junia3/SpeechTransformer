import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    ### positional encoding layer ###
    def __init__(self, d_model = 256, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad = False)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:, :length]
    
    
    
class Embedding(nn.Module):
    ### make embeddings ###
    def __init__(self, num_embeddings, pad_id, d_model = 256):
        super(Embedding, self).__init__()
        self.sqrt_dim = math.sqrt(d_model)
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx = pad_id)
    
    def forward(self, inputs):
        return self.embedding(inputs) * self.sqrt_dim
