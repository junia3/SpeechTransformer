import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from speech_transformer.modules import Linear
from torch import Tensor
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):

    """
      calculate scaled dot product attention score for batch data
      input : (Batch * head) * length * dim for query, key, value
      output : context, attention score
    """
    def __init__(self, dim: int) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask, -1e9)  ## apply maks on score query * key

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        
        
        return context, attn


class MultiheadAttention(nn.Module):
    """
      Multihead attention module used to calculate attention
      inputs : batch * frame_size * dim
      
      ----------- process ------------------
      1.    get query, key, value
            batch * frame_size * dim -> batch * frame_size * d_head * num_heads
            -> batch * frame_size * num_heads * d_head
            
      2.    change dim (join the tensor with head * batch)
            batch * frame_size * num_heads * d_head -> [batch * num_heads] * frame_size * d_head
    
      3.    Mask should be enlarged(repeat) with num_heads
      
      4.    context, attention -> attention = softmax score with key, query    ///    context = softmax score by value 
    """
    def __init__(self, d_model: int = 256, num_heads: int = 4) -> None:
        super(MultiheadAttention, self).__init__()

        assert d_model % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.query_proj = Linear(d_model, self.d_head * num_heads)
        self.key_proj = Linear(d_model, self.d_head * num_heads)
        self.value_proj = Linear(d_model, self.d_head * num_heads)
        self.sqrt_dim = np.sqrt(d_model)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)        # BxK_LENxNxD
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD
        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        
        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND

        return context, attn
