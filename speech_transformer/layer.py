import torch.nn as nn
from speech_transformer.attention import MultiheadAttention
from speech_transformer.sublayers import PositionWiseFeedForwardNet, AddNorm

class SpeechTransformerEncoderLayer(nn.Module):
    """
      speech transformer encoder layer
      
      args :
          d_model : model dimension
          num_heads : number of heads
          d_ff : feed forward dimension
          dropout_p : dropout prob
          ffnet_style : feed forwardnet style(ff only)
          
          
      input -> self_attention layer -> add and normalization -> feed forward -> add and normalization
    """
    def __init__(self,
                 d_model = 256,
                 num_heads = 4,
                 d_ff = 1024,
                 dropout_p = 0.3,
                 ffnet_style = 'ff'
                 ):
        super(SpeechTransformerEncoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiheadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff, dropout_p, ffnet_style), d_model)
        
    def forward(self, inputs, self_attn_mask):
        output, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        output = self.feed_forward(output)
        return output, attn
    
class SpeechTransformerDecoderLayer(nn.Module):
    """
      speech transformer decoder layer
      
      args :
          d_model : model dimension
          num_heads : number of heads
          d_ff : feed forward dimension
          dropout_p : dropout prob
          ffnet_style : feed forwardnet style(ff only)
          
          
      input -> self_attention layer -> add and normalization -> memory attention(with encoder output) -> add and normalization -> feed forward -> add and normalization
    """
    def __init__(self,
                 d_model = 256,
                 num_heads = 4,
                 d_ff = 1024,
                 dropout_p = 0.3,
                 ffnet_style = 'ff'):
        super(SpeechTransformerDecoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiheadAttention(d_model, num_heads), d_model)
        self.memory_attention = AddNorm(MultiheadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff, dropout_p, ffnet_style), d_model)
        
    def forward(self,
                inputs,
                memory,
                self_attn_mask = None,
                memory_mask = None):
        
        output, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        output, memory_attn = self.memory_attention(output, memory, memory, memory_mask)
        output = self.feed_forward(output)
        return output, self_attn, memory_attn
