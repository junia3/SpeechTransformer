import torch
import random
import torch.nn as nn
from speech_transformer.embedding import Embedding, PositionalEncoding
from speech_transformer.mask import get_attn_pad_mask, get_attn_subsequent_mask
from speech_transformer.layer import SpeechTransformerEncoderLayer, SpeechTransformerDecoderLayer
from speech_transformer.modules import LayerNorm, Linear

## encoder module for speech transformer
class SpeechTransformerEncoder(nn.Module):
    """
      Class for Encoder module on speech transformer model
      
      arguments : d_model, input_dim, d_ff, num_layers, num_heads, ffnet_style, dropout_p, pad_id
      
          d_model : model dimension
          d_ff : feed forward dimension
          num_layers : number of attention layers
          num_heads : number of heads
          ffnet_style : feed forward style(from this code, only 'ff' used, there is conv else)
          dropout_p : dropout used in each encoder layer
          pad_id : silence index to pad
          
      initialize -----> make same N(num_layers) layers
      
      --------------- forwarding --------------- 
      1.    input : batch * frame_length * features(input_dim)
      2.    input_proj : batch * frame_length * d_model
      3.    get attention mask : mask for padded region
      4.    inputs + positional encoding(absolute encoding)
      5.    dropout
      6.    propagate transfoermer num_layers.
                (first)    input, attention = output from dropout, attention mask
                (else)     input, attention = output from previous layer
    
      --------------- return --------------------
      outputs, input_lenghts
    """
    def __init__(self,
                 d_model = 256,     ## model dimension
                 input_dim = 94,    ## input dimension(feature)
                 d_ff = 1024,       ## feed forward dimension
                 num_layers = 6,    ## number of layer
                 num_heads = 4,     ## number of head
                 ffnet_style = 'ff', ## option for feed forward style
                 dropout_p = 0.3,   ## dropout probability
                 pad_id = 40         ## pad_id
                 ):
        super(SpeechTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pad_id = pad_id
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_proj = Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(p = dropout_p)
        self.layers = nn.ModuleList(
            [SpeechTransformerEncoderLayer(d_model, num_heads, d_ff, dropout_p, ffnet_style) for _ in range(num_layers)]
        )  ## make 6 layers for speech encoder module
        
    def forward(self, inputs, input_lengths):
        inputs = self.input_proj(inputs)
        self_attn_mask = get_attn_pad_mask(inputs, input_lengths, inputs.size(1))  ## get attention pad mask for input data
        outputs = inputs + self.positional_encoding(inputs.size(1)) ## positional encoding on embeddings
        outputs = self.input_dropout(outputs) ## dropout
        
        for layer in self.layers:
            outputs, attn = layer(outputs, self_attn_mask) ## propagate inputs into layers
            
        return outputs, input_lengths


class SpeechTransformerDecoder(nn.Module):
    """
      Class for Dencoder module on speech transformer model
      
      arguments : num_classes, d_model, d_ff, num_layers, num_heads, ffnet_style, dropout_p, pad_id, sos_id, eos_id
      
          num_classes : number of classes(phonemes)
          d_model : model dimension
          d_ff : feed forward dimension
          num_layers : number of attention layers
          num_heads : number of heads
          ffnet_style : feed forward style(from this code, only 'ff' used, there is conv else)
          dropout_p : dropout used in each encoder layer
          pad_id : silence index to pad
          sos_id : start of sequence to pad
          eos_id : end of sequence to pad
          
      initialize -----> Make same N(num_layers) layers, Fully connected layers to make scores for each phoneme classes
      
      
      ******* this decoder module is not implemented for inference mode (only training mode) *******
      
      --------------- forwarding --------------- 
      1.    Make targets except eos_id -> batch * frame_size(idx)
      2.    target length = frame_size
      3.    forward step inputs from forwarding
      
      --------------- forwarding step --------------- 
      1.    get pad mask, subsequent mask(subsequent mask is used to ignore future context)
      2.    get rid of False from pad mask and subsequent mask
      3.    Use this mask to implement same operation on encoder module
      4.    inputs into embedding + positional encoding(absolute encoding)
      5.    dropout
      6.    propagate transfoermer num_layers.
                (first)    input, attention = output from dropout, attention mask
                (else)     input, attention = output from previous layer
                
      --------------- return --------------------
      outputs from forwarding step -> log softmax to calculate probability
      
    """
    def __init__(self,
                 num_classes = 45,  ## class number = phoneme number
                 d_model = 256, ## model dimension
                 d_ff = 1024,  ## feed forward dimension
                 num_layers = 6,  ## number of layer
                 num_heads = 4,  ## number of head
                 ffnet_style = 'ff',  ## option for feed forward style
                 dropout_p = 0.3,  ## dropout
                 pad_id = 40,  ## pad_id
                 sos_id = 43,  ## sos_id
                 eos_id = 44,  ## eos_id
                 ):
        
        super(SpeechTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding = Embedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p = dropout_p)
        self.layers = nn.ModuleList(
            [SpeechTransformerDecoderLayer(d_model, num_heads, d_ff, dropout_p, ffnet_style) for _ in range(num_layers)]
        )
        
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.fc = nn.Sequential(
            LayerNorm(d_model),
            Linear(d_model, num_classes, bias = False),
        )
        
    def forward_step(self, decoder_inputs, decoder_input_lengths, encoder_outputs, encoder_output_lengths, positional_encoding_length):
        dec_self_attn_pad_mask = get_attn_pad_mask(decoder_inputs, decoder_input_lengths, decoder_inputs.size(1))
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_inputs)
        self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
            
        encoder_attn_mask = get_attn_pad_mask(encoder_outputs, encoder_output_lengths, decoder_inputs.size(1))
        outputs= self.embedding(decoder_inputs) + self.positional_encoding(positional_encoding_length)
        outputs = self.input_dropout(outputs)
            
        for layer in self.layers:
          outputs, self_attn, memory_attn = layer(outputs, encoder_outputs, self_attn_mask, encoder_attn_mask)
                
        return outputs
        
    def forward(self, encoder_outputs, targets, encoder_output_lengths, target_lengths):
        batch_size = encoder_outputs.size(0)
        targets = targets[targets != self.eos_id].view(batch_size, -1)
        target_length = targets.size(1)
       
        outputs = self.forward_step(
            decoder_inputs = targets,
            decoder_input_lengths = target_lengths,
            encoder_outputs = encoder_outputs,
            encoder_output_lengths = encoder_output_lengths,
            positional_encoding_length = target_length,
        )
        return self.fc(outputs).log_softmax(dim = -1)
    
