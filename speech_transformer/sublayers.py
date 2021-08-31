import torch.nn as nn

from speech_transformer.modules import Linear, LayerNorm

class AddNorm(nn.Module): ## Add & Layer Normalization
    """
      module with operates Residual layer + Layer Normalization
      output = inputs propates sublayer
      residual = residual inputs to simply add
      
      return (residual + output) and layer normalize it
    """
    def __init__(self, sublayer, d_model = 256):
        super(AddNorm, self).__init__()
        self.sublayer = sublayer
        self.layer_norm = LayerNorm(dim = d_model)
        
    def forward(self, *args):
        residual = args[0]
        output = self.sublayer(*args)
        
        if isinstance(output, tuple):
            return self.layer_norm(output[0] + residual), output[1]
        
        return self.layer_norm(output + residual)
    

class PositionWiseFeedForwardNet(nn.Module):
    """
      positionwise feed forward net in multihead attention model
      there is ffnet style for 'conv' also, but it is not used in experiment
      
      inputs : batch * frame_length * d_model
      process -> batch * frame_length * d_ff
              -> batch * frame_length * d_model
              -> dropout
    """
    def __init__(self, d_model = 256, d_ff = 1024, dropout_p = 0.3, ffnet_style = 'ff'):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.ffnet_style = ffnet_style.lower()
        if self.ffnet_style == 'ff':
            self.feed_forward = nn.Sequential(
                Linear(d_model, d_ff),
                nn.Dropout(dropout_p),
                nn.ReLU(),
                Linear(d_ff, d_model),
                nn.Dropout(dropout_p),
            )
        
        elif self.ffnet_style == 'conv':
            self.conv1 = nn.Conv1d(in_channels = d_model, out_channels = d_ff, kernel_size = 1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(in_channels = d_ff, out_channels = d_model, kernel_size = 1)
            
        else:
            raise ValueError("Unsupported mode: {0}".format(self.mode))
        
        
    def forward(self, inputs):
        if self.ffnet_style == 'conv':
            outputs = self.conv1(inputs.transpose(1,2))
            outputs = self.relu(outputs)
            return self.conv2(outputs).transpose(1,2)
        return self.feed_forward(inputs)
