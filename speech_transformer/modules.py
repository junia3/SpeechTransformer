import torch
import torch.nn as nn
import torch.nn.init as init

class SubSampling(nn.Module):
    """
      Subsampling module
      120(previous features) + 40 features(current) + 120(future features) from each frames
      conv1d with stride 3 to get 94(280+2 // 3) of dimension
    """
    def __init__(self, frame_number):
        super(SubSampling, self).__init__()
        self.conv = nn.Conv1d(frame_number, frame_number, 1, stride = 3) ### convolution 1D layer for subsampling
    
    def forward(self, speech):
        batch, frame, feature = speech.shape
        prev1, prev2, prev3 = speech[:, :-1, :], speech[:, :-2, :], speech[:, :-3, :]  ### previous data should get rid off future data
        next1, next2, next3 = speech[:, 1:, :], speech[:, 2:, :], speech[:, 3:, :] ### future data should get rid off previous data
        
        prev1 = torch.cat((torch.zeros(batch, 1, feature), prev1), axis = 1)
        prev2 = torch.cat((torch.zeros(batch, 2, feature), prev2), axis = 1)
        prev3 = torch.cat((torch.zeros(batch, 3, feature), prev3), axis = 1)
        
        next1 = torch.cat((next1, torch.zeros(batch, 1, feature)), axis = 1)
        next2 = torch.cat((next2, torch.zeros(batch, 2, feature)), axis = 1)
        next3 = torch.cat((next3, torch.zeros(batch, 3, feature)), axis = 1)
        
        speech_outputs = self.conv(torch.cat((prev3, prev2, prev1, speech, next1, next2, next3), axis = 2))
        
        return speech_outputs

class Linear(nn.Module): ### linear module wrapped with nn.module
    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias = bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)
    
    
class LayerNorm(nn.Module): ## layer normalization module
    def __init__(self, dim, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(dim = 1, keepdim = True)
        std = inputs.std(dim = 1, keepdim = True)
        outputs = (inputs - mean) / (std + self.eps)
        outputs = self.gamma * outputs + self.beta
    
        return outputs
    
class Transpose(nn.Module): ## transpose module
    def __init__(self, shape):
        super(Transpose, self).__init__()
        self.shape = shape
        
    def forward(self, inputs):
        return inputs.transpose(*self.shape)
    
