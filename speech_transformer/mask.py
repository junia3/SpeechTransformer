import torch

def get_attn_pad_mask(inputs, input_lengths, expand_length):
    """
      get attention pad mask for input
      inputs : batch * frame_length * d_model
      input_lengths : torch list for real lengths except silence padding
      expand_length : max length for input frame(shape[1])
      
      return : pad mask looks like batch * expanded_length * frame_length
    """
    def get_transformer_non_pad_mask(inputs, input_lengths):
        batch_size = inputs.size(0)
        
        if len(inputs.size()) == 2:
            non_pad_mask = inputs.new_ones(inputs.size()) ### batch * frame_length
        
        elif len(inputs.size()) == 3:
            non_pad_mask = inputs.new_ones(inputs.size()[:-1]) ### batch * frame_length
        
        else:
            raise ValueError(f"Unsupported input shape {inputs.size()}")
        
        for i in range(batch_size):
            non_pad_mask[i, input_lengths[i]:] = 0
            
        return non_pad_mask
    

    non_pad_mask = get_transformer_non_pad_mask(inputs, input_lengths) ### batch * frame_length
    pad_mask = non_pad_mask.lt(1) ### True if less than 1
    attn_pad_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1) ### batch * 1 * frame_length -> batch * expand_length * frame_length
    return attn_pad_mask



def get_attn_subsequent_mask(seq):
    """
      mask used in decoder module to ignore future context from encoder attention
    """
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal = 1)
    
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
        
    return subsequent_mask
